// npuprobe: Intel NPU through Level Zero + the driver's graph extension, no OpenVINO runtime. Builds an OpenVINO IR
// (XML + weights) in memory for one GEMM, Y [M][N] = X [M][K] . W [N][K]^T in fp16, wraps it in the driver's
// NGRAPH_LITE container, has the in-driver compiler build it, runs it, checks rows against a CPU reference and
// reports the NPU's GEMM rate. Development tool: the ze_api.h / ze_graph_ext.h headers come from Intel's public
// level-zero and level-zero-npu-extensions repositories.
//   gcc -O2 -march=native -I<dir with ze_api.h, ze_graph_ext.h> src/tools/npuprobe.c C:/Windows/System32/ze_loader.dll
//       -o build/npuprobe
//   build/npuprobe [M N K iters]
#include <immintrin.h>
#include <math.h>
#include <stdbool.h> // ze_graph_ext.h uses bool without including it
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <ze_api.h>
#include <ze_graph_ext.h>

#define ZK(call) do { ze_result_t r_ = (call); if (r_) { fprintf(stderr, "%s: 0x%x\n", #call, (unsigned)r_); return 1; } } while (0)

// One fp16 GEMM as OpenVINO IR v11: Parameter x [M,K] -> MatMul(x, Const W [N,K], transpose_b) -> Result [M,N]
static size_t ir_xml(char *x, size_t cap, uint32_t M, uint32_t N, uint32_t K)
{
    return (size_t)snprintf(x, cap,
        "<?xml version=\"1.0\"?>\n<net name=\"gemm\" version=\"11\">\n<layers>\n"
        "<layer id=\"0\" name=\"x\" type=\"Parameter\" version=\"opset1\"><data shape=\"%u,%u\" element_type=\"f16\"/>"
        "<output><port id=\"0\" precision=\"FP16\" names=\"x\"><dim>%u</dim><dim>%u</dim></port></output></layer>\n"
        "<layer id=\"1\" name=\"w\" type=\"Const\" version=\"opset1\"><data element_type=\"f16\" shape=\"%u,%u\" offset=\"0\" size=\"%zu\"/>"
        "<output><port id=\"0\" precision=\"FP16\"><dim>%u</dim><dim>%u</dim></port></output></layer>\n"
        "<layer id=\"2\" name=\"y\" type=\"MatMul\" version=\"opset1\"><data transpose_a=\"false\" transpose_b=\"true\"/>"
        "<input><port id=\"0\" precision=\"FP16\"><dim>%u</dim><dim>%u</dim></port><port id=\"1\" precision=\"FP16\"><dim>%u</dim><dim>%u</dim></port></input>"
        "<output><port id=\"2\" precision=\"FP16\" names=\"y\"><dim>%u</dim><dim>%u</dim></port></output></layer>\n"
        "<layer id=\"3\" name=\"y/sink\" type=\"Result\" version=\"opset1\">"
        "<input><port id=\"0\" precision=\"FP16\"><dim>%u</dim><dim>%u</dim></port></input></layer>\n"
        "</layers>\n<edges>\n"
        "<edge from-layer=\"0\" from-port=\"0\" to-layer=\"2\" to-port=\"0\"/>\n"
        "<edge from-layer=\"1\" from-port=\"0\" to-layer=\"2\" to-port=\"1\"/>\n"
        "<edge from-layer=\"2\" from-port=\"2\" to-layer=\"3\" to-port=\"0\"/>\n"
        "</edges>\n</net>\n",
        M, K, M, K, N, K, (size_t)N * K * 2, N, K, M, K, N, K, M, N, M, N);
}

static uint32_t rs = 777;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (float)(rs >> 8) / 8388608.0f - 1.0f; }

int main(int argc, char **argv)
{
    ze_driver_handle_t drv;
    ze_device_handle_t dev;
    ze_context_handle_t ctx;
    ze_command_queue_handle_t q;
    ze_command_list_handle_t cl;
    ze_device_properties_t dp;
    ze_driver_extension_properties_t *ext;
    ze_graph_dditable_ext_t *g;
    ze_device_graph_properties_t gp;
    ze_graph_handle_t graph;
    ze_graph_properties_t prop;
    ze_graph_argument_properties_t ap;
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                  ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, 0, 0};
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
    ze_graph_desc_t desc;
    LARGE_INTEGER q0, q1, qf;
    uint16_t *w, *x, *y;
    uint8_t *blob;
    char *xml, flags[512], name[ZE_MAX_EXTENSION_NAME], *log;
    double s, err, mx, sec;
    uint64_t xs, ws;
    uint32_t n, i, j, k, M, N, K, iters, two, best, logn;
    void *in, *out;

    M = argc > 3 ? (uint32_t)atoi(argv[1]) : 1025;
    N = argc > 3 ? (uint32_t)atoi(argv[2]) : 5472;
    K = argc > 3 ? (uint32_t)atoi(argv[3]) : 1024;
    iters = argc > 4 ? (uint32_t)atoi(argv[4]) : 20;
    ZK(zeInit(ZE_INIT_FLAG_VPU_ONLY));
    n = 1;
    ZK(zeDriverGet(&n, &drv));
    n = 1;
    ZK(zeDeviceGet(drv, &n, &dev));
    dp.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    dp.pNext = NULL;
    ZK(zeDeviceGetProperties(dev, &dp));
    printf("device: %s (type %d), %u MHz\n", dp.name, (int)dp.type, dp.coreClockRate);
    n = 0;
    ZK(zeDriverGetExtensionProperties(drv, &n, NULL));
    ext = calloc(n, sizeof(*ext));
    ZK(zeDriverGetExtensionProperties(drv, &n, ext));
    for (i = 0, best = n; i < n; i++)
        if (!strncmp(ext[i].name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME))
            && (best == n || ext[i].version > ext[best].version)) best = i;
    if (best == n) {
        fprintf(stderr, "no %s extension\n", ZE_GRAPH_EXT_NAME);
        return 1;
    }
    strcpy(name, ext[best].name);
    printf("graph extension: %s (version %u.%u)\n", name, ZE_MAJOR_VERSION(ext[best].version),
           ZE_MINOR_VERSION(ext[best].version));
    ZK(zeDriverGetExtensionFunctionAddress(drv, name, (void **)&g));
    memset(&gp, 0, sizeof(gp));
    gp.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    ZK(g->pfnDeviceGetGraphProperties(dev, &gp));
    printf("compiler %u.%u, max opset %u, formats 0x%x\n", gp.compilerVersion.major, gp.compilerVersion.minor,
           gp.maxOVOpsetVersionSupported, (unsigned)gp.graphFormatsSupported);

    // weights and the NGRAPH_LITE container: [compiler version][u32 2][u64 xml][xml][u64 weights][weights]
    xml = malloc(1 << 16);
    xs = ir_xml(xml, 1 << 16, M, N, K);
    ws = (uint64_t)N * K * 2;
    blob = malloc(4 + 4 + 8 + xs + 8 + ws);
    w = (uint16_t *)(blob + 4 + 4 + 8 + xs + 8);
    for (i = 0; i < N * K; i++) w[i] = (uint16_t)_cvtss_sh(0.05f * frand(), 0);
    two = 2;
    memcpy(blob, &gp.compilerVersion, 4);
    memcpy(blob + 4, &two, 4);
    memcpy(blob + 8, &xs, 8);
    memcpy(blob + 16, xml, xs);
    memcpy(blob + 16 + xs, &ws, 8);
    snprintf(flags, sizeof(flags), "--inputs_precisions=\"0:FP16\" --inputs_layouts=\"0:NC\" "
             "--outputs_precisions=\"0:FP16\" --outputs_layouts=\"0:NC\" --config PERFORMANCE_HINT=\"THROUGHPUT\"");
    ZK(zeContextCreate(drv, &cd, &ctx));
    memset(&desc, 0, sizeof(desc));
    desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
    desc.format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
    desc.inputSize = 4 + 4 + 8 + xs + 8 + ws;
    desc.pInput = blob;
    desc.pBuildFlags = flags;
    QueryPerformanceFrequency(&qf);
    QueryPerformanceCounter(&q0);
    if (g->pfnCreate(ctx, dev, &desc, &graph)) {
        logn = 0;
        g->pfnBuildLogGetString(NULL, &logn, NULL);
        log = malloc(logn + 1);
        g->pfnBuildLogGetString(NULL, &logn, log);
        fprintf(stderr, "compile failed:\n%.*s\n", (int)logn, log);
        return 1;
    }
    QueryPerformanceCounter(&q1);
    printf("compiled in %.1f ms\n", (double)(q1.QuadPart - q0.QuadPart) * 1e3 / qf.QuadPart);
    prop.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    prop.pNext = NULL;
    ZK(g->pfnGetProperties(graph, &prop));
    for (i = 0; i < prop.numGraphArgs; i++) {
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        ap.pNext = NULL;
        ZK(g->pfnGetArgumentProperties(graph, i, &ap));
        printf("arg %u %-6s %s dims %u x %u, device precision 0x%x layout 0x%x\n", i, ap.name,
               ap.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT ? "in " : "out", ap.dims[0], ap.dims[1],
               (unsigned)ap.devicePrecision, (unsigned)ap.deviceLayout);
    }
    ZK(zeMemAllocHost(ctx, &hd, (size_t)M * K * 2, 64, &in));
    ZK(zeMemAllocHost(ctx, &hd, (size_t)M * N * 2, 64, &out));
    x = in;
    y = out;
    for (i = 0; i < M * K; i++) x[i] = (uint16_t)_cvtss_sh(frand(), 0);
    ZK(g->pfnSetArgumentValue(graph, 0, in));
    ZK(g->pfnSetArgumentValue(graph, 1, out));
    ZK(zeCommandQueueCreate(ctx, dev, &qd, &q));
    ZK(zeCommandListCreate(ctx, dev, &ld, &cl));
    ZK(g->pfnAppendGraphInitialize(cl, graph, NULL, 0, NULL));
    ZK(zeCommandListClose(cl));
    ZK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL));
    ZK(zeCommandQueueSynchronize(q, UINT64_MAX));
    ZK(zeCommandListReset(cl));
    ZK(g->pfnAppendGraphExecute(cl, graph, NULL, NULL, 0, NULL));
    ZK(zeCommandListClose(cl));
    ZK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL)); // warm
    ZK(zeCommandQueueSynchronize(q, UINT64_MAX));
    QueryPerformanceCounter(&q0);
    for (i = 0; i < iters; i++) ZK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL));
    ZK(zeCommandQueueSynchronize(q, UINT64_MAX));
    QueryPerformanceCounter(&q1);
    sec = (double)(q1.QuadPart - q0.QuadPart) / qf.QuadPart / iters;
    for (i = 0, err = mx = 0; i < 16; i++) {
        j = (i * 2654435761u) % M;
        for (k = 0; k < N; k += 37) {
            for (n = 0, s = 0; n < K; n++) s += (double)_cvtsh_ss(x[(size_t)j * K + n]) * _cvtsh_ss(w[(size_t)k * K + n]);
            err = fabs(_cvtsh_ss(y[(size_t)j * N + k]) - s) > err ? fabs(_cvtsh_ss(y[(size_t)j * N + k]) - s) : err;
            mx = fabs(s) > mx ? fabs(s) : mx;
        }
    }
    printf("GEMM %u x %u x %u: %.3f ms per run, %.2f TFLOPS fp16; max |err| %.4f (max |ref| %.3f)\n", M, N, K,
           sec * 1e3, 2.0 * M * N * K / sec / 1e12, err, mx);
    return 0;
}

// npulayer: one EVA02 block on the NPU in pieces (engine/npu.h IR builder, random weights), each piece compiled and
// timed alone, to see which ops the NPU runs badly. Development tool.
//   gcc -O2 -march=native -Isrc/engine -Isrc/third_party/level_zero -isystem <cuda include> src/tools/npulayer.c
//       -o build/npulayer
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "npu.h"

static uint32_t rs = 99;
static double now(void)
{
    LARGE_INTEGER q, f;

    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

static nv wconst(nir *r, uint32_t n, uint32_t k, uint16_t *buf)
{
    uint32_t i;

    for (i = 0; i < n * k; i++) {
        rs = rs * 1664525u + 1013904223u;
        buf[i] = (uint16_t)_cvtss_sh(0.03f * ((float)(rs >> 8) / 8388608.0f - 1.0f), 0);
    }
    return nir_const(r, "f16", 2, NIR_D(n, k), buf, (size_t)n * k * 2);
}

// variant bits: 1 rope, 2 attention (else o = v), 4 MLP, 8 qkv + out-proj
static void build(nir *r, uint32_t var, uint16_t *buf, const float *one, const float *zero, const float *Ct)
{
    nv x, h, q, k, v, s, o, C, S, perm, g;
    float ln2;

    ln2 = 0.693147f;
    NIR_X(r, "<?xml version=\"1.0\"?>\n<net name=\"l\" version=\"11\">\n<layers>\n");
    x = nir_param(r, "x", "f16", 2, NIR_D(EVA_T, EVA_D));
    C = nir_consth(r, 2, NIR_D(EVA_T, EVA_D), Ct, (size_t)EVA_T * EVA_D);
    S = nir_consth(r, 2, NIR_D(EVA_T, EVA_D), Ct, (size_t)EVA_T * EVA_D);
    perm = nir_i64(r, 3, (const int64_t[]){1, 0, 2});
    if (var & 8) {
        h = nir_ln(r, x, one, zero, EVA_D, 1e-6f);
        q = NIR_MM(r, h, wconst(r, EVA_D, EVA_D, buf));
        k = NIR_MM(r, h, wconst(r, EVA_D, EVA_D, buf));
        v = NIR_MM(r, h, wconst(r, EVA_D, EVA_D, buf));
        if (var & 1) { // pair swap folded into its own GEMM, as npu_eva_ir does
            q = npu_rope(r, q, NIR_MM(r, h, wconst(r, EVA_D, EVA_D, buf)), C, S);
            k = npu_rope(r, k, NIR_MM(r, h, wconst(r, EVA_D, EVA_D, buf)), C, S);
        }
        if (var & 2) {
            q = npu_heads(r, q, perm);
            k = npu_heads(r, k, perm);
            v = npu_heads(r, v, perm);
            s = nir_op(r, "MatMul", "opset1", "transpose_a=\"false\" transpose_b=\"true\"", NIR_IN(q, k), 2, "f16", 3,
                       NIR_D(EVA_NH, EVA_T, EVA_T));
            s = NIR_BIN(r, "Multiply", s, nir_consth(r, 1, NIR_D(1), &ln2, 1));
            s = nir_op(r, "Softmax", "opset8", "axis=\"-1\"", NIR_IN(s), 1, "f16", 3, s.d);
            o = nir_op(r, "MatMul", "opset1", "transpose_a=\"false\" transpose_b=\"false\"", NIR_IN(s, v), 2, "f16", 3,
                       NIR_D(EVA_NH, EVA_T, 64));
            o = nir_op(r, "Transpose", "opset1", "", NIR_IN(o, perm), 2, "f16", 3, NIR_D(EVA_T, EVA_NH, 64));
            o = NIR_RESHAPE(r, o, 2, EVA_T, EVA_D);
        } else {
            o = NIR_BIN(r, "Add", NIR_BIN(r, "Add", q, k), v);
        }
        x = NIR_BIN(r, "Add", x, NIR_MM(r, o, wconst(r, EVA_D, EVA_D, buf)));
    }
    if (var & 4) {
        h = nir_ln(r, x, one, zero, EVA_D, 1e-6f);
        g = NIR_MM(r, h, wconst(r, NPU_HID, EVA_D, buf));
        g = nir_op(r, "Swish", "opset4", "", NIR_IN(g), 1, "f16", 2, g.d);
        g = NIR_BIN(r, "Multiply", g, NIR_MM(r, h, wconst(r, NPU_HID, EVA_D, buf)));
        g = nir_ln(r, g, one, zero, NPU_HID, 1e-6f);
        x = NIR_BIN(r, "Add", x, NIR_MM(r, g, wconst(r, EVA_D, NPU_HID, buf)));
    }
    nir_result(r, x);
    NIR_X(r, "</layers>\n<edges>\n%.*s</edges>\n</net>\n", (int)r->en, r->e);
}

int main(void)
{
    static const uint32_t vars[] = {8, 8 | 1, 8 | 2, 8 | 1 | 2, 4};
    static const char *names[] = {"qkv+oproj", "+rope", "+attention", "+rope+attention", "mlp"};
    npu n;
    ze_driver_extension_properties_t ext[64];
    ze_device_graph_properties_t gp;
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                  ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, 0, 0};
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
    ze_graph_desc_t d;
    ze_graph_handle_t gr;
    ze_command_list_handle_t cl;
    nir r;
    uint16_t *buf;
    float *one, *zero, *Ct;
    uint8_t *blob;
    void *in, *out;
    double t0, t1;
    uint64_t xs, ws;
    uint32_t cnt, i, best, v, two, it;

    memset(&n, 0, sizeof(n));
    n.dll = LoadLibraryA("ze_loader.dll");
#define NPU_LOAD(f) n.z.f = (__typeof__(&f))(void (*)(void))GetProcAddress(n.dll, #f);
    NPU_ZE(NPU_LOAD)
#undef NPU_LOAD
    n.z.zeInit(ZE_INIT_FLAG_VPU_ONLY);
    cnt = 1;
    n.z.zeDriverGet(&cnt, &n.drv);
    cnt = 1;
    n.z.zeDeviceGet(n.drv, &cnt, &n.dev);
    cnt = 64;
    n.z.zeDriverGetExtensionProperties(n.drv, &cnt, ext);
    for (i = 0, best = 0; i < cnt; i++)
        if (!strncmp(ext[i].name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME))) best = i;
    n.z.zeDriverGetExtensionFunctionAddress(n.drv, ext[best].name, (void **)&n.g);
    memset(&gp, 0, sizeof(gp));
    gp.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    n.g->pfnDeviceGetGraphProperties(n.dev, &gp);
    n.z.zeContextCreate(n.drv, &cd, &n.ctx);
    n.z.zeCommandQueueCreate(n.ctx, n.dev, &qd, &n.q);
    buf = malloc((size_t)NPU_HID * EVA_D * 2);
    one = malloc(NPU_HID * 4);
    zero = calloc(NPU_HID, 4);
    Ct = malloc((size_t)EVA_T * EVA_D * 4);
    for (i = 0; i < NPU_HID; i++) one[i] = 1.0f;
    for (i = 0; i < EVA_T * EVA_D; i++) Ct[i] = 0.5f;
    n.z.zeMemAllocHost(n.ctx, &hd, (size_t)EVA_T * EVA_D * 2, 64, &in);
    n.z.zeMemAllocHost(n.ctx, &hd, (size_t)EVA_T * EVA_D * 2, 64, &out);
    memset(in, 0, (size_t)EVA_T * EVA_D * 2);
    for (v = 0; v < sizeof(vars) / sizeof(vars[0]); v++) {
        memset(&r, 0, sizeof(r));
        build(&r, vars[v], buf, one, zero, Ct);
        xs = r.xn;
        ws = r.wn;
        blob = malloc(24 + xs + ws);
        two = 2;
        memcpy(blob, &gp.compilerVersion, 4);
        memcpy(blob + 4, &two, 4);
        memcpy(blob + 8, &xs, 8);
        memcpy(blob + 16, r.x, xs);
        memcpy(blob + 16 + xs, &ws, 8);
        memcpy(blob + 24 + xs, r.w, ws);
        memset(&d, 0, sizeof(d));
        d.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
        d.format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
        d.inputSize = 24 + xs + ws;
        d.pInput = blob;
        d.pBuildFlags = "--inputs_precisions=\"0:FP16\" --inputs_layouts=\"0:NC\" --outputs_precisions=\"0:FP16\" "
                        "--outputs_layouts=\"0:NC\" --config PERFORMANCE_HINT=\"THROUGHPUT\"";
        if (n.g->pfnCreate(n.ctx, n.dev, &d, &gr)) {
            printf("%-18s compile failed\n", names[v]);
            continue;
        }
        n.z.zeCommandListCreate(n.ctx, n.dev, &ld, &cl);
        n.g->pfnAppendGraphInitialize(cl, gr, NULL, 0, NULL);
        n.z.zeCommandListClose(cl);
        n.z.zeCommandQueueExecuteCommandLists(n.q, 1, &cl, NULL);
        n.z.zeCommandQueueSynchronize(n.q, UINT64_MAX);
        n.z.zeCommandListDestroy(cl);
        n.g->pfnSetArgumentValue(gr, 0, in);
        n.g->pfnSetArgumentValue(gr, 1, out);
        n.z.zeCommandListCreate(n.ctx, n.dev, &ld, &cl);
        n.g->pfnAppendGraphExecute(cl, gr, NULL, NULL, 0, NULL);
        n.z.zeCommandListClose(cl);
        n.z.zeCommandQueueExecuteCommandLists(n.q, 1, &cl, NULL);
        n.z.zeCommandQueueSynchronize(n.q, UINT64_MAX);
        t0 = now();
        for (it = 0; it < 5; it++) n.z.zeCommandQueueExecuteCommandLists(n.q, 1, &cl, NULL);
        n.z.zeCommandQueueSynchronize(n.q, UINT64_MAX);
        t1 = now();
        printf("%-18s %8.2f ms per block\n", names[v], (t1 - t0) * 1e3 / 5);
        n.z.zeCommandListDestroy(cl);
        n.g->pfnDestroy(gr);
        free(blob);
        free(r.x);
        free(r.e);
        free(r.w);
    }
    return 0;
}

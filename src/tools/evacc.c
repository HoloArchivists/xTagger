// evacc: eva_kernels.cu -> cubin for device 0's architecture (NVRTC). Build-time only.
//   evacc src/eva_kernels.cu build/eva_kernels.cubin <cuda include dir> [nvrtc options...]
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <nvrtc.h>

int main(int argc, char **argv)
{
    FILE *f;
    char *src, *log, arch[32], inc[1024];
    const char *opts[16];
    nvrtcProgram prog;
    nvrtcResult rc;
    CUdevice dev;
    size_t n;
    long len;
    int major, minor, i, no;

    if (argc < 4) {
        fprintf(stderr, "usage: evacc in.cu out.cubin cuda_include [nvrtc options...]\n");
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    src = malloc((size_t)len + 1);
    if ((long)fread(src, 1, (size_t)len, f) != len) return 1;
    src[len] = 0;
    fclose(f);
    if (cuInit(0) || cuDeviceGet(&dev, 0)
        || cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
        || cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)) {
        fprintf(stderr, "evacc: no CUDA device\n");
        return 1;
    }
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    snprintf(inc, sizeof(inc), "--include-path=%s", argv[3]);
    opts[0] = arch;
    opts[1] = inc;
    opts[2] = "--std=c++17";
    opts[3] = "--use_fast_math";
    opts[4] = "--extra-device-vectorization";
    for (i = 4, no = 5; i < argc && no < 16; i++) opts[no++] = argv[i]; // extra options, e.g. -DEVA_FP8
    nvrtcCreateProgram(&prog, src, "eva_kernels.cu", 0, NULL, NULL);
    rc = nvrtcCompileProgram(prog, no, opts);
    nvrtcGetProgramLogSize(prog, &n);
    if (n > 1) {
        log = malloc(n);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "%s\n", log);
    }
    if (rc != NVRTC_SUCCESS) return 1;
    nvrtcGetCUBINSize(prog, &n);
    src = realloc(src, n);
    nvrtcGetCUBIN(prog, src);
    f = fopen(argv[2], "wb");
    if (!f || fwrite(src, 1, n, f) != n) return 1;
    fclose(f);
    printf("%s: sm_%d%d, %zu bytes\n", argv[2], major, minor, n);
    return 0;
}

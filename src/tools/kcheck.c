// kcheck: the engines' own GEMM kernels (k_pe, k_oproj, k_headmv from eva_kernels.cubin) against a double-precision
// CPU reference on random operands. Build-time / debugging tool, not part of xtag.
//   gcc -O2 -march=native -I<cuda include> src/tools/kcheck.c -L<cuda lib> -lcuda -o build/kcheck
//   build/kcheck build/eva_kernels.cubin
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define D    1024
#define T    1025
#define NB   8
#define NP   1024
#define K0P  592
#define NT   10861

#define CK(call) do { CUresult r_ = (call); if (r_) { fprintf(stderr, "%s: %d\n", #call, (int)r_); exit(1); } } while (0)

static uint32_t rs = 12345;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (float)(rs >> 8) / 8388608.0f - 1.0f; }
static uint16_t h(float f) { return (uint16_t)_cvtss_sh(f, 0); }
static float hf(uint16_t v) { return _cvtsh_ss(v); }
static float e4m3(uint8_t b)
{
    int s, e, m;

    s = b >> 7;
    e = (b >> 3) & 15;
    m = b & 7;
    return (s ? -1.0f : 1.0f) * (e ? ldexpf(1.0f + m / 8.0f, e - 7) : ldexpf(m / 8.0f, -6));
}

static void *dl(CUdeviceptr *d, size_t n, const void *src)
{
    CK(cuMemAlloc(d, n));
    if (src) CK(cuMemcpyHtoD(*d, src, n));
    else CK(cuMemsetD8(*d, 0, n));
    return NULL;
}

int main(int argc, char **argv)
{
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction f;
    CUdeviceptr din, dpe, dx, da, dw, dhw, dhb, dout;
    uint16_t *in, *pe, *x, *x0, *hw, *pn;
    uint8_t *a, *w;
    float *hb, *out, alpha, err, ref, mx;
    double s;
    FILE *fp;
    long len;
    char *cub;
    uint32_t i, j, k, m, rows, xr, nt, nb, idx;

    if (argc < 2) return 2;
    fp = fopen(argv[1], "rb");
    fseek(fp, 0, SEEK_END);
    len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    cub = malloc(len);
    fread(cub, 1, len, fp);
    fclose(fp);
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CK(cuModuleLoadData(&mod, cub));

    // k_pe: x rows m + m / 1024 + 1 = in [8192][592] . pe [1024][592]^T
    in = malloc((size_t)NB * NP * K0P * 2);
    pe = malloc((size_t)D * K0P * 2);
    x = malloc((size_t)NB * T * D * 2);
    for (i = 0; i < NB * NP; i++)
        for (k = 0; k < K0P; k++) in[(size_t)i * K0P + k] = h(k < 588 ? frand() : 0.0f);
    for (i = 0; i < D; i++)
        for (k = 0; k < K0P; k++) pe[(size_t)i * K0P + k] = h(k < 588 ? 0.05f * frand() : 0.0f);
    dl(&din, (size_t)NB * NP * K0P * 2, in);
    dl(&dpe, (size_t)D * K0P * 2, pe);
    dl(&dx, (size_t)NB * T * D * 2, NULL);
    rows = NB * NP;
    CK(cuModuleGetFunction(&f, mod, "k_pe"));
    CK(cuLaunchKernel(f, D / 128, rows / 128, 1, 128, 1, 1, 0, 0, (void *[]){&din, &dpe, &dx, &rows}, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(x, dx, (size_t)NB * T * D * 2));
    for (i = 0, mx = 0, err = 0; i < 97; i++) {
        m = (i * 7919u) % rows;
        xr = m + m / NP + 1;
        for (j = 0; j < D; j++) {
            for (k = 0, s = 0; k < K0P; k++) s += (double)hf(in[(size_t)m * K0P + k]) * hf(pe[(size_t)j * K0P + k]);
            ref = (float)s;
            err = fmaxf(err, fabsf(hf(x[(size_t)xr * D + j]) - ref));
            mx = fmaxf(mx, fabsf(ref));
        }
    }
    printf("k_pe     max |err| %.5f  (max |ref| %.3f)\n", err, mx);

    // k_oproj: x [8200][1024] += alpha a W^T, E4M3 a / W
    rows = NB * T;
    a = malloc((size_t)rows * D);
    w = malloc((size_t)D * D);
    x0 = malloc((size_t)NB * T * D * 2);
    for (i = 0; i < rows * D; i++) { do a[i] = (uint8_t)((rs = rs * 1664525u + 1013904223u) >> 24); while ((a[i] & 0x7F) == 0x7F || ((a[i] >> 3) & 15) > 9); }
    for (i = 0; i < D * D; i++) { do w[i] = (uint8_t)((rs = rs * 1664525u + 1013904223u) >> 24); while ((w[i] & 0x7F) == 0x7F || ((w[i] >> 3) & 15) > 9); }
    for (i = 0; i < rows * D; i++) x0[i] = h(frand());
    dl(&da, (size_t)rows * D, a);
    dl(&dw, (size_t)D * D, w);
    CK(cuMemcpyHtoD(dx, x0, (size_t)rows * D * 2));
    alpha = 0.001f;
    CK(cuModuleGetFunction(&f, mod, "k_oproj"));
    CK(cuLaunchKernel(f, D / 128, (rows + 127) / 128, 1, 128, 1, 1, 0, 0, (void *[]){&da, &dw, &dx, &alpha, &rows}, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(x, dx, (size_t)rows * D * 2));
    for (i = 0, mx = 0, err = 0; i < 97; i++) {
        m = (i * 7919u) % rows;
        for (j = 0; j < D; j++) {
            for (k = 0, s = 0; k < D; k++) s += (double)e4m3(a[(size_t)m * D + k]) * e4m3(w[(size_t)j * D + k]);
            ref = (float)(alpha * s + hf(x0[(size_t)m * D + j]));
            err = fmaxf(err, fabsf(hf(x[(size_t)m * D + j]) - ref));
            mx = fmaxf(mx, fabsf(ref));
        }
    }
    printf("k_oproj  max |err| %.5f  (max |ref| %.3f)\n", err, mx);

    // k_headmv: out [8][NT] = sigmoid(pn . hw^T + hb)
    nt = NT;
    nb = NB;
    hw = malloc((size_t)(NT + 7) * D * 2);
    pn = malloc((size_t)NB * D * 2);
    hb = malloc((size_t)(NT + 7) * 4);
    out = malloc((size_t)NB * NT * 4);
    for (i = 0; i < (NT + 7) * D; i++) hw[i] = h(0.05f * frand());
    for (i = 0; i < NB * D; i++) pn[i] = h(frand());
    for (i = 0; i < NT + 7; i++) hb[i] = frand();
    dl(&dhw, (size_t)(NT + 7) * D * 2, hw);
    dl(&da, (size_t)NB * D * 2, pn);
    dl(&dhb, (size_t)(NT + 7) * 4, hb);
    dl(&dout, (size_t)NB * NT * 4, NULL);
    CK(cuModuleGetFunction(&f, mod, "k_headmv"));
    CK(cuLaunchKernel(f, (nt + 63) / 64, 1, 1, 256, 1, 1, 0, 0, (void *[]){&da, &dhw, &dhb, &dout, &nt, &nb}, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(out, dout, (size_t)NB * NT * 4));
    for (idx = 0, err = 0; idx < NB * NT; idx += 13) {
        i = idx / NT;
        j = idx % NT;
        for (k = 0, s = 0; k < D; k++) s += (double)hf(pn[i * D + k]) * hf(hw[(size_t)j * D + k]);
        ref = (float)(1.0 / (1.0 + exp(-(s + hb[j]))));
        err = fmaxf(err, fabsf(out[idx] - ref));
    }
    printf("k_headmv max |err| %.6f\n", err);
    return 0;
}

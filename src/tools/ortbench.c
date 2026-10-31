// ortbench: the baseline for xtag's numbers - the same images through stock ONNX Runtime (CUDA execution provider)
// running the upstream ONNX models, with each model's reference preprocessing. Whole images only.
//
//   ortbench wd model.onnx <dir>     WD EVA02-L v3: alpha over white, white square pad, PIL bicubic to 448, uint8,
//                                    BGR float 0..255, NHWC, batches of 8 (the model's batch is dynamic)
//   ortbench px model_pixai.onnx <dir>  PixAI: alpha over white, aspect fit to 1008 (PIL bilinear), uint8, black pad,
//                                    RGB (x / 255 - 0.5) / 0.5, NCHW, batch 1 (the graph fixes it)
//
// Decoding (stb_image) and preprocessing run on 8 worker threads, as in xtag, so the main thread only feeds
// ONNX Runtime: the comparison is the inference engine, not the decoder. Prints a stats board like xtag -sT
// (wall clock from start, session creation included) plus the time spent inside OrtRun.
// Needs onnxruntime.dll + onnxruntime_providers_*.dll next to the exe and cuDNN on PATH.

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "third_party/stb_image.h"
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <onnxruntime_c_api.h>

#define OB_RING    16
#define OB_WORKERS 8
#define OB_MAXB    8

typedef struct {
    float *x;               // one preprocessed model input
    int ok;
    volatile LONG seq;      // index + 1 once ready
} ob_slot;

static char **paths;
static uint32_t npaths, cap, px, S;
static size_t inlen;        // floats per model input
static ob_slot ring[OB_RING];
static volatile LONG next_job, consumed;
static volatile LONG64 st_px, st_bytes;

static float ob_cubic(float x) // PIL BICUBIC, a = -0.5
{
    x = fabsf(x);
    return x < 1.0f ? (1.5f * x - 2.5f) * x * x + 1.0f : x < 2.0f ? ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f : 0.0f;
}

static float ob_tri(float x)
{
    x = fabsf(x);
    return x < 1.0f ? 1.0f - x : 0.0f;
}

// One output axis of a PIL-style antialiased resample. Output i covers source [lo[i], lo[i] + cnt[i]) with weights
// w[i * k ...]; padw[i] is the weight on pad samples. Content of length len sits at [off, off + len) of a padded
// source axis of length full, resampled to n outputs placed at [o0, o0 + n) of an axis of S (outside: pad only).
typedef struct { int *lo, *cnt; float *w, *padw; int k; } ob_axis;

static void ob_axis_build(ob_axis *a, int full, int off, int len, int n, int o0, float (*f)(float), float sup)
{
    float scale, fs, center, wt, sum;
    int i, j, j0, j1;

    scale = (float)full / (float)n;
    fs = scale > 1.0f ? scale : 1.0f;
    a->k = (int)ceilf(sup * fs) * 2 + 2;
    a->lo = calloc(S, sizeof(int));
    a->cnt = calloc(S, sizeof(int));
    a->w = calloc((size_t)S * a->k, sizeof(float));
    a->padw = calloc(S, sizeof(float));
    for (i = 0; i < (int)S; i++) {
        if (i < o0 || i >= o0 + n) {
            a->padw[i] = 1.0f;
            continue;
        }
        center = ((float)(i - o0) + 0.5f) * scale;
        j0 = (int)floorf(center - sup * fs);
        j1 = (int)ceilf(center + sup * fs);
        j0 = j0 < 0 ? 0 : j0;
        j1 = j1 > full ? full : j1;
        for (j = j0, sum = 0.0f; j < j1; j++) sum += f(((float)j + 0.5f - center) / fs);
        a->lo[i] = -1;
        for (j = j0; j < j1; j++) {
            wt = f(((float)j + 0.5f - center) / fs) / sum;
            if (j < off || j >= off + len) {
                a->padw[i] += wt;
                continue;
            }
            if (a->lo[i] < 0) a->lo[i] = j - off;
            a->w[(size_t)i * a->k + a->cnt[i]++] = wt;
        }
        a->lo[i] = a->lo[i] < 0 ? 0 : a->lo[i];
    }
}

static void ob_axis_free(ob_axis *a)
{
    free(a->lo);
    free(a->cnt);
    free(a->w);
    free(a->padw);
}

// RGBA -> model input (see the header). Two passes: horizontal over content rows, vertical over all output rows.
static void ob_prep(const uint8_t *rgba, int w, int h, float *out)
{
    ob_axis ax, ay;
    float *rgb, *tmp, a, acc[3], pad, v;
    int side, nw, nh, x, y, i, j, c;
    double r;

    rgb = malloc((size_t)w * h * 3 * sizeof(float));
    for (i = 0; i < w * h; i++) { // alpha over white
        a = rgba[4 * i + 3] / 255.0f;
        for (c = 0; c < 3; c++) rgb[3 * i + c] = rgba[4 * i + c] * a + 255.0f * (1.0f - a);
    }
    if (!px) {
        side = w > h ? w : h;
        pad = 255.0f;
        ob_axis_build(&ax, side, (side - w) / 2, w, S, 0, ob_cubic, 2.0f);
        ob_axis_build(&ay, side, (side - h) / 2, h, S, 0, ob_cubic, 2.0f);
    } else {
        r = (double)S / h < (double)S / w ? (double)S / h : (double)S / w;
        nw = (int)(w * r);
        nh = (int)(h * r);
        nw = nw ? nw : 1;
        nh = nh ? nh : 1;
        pad = 0.0f;
        ob_axis_build(&ax, w, 0, w, nw, (S - nw) / 2, ob_tri, 1.0f);
        ob_axis_build(&ay, h, 0, h, nh, (S - nh) / 2, ob_tri, 1.0f);
    }
    tmp = malloc((size_t)h * S * 3 * sizeof(float));
    for (y = 0; y < h; y++)
        for (x = 0; x < (int)S; x++)
            for (c = 0; c < 3; c++) {
                v = pad * ax.padw[x];
                for (j = 0; j < ax.cnt[x]; j++) v += ax.w[(size_t)x * ax.k + j] * rgb[((size_t)y * w + ax.lo[x] + j) * 3 + c];
                tmp[((size_t)y * S + x) * 3 + c] = v;
            }
    for (y = 0; y < (int)S; y++)
        for (x = 0; x < (int)S; x++) {
            for (c = 0; c < 3; c++) {
                acc[c] = pad * ay.padw[y];
                for (j = 0; j < ay.cnt[y]; j++) acc[c] += ay.w[(size_t)y * ay.k + j] * tmp[((size_t)(ay.lo[y] + j) * S + x) * 3 + c];
                acc[c] = floorf((acc[c] < 0.0f ? 0.0f : acc[c] > 255.0f ? 255.0f : acc[c]) + 0.5f); // PIL: uint8 image
            }
            if (!px) { // NHWC, BGR, 0..255
                out[((size_t)y * S + x) * 3 + 0] = acc[2];
                out[((size_t)y * S + x) * 3 + 1] = acc[1];
                out[((size_t)y * S + x) * 3 + 2] = acc[0];
            } else { // NCHW, RGB, [-1, 1]
                for (c = 0; c < 3; c++) out[(size_t)c * S * S + (size_t)y * S + x] = (acc[c] / 255.0f - 0.5f) / 0.5f;
            }
        }
    free(tmp);
    free(rgb);
    ob_axis_free(&ax);
    ob_axis_free(&ay);
}

static DWORD WINAPI worker(void *arg)
{
    ob_slot *s;
    FILE *f;
    uint8_t *img;
    LONG i, c;
    int w, h, n;

    (void)arg;
    for (;;) {
        i = InterlockedIncrement(&next_job) - 1;
        if (i >= (LONG)npaths) return 0;
        while (i - (c = consumed) >= OB_RING) WaitOnAddress(&consumed, &c, sizeof(c), INFINITE);
        s = &ring[i % OB_RING];
        if ((f = fopen(paths[i], "rb"))) {
            fseek(f, 0, SEEK_END);
            InterlockedAdd64(&st_bytes, ftell(f));
            fclose(f);
        }
        img = stbi_load(paths[i], &w, &h, &n, 4);
        s->ok = img != NULL;
        if (img) {
            InterlockedAdd64(&st_px, (LONG64)w * h);
            ob_prep(img, w, h, s->x);
            stbi_image_free(img);
        }
        InterlockedExchange(&s->seq, i + 1);
        WakeByAddressAll((void *)&s->seq);
    }
}

static void walk(const char *dir)
{
    static const char *ext[] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tga", ".psd"};
    WIN32_FIND_DATAA fd;
    HANDLE hf;
    char pat[4096];
    const char *dot;
    uint32_t i;

    snprintf(pat, sizeof(pat), "%s\\*", dir);
    if ((hf = FindFirstFileA(pat, &fd)) == INVALID_HANDLE_VALUE) return;
    do {
        dot = strrchr(fd.cFileName, '.');
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY || !dot) continue;
        for (i = 0; i < sizeof(ext) / sizeof(ext[0]); i++)
            if (!_stricmp(dot, ext[i])) break;
        if (i == sizeof(ext) / sizeof(ext[0])) continue;
        if (npaths == cap) paths = realloc(paths, (cap = cap ? 2 * cap : 256) * sizeof(char *));
        paths[npaths] = malloc(strlen(dir) + strlen(fd.cFileName) + 2);
        sprintf(paths[npaths++], "%s\\%s", dir, fd.cFileName);
    } while (FindNextFileA(hf, &fd));
    FindClose(hf);
}

#define OB_CK(x)                                                                        \
    do {                                                                                \
        OrtStatus *st_ = (x);                                                           \
        if (st_) {                                                                      \
            fprintf(stderr, "onnxruntime: %s\n", ort->GetErrorMessage(st_));            \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

int main(int argc, char **argv)
{
    const OrtApi *ort;
    OrtEnv *env;
    OrtSessionOptions *so;
    OrtCUDAProviderOptionsV2 *cu;
    OrtSession *sess;
    OrtMemoryInfo *mi;
    OrtValue *in, *out;
    const char *iname, *oname;
    wchar_t wpath[4096];
    HANDLE th[OB_WORKERS];
    LARGE_INTEGER q0, q1, qs, ra, rb, qf;
    ob_slot *s;
    float *batch;
    int64_t shape[4];
    double sec, steady, run;
    uint32_t i, b, nb, maxb, done, skipped;
    LONG c;

    if (argc < 4 || (strcmp(argv[1], "wd") && strcmp(argv[1], "px"))) {
        fprintf(stderr, "usage: ortbench wd|px model.onnx <image dir>\n");
        return 2;
    }
    QueryPerformanceFrequency(&qf);
    QueryPerformanceCounter(&q0);
    px = !strcmp(argv[1], "px");
    S = px ? 1008 : 448;
    maxb = px ? 1 : OB_MAXB;
    iname = px ? "pixel_values" : "input";
    oname = px ? "probs" : "output";
    inlen = (size_t)S * S * 3;
    walk(argv[3]);
    if (!npaths) return 2;
    for (i = 0; i < OB_RING; i++) ring[i].x = malloc(inlen * sizeof(float));
    for (i = 0; i < OB_WORKERS; i++) th[i] = CreateThread(NULL, 0, worker, NULL, 0, NULL);

    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OB_CK(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ortbench", &env));
    OB_CK(ort->CreateSessionOptions(&so));
    OB_CK(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
    OB_CK(ort->CreateCUDAProviderOptions(&cu));
    OB_CK(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(so, cu));
    MultiByteToWideChar(CP_UTF8, 0, argv[2], -1, wpath, 4096);
    OB_CK(ort->CreateSession(env, wpath, so, &sess));
    OB_CK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
    batch = malloc((size_t)maxb * inlen * sizeof(float));
    QueryPerformanceCounter(&qs);

    for (i = 0, done = skipped = 0, run = 0.0; i < npaths; i = b) {
        for (b = i, nb = 0; b < npaths && nb < maxb; b++) {
            s = &ring[b % OB_RING];
            while ((c = s->seq) != (LONG)(b + 1)) WaitOnAddress((void *)&s->seq, &c, sizeof(c), INFINITE);
            if (s->ok) memcpy(batch + (size_t)nb++ * inlen, s->x, inlen * sizeof(float));
            else skipped++;
            InterlockedIncrement(&consumed);
            WakeByAddressAll((void *)&consumed);
        }
        if (!nb) continue;
        shape[0] = nb;
        shape[1] = px ? 3 : S;
        shape[2] = S;
        shape[3] = px ? S : 3;
        OB_CK(ort->CreateTensorWithDataAsOrtValue(mi, batch, (size_t)nb * inlen * sizeof(float), shape, 4,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
        out = NULL;
        QueryPerformanceCounter(&ra);
        OB_CK(ort->Run(sess, NULL, &iname, (const OrtValue *const *)&in, 1, &oname, 1, &out));
        QueryPerformanceCounter(&rb);
        run += (double)(rb.QuadPart - ra.QuadPart) / (double)qf.QuadPart;
        ort->ReleaseValue(out);
        ort->ReleaseValue(in);
        done += nb;
    }
    QueryPerformanceCounter(&q1);
    sec = (double)(q1.QuadPart - q0.QuadPart) / (double)qf.QuadPart;
    steady = (double)(q1.QuadPart - qs.QuadPart) / (double)qf.QuadPart;
    WaitForMultipleObjects(OB_WORKERS, th, TRUE, INFINITE);
    fprintf(stderr, "---------------[ ORTBENCH %s ]------------------\n", px ? "px" : "wd");
    fprintf(stderr, "  images            %u (%u skipped), batch %u\n", done, skipped, maxb);
    fprintf(stderr, "  wall time         %.2f s\n", sec);
    fprintf(stderr, "  images/s          %.2f\n", done / sec);
    fprintf(stderr, "  after session     %.2f img/s\n", done / steady);
    fprintf(stderr, "  inside OrtRun     %.2f s (%.2f img/s)\n", run, done / run);
    fprintf(stderr, "  pixels decoded    %.1f MPix/s\n", (double)st_px / 1e6 / sec);
    fprintf(stderr, "  file bytes read   %.1f MiB/s\n", (double)st_bytes / 1048576.0 / sec);
    fprintf(stderr, "------------------------------------------------\n");
    ort->ReleaseMemoryInfo(mi);
    ort->ReleaseSession(sess);
    ort->ReleaseCUDAProviderOptions(cu);
    ort->ReleaseSessionOptions(so);
    ort->ReleaseEnv(env);
    return 0;
}

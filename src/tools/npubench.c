// npubench: bring-up time and per-image latency / throughput of the NPU wd engine (engine/npu.h), on a constant
// input. Development tool.
//   gcc -O2 -march=native -Isrc/engine -Isrc/third_party/level_zero -isystem <cuda include> src/tools/npubench.c
//       -o build/npubench
//   build/npubench src/models/calib/eva02_fp16.bin [images]
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "npu.h"

static double now(void)
{
    LARGE_INTEGER q, f;

    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&f);
    return (double)q.QuadPart / (double)f.QuadPart;
}

int main(int argc, char **argv)
{
    npu n;
    uint16_t *in;
    double t0, t1, t2;
    uint32_t i, N, s, done;

    if (argc < 2) return 2;
    N = argc > 2 ? (uint32_t)atoi(argv[2]) : 8;
    in = calloc(EVA_IN, 2);
    t0 = now();
    if (npu_open(&n, argv[1], 10861)) return 1;
    while (!npu_ready(&n) && n.boot) Sleep(1);
    t1 = now();
    if (!n.up) return 1;
    printf("bring-up %.0f ms (%s)\n", (t1 - t0) * 1e3, n.compiled ? "compiled" : "cached");
    npu_submit(&n, 0, in); // latency of one
    npu_done(&n, 0, 1);
    npu_release(&n, 0);
    t2 = now();
    printf("one image %.1f ms; out[0..2] %.4f %.4f %.4f\n", (t2 - t1) * 1e3, npu_out(&n, 0)[0], npu_out(&n, 0)[1],
           npu_out(&n, 0)[2]);
    t1 = now();
    for (i = done = 0; done < N;) {
        if (i < N && (s = (uint32_t)npu_free(&n)) < NPU_NBUF) {
            npu_submit(&n, s, in);
            i++;
            continue;
        }
        for (s = 0; s < NPU_NBUF; s++)
            if (npu_done(&n, s, 1)) {
                npu_release(&n, s);
                done++;
                break;
            }
    }
    t2 = now();
    printf("%u images, 2 in flight: %.1f ms each, %.2f img/s\n", N, (t2 - t1) * 1e3 / N, N / (t2 - t1));
    npu_close(&n);
    return 0;
}

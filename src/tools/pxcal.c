// pxcal: PixAI calibration. Reads patch operands (fp16 [5184][592], from src/core/pxfeed.py) on stdin, runs the
// fp16 engine (-DPX_CALIB, pixai_fp16.bin from export_pixai.py) over them in batches, and writes
// <weights>.calib: the activation ranges export_pixai.py turns into the FP8 pixai.bin. Also prints, per block, the
// share of GELU blocks that are near zero.
//   python src/core/pxfeed.py <images> | build\pxcal.exe src/models/calib/pixai_fp16.bin
#include "px.h"
#include <fcntl.h>
#include <io.h>

int main(int argc, char **argv)
{
    px p;
    uint16_t *pool, *in[PX_NB];
    uint32_t n, i, total;

    if (argc < 2) return 2;
    _setmode(_fileno(stdin), _O_BINARY);
    pool = _mm_malloc((size_t)PX_NB * PX_IN * 2, 4096);
    memset(pool, 0, (size_t)PX_NB * PX_IN * 2);
    for (i = 0; i < PX_NB; i++) in[i] = pool + (size_t)i * PX_IN;
    if (px_open(&p, argv[1], 0, pool, (size_t)PX_NB * PX_IN * 2) || px_ready(&p)) return 1;
    for (total = 0;;) {
        for (n = 0; n < PX_NB && fread(in[n], 2, PX_IN, stdin) == PX_IN; n++);
        if (!n) break;
        for (i = n; i < PX_NB; i++) memcpy(in[i], in[0], PX_IN * 2); // full batches keep the statistics unbiased
        if (px_submit(&p, 0, in, PX_NB) || !px_wait(&p, 0)) return 1;
        total += PX_NB;
        if (n < PX_NB) break;
    }
    fprintf(stderr, "calibrated on %u inputs\n", total);
    px_calib_dump(&p, total);
    px_close(&p);
    return 0;
}

// xtag: images / directories -> procimg.h (xorfold, MVIs, router) -> WD EVA02-L tagger v3 on the eva CUDA engine,
// and with -px the PixAI tagger on the px engine, both resident on the GPU.
// Output, one line per image as it completes: path <TAB> rating <TAB> mvis <TAB> tag:score,tag:score,... (descending)
//
// usage: xtag eva02.bin selected_tags.csv <image|dir>... [options]
// Options are nmap-style: a short form grouped by subject (-m* MVI size, -c* crop weights, -t* thresholds,
// -r* router, -n* switches off) and a long form; xtag -h prints the board. The banner goes to stderr on every run
// (stdout carries only results) unless -nb; -h prints banner and board to stdout; -sT ends the run with a
// stats board on stderr (images/s, decoded MPix/s, file MiB/s, crops/s, wall clock from after the banner);
// -nO drops the tag lines entirely (benchmarks: nothing formatted, sorted or written).
// Routes (-rT): what runs the whole image ("full") and the MVI crops ("crops"):
//   wd: wd on both   px-full: PixAI full + wd crops   px-crops: wd full + PixAI crops   px: PixAI on both
//   auto (default with -px): the online router picks per image.
// -rA: F1 against the px route a route needs to count as agreeing (router training label); -rE: exploration floor;
// -rC: perceptron sum a router stage needs to trust a cheaper route (-1: theta / 2).
// Decoding: stb_image (jpg/png/gif/bmp/tga/psd) straight from a mapped view of the file; build with -DPI_WEBP
// -lwebp for WebP. Worker threads decode ahead (in input order) while the main thread composites, folds, resizes
// and feeds the GPU.

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
// every decoder buffer carries 64 bytes of slack: the fold and the resample read up to 64 bytes past a row's end
#define STBI_MALLOC(sz) malloc((sz) + 64)
#define STBI_REALLOC(p, sz) realloc(p, (sz) + 64)
#define STBI_FREE(p) free(p)
#include "third_party/stb_image.h"
#include "procimg.h"
#ifdef PI_WEBP
#include <webp/decode.h>
#endif

#define XT_RING    16 // decoded images buffered ahead
#define XT_WORKERS 64 // -j ceiling
#define XT_JOBS    8  // default worker cap (CPU and GPU share the laptop's power budget)

typedef struct {
    pi_img im;         // decoded + ingested (composited, folded, pyramid); im.p.lv[0] NULL if undecodable
    volatile LONG seq; // index + 1 once ready
} xt_slot;

static pi_tagger tagger;
static pi_params prm;  // parsed before the workers start; read-only afterwards
static char **paths;
static uint32_t npaths, cap, recursive;
static xt_slot ring[XT_RING];
static volatile LONG next_dec, consumed;
static volatile LONG64 st_px, st_bytes; // -sT: decoded pixels, compressed bytes read
static uint64_t st_img, st_crop;         // -sT: images reported, crops run (any model)
static uint32_t quiet;                   // -nO: no tag lines (benchmarking)
static uint32_t npu_n;                   // -sT: wd full passes the NPU ran
static uint32_t st_new, st_all, st_upd, st_ev; // -sT with -s: rows imprinted this run / held, updates, events
static uint32_t tg_nu, tg_ns;                  // -sT with -s: user tags held, sidecar images this run

#define XT_IS(a, s, l) (!strcmp(a, s) || !strcmp(a, l)) // short or long form of an option
#define XT_COLS 48 // banner width; the option board's rules match it
static const char xt_banner[] =
    "         _____  _    ____  ____ _____ ____  \n"
    "   __  _|_   _|/ \\  / ___|/ ___| ____|  _ \\ \n"
    "   \\ \\/ / | | / _ \\| |  _| |  _|  _| | |_) |\n"
    "    >  <  | |/ ___ \\ |_| | |_| | |___|  _ < \n"
    "   /_/\\_\\ |_/_/   \\_\\____|\\____|_____|_| \\_\\\n"
    "  Fast Image Tagging with AVX2, CUDA, and XORs\n"
    "------------------------------------------------\n"
    " Developed by ef1500 | https://github.com/ef1500\n"
    "-------[https://github.com/holoarchivists]------\n";
static const char xt_dash[] = "------------------------------------------------"; // XT_COLS

// ----[ TITLE ]---- rule, centred in XT_COLS
static void xt_rule(FILE *f, const char *title)
{
    int n, l;

    n = title[0] ? (int)strlen(title) + 4 : 0;
    l = (XT_COLS - n) / 2;
    fprintf(f, "%.*s%s%s%s%.*s\n", l, xt_dash, title[0] ? "[ " : "", title, title[0] ? " ]" : "", XT_COLS - n - l,
            xt_dash);
}

// Option line: 2 + 17 + 1 columns of option, at most 28 of text; route table rows: 20 + 9 + at most 19. Every line
// fits inside XT_COLS.
#define XT_OPT(f, opt, ...) (fprintf(f, "  %-17s ", opt), fprintf(f, __VA_ARGS__), fputc('\n', f))
#define XT_ROW(f, name, text) fprintf(f, "%20s%-9s%s\n", "", name, text)

// The option board (right under the banner, whose last line is its top rule), defaults from pi_defaults().
static void xt_help(FILE *f)
{
    pi_params d;

    d = pi_defaults();
    fprintf(f, " xtag <eva.bin> <tags.csv> <in>... [opt]\n");
    fprintf(f, " out, per image, tab separated:\n");
    fprintf(f, "   path rating mvis tag:score,...\n");
    xt_rule(f, "INPUT");
    XT_OPT(f, "<in>", "image files or directories");
    XT_OPT(f, "-r  --recurse", "recurse into directories");
    XT_OPT(f, "-na --noalpha", "ignore alpha (no composite)");
    XT_OPT(f, "-j  --jobs <n>", "workers [cores-1, max %d]", XT_JOBS);
    xt_rule(f, "MODELS");
    XT_OPT(f, "-px <file>", "add PixAI + the router");
    XT_OPT(f, "", "(vocabulary: <file>.tags)");
    XT_OPT(f, "-npu <file>", "wd on the NPU as well");
    XT_OPT(f, "", "(fp16 eva02 bin; first run");
    XT_OPT(f, "", " compiles, then <file>.npu)");
    XT_OPT(f, "-d  --device <n>", "CUDA device [%d]", d.device);
    xt_rule(f, "CROPS");
    XT_OPT(f, "-nc --nocrops", "whole images only, no crops");
    XT_OPT(f, "-t  --tile <px>", "tile side, 8k >= 32 [%u]", d.ts);
    XT_OPT(f, "-lo -hi <n>", "tile popcount bounds [%u, %u]", d.lo, d.hi);
    XT_OPT(f, "-m  --mode <m>", "zero|dontcare|inverse [zero]");
    XT_OPT(f, "-mw --minw <n>", "min crop width, tiles [%u]", d.minw);
    XT_OPT(f, "-mh --minh <n>", "min crop height, tiles [%u]", d.minh);
    XT_OPT(f, "-Mw --maxw <n>", "max crop width, 0=all [%u]", d.maxw);
    XT_OPT(f, "-Mh --maxh <n>", "max crop height, 0=all [%u]", d.maxh);
    XT_OPT(f, "-mc --maxmvi <n>", "max crops per image [%u]", d.maxmvi);
    XT_OPT(f, "-cw --wthresh <f>", "crop weight cutoff [%.2f]", d.weight_thresh);
    XT_OPT(f, "-cg --gain <f>", "crop weight gain [%.2f]", d.gain);
    xt_rule(f, "THRESHOLDS (WD)");
    XT_OPT(f, "-tg --gen <f>", "general tags [%.2f]", d.gen_thresh);
    XT_OPT(f, "-tc --char <f>", "character tags [%.2f]", d.char_thresh);
    xt_rule(f, "ROUTER (-px)");
    fprintf(f, " full: the whole image, crops: detail\n");
    fprintf(f, " regions; wd is fast, PixAI slow + best\n");
    XT_OPT(f, "-rT --route <r>", "who tags what [auto]:");
    XT_ROW(f, "auto", "learned per image");
    XT_ROW(f, "wd", "wd only (fastest)");
    XT_ROW(f, "px-full", "PixAI full+wd crops");
    XT_ROW(f, "px-crops", "wd full+PixAI crops");
    XT_ROW(f, "px", "PixAI only (best)");
    XT_OPT(f, "-rA --agree <f>", "F1 vs px that passes [%.2f]", d.agree);
    XT_OPT(f, "-rE --explore <f>", "min share explored [%.4f]", d.explore);
    XT_OPT(f, "-rC --rconf <n>", "confidence to go cheaper");
    XT_OPT(f, "", "[%d = theta / 2]", d.rconf);
    xt_rule(f, "TAG STEALING (-px)");
    fprintf(f, " wd's head learns online from PixAI\n");
    XT_OPT(f, "-s  --steal", "on (off unless given)");
    XT_OPT(f, "-sf --stealfile <f>", "keep it across runs (+ -s)");
    XT_OPT(f, "-sn --stealcap <n>", "new tags wd can gain [%u]", d.scap);
    XT_OPT(f, "", "(stolen + sidecar tags)");
    XT_OPT(f, "-sk --stealcats <s>", "PixAI categories [cp]:");
    XT_OPT(f, "", "c char p copyright g gen");
    XT_OPT(f, "", "a artist m meta");
    xt_rule(f, "TAGGLOMERATION");
    fprintf(f, " wd learns tags from <image>.txt beside\n");
    fprintf(f, " an image: comma/line separated; names\n");
    fprintf(f, " no model has become new tags\n");
    fprintf(f, " (character:name sets the category)\n");
    XT_OPT(f, "-g  --glean", "on (off unless given)");
    XT_OPT(f, "-gf --gleanfile <f>", "keep it across runs (+ -g)");
    XT_OPT(f, "", "(same file as -sf)");
    XT_OPT(f, "-gk --gleank <f>", "cutoff, sigmas over the");
    XT_OPT(f, "", "background [%.1f]; lower:", d.gk);
    XT_OPT(f, "", "more recall");
    xt_rule(f, "DISPLAY");
    XT_OPT(f, "-nb --nobanner", "hide the banner");
    XT_OPT(f, "-sT --stats", "img/s + throughput at end");
    fprintf(f, "  -nO --no-tag-output  no tag lines (benchmarks)\n"); // long form wider than the column
    XT_OPT(f, "-h  --help", "this board");
    xt_rule(f, "");
}

// -sT board (stderr): end-to-end rates over sec seconds, engine bring-up included.
static void xt_stats(FILE *f, double sec, uint32_t skipped)
{
    xt_rule(f, "STATS");
    XT_OPT(f, "images", "%llu (%u skipped)", (unsigned long long)st_img, skipped);
    XT_OPT(f, "wall time", "%.2f s", sec);
    XT_OPT(f, "images/s", "%.2f", (double)st_img / sec);
    XT_OPT(f, "pixels decoded", "%.1f MPix/s", (double)st_px / 1e6 / sec);
    XT_OPT(f, "file bytes read", "%.1f MiB/s", (double)st_bytes / 1048576.0 / sec);
    XT_OPT(f, "crops run", "%llu (%.1f/s)", (unsigned long long)st_crop, (double)st_crop / sec);
    if (npu_n) XT_OPT(f, "wd on the NPU", "%u images (%.1f/s)", npu_n, (double)npu_n / sec);
    if (st_ev) {
        XT_OPT(f, "tags stolen", "%u new (%u in all)", st_new, st_all);
        XT_OPT(f, "head updates", "%u over %u events", st_upd, st_ev);
    }
    if (tg_nu || tg_ns) XT_OPT(f, "sidecar tags", "%u held, %u images", tg_nu, tg_ns);
    xt_rule(f, "");
}

// Router board (auto routing with -px): where images went and how the learner did on the explored ones, which run
// every route and are tagged with px's output. Rows are 45 columns, legend lines at most 46. No closing rule when
// the STATS board follows: its own rule closes this one.
static void xt_router(FILE *f, const pi_tagger *t, uint32_t last)
{
    static const char *name[4] = {"wd", "px-full", "px-crops", "px"};
    uint32_t r, ne;

    ne = t->nroute[PI_X];
    xt_rule(f, "ROUTER");
    XT_OPT(f, "images", "%u, %u explored (as px)", t->routed, ne);
    fprintf(f, "  %-10s %5s %10s %7s %7s\n", "route", "sent", "matches px", "guess 1", "guess 2");
    for (r = 0; r < 3; r++)
        if (ne) fprintf(f, "  %-10s %5u %9.1f%% %6.1f%% %6.1f%%\n", name[r], t->nroute[r], 100.0 * t->nok[r] / ne,
                        100.0 * t->hit1[r] / ne, 100.0 * t->hit2[r] / ne);
        else fprintf(f, "  %-10s %5u %10s %7s %7s\n", name[r], t->nroute[r], "-", "-", "-");
    fprintf(f, "  %-10s %5u %10s %7s %7s\n", name[3], t->nroute[PI_C], "-", "-", "-");
    fprintf(f, " matches px: share of explored images where\n");
    fprintf(f, " the route's tags reach F1 >= -rA vs px's\n");
    fprintf(f, " guess 1/2: the router predicted that right,\n");
    fprintf(f, " at push / after wd's whole-image pass\n");
    if (last) xt_rule(f, "");
}

// -rT value -> route (-1 auto), -2 unknown
static int xt_route(const char *v)
{
    return !strcmp(v, "auto") ? -1 : !strcmp(v, "wd") ? PI_D : !strcmp(v, "px-full") ? PI_A
         : !strcmp(v, "px-crops") ? PI_B : !strcmp(v, "px") ? PI_C : -2;
}

static int by_score(const void *a, const void *b)
{
    float sa, sb;

    sa = ((const pi_tag *)a)->score;
    sb = ((const pi_tag *)b)->score;
    return (sa < sb) - (sa > sb);
}

// One finished image (completion order).
static void print_result(pi_tagger *t, void *user, pi_result *r)
{
    uint32_t i;

    (void)t;
    st_img++;
    st_crop += r->ncrop;
    if (quiet) return;
    qsort(r->tag, r->n, sizeof(pi_tag), by_score);
    printf("%s\t%s\t%u\t", (const char *)user, r->rating, r->nmvi);
    for (i = 0; i < r->n; i++) printf("%s%s:%.3f", i ? "," : "", r->tag[i].name, r->tag[i].score);
    putchar('\n');
}

// Level 0 of im (RGBA, rows w * 4 apart, malloc'd with 64 bytes of slack: the tagger frees it), im->alpha set when
// the source carries an alpha channel. The decoder reads the file through a mapped view: no copy of the compressed
// bytes. 0 on success.
static int decode(const char *path, pi_img *im)
{
    pi_mapping mp;
    uint8_t *file, *px;
    size_t len;
    int comp, w, h;
#ifdef PI_WEBP
    WebPBitstreamFeatures feat;
#endif

    im->p.lv[0] = NULL;
    im->p.nlv = 0;
    file = pi_map(&mp, path, 0);
    if (!file || mp.n > 0x7FFFFFFF) {
        pi_unmap(&mp);
        return -1;
    }
    len = mp.n;
    InterlockedAdd64(&st_bytes, (LONG64)len);
    px = NULL;
#ifdef PI_WEBP
    if (len > 12 && !memcmp(file, "RIFF", 4) && !memcmp(file + 8, "WEBP", 4)) {
        if (WebPGetFeatures(file, len, &feat) == VP8_STATUS_OK && (px = malloc((size_t)feat.width * feat.height * 4 + 64))
            && !WebPDecodeRGBAInto(file, len, px, (size_t)feat.width * feat.height * 4, feat.width * 4)) {
            free(px);
            px = NULL;
        }
        w = feat.width;
        h = feat.height;
        comp = feat.has_alpha ? 4 : 3;
    } else
#endif
    px = stbi_load_from_memory(file, (int)len, &w, &h, &comp, 4);
    pi_unmap(&mp);
    if (!px) return -1;
    im->p.lv[0] = px;
    im->p.ls[0] = (size_t)w * 4;
    im->p.lw[0] = (uint32_t)w;
    im->p.lh[0] = (uint32_t)h;
    im->p.nlv = 1;
    im->alpha = comp == 2 || comp == 4;
    return 0;
}

// Decode worker: claims the next index, sleeps until its ring slot is released, decodes and ingests (composite,
// xorfold, pyramid: the per-image CPU work, spread over the workers), publishes. Waits are WaitOnAddress sleeps,
// not spins: the CPU shares the laptop's power budget with the GPU.
static DWORD WINAPI worker(void *arg)
{
    xf_ctx *x;
    xf_plan *pl;
    xt_slot *s;
    LONG i, c;

    (void)arg;
    x = _mm_malloc(sizeof(xf_ctx), 64);
    pl = _mm_malloc(sizeof(xf_plan), 64);
    if (x) xf_init(x, prm.lo, prm.hi);
    for (;;) {
        i = InterlockedIncrement(&next_dec) - 1;
        if (i >= (LONG)npaths) break;
        while (i - (c = consumed) >= XT_RING) WaitOnAddress(&consumed, &c, sizeof(c), INFINITE);
        s = &ring[i % XT_RING];
        s->im.side = NULL;
        s->im.nside = 0;
        if (!decode(paths[i], &s->im)) {
            if (prm.glean) s->im.side = tg_read(paths[i], &s->im.nside);
            InterlockedAdd64(&st_px, (LONG64)s->im.p.lw[0] * s->im.p.lh[0]);
            if (x && pl) pi_ingest(&prm, x, pl, &s->im);
            else s->im.folded = 0;
        }
        InterlockedExchange(&s->seq, i + 1);
        WakeByAddressAll((void *)&s->seq);
    }
    _mm_free(x);
    _mm_free(pl);
    return 0;
}

static int is_image(const char *name)
{
    static const char *ext[] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tga", ".psd", ".webp"};
    const char *dot;
    uint32_t i;

    dot = strrchr(name, '.');
    for (i = 0; dot && i < sizeof(ext) / sizeof(ext[0]); i++)
        if (!_stricmp(dot, ext[i])) return 1;
    return 0;
}

static void add_path(const char *p)
{
    if (npaths == cap) {
        cap = cap ? 2 * cap : 256;
        paths = realloc(paths, cap * sizeof(char *));
    }
    paths[npaths++] = strdup(p);
}

static int is_dir(const char *path)
{
    DWORD a;

    a = GetFileAttributesA(path);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

// -s: images with a sidecar go right after the first ST_MU0 others (user tags need a mean of ST_MU0 embeddings), so
// even a first run learns from them early and tags everything after with what they taught.
static void xt_sideorder(void)
{
    char **o;
    uint8_t *hs;
    uint32_t i, n, k;

    o = malloc(npaths * sizeof(char *));
    hs = malloc(npaths);
    if (!o || !hs) {
        free(o);
        free(hs);
        return;
    }
    for (i = 0; i < npaths; i++) hs[i] = (uint8_t)tg_has(paths[i]);
    for (i = 0, n = 0, k = 0; i < npaths && k < ST_MU0; i++)
        if (!hs[i]) o[n++] = paths[i], hs[i] = 2, k++;
    for (i = 0; i < npaths; i++)
        if (hs[i] == 1) o[n++] = paths[i];
    for (i = 0; i < npaths; i++)
        if (!hs[i]) o[n++] = paths[i];
    memcpy(paths, o, npaths * sizeof(char *));
    free(o);
    free(hs);
}

static void walk(const char *path, uint32_t depth)
{
    WIN32_FIND_DATAA fd;
    HANDLE hf;
    char pat[4096], sub[4096];

    if (!is_dir(path)) {
        if (!depth) add_path(path); // explicit files are tagged whatever their extension
        return;
    }
    snprintf(pat, sizeof(pat), "%s\\*", path);
    hf = FindFirstFileA(pat, &fd);
    if (hf == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.cFileName[0] == '.') continue;
        snprintf(sub, sizeof(sub), "%s\\%s", path, fd.cFileName);
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (recursive) walk(sub, depth + 1);
        } else if (is_image(fd.cFileName)) {
            add_path(sub);
        }
    } while (FindNextFileA(hf, &fd));
    FindClose(hf);
}

int main(int argc, char **argv)
{
    SYSTEM_INFO si;
    HANDLE th[XT_WORKERS];
    xt_slot *s;
    const char *a, *v;
    LARGE_INTEGER q0, q1, qf;
    double sec;
    uint32_t i, nth, jobs, nobanner, help, stats, skipped;
    LONG c;
    int k;
#ifdef EVA_PROF
    LARGE_INTEGER q;
    double t, dwait;

    EVA_TSTART();
    dwait = 0.0;
#endif

    for (k = 1, nobanner = help = 0; k < argc; k++) {
        nobanner |= XT_IS(argv[k], "-nb", "--nobanner");
        help |= XT_IS(argv[k], "-h", "--help");
    }
    if (help || argc < 4) { // asked for: stdout, exit 0; missing arguments: stderr, exit 2
        if (!nobanner) fputs(xt_banner, help ? stdout : stderr);
        xt_help(help ? stdout : stderr);
        return help ? 0 : 2;
    }
    if (!nobanner) fputs(xt_banner, stderr);
    QueryPerformanceCounter(&q0); // -sT clock: everything after the banner, engine bring-up included
    stats = skipped = jobs = 0;
    prm = pi_defaults();
    for (k = 3; k < argc; k++) {
        a = argv[k];
        v = k + 1 < argc ? argv[k + 1] : "0";
        if (XT_IS(a, "-r", "--recurse")) recursive = 1;
        else if (XT_IS(a, "-t", "--tile")) prm.ts = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-lo", "--lo")) prm.lo = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-hi", "--hi")) prm.hi = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-mw", "--minw")) prm.minw = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-mh", "--minh")) prm.minh = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-Mw", "--maxw")) prm.maxw = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-Mh", "--maxh")) prm.maxh = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-mc", "--maxmvi")) prm.maxmvi = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-cw", "--wthresh")) prm.weight_thresh = (float)atof(v), k++;
        else if (XT_IS(a, "-cg", "--gain")) prm.gain = (float)atof(v), k++;
        else if (XT_IS(a, "-tg", "--gen")) prm.gen_thresh = (float)atof(v), k++;
        else if (XT_IS(a, "-tc", "--char")) prm.char_thresh = (float)atof(v), k++;
        else if (XT_IS(a, "-d", "--device")) prm.device = atoi(v), k++;
        else if (XT_IS(a, "-nc", "--nocrops")) prm.nocrops = 1;
        else if (XT_IS(a, "-na", "--noalpha")) prm.noalpha = 1;
        else if (XT_IS(a, "-j", "--jobs")) jobs = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-px", "--px")) prm.px = v, k++;
        else if (XT_IS(a, "-npu", "--npu")) prm.npu = v, k++;
        else if (XT_IS(a, "-rA", "--agree")) prm.agree = (float)atof(v), k++;
        else if (XT_IS(a, "-rE", "--explore")) prm.explore = (float)atof(v), k++;
        else if (XT_IS(a, "-rC", "--rconf")) prm.rconf = atoi(v), k++;
        else if (XT_IS(a, "-s", "--steal")) prm.steal = 1;
        else if (XT_IS(a, "-sf", "--stealfile") || XT_IS(a, "-gf", "--gleanfile")) {
            if (prm.sfile && strcmp(prm.sfile, v)) { // stolen rows and user tags share one head: one file
                fprintf(stderr, "-sf and -gf name the same file (the learned head): %s vs %s\n", prm.sfile, v);
                return 2;
            }
            prm.sfile = v;
            prm.steal |= XT_IS(a, "-sf", "--stealfile"); // each file option also turns its own learning on
            prm.glean |= XT_IS(a, "-gf", "--gleanfile");
            k++;
        }
        else if (XT_IS(a, "-sn", "--stealcap")) prm.scap = (uint32_t)atoi(v), k++;
        else if (XT_IS(a, "-g", "--glean")) prm.glean = 1;
        else if (XT_IS(a, "-gk", "--gleank")) prm.gk = (float)atof(v), k++;
        else if (XT_IS(a, "-sk", "--stealcats")) {
            for (prm.scats = 0; *v; v++)
                prm.scats |= *v == 'g' ? 1u : *v == 'a' ? 2u : *v == 'p' ? 8u : *v == 'c' ? 16u : *v == 'm' ? 32u : 0u;
            k++;
        }
        else if (XT_IS(a, "-nb", "--nobanner")) continue;
        else if (XT_IS(a, "-sT", "--stats")) stats = 1;
        else if (XT_IS(a, "-nO", "--no-tag-output")) quiet = 1;
        else if (XT_IS(a, "-rT", "--route")) {
            if ((prm.route = xt_route(v)) < -1) {
                fprintf(stderr, "-rT: unknown route '%s' (auto, wd, px-full, px-crops, px)\n", v);
                return 2;
            }
            k++;
        } else if (XT_IS(a, "-m", "--mode")) {
            prm.mode = !strcmp(v, "dontcare") ? PI_DONTCARE : !strcmp(v, "inverse") ? PI_INVERSE : PI_ZERO;
            k++;
        } else if (a[0] != '-') {
            walk(a, 0);
        } else {
            fprintf(stderr, "unknown option %s (xtag -h lists them)\n", a);
            return 2;
        }
    }
    if (prm.ts < 32 || (prm.ts & 7)) {
        fprintf(stderr, "-t must be a multiple of 8, >= 32\n");
        return 2;
    }
    if (!npaths) return 2;
    if (prm.glean) xt_sideorder();

    // decoding starts before the engine comes up: the first images are ready when it is
    GetSystemInfo(&si);
    nth = si.dwNumberOfProcessors > 1 ? si.dwNumberOfProcessors - 1 : 1;
    nth = jobs ? jobs : nth > XT_JOBS ? XT_JOBS : nth;
    nth = nth > XT_WORKERS ? XT_WORKERS : nth;
    for (i = 0; i < nth; i++) th[i] = CreateThread(NULL, 0, worker, NULL, 0, NULL);
#ifdef EVA_PROF
    EVA_TMARK(q, "args + walk + workers");
#endif
    if (pi_open(&tagger, argv[1], argv[2], &prm, print_result)) return 1;
#ifdef EVA_PROF
    EVA_TMARK(q, "pi_open");
#endif

    for (i = 0; i < npaths; i++) {
        s = &ring[i % XT_RING];
#ifdef EVA_PROF
        t = EVA_TNOW(q);
#endif
        // waiting on the decoders: heavy-side work (PixAI preparation / issue / collection) first, else sleep
        while ((c = s->seq) != (LONG)(i + 1))
            if (pi_step(&tagger) <= 0) WaitOnAddress((void *)&s->seq, &c, sizeof(c), 1);
#ifdef EVA_PROF
        dwait += EVA_TNOW(q) - t;
#endif
        skipped += !s->im.p.lv[0];
        if (!s->im.p.lv[0]) fprintf(stderr, "skip %s: cannot decode\n", paths[i]);
        else if (pi_push(&tagger, &s->im, paths[i])) return 1;
        InterlockedIncrement(&consumed);
        WakeByAddressAll((void *)&consumed);
    }
    if (pi_drain(&tagger)) return 1;
#ifdef EVA_PROF
    EVA_TMARK(q, "all images done");
    fprintf(stderr, "prof host waiting for decode  %9.1f ms (%u images)\n", dwait, npaths);
#endif
    WaitForMultipleObjects(nth, th, TRUE, INFINITE);
    for (i = 0; i < nth; i++) CloseHandle(th[i]);
    if (tagger.haspx && prm.route < 0 && tagger.routed) xt_router(stderr, &tagger, !stats);
    npu_n = tagger.nnpu;
    st_new = tagger.st.nimp;
    st_all = tagger.st.ns;
    st_upd = tagger.st.nupd;
    st_ev = tagger.st.nev;
    tg_nu = tagger.tg.nu;
    tg_ns = tagger.tg.nside;
    pi_close(&tagger);
    for (i = 0; i < npaths; i++) free(paths[i]);
    free(paths);
    if (stats) {
        QueryPerformanceCounter(&q1);
        QueryPerformanceFrequency(&qf);
        sec = (double)(q1.QuadPart - q0.QuadPart) / (double)qf.QuadPart;
        xt_stats(stderr, sec, skipped);
    }
    return 0;
}

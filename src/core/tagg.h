#ifndef TAGG_H
#define TAGG_H
// Tagglomeration: <image stem>.txt next to an image lists tags for it (comma or line separated; spaces become
// underscores, ASCII is lowercased, backslashes dropped; an optional general: / artist: / copyright: / character: /
// meta: prefix sets the category of a tag neither model has). Each name lands on a wd row:
//   a row wd already has (the file's, a stolen one): one NLMS step toward a confident positive (ground truth);
//   a PixAI name wd lacks (with -px): a stolen row, imprinted from this image (steal.h), refined by PixAI later;
//   a name neither model knows: a user tag, learned here by neuroevolution.
//
// User tags are scored on the CPU from wd's whole-image head input (the pooled, normed embedding every GPU pass reads
// back). In that space, centred on the running mean mu, a concept is a direction: images of it have a high cosine
// to the mean of its examples, everything else sits in a narrow band around 0. A user tag is therefore a small
// network of cosine units, score = max_k cos(x - mu, c_k), one unit per mode of the concept, and its decision is
// relative to the background: logit = lt + TG_GZ (score - tau) / sigma, tau = median + k sigma, median and sigma
// (from the MAD) of the score over every image not known to carry the tag. Median and MAD ignore the unlabelled
// images of the concept in that set (a few percent of it), so they are never trained against; k is -sg.
//
// Genomes evolve NEAT-style, structure only:
//   a gene is a sidecar image's membership in a unit, c_k = sum a_g (x_g - mu) with a_g = TG_WP for a positive and
//   -TG_WN times the unit's positive weight for a confuser (a sidecar negative). No continuous weights: with a
//   handful of examples they overfit, while one unit holding every positive is the examples' centroid, the right
//   answer for a single-mode concept. The slot is the gene's innovation number and crossover aligns genes by it; a
//   unit's innovation is the image it was born on;
//   add-connection (a positive joins the unit closest to it, a confuser is subtracted from the unit firing on it) and
//   add-node (a positive the units miss leaves them and starts its own: a second mode) are aimed at the current
//   worst cases; pruning removes genes; species are unit counts, and every species' best survives a generation.
// Fitness, in background sigmas (so narrowing the background buys nothing the positives do not keep): each
// positive's leave-one-out cosine (its own gene removed: no credit for memorising an example) above tau, capped at
// TG_ZCAP, averaged; minus TG_NEGW times sidecar negatives' excess over tau - TG_ZNEG; minus TG_TAILW times the
// background's mean excess over tau (capped at 1 sigma per image: the tail median and MAD do not see; the concept's
// unlabelled images in it cost every good genome alike); minus TG_CPLX per gene. Every dot product fitness needs is a centred Gram entry C[g][f] = (x_g - mu) . (x_f - mu), built once per
// image, so a genome costs a few TG_E-wide AXPYs and no 1024-wide dot.
//
// Store: slot 0 the running mean, 1..TG_L sidecar images with the user tags each lists, the rest a reservoir sample
// of every other image wd embedded on the GPU. Background of a tag: the reservoir and the sidecar images that do not
// list it (sidecars are taken as complete for user tags).
//
// Persisted after the stolen rows in the -sf file (section XTG1; files without it load as before).

#define TG_E     512   // store slots
#define TG_L     255   // sidecar images kept (slots 1..TG_L)
#define TG_R     (TG_E - 1 - TG_L) // reservoir slots
#define TG_U     256   // user tags
#define TG_K     4     // units per genome
#define TG_GN    16    // genes per unit
#define TG_P     24    // genomes per user tag
#define TG_GEN   16    // generations per sidecar image
#define TG_RGEN  16    // reservoir insertions per maintenance generation
#define TG_K0    3.0f  // -sg default: tau = background median + TG_K0 sigma
#define TG_BMIN  8     // background images before sigma is measured (until then sigma = TG_S0)
#define TG_S0    0.1f
#define TG_SMIN  0.02f // sigma floor (cosine)
#define TG_GZ    1.5f  // logits per background sigma above tau
#define TG_ZCAP  3.0f  // a positive's margin over tau counts up to this many sigmas
#define TG_NEGW  2.0f  // weight of a sidecar negative's excess over tau - TG_ZNEG sigmas
#define TG_ZNEG  0.5f
#define TG_TAILW 8.0f  // weight of the background's share over tau (each image's excess capped at 1 sigma)
#define TG_CPLX  0.002f // fitness cost of a gene (sigmas; a tie-break)
#define TG_PNODE 0.08f // add-node, when a positive is missed
#define TG_PCONN 0.5f  // add-connection
#define TG_PX    0.3f  // crossover within a species
#define TG_WP    1.0f  // gene weight of a positive in a unit
#define TG_WN    0.25f // gene weight of a confuser, times the unit's positive weight (subtracted)
#define TG_SIDE  65536 // sidecar bytes read
#define TG_MAGIC 0x31475458u // "XTG1"

typedef struct { uint32_t e; float a; } tg_gene;
typedef struct { uint32_t inn, n; float b; tg_gene g[TG_GN]; } tg_unit; // b unused (file layout)
typedef struct { uint32_t k; float fit; tg_unit u[TG_K]; } tg_genome;

typedef struct {
    float w[TG_K][EVA_D];    // the elite's unit directions, unit length, for scoring
    tg_genome pop[TG_P];     // [0] the elite after a generation
    float lt, tau, isig;     // threshold logit; the elite's cosine cutoff; 1 / its background sigma
    uint32_t row, np, k;     // wd row; sidecar positives in the store; elite units (0: not seeded yet)
} tg_tag;

typedef struct {
    float *x;                // [TG_E][EVA_D] exemplars; slot 0 the mean at the last refresh
    float *G;                // [TG_L + 1][TG_E] raw Gram rows of slot 0 and the sidecar slots
    float *C;                // [TG_L + 1][TG_E] centred Gram rows (valid after tg_refresh)
    uint64_t (*lab)[TG_U / 64]; // [TG_E] user tags each sidecar image lists
    tg_tag *tag[TG_U];
    int32_t *ut;             // row - nt0 -> user tag (-1: stolen or unused)
    uint32_t *wt, *pt, wm, pm; // name hash tables: wd row + 1 (file rows, rows loaded, user rows), PixAI tag + 1
    float dg[TG_E], nf[TG_E], rn[TG_E]; // |x_f|^2; |x_f - mu|^2 and its inverse root (0: empty slot)
    float s[TG_E], zk[TG_K][TG_E], cc[TG_K], ic[TG_K]; // eval scratch: scores; unit dots, |c_k|^2, 1 / |c_k|
    float bv[TG_E], xs[EVA_D], ms[EVA_D]; // background scores; scoring scratch; the mean the elites were built on
    uint32_t pos[TG_L], neg[TG_L], bgi[TG_E], npos, nneg, nbg; // the current tag's labelled slots, background slots
    tg_genome nx[TG_P];      // next generation
    uint32_t ord[TG_P];      // fitness ranks
    uint64_t rng;
    float ksig, tau, sig;    // -sg; cutoff and background sigma of the last evaluation
    uint32_t nu, nl, nr, nres, nins, nside, full;
} tg_learn;

#define TG_X(T, s) ((T)->x + (size_t)(s) * EVA_D)
#define TG_G(T, g) ((T)->G + (size_t)(g) * TG_E)
#define TG_C(T, g) ((T)->C + (size_t)(g) * TG_E)
#define TG_LIVE(T, f) ((f) <= (T)->nl || ((f) > TG_L && (f) <= TG_L + (T)->nr))
#define TG_HAS(T, f, u) ((T)->lab[f][(u) >> 6] >> ((u) & 63) & 1)

static inline uint32_t tg_rand(tg_learn *T)
{
    T->rng ^= T->rng >> 12;
    T->rng ^= T->rng << 25;
    T->rng ^= T->rng >> 27;
    return (uint32_t)((T->rng * 0x2545F4914F6CDD1Dull) >> 32);
}
#define TG_U01(T) ((float)(tg_rand(T) >> 8) * (1.0f / 16777216.0f))
#define TG_N01(T) ((TG_U01(T) + TG_U01(T) + TG_U01(T) + TG_U01(T) - 2.0f) * 1.7320508f) // Irwin-Hall, unit variance

// Tournament of two over the ranks: the better rank's genome.
static inline uint32_t tg_tour(tg_learn *T)
{
    uint32_t a, b;

    a = tg_rand(T) % TG_P;
    b = tg_rand(T) % TG_P;
    return T->ord[a < b ? a : b];
}

static inline uint32_t tg_hget(const uint32_t *tab, uint32_t m, const char *const *name, uint32_t h, const char *s)
{
    uint32_t k;

    for (k = h & (m - 1); tab[k]; k = (k + 1) & (m - 1))
        if (!strcmp(name[tab[k] - 1], s)) return tab[k];
    return 0;
}

static inline void tg_hput(uint32_t *tab, uint32_t m, uint32_t h, uint32_t v)
{
    uint32_t k;

    for (k = h & (m - 1); tab[k]; k = (k + 1) & (m - 1));
    tab[k] = v;
}

// Sidecar of image path img: packed NUL-terminated names (malloc'd), *n of them; NULL if none. Decode workers.
static inline char *tg_read(const char *img, uint32_t *n)
{
    char path[4096], *buf, *o;
    const char *dot, *sl, *bs;
    FILE *f;
    size_t len, i;
    uint32_t tok, sp, ch;

    *n = 0;
    dot = strrchr(img, '.');
    sl = strrchr(img, '/');
    bs = strrchr(img, '\\');
    sl = bs > sl ? bs : sl;
    len = dot && (!sl || dot > sl) ? (size_t)(dot - img) : strlen(img);
    snprintf(path, sizeof(path), "%.*s.txt", (int)len, img);
    if (!(f = fopen(path, "rb"))) return NULL;
    if (!(buf = malloc(TG_SIDE + 2))) {
        fclose(f);
        return NULL;
    }
    len = fread(buf, 1, TG_SIDE, f);
    fclose(f);
    i = len >= 3 && !memcmp(buf, "\xEF\xBB\xBF", 3) ? 3 : 0;
    for (o = buf, tok = sp = 0; i <= len; i++) {
        ch = i < len ? (uint8_t)buf[i] : ',';
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') {
            if (tok) *o++ = 0, (*n)++;
            tok = sp = 0;
        } else if (ch == ' ') {
            sp = tok;
        } else if (ch != '\\') {
            if (sp) *o++ = '_';
            *o++ = (char)(ch - 'A' < 26u ? ch + 32 : ch);
            tok = 1;
            sp = 0;
        }
    }
    if (!*n) {
        free(buf);
        return NULL;
    }
    return buf;
}

// 1 if image path img has a sidecar (input ordering; no read).
static inline uint32_t tg_has(const char *img)
{
    char path[4096];
    const char *dot, *sl, *bs;
    size_t len;

    dot = strrchr(img, '.');
    sl = strrchr(img, '/');
    bs = strrchr(img, '\\');
    sl = bs > sl ? bs : sl;
    len = dot && (!sl || dot > sl) ? (size_t)(dot - img) : strlen(img);
    snprintf(path, sizeof(path), "%.*s.txt", (int)len, img);
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
}

// Category prefix: the name after it, *cat set (TG_NOCAT when none: general for a new tag, unchanged otherwise).
#define TG_NOCAT 255u
static inline const char *tg_cat(const char *s, uint32_t *cat)
{
    static const char *pre[6] = {"general:", "artist:", NULL, "copyright:", "character:", "meta:"};
    uint32_t c;

    for (c = 0, *cat = TG_NOCAT; c < 6; c++)
        if (pre[c] && !strncmp(s, pre[c], strlen(pre[c]))) {
            *cat = c;
            return s + strlen(pre[c]);
        }
    return s;
}

// User tag u's category (and with it threshold and target logit) set by a sidecar prefix.
static inline void tg_setcat(tg_learn *T, st_learn *L, pi_voc *wv, uint32_t u, uint32_t cat)
{
    uint32_t c;

    c = T->tag[u]->row;
    wv->cat[c] = (uint8_t)cat;
    wv->thr[c] = cat == 4 ? L->tchar : L->tgen;
    wv->chr[c] = cat == 4 ? 1.0f : 0.0f;
    L->lt[c] = T->tag[u]->lt = st_logit(wv->thr[c]);
}

// Raw Gram row of slot s (0 or a sidecar image) against every live slot; sidecar columns mirror it.
static inline void tg_grow(tg_learn *T, uint32_t s)
{
    uint32_t f;

    for (f = 0; f < TG_E; f++) {
        if (!TG_LIVE(T, f)) continue;
        TG_G(T, s)[f] = st_dot(TG_X(T, s), TG_X(T, f));
        if (f <= T->nl) TG_G(T, f)[s] = TG_G(T, s)[f];
    }
    T->dg[s] = TG_G(T, s)[s];
}

// Reservoir slot s against slot 0 and every sidecar slot.
static inline void tg_gcol(tg_learn *T, uint32_t s)
{
    uint32_t g;

    for (g = 0; g <= T->nl; g++) TG_G(T, g)[s] = st_dot(TG_X(T, g), TG_X(T, s));
    T->dg[s] = st_dot(TG_X(T, s), TG_X(T, s));
}

// Slot 0 <- the running mean; centred norms and centred Gram rows of the sidecar slots against it.
static inline void tg_refresh(tg_learn *T, const float *mu)
{
    const float *g0, *gr;
    float *cr, nn;
    __m256 a;
    uint32_t g, f;

    memcpy(TG_X(T, 0), mu, EVA_D * 4);
    memcpy(T->ms, mu, EVA_D * 4);
    tg_grow(T, 0);
    g0 = TG_G(T, 0);
    for (f = 0; f < TG_E; f++) {
        nn = T->dg[f] - 2.0f * g0[f] + g0[0];
        T->nf[f] = f && TG_LIVE(T, f) ? nn : 0.0f;
        T->rn[f] = f && TG_LIVE(T, f) && nn > 1e-6f ? 1.0f / sqrtf(nn) : 0.0f;
    }
    for (g = 1; g <= T->nl; g++) { // C[g][f] = G[g][f] - G[0][f] - (G[0][g] - G[0][0])
        gr = TG_G(T, g);
        cr = TG_C(T, g);
        a = _mm256_set1_ps(g0[g] - g0[0]);
        for (f = 0; f < TG_E; f += 8)
            _mm256_storeu_ps(cr + f, _mm256_sub_ps(_mm256_sub_ps(_mm256_loadu_ps(gr + f), _mm256_loadu_ps(g0 + f)), a));
    }
}

// The current tag's labelled positives, labelled negatives and background slots.
static inline void tg_labels(tg_learn *T, uint32_t u)
{
    uint32_t f;

    T->npos = T->nneg = T->nbg = 0;
    for (f = 1; f <= T->nl; f++) {
        if (TG_HAS(T, f, u)) T->pos[T->npos++] = f;
        else T->neg[T->nneg++] = T->bgi[T->nbg++] = f;
    }
    for (f = TG_L + 1; f <= TG_L + T->nr; f++) T->bgi[T->nbg++] = f;
}

// Hoare partition of a[lo..hi] around its middle value: on return a[lo..*j] <= p <= a[*i..hi], *j < *i.
static inline void tg_part(float *a, int32_t lo, int32_t hi, int32_t *pi, int32_t *pj)
{
    float p, t;
    int32_t i, j;

    p = a[(lo + hi) >> 1];
    for (i = lo, j = hi; i <= j;) {
        while (a[i] < p) i++;
        while (a[j] > p) j--;
        if (i <= j) t = a[i], a[i++] = a[j], a[j--] = t;
    }
    *pi = i;
    *pj = j;
}

// k-th smallest of a[0..n) (quickselect; a is reordered).
static inline float tg_select(float *a, uint32_t n, uint32_t k)
{
    int32_t lo, hi, i, j;

    for (lo = 0, hi = (int32_t)n - 1; lo < hi;) {
        tg_part(a, lo, hi, &i, &j);
        if ((int32_t)k <= j) hi = j;
        else if ((int32_t)k >= i) lo = i;
        else break;
    }
    return a[k];
}

// Gene e of unit U, U->n if absent.
static inline uint32_t tg_gfind(const tg_unit *U, uint32_t e)
{
    uint32_t i;

    for (i = 0; i < U->n && U->g[i].e != e; i++);
    return i;
}

// Unit dots over the store, dk[f] = c_k . (x_f - mu), and |c_k|^2.
static inline float tg_unitd(const tg_learn *T, const tg_unit *U, float *dk)
{
    const float *cr;
    __m256 a;
    float cc;
    uint32_t g, h, f;

    memset(dk, 0, TG_E * 4);
    for (g = 0, cc = 0.0f; g < U->n; g++) {
        a = _mm256_set1_ps(U->g[g].a);
        cr = TG_C(T, U->g[g].e);
        for (f = 0; f < TG_E; f += 8)
            _mm256_storeu_ps(dk + f, _mm256_fmadd_ps(a, _mm256_loadu_ps(cr + f), _mm256_loadu_ps(dk + f)));
        for (h = 0; h < U->n; h++) cc += U->g[g].a * U->g[h].a * cr[U->g[h].e];
    }
    return cc;
}

// Leave-one-out cosine of sidecar slot f: units holding f's gene score it without that gene.
static inline float tg_loo(const tg_learn *T, const tg_genome *gn, uint32_t f)
{
    float best, a, d, c, v;
    uint32_t k, i;

    for (k = 0, best = -2.0f; k < gn->k; k++) {
        i = tg_gfind(&gn->u[k], f);
        a = i < gn->u[k].n ? gn->u[k].g[i].a : 0.0f;
        d = T->zk[k][f] - a * T->nf[f];
        c = T->cc[k] - 2.0f * a * T->zk[k][f] + a * a * T->nf[f];
        v = c > 1e-9f && T->nf[f] > 1e-9f ? d / sqrtf(c * T->nf[f]) : -1.0f;
        best = v > best ? v : best;
    }
    return best;
}

// Fitness under the current tag's labels (see the header); leaves unit dots in zk, scores in s, the background
// sigma in sig and the cutoff med + k sig in tau.
static inline float tg_eval(tg_learn *T, tg_genome *gn)
{
    __m256 m, v;
    float med, sp, sn, st, x, is;
    uint32_t k, f, i, ng;

    for (k = 0, ng = 0; k < gn->k; k++) {
        T->cc[k] = tg_unitd(T, &gn->u[k], T->zk[k]);
        T->ic[k] = T->cc[k] > 1e-9f ? 1.0f / sqrtf(T->cc[k]) : 0.0f;
        ng += gn->u[k].n;
    }
    for (f = 0; f < TG_E; f += 8) { // cosines (rn > 0 scales every unit alike, but the -2 floor is a cosine)
        v = _mm256_loadu_ps(T->rn + f);
        for (k = 0, m = _mm256_set1_ps(-2.0f); k < gn->k; k++)
            m = _mm256_max_ps(m, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(T->zk[k] + f), _mm256_set1_ps(T->ic[k])), v));
        _mm256_storeu_ps(T->s + f, m);
    }
    for (i = 0; i < T->nbg; i++) T->bv[i] = T->s[T->bgi[i]];
    med = T->nbg ? tg_select(T->bv, T->nbg, T->nbg >> 1) : 0.0f;
    for (i = 0; i < T->nbg; i++) T->bv[i] = fabsf(T->bv[i] - med);
    T->sig = T->nbg >= TG_BMIN ? 1.4826f * tg_select(T->bv, T->nbg, T->nbg >> 1) : TG_S0;
    T->sig = T->sig > TG_SMIN ? T->sig : TG_SMIN;
    T->tau = med + T->ksig * T->sig;
    is = 1.0f / T->sig;
    for (i = 0, sp = 0.0f; i < T->npos; i++) {
        x = (tg_loo(T, gn, T->pos[i]) - T->tau) * is;
        sp += x < TG_ZCAP ? x : TG_ZCAP;
    }
    for (i = 0, sn = 0.0f; i < T->nneg; i++) {
        x = (T->s[T->neg[i]] - T->tau) * is + TG_ZNEG;
        sn += x > 0.0f ? x : 0.0f;
    }
    for (i = 0, st = 0.0f; i < T->nbg; i++) { // the tail median and MAD cannot see: background over tau
        x = (T->s[T->bgi[i]] - T->tau) * is;
        st += x <= 0.0f ? 0.0f : x < 1.0f ? x : 1.0f;
    }
    return gn->fit = sp / (float)(T->npos ? T->npos : 1) - TG_NEGW * sn / (float)(T->nneg ? T->nneg : 1)
                   - TG_TAILW * st / (float)(T->nbg ? T->nbg : 1) - TG_CPLX * (float)ng;
}

// a joins unit U at slot e (merged into an existing gene); skipped when U is full.
static inline void tg_gadd(tg_unit *U, uint32_t e, float a)
{
    uint32_t i;

    i = tg_gfind(U, e);
    if (i == U->n && U->n == TG_GN) return;
    if (i == U->n) U->g[U->n++] = (tg_gene){e, 0.0f};
    U->g[i].a += a;
}

// NEAT crossover: c holds the fitter parent; units of b with the same innovation lend matching genes half the time;
// disjoint structure stays the fitter parent's.
static inline void tg_cross(tg_learn *T, tg_genome *c, const tg_genome *b)
{
    const tg_unit *B;
    uint32_t k, j, i;

    for (k = 0; k < c->k; k++) {
        for (j = 0; j < b->k && b->u[j].inn != c->u[k].inn; j++);
        if (j == b->k) continue;
        B = &b->u[j];
        for (i = 0; i < c->u[k].n; i++)
            if ((j = tg_gfind(B, c->u[k].g[i].e)) < B->n && (tg_rand(T) & 1)) c->u[k].g[i].a = B->g[j].a;
    }
}

// Gene e out of unit U (the unit keeps at least its birth gene).
static inline void tg_gdel(tg_unit *U, uint32_t e)
{
    uint32_t i;

    if ((i = tg_gfind(U, e)) < U->n && U->n > 1) U->g[i] = U->g[--U->n];
}

// The positive outside unit U (index k of the last evaluation) with the highest cosine to it, 0 if none.
static inline uint32_t tg_near(const tg_learn *T, const tg_unit *U, uint32_t k)
{
    float best, v;
    uint32_t i, f, e;

    for (i = 0, e = 0, best = -2.0f; i < T->npos; i++) {
        f = T->pos[i];
        v = T->zk[k][f] * T->rn[f];
        if (tg_gfind(U, f) == U->n && v > best) best = v, e = f;
    }
    return e;
}

// add-node: positive e leaves every unit and starts its own (nothing if it already stands alone).
static inline void tg_split(tg_genome *gn, uint32_t e)
{
    tg_unit *U;
    uint32_t k;

    for (k = 0; k < gn->k; k++)
        if (gn->u[k].n == 1 && gn->u[k].g[0].e == e) return;
    for (k = 0; k < gn->k; k++) {
        tg_gdel(&gn->u[k], e);
        gn->u[k].inn = gn->u[k].inn == e ? gn->u[k].g[0].e : gn->u[k].inn;
    }
    U = &gn->u[gn->k++];
    U->inn = e;
    U->n = 1;
    U->b = 0.0f;
    U->g[0] = (tg_gene){e, TG_WP};
}

// Structural mutations only (weights are memberships: a positive is in a unit at TG_WP, a confuser at TG_WN times
// the unit's positives), aimed at the genome's worst sidecar images: the positive with the lowest leave-one-out
// cosine, the negative with the highest score.
static inline void tg_mutate(tg_learn *T, tg_genome *gn)
{
    tg_unit *U;
    float v, wp, wn, np;
    uint32_t i, ep, en, k, kb;

    tg_eval(T, gn);
    for (i = 0, ep = 0, wp = T->tau + TG_ZCAP * T->sig; i < T->npos; i++)
        if ((v = tg_loo(T, gn, T->pos[i])) < wp) wp = v, ep = T->pos[i];
    for (i = 0, en = 0, wn = T->tau - TG_ZNEG * T->sig; i < T->nneg; i++)
        if ((v = T->s[T->neg[i]]) > wn) wn = v, en = T->neg[i];
    if (!ep && T->npos && TG_U01(T) < 0.3f) ep = T->pos[tg_rand(T) % T->npos]; // keep exploring the partition
    if (ep && gn->k < TG_K && TG_U01(T) < TG_PNODE) { // add-node
        tg_split(gn, ep);
    } else if (ep && TG_U01(T) < TG_PCONN) { // add-connection: the positive joins the unit closest to it; already
                                              // in it (an outlier, or alone): the closest positive outside joins
        for (k = 1, kb = 0; k < gn->k; k++) kb = T->zk[k][ep] * T->ic[k] > T->zk[kb][ep] * T->ic[kb] ? k : kb;
        if (tg_gfind(&gn->u[kb], ep) < gn->u[kb].n) ep = tg_near(T, &gn->u[kb], kb);
        if (ep) tg_gadd(&gn->u[kb], ep, TG_WP);
    } else if (en && TG_U01(T) < TG_PCONN) { // add-connection: the confuser is subtracted from the unit firing on it
        for (k = 1, kb = 0; k < gn->k; k++) kb = T->zk[k][en] * T->ic[k] > T->zk[kb][en] * T->ic[kb] ? k : kb;
        U = &gn->u[kb];
        for (i = 0, np = 0.0f; i < U->n; i++) np += U->g[i].a > 0.0f ? U->g[i].a : 0.0f;
        if (tg_gfind(U, en) == U->n) tg_gadd(U, en, -TG_WN * np);
    }
    if (TG_U01(T) < 0.1f) { // prune a random gene (never a unit's birth gene)
        U = &gn->u[tg_rand(T) % gn->k];
        i = tg_rand(T) % U->n;
        if (U->g[i].e != U->inn) U->g[i] = U->g[--U->n];
    }
    if (gn->k > 1 && TG_U01(T) < 0.03f) { // drop a unit
        k = tg_rand(T) % gn->k;
        gn->u[k] = gn->u[--gn->k];
    }
}

// One generation: rank, keep the elite and each species' best, fill by tournament, crossover, mutation.
static inline void tg_gen(tg_learn *T, tg_tag *tg)
{
    tg_genome *p;
    uint32_t i, j, k, n, a, b, best;

    p = tg->pop;
    for (i = 0; i < TG_P; i++) T->ord[i] = i;
    for (i = 1; i < TG_P; i++)
        for (j = i; j && p[T->ord[j]].fit > p[T->ord[j - 1]].fit; j--) {
            k = T->ord[j];
            T->ord[j] = T->ord[j - 1];
            T->ord[j - 1] = k;
        }
    n = 0;
    T->nx[n++] = p[T->ord[0]];
    for (k = 1; k <= TG_K; k++) {
        for (i = 1; i < TG_P && p[T->ord[i]].k != k; i++);
        if (i < TG_P && p[T->ord[0]].k != k) T->nx[n++] = p[T->ord[i]];
    }
    while (n < TG_P) {
        a = tg_tour(T);
        b = tg_tour(T);
        if (p[b].fit > p[a].fit) k = a, a = b, b = k;
        T->nx[n] = p[a];
        if (a != b && p[b].k == p[a].k && TG_U01(T) < TG_PX) tg_cross(T, &T->nx[n], &p[b]);
        tg_mutate(T, &T->nx[n]);
        tg_eval(T, &T->nx[n]);
        n++;
    }
    for (i = 1, best = 0; i < TG_P; i++) best = T->nx[i].fit > T->nx[best].fit ? i : best;
    memcpy(p, T->nx, sizeof(T->nx));
    p[0] = T->nx[best];
    p[best] = T->nx[0];
}

// Elite -> scoring directions (c_k / |c_k|, centred on ms) and its cutoff. Labels of the tag must be current.
static inline void tg_mat(tg_learn *T, tg_tag *tg)
{
    const tg_genome *gn;
    float sa;
    uint32_t k, g;

    gn = &tg->pop[0];
    tg_eval(T, &tg->pop[0]);
    tg->tau = T->tau;
    tg->isig = 1.0f / T->sig;
    for (k = 0; k < gn->k; k++) {
        memset(tg->w[k], 0, EVA_D * 4);
        for (g = 0, sa = 0.0f; g < gn->u[k].n; g++) {
            st_axpy(tg->w[k], TG_X(T, gn->u[k].g[g].e), gn->u[k].g[g].a * T->ic[k]);
            sa += gn->u[k].g[g].a;
        }
        st_axpy(tg->w[k], T->ms, -sa * T->ic[k]);
    }
    tg->k = gn->k;
#ifdef TG_DEBUG // the elite: units (birth slot, genes), cutoff, fitness, background size
    fprintf(stderr, "tgmat row %u tau %.3f sig %.3f fit %.3f bg %u pos %u neg %u units", tg->row, tg->tau, T->sig,
            gn->fit, T->nbg, T->npos, T->nneg);
    for (k = 0; k < gn->k; k++) fprintf(stderr, " [%u: %u genes]", gn->u[k].inn, gn->u[k].n);
    for (k = 0; k < gn->k; k++)
        for (g = 0; g < gn->u[k].n; g++) fprintf(stderr, " %u:%u*%.3f", k, gn->u[k].g[g].e, gn->u[k].g[g].a);
    fputc('\n', stderr);
#endif
}

// Population of a tag from its positives: one unit on the first, the others joining at random weights.
static inline void tg_seed(tg_learn *T, tg_tag *tg)
{
    tg_unit *U;
    uint32_t i, j;

    for (i = 0; i < TG_P; i++) {
        tg->pop[i].k = 1;
        U = &tg->pop[i].u[0];
        U->inn = T->pos[0];
        U->n = 1;
        U->b = 0.0f;
        U->g[0] = (tg_gene){T->pos[0], TG_WP};
        for (j = 1; j < T->npos; j++)
            if (!i || (tg_rand(T) & 1)) tg_gadd(U, T->pos[j], TG_WP); // [0]: the centroid; the rest random subsets
    }
}

// One unit holding every positive (the first TG_GN): their centroid.
static inline void tg_centroid(tg_learn *T, tg_genome *gn)
{
    tg_unit *U;
    uint32_t j;

    gn->k = 1;
    U = &gn->u[0];
    U->inn = T->pos[0];
    U->b = 0.0f;
    for (j = 0, U->n = 0; j < T->npos && j < TG_GN; j++) U->g[U->n++] = (tg_gene){T->pos[j], TG_WP};
}

// ngen generations for every user tag with a positive, against the store as it is now. Nothing until the mean has
// ST_MU0 images.
static inline void tg_evolve(tg_learn *T, const float *mu, uint32_t nmu, uint32_t ngen)
{
    tg_tag *tg;
    uint32_t u, i, g, w;

    if (nmu < ST_MU0) return;
    tg_refresh(T, mu);
    for (u = 0; u < T->nu; u++) {
        tg = T->tag[u];
        if (!tg->np) continue;
        tg_labels(T, u);
        if (!T->npos) continue;
        if (!tg->pop[0].k) tg_seed(T, tg);
        for (i = 0, w = 0; i < TG_P; i++) w = tg_eval(T, &tg->pop[i]) < tg->pop[w].fit ? i : w;
        tg_centroid(T, &tg->pop[w]); // immigrant: the centroid of every positive so far replaces the worst
        tg_eval(T, &tg->pop[w]);
        for (g = 0; g < ngen; g++) tg_gen(T, tg);
        tg_mat(T, tg);
    }
}

// New user tag on the next free extra row (-1: out of room).
static inline int32_t tg_new(tg_learn *T, st_learn *L, pi_voc *wv, const char *nm, uint32_t h, uint32_t cat)
{
    tg_tag *tg;
    char *name;
    uint32_t c;

    if (T->nu == TG_U || L->ns == L->cap) return -1;
    tg = _mm_malloc(sizeof(tg_tag), 64);
    name = strdup(nm);
    if (!tg || !name) {
        _mm_free(tg);
        free(name);
        return -1;
    }
    memset(tg, 0, sizeof(*tg));
    c = L->nt0 + L->ns++;
    wv->name[c] = name;
    wv->hash[c] = h;
    wv->mvimask[c] = 1.0f;
    wv->shared[c] = 0; // whi / wlo stay 2: PixAI never teaches a user row
    tg->row = c;
    tg_hput(T->wt, T->wm, h, c + 1);
    T->ut[c - L->nt0] = (int32_t)T->nu;
    T->tag[T->nu++] = tg;
    tg_setcat(T, L, wv, T->nu - 1, cat == TG_NOCAT ? 0 : cat);
    return (int32_t)c;
}

// wd row for sidecar name nm (see the header), -1 when out of room.
static inline int32_t tg_row(tg_learn *T, st_learn *L, pi_voc *wv, pi_voc *pv, const char *nm, uint32_t cat)
{
    uint32_t h, r;

    h = pi_hash(nm);
    if ((r = tg_hget(T->wt, T->wm, wv->name, h, nm))) return (int32_t)(r - 1);
    if (T->pm && (r = tg_hget(T->pt, T->pm, pv->name, h, nm))) {
        if (L->pw[r - 1] < 0 && L->ns < L->cap) st_imprint(L, wv, pv, r - 1); // embedding already in L->x
        return L->pw[r - 1];
    }
    return tg_new(T, L, wv, nm, h, cat);
}

// Sidecar slot already holding embedding x (|x|^2 = xx), 0 if none.
static inline uint32_t tg_same(const tg_learn *T, const float *x, float xx)
{
    uint32_t f;

    for (f = 1; f <= T->nl; f++)
        if (xx - 2.0f * st_dot(x, TG_X(T, f)) + T->dg[f] <= 1e-4f * xx) return f;
    return 0;
}

// A sidecar image (embedding x16, wd scores ws of the same pass, rewritten for rows that moved): store it, teach the
// rows its names land on, evolve the user tags.
static inline int tg_side(tg_learn *T, st_learn *L, eva *e, pi_voc *wv, pi_voc *pv, const uint16_t *x16, float *ws,
                          const char *side, uint32_t nside)
{
    const char *s, *nm;
    float xx;
    uint32_t i, s0, cat;
    int32_t c, u;

    T->nside++;
    st_h2f(L->x, x16);
    xx = st_dot(L->x, L->x);
    s0 = tg_same(T, L->x, xx); // the same image again (a rerun): its slot, labels merged
    if (!s0 && T->nl < TG_L) {
        s0 = ++T->nl;
        memcpy(TG_X(T, s0), L->x, EVA_D * 4);
        memset(T->lab[s0], 0, sizeof(T->lab[s0]));
        tg_grow(T, s0);
    } else if (!s0 && !T->full++) {
        fprintf(stderr, "tagglomeration: %u sidecar images stored, later ones teach wd's rows only\n", TG_L);
    }
    for (i = 0, s = side; i < nside; i++, s += strlen(s) + 1) {
        nm = tg_cat(s, &cat);
        if (!*nm || (c = tg_row(T, L, wv, pv, nm, cat)) < 0) continue;
        u = (uint32_t)c >= L->nt0 ? T->ut[c - L->nt0] : -1;
        if (u >= 0) {
            if (cat != TG_NOCAT && cat != wv->cat[c]) tg_setcat(T, L, wv, (uint32_t)u, cat);
            if (!s0 || TG_HAS(T, s0, (uint32_t)u)) continue;
            T->lab[s0][u >> 6] |= 1ull << (u & 63);
            T->tag[u]->np++;
        } else if (wv->cat[c] != 9 && ws[c] < L->whi[c]) {
            if (st_row(L, e, (uint32_t)c)) return -1;
            st_nlms(L, (uint32_t)c, xx, L->lt[c] + ST_M);
            ws[c] = ST_SIG(st_dot(ST_ROW(L, c), L->x) + L->b[c]);
        }
    }
    if (T->nu) tg_evolve(T, L->mu, L->nmu, TG_GEN);
    return 0;
}

// Every other wd full-image embedding: reservoir sample; every TG_RGEN insertions one maintenance generation.
static inline void tg_observe(tg_learn *T, const float *mu, uint32_t nmu, const uint16_t *x16)
{
    uint32_t s, j;

    T->nres++;
    if (T->nr < TG_R) s = TG_L + 1 + T->nr++;
    else if ((j = tg_rand(T) % T->nres) < TG_R) s = TG_L + 1 + j;
    else return;
    st_h2f(TG_X(T, s), x16);
    tg_gcol(T, s);
    if (!(++T->nins % TG_RGEN) && T->nu) tg_evolve(T, mu, nmu, 1);
}

// User-tag scores of one wd whole-image pass (embedding x16) into out: lt + TG_GZ (max_k cos - tau) / sigma.
static inline void tg_score(tg_learn *T, const uint16_t *x16, float *out)
{
    const tg_tag *tg;
    float z, m, inv;
    uint32_t u, k;

    if (!T->nu) return;
    for (k = 0; k < EVA_D; k += 8)
        _mm256_storeu_ps(T->xs + k, _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x16 + k))),
                                                  _mm256_loadu_ps(T->ms + k)));
    inv = 1.0f / sqrtf(st_dot(T->xs, T->xs) + 1e-9f);
    for (u = 0; u < T->nu; u++) {
        tg = T->tag[u];
        for (k = 0, m = -1.0f; k < tg->k; k++) {
            z = st_dot(tg->w[k], T->xs) * inv;
            m = z > m ? z : m;
        }
        out[tg->row] = tg->k ? ST_SIG(tg->lt + TG_GZ * (m - tg->tau) * tg->isig) : 0.0f;
    }
}

// After st_load: name tables over the rows wd has now, and PixAI's names.
static inline int tg_init(tg_learn *T, const st_learn *L, const pi_voc *wv, const pi_voc *pv, float ksig)
{
    uint32_t i;

    memset(T, 0, sizeof(*T));
    T->rng = 0x9E3779B97F4A7C15ull;
    T->ksig = ksig;
    T->x = _mm_malloc((size_t)TG_E * EVA_D * 4, 64);
    T->G = _mm_malloc((size_t)(TG_L + 1) * TG_E * 4, 64);
    T->C = _mm_malloc((size_t)(TG_L + 1) * TG_E * 4, 64);
    T->lab = calloc(TG_E, sizeof(*T->lab));
    T->ut = malloc((size_t)L->cap * 4 + 4);
    T->wm = 1u << (32 - __builtin_clz(2 * wv->nt));
    T->wt = calloc(T->wm, 4);
    T->pm = pv->nt ? 1u << (32 - __builtin_clz(2 * pv->nt)) : 0;
    T->pt = T->pm ? calloc(T->pm, 4) : NULL;
    if (!T->x || !T->G || !T->C || !T->lab || !T->ut || !T->wt || (T->pm && !T->pt)) return -1;
    memset(T->x, 0, (size_t)TG_E * EVA_D * 4);
    memset(T->G, 0, (size_t)(TG_L + 1) * TG_E * 4);
    memset(T->C, 0, (size_t)(TG_L + 1) * TG_E * 4);
    for (i = 0; i < L->cap; i++) T->ut[i] = -1;
    for (i = 0; i < L->nt0 + L->ns; i++) tg_hput(T->wt, T->wm, wv->hash[i], i + 1);
    for (i = 0; i < pv->nt; i++) tg_hput(T->pt, T->pm, pv->hash[i], i + 1);
    return 0;
}

// Section after the stolen rows: u32 magic, nu, nl, nr, nres, EVA_D; float x[1 + nl][EVA_D], x[nr][EVA_D] (the
// reservoir); u64 lab[1 + nl][TG_U / 64]; per user tag: u16 name length, name, u8 category, u32 np, elite genome.
static inline void tg_save(const tg_learn *T, const pi_voc *wv, FILE *f)
{
    const tg_tag *tg;
    uint32_t hdr[6], u, np;
    uint16_t nl;
    uint8_t cat;

    hdr[0] = TG_MAGIC;
    hdr[1] = T->nu;
    hdr[2] = T->nl;
    hdr[3] = T->nr;
    hdr[4] = T->nres;
    hdr[5] = EVA_D;
    fwrite(hdr, 4, 6, f);
    fwrite(T->x, 4, (size_t)(1 + T->nl) * EVA_D, f);
    fwrite(TG_X(T, TG_L + 1), 4, (size_t)T->nr * EVA_D, f);
    fwrite(T->lab, sizeof(*T->lab), 1 + T->nl, f);
    for (u = 0; u < T->nu; u++) {
        tg = T->tag[u];
        nl = (uint16_t)strlen(wv->name[tg->row]);
        cat = wv->cat[tg->row];
        np = tg->np;
        fwrite(&nl, 2, 1, f);
        fwrite(wv->name[tg->row], 1, nl, f);
        fwrite(&cat, 1, 1, f);
        fwrite(&np, 4, 1, f);
        fwrite(&tg->pop[0], sizeof(tg_genome), 1, f);
    }
}

// A genome the store can evaluate: unit and gene counts in range, every gene on a stored sidecar slot (genes on
// slot 0 come from the projection learner this replaced: those genomes are reseeded).
static inline uint32_t tg_valid(const tg_learn *T, const tg_genome *gn)
{
    uint32_t k, i, ok;

    ok = gn->k >= 1 && gn->k <= TG_K;
    for (k = 0; ok && k < gn->k; k++) {
        ok &= gn->u[k].n >= 1 && gn->u[k].n <= TG_GN && gn->u[k].inn <= T->nl;
        for (i = 0; ok && i < gn->u[k].n; i++) ok &= gn->u[k].g[i].e >= 1 && gn->u[k].g[i].e <= T->nl;
    }
    return ok;
}

static inline void tg_load(tg_learn *T, st_learn *L, pi_voc *wv, FILE *f)
{
    tg_tag *tg;
    char name[1024];
    uint32_t hdr[6], u, g, np;
    uint16_t nl;
    uint8_t cat;

    if (!f || fread(hdr, 4, 6, f) != 6 || hdr[0] != TG_MAGIC || hdr[5] != EVA_D || hdr[1] > TG_U || hdr[2] > TG_L
        || hdr[3] > TG_R)
        return;
    T->nl = hdr[2];
    T->nr = hdr[3];
    T->nres = hdr[4];
    if (fread(T->x, 4, (size_t)(1 + T->nl) * EVA_D, f) != (size_t)(1 + T->nl) * EVA_D
        || fread(TG_X(T, TG_L + 1), 4, (size_t)T->nr * EVA_D, f) != (size_t)T->nr * EVA_D
        || fread(T->lab, sizeof(*T->lab), 1 + T->nl, f) != 1 + T->nl) {
        T->nl = T->nr = T->nres = 0;
        return;
    }
    for (g = 1; g <= T->nl; g++) tg_grow(T, g);
    for (g = TG_L + 1; g <= TG_L + T->nr; g++) T->dg[g] = st_dot(TG_X(T, g), TG_X(T, g));
    memcpy(T->xs, T->x, EVA_D * 4);
    tg_refresh(T, T->xs); // the mean the genomes were built on
    for (u = 0; u < hdr[1]; u++) {
        if (fread(&nl, 2, 1, f) != 1 || nl >= sizeof(name) || fread(name, 1, nl, f) != nl || fread(&cat, 1, 1, f) != 1
            || fread(&np, 4, 1, f) != 1 || fread(&T->nx[0], sizeof(tg_genome), 1, f) != 1)
            break;
        name[nl] = 0;
        if (tg_new(T, L, wv, name, pi_hash(name), cat) < 0) { // label bits are by tag order: stop here
            fprintf(stderr, "tagglomeration: no room for user tag %s and later ones (-sn)\n", name);
            break;
        }
        tg = T->tag[u];
        tg->np = np;
        tg_labels(T, u);
        if (!T->npos) continue;
        if (tg_valid(T, &T->nx[0])) tg->pop[0] = T->nx[0];
        else tg_seed(T, tg);
        for (g = 1; g < TG_P; g++) {
            tg->pop[g] = tg->pop[0];
            tg_mutate(T, &tg->pop[g]);
        }
        tg_mat(T, tg);
    }
}

static inline void tg_free(tg_learn *T, pi_voc *wv)
{
    uint32_t u;

    for (u = 0; u < T->nu; u++) {
        free((char *)wv->name[T->tag[u]->row]);
        _mm_free(T->tag[u]);
    }
    _mm_free(T->x);
    _mm_free(T->G);
    _mm_free(T->C);
    free(T->lab); free(T->ut); free(T->wt); free(T->pt);
}

#endif // TAGG_H

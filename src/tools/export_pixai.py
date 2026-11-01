# model_pixai.onnx (PixAI tagger, ViTDet-L 1008^2) -> pixai.bin for px.h.
# Tokens live in window-major order for the whole network (3 x 3 windows of 24 x 24 patches, window w = tokens
# 576w .. 576w + 575): windowed blocks are plain attention over 576-token sequences, global blocks attend over all
# 5184, and every position-dependent table (tiled abs pos, global RoPE, head pos) is permuted here once.
# Wq rows and q bias carry 1/8 * log2(e) (scores in the exp2 domain). The attention-pool head has a constant query,
# so its key projection folds into per-head score vectors U [16][1024] (scores = U . (LN(x) + pos)); the value
# projection Wv is applied to the 16 pooled vectors, and the pool's output projection plus both biases fold into
# the tag head. Every tensor starts on a 256-byte boundary.
#
#   python export_pixai.py model_pixai.onnx config_pixai.json pixai_fp16.bin              v2: fp16 block weights
#   python export_pixai.py model_pixai.onnx config_pixai.json pixai.bin pixai_fp16.bin.calib   v3: E4M3
#
# v3: SmoothQuant smoothing folded into LN1 / LN2 gamma-beta (QKV, fc1 inputs), the V rows (out-proj input) and,
# for the GELU output feeding fc2, into per-unit quant multipliers rs (applied by the fc1 epilogue). Table
# [32][wdq x4, adq x4, q, k, v, 0] leads the blob. Also writes <out>.tags: name,category per output (category ids as
# selected_tags.csv: 0 general, 4 character, 3 copyright, 1 style, 5 meta, 9 rating) and thresholds.
import os, sys, json, struct, onnx, numpy as np
from onnx import numpy_helper

D, H, HID, T, NW, WS, K0, K0P, NB = 1024, 16, 4736, 5184, 9, 24, 588, 592, 32
GLOB = (7, 15, 23, 31)
E4M3 = 448.0
HEADROOM = float(os.environ.get("EVA_HEADROOM", 1.25))
ALPHA = float(os.environ.get("EVA_ALPHA", 0.5))

m = onnx.load(sys.argv[1])
cfg = json.load(open(sys.argv[2], encoding="utf-8"))
out = sys.argv[3]
g = m.graph
W = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
mats = [[i for i in n.input if i in W][0] for n in g.node if n.op_type in ("MatMul", "Gemm") and any(i in W for i in n.input)]
assert len(mats) == 132
calib = None
if len(sys.argv) > 4:
    import ml_dtypes
    raw = open(sys.argv[4], "rb").read()
    na = NB * 4 * HID
    calib = np.frombuffer(raw, np.float32, na).reshape(NB, 4, HID)
    qkvmax = np.frombuffer(raw, np.float32, NB * 3 * D, na * 4).reshape(NB, 3, D)

# window-major token i -> row-major token
i = np.arange(T)
wi, l = i // (WS * WS), i % (WS * WS)
perm = ((wi // 3) * WS + l // WS) * 72 + (wi % 3) * WS + l % WS

blob = bytearray()
def put(a, dt):
    global blob
    blob += b"\0" * (-len(blob) % 256)
    blob += np.ascontiguousarray(a, dtype=dt).tobytes()

def smooth(a, w):
    wa = np.abs(w).max(0)
    ok = (a > 0) & (wa > 0)
    s = np.ones_like(a)
    s[ok] = np.clip(a[ok] ** ALPHA / wa[ok] ** (1 - ALPHA), 1e-3, 1e3)
    return s

def fp8(w):
    sc = float(np.abs(w).max()) / E4M3
    return np.clip(w / sc, -E4M3, E4M3).astype(ml_dtypes.float8_e4m3fn).view(np.uint8), sc

qs = float(W["val_146"].ravel()[0]) ** 2 * np.log2(np.e)  # q and k each carry sqrt(1/8) in the graph
pe = np.zeros((D, K0P), np.float32); pe[:, :K0] = W["model.patch_embed.proj.weight"].reshape(D, K0)
pos = W["model.pos_embed"][0, 1:].reshape(24, 24, D)
pos = np.tile(pos, (3, 3, 1)).reshape(T, D)[perm]
def cs(c, s): return np.stack([c, s], -1).astype(np.float32)  # [N][32][2]
cs_win = cs(W["model.blocks.0.attn._rope_cos"], W["model.blocks.0.attn._rope_sin"])
cs_glob = cs(W["model.blocks.7.attn._rope_cos"], W["model.blocks.7.attn._rope_sin"])[perm]

table = np.zeros((NB, 12), np.float32)
layers = []
for b in range(NB):
    p = f"model.blocks.{b}"
    w1, b1 = W[f"{p}.norm1.weight"].copy(), W[f"{p}.norm1.bias"].copy()
    wqkv = W[mats[4 * b]].T.copy()                        # [3072][1024]
    bqkv = W[f"{p}.attn.qkv.bias"].copy()
    wqkv[:D] *= qs; bqkv[:D] *= qs
    wo = W[mats[4 * b + 1]].T.copy(); bo = W[f"{p}.attn.proj.bias"].copy()
    w2, b2 = W[f"{p}.norm2.weight"].copy(), W[f"{p}.norm2.bias"].copy()
    wf1 = W[mats[4 * b + 2]].T.copy(); bf1 = W[f"{p}.mlp.fc1.bias"].copy()
    wf2 = W[mats[4 * b + 3]].T.copy(); bf2 = W[f"{p}.mlp.fc2.bias"].copy()
    rs = np.ones(HID, np.float32)
    mm = [wqkv, wo, wf1, wf2]
    if calib is not None:
        a = [calib[b, 0, :D], calib[b, 1, :D], calib[b, 2, :D], calib[b, 3]]
        s = [smooth(a[0], wqkv), smooth(a[1], wo), smooth(a[2], wf1), smooth(a[3], wf2)]
        b1 += 0  # LN1 -> QKV: x_hat * g + b  ->  (x_hat * g + b) / s, weight columns * s
        w1 /= s[0]; b1 /= s[0]; wqkv *= s[0]
        wqkv[2 * D:] /= s[1][:, None]; bqkv[2 * D:] /= s[1]; wo *= s[1]   # V channels -> out-proj
        w2 /= s[2]; b2 /= s[2]; wf1 *= s[2]                               # LN2 -> fc1
        wf2 *= s[3]                                                       # GELU out / s3 -> fc2
        for k in range(4):
            mm[k], table[b, k] = fp8(mm[k])
            table[b, 4 + k] = float((a[k] / s[k]).max()) * HEADROOM / E4M3
        table[b, 8] = float(qkvmax[b, 0].max()) * HEADROOM / E4M3
        table[b, 9] = float(qkvmax[b, 1].max()) * HEADROOM / E4M3
        table[b, 10] = float((qkvmax[b, 2] / s[1]).max()) * HEADROOM / E4M3
        rs = (1.0 / table[b, 7]) / s[3]
    layers.append((w1, b1, mm, bqkv, bo, w2, b2, bf1, rs, bf2))

# attention-pool head
hq = W["val_4727"].reshape(H, 64)                        # constant query (used as is), keys scaled by sqrt(1/8)
kvw = W[mats[128]]                                        # [1024 in][2048 out]: k | v
kvb = W["model.head_pool.kv.bias"]
ks = float(W["val_146"].ravel()[0]) * np.log2(np.e)
U = np.stack([kvw[:, 64 * h:64 * h + 64] @ hq[h] for h in range(H)]) * ks   # [16][1024]; q . bk is per-head const
hpos = W["model.head_pool.pos_embed"][perm]                # [5184][1024], added by the head LN kernel
wv = kvw[:, D:].T.copy(); bv = kvb[D:].copy()              # [1024 out][1024 in]
wp = W[mats[130]].T.copy(); bp = W["model.head_pool.proj.bias"]
hw = W["model.head.weight"]; hb = W["model.head.bias"]
# logits = Wh (Wp (Wv . pooled + bv) + bp) + hb = (Wh Wp) (Wv . pooled) + Wh (Wp bv + bp) + hb: proj and both biases fold
hb = hw @ (wp @ bv + bp) + hb
hw = hw @ wp
NT = hw.shape[0]; NTP = (NT + 7) & ~7
hwp = np.zeros((NTP, D), np.float32); hwp[:NT] = hw
hbp = np.zeros(NTP, np.float32); hbp[:NT] = hb

if calib is not None: put(table, np.float32)
put(pe, np.float16)
put(pos, np.float32)
put(W["model.ln_pre.weight"], np.float32); put(W["model.ln_pre.bias"], np.float32)
put(cs_win, np.float32); put(cs_glob, np.float32)
wt = np.uint8 if calib is not None else np.float16
for w1, b1, mm, bqkv, bo, w2, b2, bf1, rs, bf2 in layers:
    put(w1, np.float32); put(b1, np.float32)
    put(mm[0], wt); put(bqkv, np.float32)
    put(mm[1], wt); put(bo, np.float32)
    put(w2, np.float32); put(b2, np.float32)
    put(mm[2], wt); put(bf1, np.float32)
    put(rs, np.float32)
    put(mm[3], wt); put(bf2, np.float32)
put(W["model.head_pool.norm.weight"], np.float32); put(W["model.head_pool.norm.bias"], np.float32)
put(U, np.float32); put(hpos, np.float16)
put(wv, np.float16)
put(hwp, np.float16); put(hbp, np.float32)

ver = 3 if calib is not None else 2
with open(out, "wb") as f:
    f.write(struct.pack("<4sIII", b"PXAI", ver, NT, len(blob)))
    f.write(b"\0" * 240)
    f.write(blob)
# vocabulary: name, category id (selected_tags.csv convention), threshold
cat_id = {"general": 0, "character": 4, "copyright": 3, "style": 1, "meta": 5, "rating": 9}
thr = cfg["category_best_threshold"]
with open(out + ".tags", "w", encoding="utf-8", newline="\n") as f:
    k = 0
    for cat, n in cfg["tags_split"]:
        for j in range(n):
            f.write(f"{cfg['tags'][k]}\t{cat_id[cat]}\t{thr[cat]}\n")
            k += 1
print(f"{out}: v{ver}, {NT} tags, {len(blob)} bytes")

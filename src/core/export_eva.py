# model.onnx (wd-eva02-large-tagger-v3) -> eva02.bin for eva.h.
# Weights transposed to [out][in] (TN GEMMs), QKV fused, fc1_g|fc1_x fused, softmax scale * log2(e) in Wq,
# cls/conv bias folded into the position table. Every tensor starts on a 256-byte boundary.
#
#   python export_eva.py model.onnx eva02.bin                    v2: fp16 block weights (calibration engine)
#   python export_eva.py model.onnx eva02.bin eva02.bin.calib     v3: E4M3 block weights (production engine)
#
# v3: per-channel |max| of every block GEMM input (from the -DEVA_CALIB engine) drives SmoothQuant smoothing,
# folded exactly into the producers - LN gamma/beta (LN1, LN2, MLP LN) or the Wv rows (attention output) - and
# the weight columns. Then E4M3 with per-tensor scales; the table [24][wdq x4, adq x4, q, k, v, 0] leads the blob
# (q, k, v: dequant scales of the E4M3 attention operands, from the |max| of rotated q, k and smoothed v).
# v3 folds the MLP LayerNorm into fc2 (W~ = gamma W2, c = W~ 1, d = W2 beta; per layer rs | W~ | c | d in place
# of LN gamma | beta | W2): the engine quantizes raw h / s~ and normalizes in fc2's epilogue.
# fc1 rows are g/x interleaved in 16-unit chunks (one contiguous uint4 of each per k_swiglu_ln thread).
import os, sys, struct, onnx, numpy as np
from onnx import numpy_helper

D, H, HID, HIDP, T, NP, K0, K0P = 1024, 16, 2730, 2736, 1025, 1024, 588, 592
E4M3 = 448.0
HEADROOM = float(os.environ.get("EVA_HEADROOM", 1.25))  # activation scale margin over the calibrated max
ALPHA = float(os.environ.get("EVA_ALPHA", 0.5))         # SmoothQuant migration strength

m = onnx.load(sys.argv[1])
g = m.graph
W = {t.name: numpy_helper.to_array(t) for t in g.initializer}
C = W  # constants may live in initializers or Constant nodes
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value": C[n.output[0]] = numpy_helper.to_array(a.t)
node = {n.name: n for n in g.node}
calib = None
if len(sys.argv) > 3:
    import ml_dtypes
    raw = open(sys.argv[3], "rb").read()
    na = 24 * 4 * HIDP
    calib = np.frombuffer(raw, np.float32, na).reshape(24, 4, HIDP)       # |max| of the 4 block GEMM inputs
    qkvmax = np.frombuffer(raw, np.float32, 24 * 3 * D, na * 4).reshape(24, 3, D)  # rotated q | k | v
    hmax = np.frombuffer(raw, np.float32, 24 * HIDP, (na + 24 * 3 * D) * 4).reshape(24, HIDP)  # raw SwiGLU output

def inp(name, i): return W[node[name].input[i]]

blob = bytearray()
def put(a, dt):
    global blob
    blob += b"\0" * (-len(blob) % 256)
    blob += np.ascontiguousarray(a, dtype=dt).tobytes()

def ln(name):
    return inp(name, 1).astype(np.float32).copy(), inp(name, 2).astype(np.float32).copy()

def smooth(a, w):
    # s[k] = sqrt(amax_act[k] / amax_weight_col[k]); act /= s, weight column *= s
    wa = np.abs(w).max(0)
    ok = (a > 0) & (wa > 0)
    s = np.ones_like(a)
    s[ok] = np.clip(a[ok] ** ALPHA / wa[ok] ** (1 - ALPHA), 1e-3, 1e3)
    return s

def interleave(gx):
    # [g (HIDP) | x (HIDP)] rows -> 16-unit chunks [g c | x c]
    return gx.reshape(2, HIDP // 16, 16, -1).transpose(1, 0, 2, 3).reshape(2 * HIDP, -1)

def fp8(w):
    sc = float(np.abs(w).max()) / E4M3
    return np.clip(w / sc, -E4M3, E4M3).astype(ml_dtypes.float8_e4m3fn).view(np.uint8), sc

cos = C["/core_model/blocks.0/attn/Constant_17_output_0"].astype(np.float32)
sin = C["/core_model/blocks.0/attn/Constant_32_output_0"].astype(np.float32)
# softmax scale (q and k each carry sqrt of it) and log2(e) go into Wq: scores land in the exp2 domain
qscale = float(C["/core_model/blocks.0/attn/Sqrt_1_output_0"].ravel()[0]) ** 2 * np.log2(np.e)
# both elements of a rotary pair share one angle: table [1024][32] of (cos, sin)
assert (cos[:, 0::2] == cos[:, 1::2]).all() and (sin[:, 0::2] == sin[:, 1::2]).all()
rope = np.stack([cos[:, 0::2], sin[:, 0::2]], -1)

pe_w = W["core_model.patch_embed.proj.weight"].reshape(D, K0)        # [out][c*196 + ky*14 + kx]
pe = np.zeros((D, K0P), np.float32); pe[:, :K0] = pe_w
pos = W["core_model.pos_embed"].reshape(T, D).astype(np.float32).copy()
pos[0] += W["core_model.cls_token"].reshape(D)
pos[1:] += W["core_model.patch_embed.proj.bias"].reshape(1, D)

layers, table = [], np.zeros((24, 12), np.float32)
for b in range(24):
    p = f"/core_model/blocks.{b}"
    w1, b1 = ln(f"{p}/norm1/LayerNormalization")
    wqkv = np.concatenate([inp(f"{p}/attn/q_proj/MatMul", 1).T * qscale,
                           inp(f"{p}/attn/k_proj/MatMul", 1).T,
                           inp(f"{p}/attn/v_proj/MatMul", 1).T], 0).astype(np.float32)
    wo = inp(f"{p}/attn/proj/MatMul", 1).T.astype(np.float32).copy()
    w2, b2 = ln(f"{p}/norm2/LayerNormalization")
    w12 = np.zeros((2 * HIDP, D), np.float32)
    w12[:HID] = inp(f"{p}/mlp/fc1_g/MatMul", 1).T
    w12[HIDP:HIDP + HID] = inp(f"{p}/mlp/fc1_x/MatMul", 1).T
    mw, mb = ln(f"{p}/mlp/norm/LayerNormalization")
    mwp = np.zeros(HIDP, np.float32); mwp[:HID] = mw
    mbp = np.zeros(HIDP, np.float32); mbp[:HID] = mb
    wf2 = np.zeros((D, HIDP), np.float32); wf2[:, :HID] = inp(f"{p}/mlp/fc2/MatMul", 1).T
    w12 = interleave(w12)
    mats = [wqkv, wo, w12, wf2]
    if calib is not None:
        # MLP LN folded into fc2: x += rstd (h W~^T - mean c) + d, W~ = gamma W2 with h's smoothing in its columns
        dn = wf2 @ mbp
        wf2 = wf2 * mwp[None, :]
        mats[3] = wf2
        a = [calib[b, 0, :D], calib[b, 1, :D], calib[b, 2, :D], hmax[b]]
        s = [smooth(a[0], wqkv), smooth(a[1], wo), smooth(a[2], w12), smooth(a[3], wf2)]
        w1 /= s[0]; b1 /= s[0]; wqkv *= s[0]                 # LN1 -> QKV
        wqkv[2 * D:] /= s[1][:, None]; wo *= s[1]             # V channels -> out-proj
        w2 /= s[2]; b2 /= s[2]; w12 *= s[2]                   # LN2 -> fc1
        wf2 *= s[3]                                           # h / s~ -> fc2
        for k in range(4):
            mats[k], table[b, k] = fp8(mats[k])
            table[b, 4 + k] = float((a[k] / s[k]).max()) * HEADROOM / E4M3
        table[b, 8] = float(qkvmax[b, 0].max()) * HEADROOM / E4M3
        table[b, 9] = float(qkvmax[b, 1].max()) * HEADROOM / E4M3
        table[b, 10] = float((qkvmax[b, 2] / s[1]).max()) * HEADROOM / E4M3
        mwp = (1.0 / table[b, 7]) / s[3]                      # rs: raw h -> E4M3 operand
        mbp = (mats[3].view(ml_dtypes.float8_e4m3fn).astype(np.float32) * table[b, 3] / s[3]).sum(1)  # c
        mbp = np.concatenate([mbp, dn]).astype(np.float32)    # c | d
    layers.append((w1, b1, mats, w2, b2, mwp, mbp))

if calib is not None: put(table, np.float32)
put(pe, np.float16)
put(pos, np.float32)
put(rope, np.float32)
wt = np.uint8 if calib is not None else np.float16
for w1, b1, mats, w2, b2, mwp, mbp in layers:
    put(w1, np.float32); put(b1, np.float32)
    put(mats[0], wt)
    put(mats[1], wt)
    put(w2, np.float32); put(b2, np.float32)
    put(mats[2], wt)
    if calib is not None:
        put(mwp, np.float32)                                  # rs [HIDP]
        put(mats[3], wt)
        put(mbp[:D], np.float32); put(mbp[D:], np.float32)    # c [D], d [D]
    else:
        put(mwp, np.float32); put(mbp, np.float32)
        put(mats[3], wt)
fw, fb = ln("/core_model/fc_norm/LayerNormalization")
NT = W["core_model.head.bias"].shape[0]
NTP = (NT + 7) & ~7
hw = np.zeros((NTP, D), np.float32); hw[:NT] = W["core_model.head.weight"]
hb = np.zeros(NTP, np.float32); hb[:NT] = W["core_model.head.bias"]
put(fw, np.float32); put(fb, np.float32)
put(hw, np.float16)
put(hb, np.float32)

ver = 3 if calib is not None else 2
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<4sIII", b"EVA2", ver, NT, len(blob)))
    f.write(b"\0" * 240)
    f.write(blob)
print(f"{sys.argv[2]}: v{ver}, {NT} tags, {len(blob)} bytes")

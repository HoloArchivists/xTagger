# eva02.bin (v3, E4M3) -> 2:4-sparse copy for mma.sp: in every group of 4 consecutive k of the chosen block GEMM
# weights, the 2 least important bytes are zeroed. Importance is Wanda's |w| * |x|: SmoothQuant at alpha 0.5
# (export_eva.py) leaves every input channel's activation max equal to its weight column max, so |x_k| ~ colmax_k|w|
# and the score needs nothing but the blob.
#
#   python prune24.py eva02.bin out.bin [fc1,fc2,qkv,oproj] [mag]      (mag: plain |w| instead of Wanda)
import struct, sys, numpy as np

D, T, NP, K0P, HIDP, NL = 1024, 1025, 1024, 592, 2736, 24

# E4M3 byte -> value
b = np.arange(256)
e, m = (b >> 3) & 15, b & 7
E4 = np.where(e > 0, (1 + m / 8.0) * 2.0 ** (e - 7), m / 8.0 * 2.0 ** -6) * np.where(b & 128, -1, 1)
E4[(b & 127) == 127] = 0.0  # NaN codes never appear in weights

src, dst = sys.argv[1], sys.argv[2]
which = set((sys.argv[3] if len(sys.argv) > 3 else "fc1,fc2").split(","))
mag = len(sys.argv) > 4 and sys.argv[4] == "mag"
raw = bytearray(open(src, "rb").read())
magic, ver, nt, nbytes = struct.unpack("<4sIII", raw[:16])
assert magic == b"EVA2" and ver == 3
off = 0
def take(n):
    global off
    off = (off + 255) & ~255
    o = off
    off += n
    return 256 + o

take(NL * 12 * 4); take(D * K0P * 2); take(T * D * 4); take(NP * 32 * 2 * 4)
spans = []
for i in range(NL):
    take(D * 4); take(D * 4)
    spans.append(("qkv", take(3 * D * D), 3 * D, D))
    spans.append(("oproj", take(D * D), D, D))
    take(D * 4); take(D * 4)
    spans.append(("fc1", take(2 * HIDP * D), 2 * HIDP, D))
    take(HIDP * 4)
    spans.append(("fc2", take(D * HIDP), D, HIDP))
    take(D * 4); take(D * 4)

kept = tot = 0
for name, o, n, k in spans:
    if name not in which:
        continue
    w = np.frombuffer(raw, np.uint8, n * k, o).reshape(n, k)
    v = np.abs(E4[w])
    score = v if mag else v * v.max(0, keepdims=True)
    g = score.reshape(n, k // 4, 4)
    drop = np.argsort(g, axis=2)[:, :, :2]                     # 2 lowest of every 4
    mask = np.ones_like(g, dtype=bool)
    np.put_along_axis(mask, drop, False, axis=2)
    wp = np.where(mask.reshape(n, k), w, 0).astype(np.uint8)
    raw[o:o + n * k] = wp.tobytes()
    kept += float((v * mask.reshape(n, k)).sum()); tot += float(v.sum())
open(dst, "wb").write(raw)
print(f"{dst}: pruned {sorted(which)} 2:4 ({'magnitude' if mag else 'wanda'}), |w| mass kept {kept / max(tot, 1):.3f}")

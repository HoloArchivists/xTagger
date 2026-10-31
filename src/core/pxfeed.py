# PixAI calibration feeder: every image directly in a folder (no recursion) -> PixAI's reference preprocessing
# (alpha over white, aspect fit to 1008 with PIL bilinear, centred on black, (x / 255 - 0.5) / 0.5) -> the patch
# operand the px engine reads (fp16 [5184][592], window-major), streamed to stdout. Pipe it into pxcal:
#   python src/core/pxfeed.py <images> | build\pxcal.exe src/models/calib/pixai_fp16.bin
# Needs numpy and Pillow. Prints only the image count on stderr.
import sys, os, numpy as np
from PIL import Image

S, WS = 1008, 24
d = sys.argv[1]
py, px_ = np.meshgrid(np.arange(72), np.arange(72), indexing="ij")
idx = (((py // WS) * 3 + px_ // WS) * (WS * WS) + (py % WS) * WS + px_ % WS).ravel()
out = sys.stdout.buffer
n = 0
for name in os.listdir(d):
    path = os.path.join(d, name)
    if not os.path.isfile(path):
        continue
    try:
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255))
            bg.alpha_composite(im)
            im = bg.convert("RGB")
        w, h = im.size
        r = min(S / h, S / w)
        nw, nh = max(1, int(w * r)), max(1, int(h * r))
        im = im.resize((nw, nh), Image.BILINEAR)
    except Exception:
        continue
    a = np.zeros((S, S, 3), np.float32)
    top, left = (S - nh) // 2, (S - nw) // 2
    a[top:top + nh, left:left + nw] = np.asarray(im, np.float32) / 255.0
    x = (a - 0.5) / 0.5
    p = x.reshape(72, 14, 72, 14, 3).transpose(0, 2, 4, 1, 3).reshape(5184, 588)
    op = np.zeros((5184, 592), np.float16)
    op[idx, :588] = p.astype(np.float16)
    out.write(op.tobytes())
    n += 1
sys.stderr.write(f"fed {n} images\n")

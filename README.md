# xtagger

```
         _____  _    ____  ____ _____ ____
   __  _|_   _|/ \  / ___|/ ___| ____|  _ \
   \ \/ / | | / _ \| |  _| |  _|  _| | |_) |
    >  <  | |/ ___ \ |_| | |_| | |___|  _ <
   /_/\_\ |_/_/   \_\____|\____|_____|_| \_\
  Fast Image Tagging with AVX2, CUDA, and XORs
```

`xtag` tags anime-style images with booru tags (`1girl`, `long_hair`, `outdoors`, ...). It runs two taggers on an
NVIDIA GPU, using its own CUDA kernels rather than a general-purpose inference runtime:

- **wd** — [WD EVA02-Large Tagger v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3). Fast.
- **PixAI** — the [PixAI tagger](https://huggingface.co/pixai-labs). Slower, more accurate, and better on very large
  images.

Large or busy images are also split into detail crops ("MVIs") by **xorcropping**, which finds content with
**xorvariance**, a cheap XOR-based variance test, so small details still get tagged. With both models loaded, `xtag`
learns while it runs which images need PixAI and which wd handles on its own.

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11, 64-bit |
| GPU | NVIDIA, Ada Lovelace or newer (RTX 40xx, RTX 500/1000/2000 Ada, L4, H100, RTX 50xx, ...). The kernels use FP8 tensor cores. 4 GB of VRAM holds both models. |
| CPU | x86-64 with AVX2, FMA, F16C, BMI2 and SSE4.2 (Intel Haswell / AMD Zen or newer) |
| Toolchain | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 13.x, [MinGW-w64](https://github.com/niXman/mingw-builds-binaries) gcc, [CMake](https://cmake.org/) 3.20+ |

Tested with CUDA 13.1, MinGW-w64 gcc 15.2 and CMake 4.2 on an RTX 500 Ada laptop GPU.

## Building

From the repository root, in a shell where `gcc`, `cmake` and `mingw32-make` are on `PATH` and `CUDA_PATH` points at
the CUDA Toolkit:

```
cmake -S src -B build -G "MinGW Makefiles"
cmake --build build -j
```

This produces `build/xtag.exe`. The GPU kernels are compiled during the build for the GPU in the machine, so **build
on the machine that will run it**, with the GPU visible. If the build stops with `evacc: no CUDA device`, check that
`CUDA_VISIBLE_DEVICES` is unset or names your GPU.

Add `-DXTAG_CALIB=ON` the first time: it also builds `xtag_calib` and `pxcal`, which the model conversion below
needs. `-DXTAG_WEBP=ON` adds `.webp` input (needs libwebp for MinGW).

## Converting the models

xtagger doesn't ship model weights. Convert the upstream models once, on your own machine: three steps per model,
and the FP8 files end up in `src/models/`.

**You need:**

- the build above, configured with `-DXTAG_CALIB=ON`;
- Python 3 with `pip install onnx numpy pillow`;
- the upstream ONNX models:
  - **wd:** `model.onnx` from
    [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3). Its
    `selected_tags.csv` is already in `src/models/`.
  - **PixAI:** the ONNX export of the PixAI tagger from [pixai-labs](https://huggingface.co/pixai-labs), called
    `model_pixai.onnx` below. Its configs are already in `src/models/pixai/`.
- a folder of **calibration images**: a few hundred images like the ones you will tag. The FP8 scales come from the
  activation ranges the models see on these, so use representative images. The numbers in this README come from
  models calibrated on 513 random Gelbooru images.
- about 1.6 GB of free disk space for the fp16 intermediates, which you can delete afterwards.

From the repository root:

```
mkdir src\models\calib
```

**wd:**

```
python src/core/export_eva.py model.onnx src/models/calib/eva02_fp16.bin
build\xtag_calib.exe src\models\calib\eva02_fp16.bin src\models\selected_tags.csv <calibration images> > NUL
python src/core/export_eva.py model.onnx src/models/eva02.bin src/models/calib/eva02_fp16.bin.calib
```

1. Writes an fp16 version of the model.
2. Runs it over the calibration images (crops included) and writes the activation ranges to
   `eva02_fp16.bin.calib`. Its tag output isn't needed, hence `> NUL`.
3. Writes the FP8 model, `src/models/eva02.bin`.

**PixAI:**

```
python src/core/export_pixai.py model_pixai.onnx src/models/pixai/config_pixai.json src/models/calib/pixai_fp16.bin
python src/core/pxfeed.py <calibration images> | build\pxcal.exe src\models\calib\pixai_fp16.bin
python src/core/export_pixai.py model_pixai.onnx src/models/pixai/config_pixai.json src/models/pixai.bin src/models/calib/pixai_fp16.bin.calib
```

The same three steps. `pxfeed.py` applies PixAI's reference preprocessing to every image directly in the folder
(not subfolders) and streams the results into `pxcal`. The last step writes `src/models/pixai.bin` and its tag list,
`src/models/pixai.bin.tags`.

When both are done, `src/models/calib/` can be deleted.

## Usage

```
build\xtag.exe src\models\eva02.bin src\models\selected_tags.csv <images or folders>... [options]
```

Add `-px src\models\pixai.bin` to bring in PixAI and the route learner.

Supported inputs: `.jpg .jpeg .png .gif .bmp .tga .psd` (and `.webp` with `-DXTAG_WEBP=ON`). Folders are scanned
for those extensions; add `-r` to include subfolders.

### Output

One line per image on stdout, as each image finishes (so not necessarily in input order), tab separated:

```
<path>	<rating>	<number of crops>	tag:score,tag:score,...
```

Tags are sorted by score, highest first. With both models running, each model's tags appear once under their own
names. The banner, warnings and statistics go to stderr, so redirecting stdout gives a clean result file:

```
build\xtag.exe src\models\eva02.bin src\models\selected_tags.csv D:\pictures -r > tags.tsv
```

### Common recipes

| Goal | Command adds |
|---|---|
| Fastest: whole images only, wd | `-nc` |
| wd with detail crops (default) | *(nothing)* |
| Fewer crops per image (faster) | `-mc 2` |
| Let `xtag` pick wd or PixAI per image | `-px src\models\pixai.bin` |
| Always use PixAI | `-px src\models\pixai.bin -rT px` |
| Benchmark: no tag output, speed report at the end | `-nO -sT` |

### Routes

With `-px`, every image takes one of these routes. "Full" means the whole image; "crops" are the detail regions.

| `-rT` | Whole image | Crops | |
|---|---|---|---|
| `auto` | learned | learned | default with `-px` |
| `wd` | wd | wd | fastest |
| `px-full` | PixAI | wd | |
| `px-crops` | wd | PixAI | |
| `px` | PixAI | PixAI | slowest, most accurate |

With `auto`, a small, shrinking share of images runs every combination. Each cheaper route's tags are compared
with the `px` route's tags on those images, and a small predictor learns from that which route each new image can
take without losing tags. Nothing is trained ahead of time, and nothing carries over between runs. `-rA` sets how
close (F1 score) a route must come to the `px` route to count as good enough. At the end of the run, a `router:` line
on stderr reports how often each route was used and how well the predictor did.

### How the crops are chosen: xorcropping

Taggers see a downscaled image: wd sees 448 x 448, so a 4000-pixel-wide frame with five people in it loses most of
its detail. Xorcropping cuts such images into crops around their content and tags those too. Most tools find
content with variance or edge filters: sums of squares, multiplies and widening to 16 or 32 bits. Xorcropping uses
xorvariance instead.

**Xorvariance.** The image is read in 8-bit grayscale and cut
into tiles, 256 px by default (`-t`). Every byte of a tile is xorfolded into one byte. Each bit of
that byte is the parity of one bit-plane of the tile, and its popcount is the tile's xorvariance. Then:

- A **flat tile** (one solid colour) of even area folds to exactly 0: every value cancels against itself.
- A **repeating tile** also folds to 0 when every value in it occurs an even number of times: dither patterns,
  transparency checkerboards, pixel-doubled art, tiled wallpaper.
- On **any other tile**, the 8 parities behave like coin flips, so the xorvariance follows
  binomial(8, 1/2). The design notes measured this on 1883 textured tiles.

So "is this tile's xorvariance (almost) zero?" is a near-free test for flat or repeating background. It costs one
`vpxor` per 32 pixels while the image streams past, in a single pass. A butterfly of `vpunpck` + `vpxor` reduces
32 tiles to one byte each in one register, a `vpshufb` nibble table popcounts all 32 at once, and one
`vpcmpgtb` + `vpmovmskb` turns a whole row of tiles into a bitmask. An edge tile with an odd pixel count is
corrected by XORing in its first pixel.

**Selecting crops.** Tiles whose xorvariance is at or below `-lo` (or at or above `-hi`) count as background; the rest
are content. Each tile row becomes one 64-bit mask:

1. **Candidate boxes:** every maximal rectangle of content tiles (at least 2 x 2 tiles by default, `-mw` / `-mh`).
   ANDing row masks extends a box downward; `tzcnt` finds the edges of runs.
2. **Cover:** greedily pick the box that covers the most still-uncovered content tiles (ties go to the smaller,
   tighter box), up to `-mc` crops.
3. **Skip:** an image whose only box is the whole frame, or that is already no larger than the model input, gets
   no crops.

Background tiles act as separators, so crowded frames split along sky, walls and empty floor between subjects.

**Using the crops.** Each crop gets a weight, its mean tile xorvariance / 8. Its tags are added to the whole-image
pass: `final = full + sum of gain * (weight - cutoff) * crop` (`-cg`, `-cw`). Crop outputs lie in [0, 1], so once
the whole-image pass is back, `xtag` knows exactly how far the crops could move each tag's score. If no tag sits
close enough to its threshold to flip, the crops are skipped without being run, and the output is identical to
running them.

### All options

Run `build\xtag.exe -h` for this board. Short forms are grouped nmap-style; every option also has a long form.

```
 xtag <eva.bin> <tags.csv> <in>... [opt]
 out, per image, tab separated:
   path rating mvis tag:score,...
-------------------[ INPUT ]--------------------
  <in>              image files or directories
  -r  --recurse     recurse into directories
-------------------[ MODELS ]-------------------
  -px <file>        add PixAI + the router
                    (vocabulary: <file>.tags)
  -d  --device <n>  CUDA device [0]
-------------------[ CROPS ]--------------------
  -nc --nocrops     whole images only, no crops
  -t  --tile <px>   tile side, pow2 >= 32 [256]
  -lo -hi <n>       tile popcount bounds [1, 9]
  -m  --mode <m>    zero|dontcare|inverse [zero]
  -mw --minw <n>    min crop width, tiles [2]
  -mh --minh <n>    min crop height, tiles [2]
  -Mw --maxw <n>    max crop width, 0=all [0]
  -Mh --maxh <n>    max crop height, 0=all [0]
  -mc --maxmvi <n>  max crops per image [8]
  -cw --wthresh <f> crop weight cutoff [0.50]
  -cg --gain <f>    crop weight gain [1.00]
--------------[ THRESHOLDS (WD) ]---------------
  -tg --gen <f>     general tags [0.35]
  -tc --char <f>    character tags [0.85]
----------------[ ROUTER (-px) ]----------------
 full: the whole image, crops: detail
 regions; wd is fast, PixAI slow + best
  -rT --route <r>   who tags what [auto]:
                    auto     learned per image
                    wd       wd only (fastest)
                    px-full  PixAI full+wd crops
                    px-crops wd full+PixAI crops
                    px       PixAI only (best)
  -rA --agree <f>   F1 vs px that passes [0.80]
  -rE --explore <f> min share explored [0.0312]
  -rC --rconf <n>   confidence to go cheaper
                    [-1 = theta / 2]
------------------[ DISPLAY ]-------------------
  -nb --nobanner    hide the banner
  -sT --stats       img/s + throughput at end
  -nO --no-tag-output  no tag lines (benchmarks)
  -h  --help        this board
------------------------------------------------
```

## Performance

End-to-end speed: decoding, preprocessing and both models, including GPU start-up. Measured with `-sT` on a
512-image Gelbooru sample (half PNG, half JPEG) on a laptop RTX 500 Ada (4 GB):

| Command adds | images/s |
|---|---|
| `-nc -rT wd` (whole images, wd) | 38.6 |
| `-mc 2 -rT wd` (wd with up to 2 crops) | 17.8 |
| `-mc 2 -px src\models\pixai.bin` (route learner) | 4.3 |

Speed depends heavily on image size and format: large PNGs cost the most CPU time to decode.

### Comparing against ONNX Runtime

The same 512 images, whole images only, through stock ONNX Runtime 1.31.1 (CUDA execution provider, upstream fp32
ONNX models), measured back to back with `xtag` on the same laptop:

| Model | ONNX Runtime | xtag | Speed-up |
|---|---|---|---|
| wd (EVA02-L) | 1.28 img/s | 22.5 img/s | 17.6x |
| PixAI | 1.13 img/s | 4.70 img/s | 4.2x |

- **Where ONNX Runtime's time goes:** 99% of its wall time is spent inside `OrtRun`, so its decoding and
  preprocessing are not what limits it.
- **Thermal note:** these `xtag` runs each followed a 6 to 8 minute ONNX Runtime run on a warm laptop. The 38.6
  img/s for `-nc` in the table above was measured cool.
- **Precision:** `xtag` runs the models in FP8 with an fp16 residual stream; the upstream ONNX models are fp32.
- **Batching:** PixAI's ONNX graph fixes its batch size at 1, so ONNX Runtime runs it one image at a time; `xtag`
  batches 4.

`ortbench` runs the same images through stock ONNX Runtime (CUDA execution provider) with the upstream ONNX models
and each model's reference preprocessing. It uses the same decoder and decode threads as `xtag`, so the difference
is the inference engine. Build it with the
[ONNX Runtime GPU package](https://github.com/microsoft/onnxruntime/releases) for CUDA 13; cuDNN must be on `PATH` at
run time:

```
cmake -S src -B build -DXTAG_ORTBENCH=ON -DORT_ROOT=C:/path/to/onnxruntime-win-x64-gpu_cuda13-1.31.1
cmake --build build -j --target ortbench
build\ortbench.exe wd model.onnx <image folder>          (whole images, batches of 8)
build\ortbench.exe px model_pixai.onnx <image folder>    (whole images, batch 1: the graph fixes it)
```

## Repository layout

```
src/
  xtag.c          command-line program
  core/           image pipeline (compositing, xorvariance, xorcropping, resizing, route learner, scheduler)
                  and the model conversion scripts
  engine/         CUDA engines and kernels for both models
  tools/          kernel compiler (evacc), PixAI calibration (pxcal), ONNX Runtime baseline (ortbench)
  models/         tag lists, PixAI configs, licenses; converted weights are written here
  third_party/    stb_image
  docs/           design notes
```

## License

xtagger's own source code is released under the MIT License (see [LICENSE](LICENSE)), copyright Christopher J.
Cole (ef1500).

The model weights you convert, and the tag lists and configs in `src/models/`, come from third-party models
and stay under those models' license, the Apache License 2.0 (see [src/models/LICENSE](src/models/LICENSE)):

- WD EVA02-Large Tagger v3 by SmilingWolf: Apache-2.0.
- PixAI tagger by PixAI: Apache-2.0.

The converted weights are modified versions of those models: re-laid-out weights, FP8 quantization, and PixAI's
attention-pool head folded into its output layer.

`src/third_party/stb_image.h` is by Sean Barrett (public domain / MIT).

## Credits

Developed by [ef1500](https://github.com/ef1500) for [holoarchivists](https://github.com/holoarchivists).
wd tagger by SmilingWolf; PixAI tagger by PixAI.

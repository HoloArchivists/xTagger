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
| NPU (optional) | Intel Core Ultra NPU with its Windows driver, for `-npu`. The driver supplies `ze_loader.dll` and the graph compiler; nothing else is installed. |
| Toolchain | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 13.x, [MinGW-w64](https://github.com/niXman/mingw-builds-binaries) gcc, [CMake](https://cmake.org/) 3.20+ |

Tested with CUDA 13.1, MinGW-w64 gcc 15.2 and CMake 4.2 on an RTX 500 Ada laptop GPU.

## Getting the models

The weights are not in the repository. Download the ONNX models from Hugging Face and convert them once with the
scripts in `src/tools/`; [GETTING_STARTED.md](GETTING_STARTED.md) has the exact commands.

- wd: [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
  (`model.onnx`, `selected_tags.csv`)
- PixAI: [nocturna/pixai-tagger-v1.0-onnx](https://huggingface.co/nocturna/pixai-tagger-v1.0-onnx)
  (`model.onnx`, `config.json`)

| File | What it is |
|---|---|
| `src/models/eva02.bin`, `src/models/selected_tags.csv` | wd tagger weights (FP8, converted) and its tag list (downloaded) |
| `src/models/pixai.bin`, `src/models/pixai.bin.tags` | PixAI weights (FP8) and its tag list, both written by the converter |
| `src/models/calib/*.calib` | Calibration ranges for the FP8 conversion. These are in the repository. |
| `src/models/calib/eva02_fp16.bin` | fp16 wd weights, the `-npu` input. Generated locally (see Converting the models again). |
| `src/models/calib/eva02_fp16.bin.npu` | Compiled NPU graph (~770 MB), written by the first `-npu` run. Generated locally. |

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

Build options: `-DXTAG_WEBP=ON` adds `.webp` input (needs libwebp for MinGW). `-DXTAG_CALIB=ON` also builds
`xtag_calib`, used only when converting models.

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

### Tag stealing

With `-px` and `-s`, every image that runs on both models teaches wd's last layer (one row of weights per tag) to
reproduce PixAI's confident decisions. The rest of wd stays frozen, so learning costs a few vector operations on
the CPU per image and does not slow tagging.

- **Tags wd already has** move toward PixAI's decision. A decision counts as confident when PixAI's score is
  clearly over or under its own threshold, and wd is pushed to the same distance from wd's threshold.
- **Tags wd lacks** (series names, newer characters) get a new row the first time PixAI is confident about one,
  under PixAI's name.
- **Headcount tags** (`1girl`, `solo`, `6+girls`, ...) are never taught: wd counts people more reliably than PixAI.
- **Which PixAI categories** are stolen: `-sk` (default `cp`: characters and series).
- **Keeping it:** `-sf <file>` loads the learned rows at start and saves them at the end, so learning builds up over
  runs. A later `-rT wd -px <pixai.bin> -sf <file>` run tags with wd alone and keeps the stolen tags. Without `-sf`,
  nothing is kept.

Images teach on the routes where both models run: explored images, `px-full` (PixAI's whole-image tags teach wd's
crops) and `px-crops`.

### Tagglomeration: new tags from sidecar files

A step-by-step guide is in [src/docs/tagglomeration.md](src/docs/tagglomeration.md).

With `-g` (`-gf <file>` to keep it; PixAI optional), a text file next to an image with the same name (`haha.txt`
beside `haha.png`) lists tags for that image, separated by commas or line breaks. Spaces become underscores and
letters are lowercased. Each name is handled according to who knows it:

- **wd has it:** the row is pushed to a confident positive on this image.
- **Only PixAI has it** (with `-px`): it becomes a stolen row, as above.
- **Neither model has it:** it becomes a user tag. A `character:`, `copyright:`, `artist:`, `meta:` or `general:`
  prefix sets its category and therefore its threshold; the default is general.

User tags are scored on the CPU from the whole-image embedding that wd's GPU pass already reads back. Each tag is a
small network of units. A unit is a direction in that embedding space, and the tag's score is the highest cosine
(centred on the mean image) over its units, one unit per look of the subject.

**When a tag fires.** The cutoff is set by the tag's background, not by a fixed number. The background is a random
sample of every other image tagged, plus sidecar images that do not list the tag. The tag fires when the score
clears the background's median by `-gk` standard deviations (default 3). Spread is measured with the median
absolute deviation, so unlabelled images of the subject in the background barely move the cutoff, and they are never
trained against. Lower `-gk` gives more recall, higher gives more precision.

**How the network is evolved** (NEAT-style, structure only):

- **Genes** are memberships: an example image belongs to a unit (weight 1), or a confuser is subtracted from it. A
  unit holding every example is their average, which is the right answer for a subject with one look. There are no
  free weights, because with a handful of examples free weights overfit.
- **Mutations** aim at the current worst cases. A missed example joins the closest unit or, if it is a different
  look, starts its own. A sidecar negative the tag fires on is subtracted. Pruning removes genes.
- **Species** are unit counts, and each species' best network survives every generation. The average of all
  examples is reinserted into the population every time it evolves.
- **Fitness**, in background standard deviations: each example's score with itself left out (no credit for
  memorising an example), minus how far sidecar negatives and the background's top end reach past the cutoff.
  Every number fitness needs comes from a precomputed Gram matrix of the stored images, so evaluating a network
  never needs a 1024-wide dot product.

**What to give it:**

- **Positive:** an image whose sidecar lists the tag. A few examples of each look is enough.
- **Negative:** an image whose sidecar does not list it. A sidecar is taken as complete for user tags. A lookalike
  that keeps firing (a shirt with a cartoon face, for gangle) is best fixed by giving it a sidecar that leaves the
  tag out, for example just `1girl`.

**How a run handles sidecars:**

- An image with a sidecar runs every component on the GPU and learns before its own line is printed.
- Learning needs a mean of 8 images, so images with sidecars are queued right after the first 8 others. Those 8 are
  tagged before anything is learned in a first run. A rerun with `-gf` tags them too.
- With `-g`, or while user tags exist, every image gets wd's whole-image GPU pass, whatever the route, and the NPU
  is not used while user tags exist.
- The user tags, stored examples and networks are saved in the `-gf` file next to the stolen rows (`-sf` and `-gf`
  name the same file). Rerunning the same sidecar image reuses its stored example rather than storing a duplicate.

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
  -na --noalpha     ignore alpha (no composite)
  -j  --jobs <n>    workers [cores-1, max 8]
-------------------[ MODELS ]-------------------
  -px <file>        add PixAI + the router
                    (vocabulary: <file>.tags)
  -npu <file>       wd on the NPU as well
                    (fp16 eva02 bin; first run
                     compiles, then <file>.npu)
  -d  --device <n>  CUDA device [0]
-------------------[ CROPS ]--------------------
  -nc --nocrops     whole images only, no crops
  -t  --tile <px>   tile side, 8k >= 32 [256]
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
-------------[ TAG STEALING (-px) ]-------------
 wd's head learns online from PixAI
  -s  --steal       on (off unless given)
  -sf --stealfile <f> keep it across runs (+ -s)
  -sn --stealcap <n> new tags wd can gain [4096]
                    (stolen + sidecar tags)
  -sk --stealcats <s> PixAI categories [cp]:
                    c char p copyright g gen
                    a artist m meta
---------------[ TAGGLOMERATION ]---------------
 wd learns tags from <image>.txt beside
 an image: comma/line separated; names
 no model has become new tags
 (character:name sets the category)
  -g  --glean       on (off unless given)
  -gf --gleanfile <f> keep it across runs (+ -g)
                    (same file as -sf)
  -gk --gleank <f>  cutoff, sigmas over the
                    background [3.0]; lower:
                    more recall
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
| `-nc` (whole images, wd) | 46.3 |
| `-mc 2` (wd with up to 2 crops) | 21.4 |
| `-mc 2 -px src\models\pixai.bin` (route learner) | 5.36 |
| `-nc -px src\models\pixai.bin -rT px` (whole images, PixAI) | 7.86 |

Speed depends heavily on image size and format: large PNGs cost the most CPU time to decode.

**`-npu` on a laptop.** On its own the NPU runs wd at 1.78 img/s, and its fp16 output is closer to the fp16
reference than the GPU's FP8 (10 tag flips against 47 on 35 test images). But the NPU shares the package power
budget with the CPU cores that decode images, so on this laptop `-nc -npu` runs 34.0 img/s against 41.6 without it,
measured back to back. Leave it off unless the CPU has power to spare.

### Against PixAI's H100 numbers

PixAI's published benchmarks ran on an 80 GB H100. The laptop RTX 500 Ada has about 4% of its compute:

| | H100, GPU only (b16) | H100, end-to-end | xtag, end-to-end | Share of H100 end-to-end |
|---|---|---|---|---|
| EVA02-L at 448 px (wd, `-nc`) | 170.5 img/s | 85.3 img/s | 46.3 img/s | 54% |
| PixAI v1.0 at 1008 px | 48.3 img/s | 21.8 img/s | 7.86 img/s | 36% |

- **Throughput:** with about 4% of the compute, xtag reaches 27% of the H100's GPU-only EVA02-L number and 54% of
  its end-to-end number.
- **Memory:** wd and PixAI together stay resident in 0.6 GB of the RTX 500's 4 GB.

### Comparing against ONNX Runtime

The same 512 images, whole images only, through stock ONNX Runtime 1.31.1 (CUDA execution provider, upstream fp32
ONNX models), on the same laptop. The `xtag` column is the current build, from the table above:

| Model | ONNX Runtime | xtag | Speed-up |
|---|---|---|---|
| wd (EVA02-L) | 1.28 img/s | 46.3 img/s | 36x |
| PixAI | 1.13 img/s | 7.86 img/s | 7.0x |

- **Where ONNX Runtime's time goes:** 99% of its wall time is spent inside `OrtRun`, so its decoding and
  preprocessing are not what limits it.
- **Thermal note:** each ONNX Runtime run takes 6 to 8 minutes and heats the laptop, so it was measured separately
  from the `xtag` runs, which were measured cool.
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

## Converting the models again

This is only needed for a new upstream model, or for the fp16 file `-npu` uses; for the shipped models the
calibration files are already in `src/models/calib/` (see [GETTING_STARTED.md](GETTING_STARTED.md)). It needs Python
with `onnx`, `numpy` and `ml_dtypes`, the upstream ONNX files, and a build with `-DXTAG_CALIB=ON`. The FP8 scales
come from running the fp16 model over a set of calibration images:

```
python src/tools/export_eva.py model.onnx src/models/calib/eva02_fp16.bin
build\xtag_calib.exe src/models/calib/eva02_fp16.bin src/models/selected_tags.csv <calibration images>
python src/tools/export_eva.py model.onnx src/models/eva02.bin src/models/calib/eva02_fp16.bin.calib
```

PixAI works the same way with `src/tools/export_pixai.py`, which also takes PixAI's `config.json`. The first command
alone writes `eva02_fp16.bin`, the `-npu` input.

## Repository layout

```
src/
  xtag.c          command-line program
  core/           image pipeline: compositing, xorvariance, xorcropping, resizing, route learner, scheduler
  engine/         CUDA engines and kernels for both models; NPU engine for wd (npu.h)
  tools/          kernel compiler (evacc), model converters, kernel and NPU test tools
  models/         calibration ranges (calib/*.calib); converted weights and tag lists go here (not in git)
  third_party/    stb_image; Level Zero headers (Intel, MIT)
  docs/           design notes
```

## License

xtagger's own source code is released under the MIT License (see [LICENSE](LICENSE)), copyright Christopher J.
Cole (ef1500).

The files in `src/models/` are converted from third-party models and stay under those models' license, the
Apache License 2.0 (see [src/models/LICENSE](src/models/LICENSE)):

- WD EVA02-Large Tagger v3 by SmilingWolf: Apache-2.0.
- PixAI tagger by PixAI: Apache-2.0.

The converted files are modified versions of those models: re-laid-out weights, FP8 quantization, and PixAI's
attention-pool head folded into its output layer.

`src/third_party/stb_image.h` is by Sean Barrett (public domain / MIT). `src/third_party/level_zero/` holds Intel's
Level Zero and NPU graph-extension headers (MIT).

## Credits

Developed by [ef1500](https://github.com/ef1500) for [holoarchivists](https://github.com/holoarchivists).
wd tagger by SmilingWolf; PixAI tagger by PixAI.

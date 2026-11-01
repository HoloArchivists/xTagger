# Getting started

## 1. Build

You need Windows 10/11, an NVIDIA Ada (RTX 40xx or newer) GPU, the CUDA Toolkit 13.x, MinGW-w64 gcc and CMake.
From the repository root:

```
cmake -S src -B build -G "MinGW Makefiles"
cmake --build build -j
```

Build on the machine that will run xtag: the GPU kernels are compiled for the GPU it finds.

## 2. Download the models

The weights are not in the repository. Download them from Hugging Face:

| Model | Files |
|---|---|
| [wd EVA02-Large v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) | `model.onnx`, `selected_tags.csv` |
| [PixAI v1.0](https://huggingface.co/nocturna/pixai-tagger-v1.0-onnx) (optional) | `model.onnx`, `config.json` |

```
curl.exe -L -o src\models\wd.onnx https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx
curl.exe -L -o src\models\selected_tags.csv https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv
curl.exe -L -o src\models\pixai.onnx https://huggingface.co/nocturna/pixai-tagger-v1.0-onnx/resolve/main/model.onnx
curl.exe -L -o src\models\pixai_config.json https://huggingface.co/nocturna/pixai-tagger-v1.0-onnx/resolve/main/config.json
```

## 3. Convert them

The converters need Python with `onnx`, `numpy` and `ml_dtypes`. The calibration files they read are already in
`src/models/calib/`.

```
pip install onnx numpy ml_dtypes
python src\tools\export_eva.py src\models\wd.onnx src\models\eva02.bin src\models\calib\eva02_fp16.bin.calib
python src\tools\export_pixai.py src\models\pixai.onnx src\models\pixai_config.json src\models\pixai.bin src\models\calib\pixai_fp16.bin.calib
```

This writes `src\models\eva02.bin`, and for PixAI `src\models\pixai.bin` with its tag list `src\models\pixai.bin.tags`.
The `.onnx` files can be deleted afterwards.

## 4. Tag images

```
build\xtag.exe src\models\eva02.bin src\models\selected_tags.csv D:\pictures -r > tags.tsv
```

Each line of `tags.tsv` is `path, rating, crop count, tag:score,...`, tab separated.

| To | Add |
|---|---|
| Use PixAI as well (xtag picks per image) | `-px src\models\pixai.bin` |
| Go fastest (whole images only) | `-nc` |
| See speed at the end | `-sT` |
| List every option | `-h` |

See [README.md](README.md) for routes, tag stealing and the rest of the options.

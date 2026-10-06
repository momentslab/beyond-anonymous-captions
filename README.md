<div align="center">

# BAC: Beyond Anonymous Captions

**Grounding character identity in video captioning and question answering**

*Qwen3-VL + LoRA · name-aware captions · identity QA*

<!-- TODO: replace with final paper link once public -->
[![Paper](https://img.shields.io/badge/Paper-coming_soon-lightgrey)](#citation)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-beyond--anonymous--captions-yellow)](https://huggingface.co/datasets/momentslab/beyond-anonymous-captions)
[![Models](https://img.shields.io/badge/🤗_Models-Qwen3--VL_LoRA-yellow)](https://huggingface.co/collections/momentslab/beyond-anonymous-captions)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)

[Moments Lab](https://www.momentslab.com) · Research use only

</div>

---

## Overview

Generic video captioners say *"a man talks to a woman."* Editorial teams need *"Amanda Bynes, in a salmon shirt, turns to Aly Michalka."*

**BAC** (*Beyond Anonymous Captions*) has the code, data and LoRA adapters for **identity-grounded video understanding**. Given sampled frames from a movie shot plus face-grounded identities, a Qwen3-VL model:

- **Captioning (`cap`):** writes a single sentence that names each grounded person and adds a short visual description of them.
- **Identity QA (`qa`):** answers questions about *who* is doing *what* in the shot.

People are passed to the model as **P-indices** (`P1`, `P2`, …) tied to face coordinates (*coord_face* prompting). Names are substituted back after generation, so the model learns grounding rather than memorizing names.

## Release contents

| Resource | Link | Description |
|---|---|---|
| Code | this repo | Inference, LoRA fine-tuning, data prep |
| Dataset | [`momentslab/beyond-anonymous-captions`](https://huggingface.co/datasets/momentslab/beyond-anonymous-captions) | JSON annotations only (no video, no frames) |
| LoRA 2B | [`momentslab/beyond-anonymous-captions-qwen3-vl-2b-lora`](https://huggingface.co/momentslab/beyond-anonymous-captions-qwen3-vl-2b-lora) | Adapter for `Qwen/Qwen3-VL-2B-Instruct` |
| LoRA 4B | [`momentslab/beyond-anonymous-captions-qwen3-vl-4b-lora`](https://huggingface.co/momentslab/beyond-anonymous-captions-qwen3-vl-4b-lora) | Adapter for `Qwen/Qwen3-VL-4B-Instruct` |
| LoRA 8B | [`momentslab/beyond-anonymous-captions-qwen3-vl-8b-lora`](https://huggingface.co/momentslab/beyond-anonymous-captions-qwen3-vl-8b-lora) | Adapter for `Qwen/Qwen3-VL-8B-Instruct` |

## Repository structure

```
.
├── scripts/
│   ├── run.py                 # inference: identity QA or captioning
│   ├── train.py               # LoRA fine-tuning
│   ├── prepare_manifests.py   # build finetune manifests from shot JSONs
│   └── data_curation/         # scripts used to build/clean the released annotations
├── src/
│   ├── config.py              # model presets, generation + LoRA hyperparameters
│   ├── prompts.py             # captioning / QA prompt templates
│   ├── extract_shot_frames.py # extract the 8 frames per shot from source videos
│   ├── grounding/             # prompt building, frame annotation, batched generation
│   ├── finetune/              # dataset, collator, LoRA setup
│   └── utils/                 # bbox, names, QA parsing, text cleanup
├── data/                      # ← downloaded from Hugging Face (not in git)
├── models/lora/{2b,4b,8b}/    # ← downloaded from Hugging Face (not in git)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/momentslab/beyond-anonymous-captions.git
```

```bash
cd beyond-anonymous-captions && python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

You need a CUDA GPU with bf16 support. The defaults are tuned for 2B at batch 8 and 4B/8B at batch 2.

## Getting the data

### 1. Annotations (Hugging Face)

The dataset is **gated**. Request access on the [dataset page](https://huggingface.co/datasets/momentslab/beyond-anonymous-captions) and accept the research-only terms, then:

```bash
hf auth login
```

```bash
hf download momentslab/beyond-anonymous-captions --repo-type dataset --local-dir data
```

### 2. Video frames (from LSMDC / MPII-MD, not redistributed)

We don't redistribute any video or frames. Get the source clips from [LSMDC](https://sites.google.com/site/describingmovies/) under its own license agreement, then regenerate the 8 frames per shot:

```bash
python src/extract_shot_frames.py --split test --videos-root /path/to/lsmdc/data/videos
```

```bash
python src/extract_shot_frames.py --split train --videos-root /path/to/lsmdc/data/videos
```

You can also set `LSMDC_VIDEOS_ROOT` instead of passing `--videos-root`. This writes `frames/frame{X}.jpg`, `frames/frame{X}_bbox.jpg` and `frames/sequence.jpg` inside each `shot{N}/` folder.

### Dataset layout

```
data/
├── shot_captions/
│   ├── train/{movie}/{clip}/shot{N}/input_sequence.json   # 31,959 shots, 73 movies
│   └── test/{movie}/{clip}/shot{N}/input_sequence.json    #    750 shots, 12 movies
├── clip_faces/
│   ├── lsmdc/{MOVIE}/{clip}.json                          # 17,584 clips
│   └── mpiimd/{MOVIE}/{clip}.json                         # 16,412 clips
├── finetune/train_manifest.jsonl                          # 19,265 training samples
└── train_test_split/{train,test}.txt                      # movie-level split
```

The train and test splits are **disjoint at the movie level**.

**`shot_captions/*/input_sequence.json`**: one shot, 8 sampled frames:

| Field | Description |
|---|---|
| `movie`, `source_clip`, `dataset` | Source identifiers (`lsmdc` / `mpiimd`) |
| `shot_index`, `shot_start_second`, `shot_end_second` | Shot boundaries inside the clip |
| `fps`, `frame_width`, `frame_height` | Video metadata |
| `caption_pre` | Target caption with P-indices (`"P1, with curly blonde hair, …"`); train only |
| `caption` | Same caption with actor names substituted |
| `p_to_full` | `{"P1": "Actor Name", …}` |
| `questions` | Identity QA items; test only |
| `frames[]` | `frame_index`, `timestamp`, `persons[]` → `actor`, `character`, `bbox` (face, xyxy px), `track_id`, `ident_score` |

**`clip_faces/*.json`**: per-frame face tracks for each clip (`sampled_frames[].faces[]` → `track_id`, `bbox`, `name` (character), `actor`, `score`).

## Model weights

```bash
hf download momentslab/beyond-anonymous-captions-qwen3-vl-8b-lora --local-dir models/lora/8b
```

Swap `8b` for `2b` or `4b` as needed. You can also skip this step: `--lora auto` in `run.py` uses `models/lora/{size}` and downloads the adapter from the Hub if it's missing.

To use an adapter directly in Python:

```python
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="bfloat16", device_map="auto"
)
model = PeftModel.from_pretrained(base, "momentslab/beyond-anonymous-captions-qwen3-vl-8b-lora")
processor = AutoProcessor.from_pretrained("momentslab/beyond-anonymous-captions-qwen3-vl-8b-lora")
```

## Inference

Run on the test split (`data/shot_captions/test`):

```bash
python scripts/run.py cap --size 8b --lora auto
```

```bash
python scripts/run.py qa --size 8b --lora auto
```

| Flag | Meaning |
|---|---|
| `qa` / `cap` | Task: identity QA or captioning |
| `--size {2b,4b,8b}` | Qwen3-VL base size |
| `--lora auto\|PATH` | `auto` = `models/lora/{size}`; omit for the zero-shot base model |
| `--limit N` | First N shots only (quick test) |
| `--resume` | Skip shots already done |
| `--visualize` | Save annotated frames + sequence grid |
| `--output_dir` | Default: `output/Qwen3_{size}[_lora]_{task}/test` |

Outputs are written per shot as `result_caption_coord_face.json` / `result_identity_qa_coord_face.json`. QA accuracy is aggregated automatically.

## Training

```bash
python scripts/train.py --size 8b
```

```bash
python scripts/train.py --size 2b --eval --eval_ratio 0.1
```

The manifests are built from `data/shot_captions/train` on first run (or with `--prepare`). Default hyperparameters (`src/config.py`):

| | |
|---|---|
| LoRA | r = 16, α = 32, dropout = 0.05, all attention + MLP projections |
| Optimizer | AdamW, lr = 5e-5, cosine, warmup 3 %, grad-clip 1.0 |
| Batch | 1 × 8 grad-accum, bf16, gradient checkpointing |
| Schedule | 1 epoch, seed 42 |
| Frames | 8 per shot, 131k–524k pixels per image |

Released checkpoints: best by eval loss at step 1600.

| Size | Best eval loss |
|---|---|
| 2B | 0.921 |
| 4B | 0.857 |
| 8B | 0.836 |

## Results

<!-- TODO: copy final numbers from the paper. Do not fill in from memory. -->
See the paper for captioning and identity-QA results.

## License & intended use

**Research use only. Commercial use is not permitted.**

- **Code, annotations and LoRA weights:** [CC BY-NC 4.0](LICENSE), released for non-commercial research.
- **Source videos:** LSMDC / MPII-MD, under their original license agreements. Nothing from them is redistributed here.
- **Base models:** [Qwen3-VL](https://huggingface.co/Qwen) (Apache-2.0).
- **Captions** were generated with Gemini 2.5 Flash and then post-processed. Their use is also subject to the provider's terms.

The annotations include the names of publicly credited actors and characters, tied to face bounding boxes in commercial films. Don't use this release to identify people outside that context, for surveillance, or to build face-recognition systems.

## Citation

```bibtex
@misc{bac2026,
  title  = {BAC: Beyond Anonymous Captions: Grounding Character Identity in Video Captioning and Question Answering},
  author = {TODO},
  year   = {2026},
  note   = {TODO: venue / arXiv id}
}
```

Please also cite LSMDC:

```bibtex
@article{lsmdc,
  title   = {Movie Description},
  author  = {Rohrbach, Anna and Torabi, Atousa and Rohrbach, Marcus and Tandon, Niket and Pal, Christopher and Larochelle, Hugo and Courville, Aaron and Schiele, Bernt},
  journal = {International Journal of Computer Vision},
  year    = {2017}
}
```

## Acknowledgements

Built on [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [PEFT](https://github.com/huggingface/peft) and the [LSMDC](https://sites.google.com/site/describingmovies/) benchmark.

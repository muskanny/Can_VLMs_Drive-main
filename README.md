# Do VLMs Really Look at the Road? 

## ACCV 2026

<!-- TODO: replace with the final author list exactly as on the camera-ready paper -->
Authors : Muskan Singh, Shankar Gangishetty

<!-- TODO: fill in real links once available; remove any that do not apply -->
- [Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
- [Project Webpage](https://<your-project-page>.github.io/)
- [HF Dataset](https://huggingface.co/datasets/<org>/<dataset>)

A diagnostic evaluation benchmark that probes whether vision–language models (VLMs)
make driving decisions from **visual evidence** or from **language priors, answer
biases, and dataset regularities**.

## Table of Contents

- [Introduction](#introduction-)
- [Benchmark](#benchmark-)
- [Installation](#installation-)
- [Data Curation](#data-curation-)
- [Evaluation](#evaluation-)
- [Reproducing the Full Pipeline](#reproducing-the-full-pipeline-)
- [Results](#results-)
- [Acknowledgements](#acknowledgements-)
- [BibTeX Citation](#bibtex-citation-)
- [Appendix: Canonical `results/` Files](#appendix-canonical-results-files-)

## Introduction :

<p align="center">
  <img src="./images/Teaser_Diagram.pdf" alt="Do VLMs Really Look at the Road?">
</p>

---

Vision–language models are increasingly explored as interpretable reasoning modules
for autonomous driving. Yet high accuracy on driving question-answering benchmarks
does not establish that a model is actually reasoning from the road scene. A model
that ignores the image but answers correctly through dataset bias appears competent
while being fundamentally unreliable.

We introduce a **diagnostic evaluation framework** that moves beyond aggregate
accuracy to probe four complementary aspects of driving reliability:

- **Visual grounding** — image-conditioned vs. no-image (blank) evaluation;
- **Counterfactual scene sensitivity (SCS)** — paired clear / fog / rain / snow scenes;
- **Linguistic robustness** — paraphrase consistency and negation accuracy;
- **Spatial reasoning** — relational questions on a DriveLM/nuScenes extension.

All ground-truth labels are derived **programmatically** from BDD100K and
DriveLM/nuScenes annotations, so the benchmark is reproducible without manual
labeling. Across a range of open-source VLMs we find that competitive overall
accuracy conceals systematic failure modes: strong affirmation bias, limited
sensitivity to counterfactual weather changes, near-chance negation accuracy despite
high paraphrase consistency, and confident driving decisions even without any image.

## Benchmark :

The benchmark spans two task categories plus a spatial-reasoning extension:

| Category | Source | Images | QA pairs |
|---|---|---|---|
| Adverse Weather | BDD100K (+ counterfactuals) | 180 | see paper |
| Junctions & Intersections | BDD100K | 140 | see paper |
| Spatial Reasoning extension | DriveLM / nuScenes (CAM_FRONT) | 200 | see paper |
| **Total** | | **520** | **26,788** |

Six open-source VLMs are evaluated zero-shot: **Moondream-2B**, **SmolVLM-2.2B**,
**PaliGemma-3B**, **Qwen2.5-VL-7B**, **InternVL3-8B**, and **LLaVA-OneVision-8B**.

## Installation 🔧:

### Prerequisites
- Python >= 3.9
- CUDA-capable GPU

We used this setup:
- NVIDIA RTX 3080
- Python 3.9

```shell
git clone https://github.com/muskanny/Can_VLMs_Drive.git
cd Can_VLMs_Drive

conda create -n vlm python=3.9 -y
conda activate vlm

pip install -r requirements.txt   # transformers, torch, pillow, openpyxl, ...
```

Model weights are downloaded from the Hugging Face Hub on first run. Each VLM is
loaded with the Hugging Face `transformers` library in `bfloat16` precision.

## Data Curation :

Ground truth is generated entirely through **deterministic derivation functions** on
existing dataset annotations — no manual question-answer labeling. Each question is
mapped to a binary label from annotation attributes (weather, time of day, scene
type, traffic-light state, per-class object counts, bounding-box geometry, occlusion
metadata; and, for nuScenes, camera-attributed 2D object positions in CAM_FRONT).

Questions are written from the **driver's perspective** (e.g. *"Should the driver
turn on the headlights?"* rather than *"Is it raining?"*), and the correct answer is
designed to vary across visually similar scenes so that models cannot rely on
condition labels alone. Each question is further expanded into two meaning-preserving
paraphrases and one negated variant.

**-** The actual repository layout (as shipped in this snapshot):

```
Can_VLMs_Drive
├── manifests/                      # the 7 manifests actually used by eval.py (see pipeline section)
│    ├── aw_manifest_v2.json
│    ├── aw_manifest_cf_jarvis.json
│    ├── aw_phase2_manifest_extended.json
│    ├── ji_manifest_v1.json
│    ├── ji_manifest_linguistic.json
│    ├── ji_phase2_manifest.json
│    └── ns_manifest_200_extended.json
├── JSONs/adverse_weather/          # a duplicate copy of aw_manifest_v2.json + per-model analysis JSONs
├── images/                         # benchmark images (double-nested — see pipeline section)
│    ├── adverse_weather/{bdd, counterfactual_jarvis}/
│    ├── junctions/bdd/
│    └── nuscenes/images/
├── code/                           # eval / fix-up / analysis / export scripts (~25 scripts)
├── scripts/                        # per-model, per-category SLURM launch scripts + bundle collectors
└── results/                        # NOT included — created by you when you run eval.py (see below)
```

See **[Reproducing the Full Pipeline](#reproducing-the-full-pipeline-)** below for exactly which
script consumes which manifest and in what order.

## Evaluation :

Each model is evaluated across the full protocol matrix — with-image (Mode A, the
default), no-image (Mode C, `--no-image`), counterfactual weather pairs, and
linguistic variants — writing one CSV per (model, category, manifest, mode) to
`results/` (create this directory yourself; it is not shipped in the repo and every
script below assumes it already exists). `code/eval.py` does **not** have a
`--mode` flag — Mode A/C is controlled by the presence of `--no-image`:

```shell
mkdir -p results

# Adverse Weather — with image (Mode A) / no image (Mode C)
python code/eval.py --model internvl3 --category adverse_weather \
    --manifest manifests/aw_manifest_v2.json --suffix _v2 --img-dir images/adverse_weather
python code/eval.py --model internvl3 --category adverse_weather \
    --manifest manifests/aw_manifest_v2.json --suffix _v2 --img-dir images/adverse_weather --no-image

# Adverse Weather — counterfactual clear/fog/rain/snow pairs
python code/eval.py --model internvl3 --category adverse_weather \
    --manifest manifests/aw_manifest_cf_jarvis.json --suffix _cf --img-dir images/adverse_weather

# Junctions & Intersections — baseline questions, then paraphrase + negation variants
python code/eval.py --model internvl3 --category junctions \
    --manifest manifests/ji_manifest_v1.json --suffix _ji_v1 --img-dir images/junctions
python code/eval.py --model internvl3 --category junctions \
    --manifest manifests/ji_manifest_linguistic.json --suffix _ji_ling --img-dir images/junctions

# nuScenes spatial reasoning
python code/eval.py --model internvl3 --category nuscenes \
    --manifest manifests/ns_manifest_200_extended.json --suffix _ns --img-dir images/nuscenes
```

`--img-dir images/adverse_weather` etc. only resolves correctly once you've flattened
the double-nested image folders shipped in this snapshot — see step 3 of
**[Reproducing the Full Pipeline](#reproducing-the-full-pipeline-)** before running any
of the above. Swap `--model internvl3` for any key in `MODEL_LOADERS` inside
`code/eval.py` (`moondream`, `smolvlm`, `paligemma`, `llava_ov`, `gemma`, `qwen3`,
`qwen3_thinking`, `kimi`).

There is no single `code/analyze.py` entry point — each evaluation protocol has its
own analysis script (`code/analyze_junctions.py`, `code/analyze_aw_cf.py`,
`code/analyze_aw_linguistic.py`, …) that computes the metrics defined in the paper
(overall / GT-conditioned accuracy, affirmation gap, Scene-Change Sensitivity,
paraphrase consistency, negation accuracy). See the full pipeline section below for
which script to run on which CSVs.

## Reproducing the Full Pipeline :

This section walks through the codebase end-to-end — manifests → inference → post-hoc
fixes → metrics → tables — so you can reproduce (or extend) the paper's results. It
documents what is actually in this repository, including a few packaging quirks you'll
need to work around (flagged below).

### Naming convention cheat sheet

Manifest, script, and output filenames all follow the same shorthand:

| Token | Meaning |
|---|---|
| `aw` | Adverse Weather category |
| `ji` | Junctions & Intersections category |
| `ns` | nuScenes / DriveLM spatial-reasoning extension |
| `v1` | baseline question set for a category (e.g. `ji_manifest_v1.json`) |
| `v2` | the extended Adverse Weather manifest (real + synthetic images, linguistic variants included) |
| `cf` | counterfactual weather pairs (same scene, clear → fog/rain/snow) |
| `_ling` / `linguistic` | paraphrase + negation variants of a category's questions |
| `phase2` | small-subset, verbose free-text (evidence/rationale) re-evaluation, as opposed to the main forced yes/no sweep |
| Mode A | with-image evaluation (default `eval.py` behaviour) |
| Mode C | no-image / blank-image baseline (`eval.py --no-image`) |

### Step 0 — Environment

No `requirements.txt` is checked in. Install at minimum:

```shell
pip install torch torchvision transformers pillow openpyxl diffusers \
            accelerate bitsandbytes huggingface_hub qwen_vl_utils
```

(`bitsandbytes` is only needed for the 4-bit InternVL3 / LLaVA-OneVision loaders;
`qwen_vl_utils` only for the Qwen3 loaders; `diffusers` only if you re-run the
Stable-Diffusion weather augmentation in Step 2.)

### Step 1 — Repository map: what's already built vs. what you'd rebuild

The manifests in `manifests/` are **already built** — ground truth, paraphrases, and
negations are baked in as static JSON, so you can jump straight to Step 4
(evaluation) using the provided manifests and images. The manifest-*construction*
scripts in `code/` are documented here for completeness / extending the benchmark to
new images, not because you need to re-run them:

- **`code/build_manifests.py`** — the original, from-scratch builder. Reads a raw
  BDD100K `bdd100k_labels_images_val.json` annotation file and writes four manifests
  (`adverse_weather`, `junctions`, `human_behaviour`, `negation_bias`) using
  hand-written deterministic GT-derivation rules keyed off `weather`/`timeofday`/
  `scene`/object-category attributes. This is the ancestor of the shipped `ji_manifest_v1.json`;
  the `human_behaviour`/`negation_bias` categories it builds are **not** present among
  the shipped manifests (superseded by the linguistic-variation manifests below).
- **`code/select_aw_images.py`** → **`code/download_ip2p.py`** +
  **`code/augment_weather_sd.py`** → **`code/build_aw_new_manifest.py`** — the pipeline
  that produced the extended `aw_manifest_v2.json`: sample real rainy/snowy/foggy BDD100K
  images plus a pool of clear images (`select_aw_images.py`), synthesize weather variants
  of the clear pool with a Stable-Diffusion img2img model (`augment_weather_sd.py` —
  `download_ip2p.py` is a superseded InstructPix2Pix alternative, unused in the final
  path), then merge real + synthetic images into one manifest with both "easy"
  (auto-derived GT) and "hard" (placeholder GT, needs manual annotation) questions
  (`build_aw_new_manifest.py`).
- **`code/update_scs_snowy.py`** — a later correction: swaps the Scene-Change-Sensitivity
  source for the "snowy" condition from the JarvisIR counterfactual to the SD-augmented
  image (JarvisIR's snow effect was judged too subtle), patching `aw_cf_analysis.json`
  in place. Already reflected in `analyze_aw_cf_v2.py`; nothing to re-run.
- **`code/eval.py.bak`**, **`code/eval_old.py`**, and the stray zero-byte `code/sed*` /
  `scripts/sed*` files are leftover editing artifacts, not part of the pipeline — safe
  to ignore or delete.

### Step 2 — Manifests reference table

| Manifest | Images | Distinguishing content | Consumed by |
|---|---|---|---|
| `manifests/aw_manifest_v2.json` | 180 | real + SD-augmented weather images; questions tagged `use_for: main_eval` or `linguistic_variation` | `eval.py --category adverse_weather --suffix _v2` |
| `manifests/aw_manifest_cf_jarvis.json` | 72 | JarvisIR-synthesized clear→fog/rain/snow pairs (`source_image_id` links back to the clear original) | `eval.py --category adverse_weather --suffix _cf` |
| `manifests/aw_phase2_manifest_extended.json` | 19 | small AW subset for verbose-reasoning re-evaluation | `eval_phase2_aw.py` |
| `manifests/ji_manifest_v1.json` | 140 | baseline junction questions | `eval.py --category junctions --suffix _ji_v1` |
| `manifests/ji_manifest_linguistic.json` | 140 | same images, questions replaced with paraphrase/negation variants (`source_q_id` points back to the `v1` question) | `eval.py --category junctions --suffix _ji_ling` |
| `manifests/ji_phase2_manifest.json` | 23 | small JI subset for verbose-reasoning re-evaluation | `eval_phase2_ji.py` |
| `manifests/ns_manifest_200_extended.json` | 200 | nuScenes/DriveLM `CAM_FRONT` spatial-reasoning questions | `eval.py --category nuscenes --suffix _ns` |

`JSONs/adverse_weather/aw_manifest_v2.json` is a byte-identical duplicate of
`manifests/aw_manifest_v2.json`. The five `aw_<model>_v2_analysis.json` files alongside
it (per-model accuracy/bias/SCS/linguistic summaries) are consumed by
`build_excel_v2.py` / `build_excel_final.py`, but no script in this repo currently
*produces* them in that exact shape — treat them as pre-computed reference outputs
rather than something you regenerate from scratch.

### Step 3 — Images: fix the double-nested folders first

The AW and JI image folders in this snapshot have an accidental extra nesting level,
and every manifest's `image_path` field assumes a flatter layout (`eval.py` resolves an
image as `os.path.join(--img-dir, entry["image_path"])`). Flatten them once, up front:

```shell
mkdir -p images/adverse_weather/images
mv images/adverse_weather/bdd/bdd images/adverse_weather/images/bdd
mv images/adverse_weather/counterfactual_jarvis/counterfactual_jarvis images/adverse_weather/images/counterfactual_jarvis
rmdir images/adverse_weather/bdd images/adverse_weather/counterfactual_jarvis

mkdir -p images/junctions/images
mv images/junctions/bdd/bdd images/junctions/images/bdd
rmdir images/junctions/bdd
```

`images/nuscenes/images/` is already flat and needs no change. After this fix,
`--img-dir images/adverse_weather`, `--img-dir images/junctions`, and
`--img-dir images/nuscenes` (as used in the Evaluation section above and in Step 4)
all resolve correctly — verify with e.g.
`python3 -c "import json,os; m=json.load(open('manifests/aw_manifest_v2.json')); print(os.path.exists(os.path.join('images/adverse_weather', m[0]['image_path'])))"`.

### Step 4 — Run the primary evaluation sweep

For each model and each manifest, run Mode A and Mode C (Mode C only applies to
AW and JI — nuScenes and the counterfactual/linguistic manifests are Mode A only in
this codebase). See the commands in the [Evaluation](#evaluation-) section above;
`scripts/run_<model>_<aw|ji|ns>[_v1|_v2|_cf].sh` are SLURM wrappers around exactly
those `eval.py` calls, one pair (Mode A + Mode C) per model/category/manifest
combination. They're written for the original cluster (`#SBATCH` directives,
hard-coded `/home2/muskan.singh/...` paths) — copy the `python code/eval.py ...`
lines out and substitute your own paths rather than running the `.sh` files as-is.
Each run resumes from existing rows in its output CSV (keyed on `image_id`+`q_id`), so
interrupting and re-launching a script is safe.

### Step 5 — Phase 2: verbose-reasoning re-evaluation (optional)

`code/eval_phase2_aw.py --model <name> [--no-image]` and
`code/eval_phase2_ji.py --model <name> [--no-image]` (choices: `paligemma`, `smolvlm`,
`llava_ov`, `internvl3`) re-run a small subset of each category
(`aw_phase2_manifest_extended.json` / `ji_phase2_manifest.json`) asking for a one-line
`Answer: Yes/No/Unanswerable` plus a short evidence rationale. These feed the paper's
qualitative "targeted reasoning failure" analysis (hallucination / safety-prior /
negation-blindness examples), not the headline accuracy numbers. Both scripts hard-code
their manifest/image/results paths at the top of the file (`MANIFEST`, `IMG_BASE`,
`RESULTS`) — edit those constants to point at your local `manifests/` and `images/`
directories before running.

### Step 6 — Post-hoc response-extraction fixes (run before analysis)

`eval.py`'s `extract_yes_no` is a simple heuristic and misclassifies some free-text
responses as `unclear`. Before computing metrics, run the category-specific fixers,
which re-parse `full_response` with stricter rules and recompute `correct` — they all
read/write CSVs in `results/`, with paths hard-coded at the top of each file:

- **`code/fix_extractor_junctions.py`** — re-parses `unclear` rows across all 5 models'
  `junctions_<model>_ji_v1_*.csv` files, writing `junctions_<model>_ji_v1_fixed.csv`
  (the file the JI analysis scripts actually read).
- **`code/fix_extractor_ns_moondream.py`** — same idea, scoped to
  `nuscenes_moondream_ns.csv` → `nuscenes_moondream_ns_fixed2.csv`.
- **`code/reextract_moondream_v2.py`** / **`code/reextract_paligemma_v2.py`** — re-apply
  an improved extractor to `adverse_weather_{moondream,paligemma}_v2.csv` **in place**
  (the original is backed up to `..._v2_raw.csv` first).
- **`code/classify_responses.py`** / **`code/classify_responses_v2.py`** — not
  correctness fixers; these bucket `full_response` text into qualitative failure/success
  modes (hallucination, safety-prior reasoning, language-prior reliance, visually
  grounded, …) used for the paper's qualitative figures.
- **`code/check_ji_ns_response_style.py`** — audits response length/style across all JI
  and NS result CSVs; writes a `results/response_style_audit/` report. Informational only.

### Step 7 — Compute the diagnostic metrics

Each of these reads fixed/re-extracted CSVs plus the relevant manifest and writes a
JSON summary to `results/`. **All of them hard-code `RESULTS_DIR`/`OUTPUT_DIR`/
`MANIFEST_PATH` as `/home2/muskan.singh/...` constants near the top of the file** — edit
those three or four lines to point at your own `results/` and `manifests/` paths before
running (none of them take CLI arguments except `analyze_results.py`):

| Script | Reads | Computes | Writes |
|---|---|---|---|
| `analyze_junctions.py` | `junctions_<model>_ji_v1_fixed.csv` (Mode A) + `_noimage_ji_v1.csv` (Mode C), all 5 models | overall/GT-conditioned accuracy, affirmation gap, per-question accuracy, unclear rate | `junctions_analysis.json` |
| `analyze_aw_cf.py` | `adverse_weather_<model>_v2.csv` (clear) + `_cf.csv` / `_noimage_cf.csv` (JarvisIR pairs) | Scene-Change Sensitivity (Eq. 3 in the paper) | `aw_cf_analysis.json` |
| `analyze_aw_cf_v2.py` | same, but snow comes from the SD-augmented `_v2.csv` rows instead of JarvisIR | SCS (corrected snow source) | `aw_cf_analysis_v2.json` |
| `analyze_aw_linguistic.py` | `aw_manifest_v2.json` + `adverse_weather_<model>_v2.csv` | paraphrase consistency, negation accuracy (Eq. 4–5) | `aw_linguistic_analysis.json` |
| `analyze_ji_linguistic.py` | `ji_manifest_linguistic.json` + `junctions_<model>_ji_v1_fixed.csv` + `junctions_<model>_ji_ling.csv` | paraphrase consistency, negation accuracy for JI | `junctions_linguistic_analysis.json` |
| `analyze_results.py` | any category's manifest + CSVs (`--results_dir`/`--model`/`--category` flags) | generic overall/GT-conditioned accuracy; also still supports the older `negation_bias` category from `build_manifests.py` | printed report |

### Step 8 — Export tables / spreadsheets

Final, human-facing reporting layer, all AW-focused and all read from `results/`:

- **`code/extract_aw_tables.py`** (6 tables) and **`code/extract_aw_tables_final.py`**
  (11 tables, adds behavioral/bias classification) print TSV to stdout for pasting into
  a spreadsheet: `python3 code/extract_aw_tables_final.py > aw_tables.tsv`.
- **`code/build_excel.py`** → **`build_excel_v2.py`** → **`build_excel_final.py`** are
  successive versions of an `openpyxl` workbook builder (Moondream only → +PaliGemma →
  +SmolVLM), writing `results/adverse_weather_results[_v2|_v3].xlsx`. Use
  `build_excel_final.py` for the most complete workbook; the earlier versions are kept
  for reference.

### Step 9 — Bundling results (optional)

`scripts/collect_aw_exact_files.sh`, `scripts/collect_ji_ns_exact_files.sh`, and
`scripts/collect_phase2_bundle.sh` are plain file-copy + `tar.gz` utilities (no
inference or analysis) that gather a fixed, named list of CSV/JSON outputs from
`results/` into a timestamped archive — useful for moving results off a SLURM cluster,
not a required pipeline step.

## Results :

Diagnostic metrics for all evaluated models are reported in the paper. Raw
per-question outputs are provided under `results/`, organized by category
(Adverse Weather, Junctions & Intersections, nuScenes) and evaluation setting.

## Acknowledgements 

<!-- TODO: confirm funding / institutional acknowledgements -->
This work was supported by IIIT Hyderabad.

The evaluation builds on the following open-source VLMs and datasets:
[BDD100K](https://bdd-data.berkeley.edu/), [DriveLM](https://github.com/OpenDriveLab/DriveLM),
[nuScenes](https://www.nuscenes.org/), and the Hugging Face `transformers` ecosystem.

## BibTeX Citation 

If you use this benchmark, please cite our paper:

<!-- TODO: update page numbers / details once proceedings are published -->
```bibtex
@inproceedings{vlmsdrive2026,
  title     = {Do VLMs Really Look at the Road? A Diagnostic Evaluation for Autonomous Driving},
  author    = {TODO: author list},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```


## Appendix: Canonical `results/` Files :

After running the full pipeline above, your `results/` directory should contain these
58 files (grouped by evaluation protocol); anything else under `results/` is an
intermediate or deprecated artifact (e.g. pre-`fix_extractor`/`reextract` raw CSVs).

**Adverse Weather (AW) — Phase 1**
```
adverse_weather_moondream_v2.csv
adverse_weather_moondream_noimage_v2.csv
adverse_weather_paligemma_v2.csv
adverse_weather_paligemma_noimage_v2.csv
adverse_weather_smolvlm_v2.csv
adverse_weather_smolvlm_noimage_v2.csv
adverse_weather_llava_ov_v2.csv
adverse_weather_llava_ov_noimage_v2.csv
adverse_weather_internvl3_v2.csv
adverse_weather_internvl3_noimage_v2.csv
```

**AW Counterfactual**
```
adverse_weather_moondream_cf.csv
adverse_weather_moondream_noimage_cf.csv
adverse_weather_paligemma_cf.csv
adverse_weather_paligemma_noimage_cf.csv
adverse_weather_smolvlm_cf.csv
adverse_weather_smolvlm_noimage_cf.csv
adverse_weather_llava_ov_cf.csv
adverse_weather_llava_ov_noimage_cf.csv
adverse_weather_internvl3_cf.csv
adverse_weather_internvl3_noimage_cf.csv
```

**Junctions & Intersections (JI) — Phase 1**
```
junctions_moondream_ji_v1_fixed.csv
junctions_moondream_noimage_ji_v1.csv
junctions_paligemma_ji_v1_fixed.csv
junctions_paligemma_noimage_ji_v1.csv
junctions_smolvlm_ji_v1_fixed.csv
junctions_smolvlm_noimage_ji_v1.csv
junctions_llava_ov_ji_v1_fixed.csv
junctions_llava_ov_noimage_ji_v1.csv
junctions_internvl3_ji_v1_fixed.csv
junctions_internvl3_noimage_ji_v1.csv
```

**JI Linguistic**
```
junctions_moondream_ji_ling.csv
junctions_moondream_noimage_ji_ling.csv
junctions_paligemma_ji_ling.csv
junctions_paligemma_noimage_ji_ling.csv
junctions_smolvlm_ji_ling.csv
junctions_smolvlm_noimage_ji_ling.csv
junctions_llava_ov_ji_ling.csv
junctions_llava_ov_noimage_ji_ling.csv
junctions_internvl3_ji_ling.csv
junctions_internvl3_noimage_ji_ling.csv
```

**nuScenes (NS)**
```
nuscenes_moondream_ns.csv
nuscenes_paligemma_ns_fixed.csv
nuscenes_smolvlm_ns_fixed.csv
nuscenes_llava_ov_ns.csv
nuscenes_internvl3_ns.csv
```

**Phase 2 Verbose**
```
phase2_aw_llava_ov.csv
phase2_aw_llava_ov_noimage.csv
phase2_aw_internvl3.csv
phase2_aw_internvl3_noimage.csv
phase2_ji_llava_ov.csv
phase2_ji_llava_ov_noimage.csv
phase2_ji_internvl3.csv
phase2_ji_internvl3_noimage.csv
```

**Analysis JSONs**
```
aw_cf_analysis.json
aw_linguistic_analysis.json
junctions_analysis.json
junctions_linguistic_analysis.json
```


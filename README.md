# Ouroboros

[Versione italiana](README_IT.md)

Python implementation of the thesis gender bias experiment, with independent search and verification. The project separates the experimental loop from models and analysis; it uses no agent frameworks, databases, or paid services.

**Selected models**

| Role | Model | Execution |
|---|---|---|
| Target 1 | `black-forest-labs/flux-3-image-klein` | Distilled modular Diffusers, 16 steps, no CFG |
| Target 2 | `Qwen/Qwen-Image-2.1` | Diffusers, 40 steps |
| Attacker | `ministral-3:8b` | Ollama, temperature 1 |
| Judge | `qwen3-vl:8b-instruct-q4_K_M` | Ollama, temperature 0 |
| Secondary measurement | FairFace ResNet-34 | PyTorch + dlib, post hoc |

The attacker uses Ministral 3 8B and the judge uses Qwen3-VL 8B, which belong to different families. Their suitability must be checked in the pilot; using a different family does not guarantee independent errors. Model names are configurable. The attacker receives only text and search feedback.

## Installation

Use Python 3.11 or later, preferably in a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This is sufficient for tests, mocks, and core statistics. For inference, install PyTorch for your GPU following the [official instructions](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e '.[models]'
pip install -r requirements-models.txt
ollama pull ministral-3:8b
ollama pull qwen3-vl:8b-instruct-q4_K_M
```

Ollama must be running (`ollama serve`, if the app has not already started it). Its version must support Qwen3-VL and the `think: false` parameter. Ollama models are not downloaded automatically; Diffusers downloads the target weights on first use. Qwen Image **2.1** uses `QwenImage21Pipeline`, not the pipeline for the previous version. The Diffusers revision is pinned in `requirements-models.txt`; after preparing the environment, also save `pip freeze > requirements-lock.txt`.

FLUX 3 uses `Flux3DistilledAutoBlocks`, separate component loading, and
`ComponentsManager` for CUDA offloading. The weights require accepting the
license on Hugging Face and an authorized `HF_TOKEN`.
The upsampler is explicitly excluded: this audit measures the response to the
frozen prompt, without rewrites that could introduce gender attributes. This is
an experimental choice that differs from the BFL recipe, which recommends extended captions;
results with short prompts do not represent the quality of the full BFL pipeline.
Use new runs for FLUX 3: replaying FLUX 2 checkpoints requires the original code.
Measure pilot runtime and memory again before committing the budget.

The presets use the documented resolution for each model: 1024² for FLUX, 2048² for Qwen. The comparison controls the number of images, **not** GPU seconds or FLOPs. Change width/height before the run for a common-resolution design, and disclose that choice.

`device: auto` selects CUDA, then MPS, then CPU. CPU offloading is enabled on CUDA; on MPS, the model is moved to the device without that mechanism. MPS and CPU paths are implemented but have not been verified with real weights in this session. Sequential execution unloads models between phases: this reduces peak memory but increases reload time. For a static check, set `available_memory_gb` and a conservative `target_estimate_gb` in the JSON: these are not automatic VRAM and RAM estimates. The default value of 0 disables the check.

## Quick check without a GPU

```bash
python -m unittest discover -s tests -v
python -m ouroboros run --config configs/flux-test.json --mock --output runs/mock
```

The command prints the run directory, for example `runs/mock/<id>`. Pass that path to subsequent commands:

```bash
python -m ouroboros report runs/mock/<id>
python -m ouroboros replay runs/mock/<id> --limit 8
```

The mock produces solid-color images and artificial labels. It is only intended to verify the software; its reports are marked **MOCK** and are not thesis results.

## 1. Judge validation

Download `hf_test_fairness_generated.json` and `test.zip` from the [official T2ISafety dataset](https://huggingface.co/datasets/OpenSafetyLab/t2i_safety_dataset/tree/main), then extract the ZIP. The test set also includes files for real images; the following preset uses only generated images. Do not use captions as labels or mix training and test data.

```bash
hf download OpenSafetyLab/t2i_safety_dataset \
  hf_test_fairness_generated.json test.zip \
  --repo-type dataset --local-dir data/t2isafety
unzip data/t2isafety/test.zip -d data/t2isafety
python -m ouroboros import-t2isafety \
  --annotations data/t2isafety/hf_test_fairness_generated.json \
  --image-root data/t2isafety \
  --output data/t2isafety/manifest.json
python -m ouroboros validate-judge \
  --config configs/flux-test.json \
  --manifest data/t2isafety/manifest.json --sample 150 \
  --output runs/judge-validation.json
```

`--image-root` must contain the `evaluator_test` folder referenced in the annotations; if the ZIP adds an outer directory, adjust the path. The importer checks every file. It reads only the dataset's `Gender`, `Age`, and `Race` annotations: no annotations are invented. The sample is random and reproducible, not stratified; small subgroups should be interpreted with caution.

Validation returns accuracy, Cohen's kappa, a confusion matrix, errors, and accuracy by age/race with bootstrapping. It is binary and closed-set, like the dataset; in production, the judge can use `unclear`. The file allows the pilot to resume without repeating images already processed. Failed classifications remain visible errors; use a new output to retry the pilot.

Examine the results before running the experiment. The code checks that a validation exists with the same judge digest; it **does not automatically certify** that accuracy is sufficient: the thesis does not prescribe a numerical threshold. A run without a pilot requires `--exploratory` and is recorded as preliminary.

## 2. Paired experiment

```bash
python -m ouroboros doctor --config configs/flux-test.json
python -m ouroboros run --config configs/flux-test.json \
  --validation runs/judge-validation.json
```

After testing, use the full presets for each target:

```bash
python -m ouroboros run --config configs/flux-full.json \
  --validation runs/judge-validation.json
python -m ouroboros run --config configs/qwen-full.json \
  --validation runs/judge-validation.json
```

| Mode | Professions | Images/batch (M) | Search (B) | Verification per prompt (V) | Total images/target |
|---|---:|---:|---:|---:|---:|
| test | 10 | 2 | 5 | 5 | 300 |
| full | 175 | 8 | 5 | 5 | 21,000 |

For each profession:

1. Search tests B prompts, without early stopping, guided by `abs(n_female - n_male) / M`. Unclear images remain in the denominator.
2. The highest-scoring prompt is frozen; ties go to the first. Batches containing only unclear images and errors cannot win.
3. The selected prompt and the neutral prompt each generate V*M new images, with seeds paired across conditions and distinct from search seeds. The judge sees only the images. Verification feedback is not returned to the attacker.
4. The report aggregates all verification images and compares female share, absolute skew, and coverage. It does not use the maximum across batches or a success threshold.

In the JSON, `iterations` is B and `verification_batches` is V. `max_generations` limits target calls, including interrupted or failed calls. The plan is `professions * M * (B + 2*V)`: search is an additional cost on top of paired verification. The full values are a starting point to size using a GPU pilot, not a guarantee of staying within budget.

Repeating the same command resumes the checkpoint. Selection is saved before verification and is not recalculated using the holdout. The configuration and model digests identify the run; code and image hashes block incompatible resumption. Use only one process per run directory. An interrupted call is counted even if it saves no image: allow headroom in the initial cap.

Before starting a profession, the budget is checked to ensure search and verification can be completed. If it is insufficient, the run stops. Errors and refusals consume a slot; if no candidate is eligible for selection, verification is skipped and this is recorded. The comparison uses only complete, classifiable pairs and exposes all exclusions. Old JSON files with `quorum` or `baseline` cannot be reused: use `main` for those runs.

## 3. FairFace and reports

```bash
pip install -e '.[fairface]'
```

Download the **7-race** checkpoint `fairface_alldata_20191111.pt` and the two dlib files `mmod_human_face_detector.dat` and `shape_predictor_5_face_landmarks.dat` from the [official FairFace repository](https://github.com/dchen236/FairFace) (decompress them if necessary). Do not use the 4-race checkpoint: the head positions differ. Building dlib may require CMake and a C++ compiler.

```bash
python -m ouroboros fairface runs/<id> \
  --weights models/fairface_alldata_20191111.pt \
  --detector models/mmod_human_face_detector.dat \
  --landmarks models/shape_predictor_5_face_landmarks.dat
python -m ouroboros report runs/<id>
python -m ouroboros replay runs/<id> --limit 8
```

FairFace is applied to all saved images without feeding into the attacker's feedback. It classifies each face using dlib alignment; it preserves the face count and returns only perceived gender. Per-image kappa uses exactly one face; zero faces are not converted to `male`. Weight and image files are identified by hashes.

Optional strategy analysis uses MiniLM + HDBSCAN:

```bash
pip install -e '.[analysis]'
python -m ouroboros report runs/<id> --cluster-strategies
```

Main outputs: `checkpoint.json` (history, frozen selection, and raw responses), `images/`, `report.md`, `report.json`, `per_profession.csv`; after the corresponding commands, also `fairface.json`, `strategies.json`, and `replay.json`. Reports contain counts, shares, and skew aggregated over verification, paired differences, and 95% bootstrap intervals across professions. Per-profession shares include Wilson confidence intervals. Undefined metrics are `null`; errors do not become zeros. FairFace compares only verification images in the report.

## Navigating the code

- `config.py`: parameters and run identification.
- `experiment.py`: the main loop and memory management.
- `models.py`, `prompts.py`: model access and versioned prompts.
- `storage.py`: checkpoints and hashes.
- `validation.py`, `fairface.py`: the two judge checks.
- `analysis.py`, `replay.py`: metrics, strategies, and reproducibility.
- `cli.py`: terminal commands.

Methodological choices and differences from the thesis formulation are described in [docs/protocol.md](docs/protocol.md). The provenance of the 175 names is recorded in `data/provenance.json`: the list comes from the actual values of the `profession` column, not the incomplete list in the dataset card.

## Verified references

- [FLUX 3 Image Klein: model card and inference](https://huggingface.co/black-forest-labs/flux-3-image-klein).
- [Qwen Image 2.1: model card and inference](https://huggingface.co/Qwen/Qwen-Image-2.1).
- [Ministral 3 8B Q4](https://ollama.com/library/ministral-3:8b), [Qwen3-VL-8B Instruct Q4](https://ollama.com/library/qwen3-vl:8b-instruct-q4_K_M).
- [Stable Bias](https://huggingface.co/datasets/stable-bias/professions), CC-BY-SA-4.0 license for derived data; [T2ISafety](https://github.com/adwardlee/t2i_safety); [FairFace](https://github.com/dchen236/FairFace).

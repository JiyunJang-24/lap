# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LAP (Language-Action Pre-Training) is a Vision-Language-Action (VLA) model for robot manipulation that represents robot actions as language tokens, enabling zero-shot cross-embodiment transfer. It is built on top of [OpenPI](https://github.com/Physical-Intelligence/openpi) (in `third_party/openpi/`), extending it with language-action pre-training, VQA datasets, and a chain-of-thought (CoT) training pipeline.

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Python 3.11 is required (not 3.12+).

```bash
# Clone with submodules
git submodule update --init --recursive

# Install main environment
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

LIBERO evaluation uses a **separate Python 3.8 virtual environment** (`scripts/libero/.venv`):

```bash
source setting.sh  # sets up the libero venv
```

## Common Commands

### Training

```bash
# GPU training
JAX_PLATFORMS=cuda uv run --group cuda scripts/train.py lap_libero --exp-name=lap_libero --data.rlds_data_dir=<your_data_dir>

# TPU training
uv run scripts/train.py lap_libero --exp-name=lap_libero --data.rlds_data_dir=<your_data_dir>

# Custom data mix
JAX_PLATFORMS=cuda uv run --group cuda scripts/train.py lap --exp-name=lap_custom --data.rlds_data_dir=<data_dir> --data.data-mix=<mix_name>
```

### Inference / Policy Server

```bash
# Start policy server (default LAP checkpoint)
JAX_PLATFORMS=cuda uv run --group cuda scripts/serve_policy.py --env=LAP

# Start policy server from checkpoint
JAX_PLATFORMS=cuda uv run --group cuda --active scripts/serve_policy.py policy:checkpoint --policy.config=lap_libero --policy.dir=checkpoints/lap --policy.type=flow
```

### LIBERO Evaluation (two terminals)

```bash
# Terminal 1: policy server
JAX_PLATFORMS=cuda uv run --group cuda --active scripts/serve_policy.py policy:checkpoint --policy.config=lap_libero --policy.dir=checkpoints/LAP-3B-Libero --policy.type=flow

# Terminal 2: simulator (uses libero venv)
source $PWD/scripts/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=$PWD/third_party/openpi/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/openpi/third_party/libero
python scripts/libero/main.py
```

Or use the convenience script: `bash eval.sh`

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Tests

```bash
uv run pytest                        # all tests
uv run pytest src/lap/...            # specific module
uv run pytest -m "not manual"        # skip manual tests
```

## Architecture

### Relationship to OpenPI

LAP wraps and extends `third_party/openpi/`. The key pattern is that LAP defines its own model, policy, and data-loading classes that inherit from or adapt the upstream OpenPI equivalents. When reading code, always check if a class is defined in `src/lap/` or comes from `openpi`.

### Key Source Modules (`src/lap/`)

- **`models/`** — Core model code:
  - `lap_config.py`: `LAPConfig` dataclass — the primary knob for all model hyperparameters (action horizon, loss weights, whether to use Gemma2 vs Gemma3, LoRA, etc.)
  - `lap.py`: `LAP` model (PaliGemma/Gemma2 backbone), extends `pi0.Pi0`
  - `lap_gemma3.py`: `LAPGemma3` model variant (Gemma3 backbone)
  - `model_adapter.py`: `CoTObservation` — extends the upstream `Observation` with tokenized reasoning fields
  - `tokenizer.py`: `PaligemmaTokenizer`, `Gemma3Tokenizer`, `FASTTokenizer`, etc.
  - `backbones/`: Gemma2 and Gemma3 backbone wrappers

- **`training/`** — Training infrastructure:
  - `config.py`: `TrainConfig` and `DataConfig` dataclasses, named experiment configs (e.g., `lap_libero`), and `ModelTransformFactory`
  - `state.py`: `TrainState` (wraps NNX params, optimizer state, EMA params)
  - `checkpoints.py`: Orbax-based checkpoint save/restore
  - `mh_sharding.py`: FSDP mesh setup for multi-host/TPU training

- **`datasets/`** — Data pipeline:
  - `data_loader.py`: `RLDSDataLoader` wrapping TensorFlow RLDS datasets
  - `dataset_mixer.py`: `OXEDatasets` — mixes multiple RLDS datasets by the weights in `utils/mixtures.py`
  - `utils/mixtures.py`: Named dataset mixtures (e.g., `oxe_magic_soup`, `libero_finetune`)
  - `robot/`: DROID-specific and OXE dataset wrappers
  - `vqa/`: VQA datasets (COCO, LVIS, PACO, PixMo, VQAv2, bbox tasks) used for visual language pre-training

- **`policies/`** — Inference policy transforms:
  - `transforms/`: `CoTInputs` and `CoTOutputs` — transform raw observations into tokenized model inputs and decode outputs back to actions
  - `policy_config_adapter.py`: Maps environment names to checkpoint configs for `serve_policy.py`
  - `lang_action_formats.py`: Defines how actions are encoded as text strings

### Training Flow

`scripts/train.py` → `TrainConfig` (from `src/lap/training/config.py`) → creates model via `LAPConfig.create()` → loads RLDS data via `RLDSDataLoader` → runs JAX-JIT'd training steps with FSDP sharding → saves checkpoints via Orbax.

### Data Flow

RLDS datasets (TF format) → `OXEDatasets` mixer → per-sample transforms (normalization, image augmentation, tokenization of language-action strings) → `CoTObservation` batch → model `compute_loss()`.

### Adding a Custom Dataset

1. Add dataset weights to `src/lap/datasets/utils/mixtures.py`
2. Register dataset config in `src/lap/datasets/utils/configs.py`
3. Place RLDS data at `<rlds_data_dir>/<dataset_name>/`

### Checkpoints

- Default LAP-3B checkpoint: `checkpoints/lap/` (download from `lihzha/LAP-3B` on HuggingFace)
- LIBERO fine-tuned: `checkpoints/lap_libero/` (download from `lihzha/LAP-3B-Libero`)
- Additional assets cached at `~/.cache/openpi` (override with `OPENPI_DATA_HOME`)

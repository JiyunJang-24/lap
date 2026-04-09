# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 연구 맥락 (Research Context)

이 팀은 LAP 코드베이스를 기반으로 모델의 문제점을 분석하고 연구 주제로 확장하는 것이 목표다.

**문제:** LAP은 학습 데이터 대부분이 **DROID 환경** 기반이라, LIBERO 환경에서 Zero-shot 테스트 시 성능이 낮다.

**실험 방향 (Ablation Study):** LIBERO 환경에서 **DROID 셋업의 Camera Viewpoint와 Gripper로만 교체**하여, "Unseen Environment" 변수만 단독으로 격리해 Zero-shot 실패 원인이 환경 차이인지 검증한다.

**`scripts/libero/main_custom.py`:** 위 Ablation용 스크립트. LIBERO 시뮬레이터에서 카메라 시점(`viewpoint_rotate`)과 그리퍼(`gripper_type`)를 DROID 셋업으로 교체하는 역할. 작성자: Jiyun (팀 내부, LAP 원저자 팀 아님).

**`xyg_scripts` 모듈:** Jiyun이 작성한 비공개 모듈 (`rotate_recolor_dataset`, `cross_embodiment_utils`). git에 포함되지 않고 zip으로 별도 배포됨. 설치 위치: `third_party/openpi/third_party/libero/xyg_scripts/` (해당 경로가 PYTHONPATH에 추가되기 때문).

## Project Overview

LAP (Language-Action Pre-Training) is a Vision-Language-Action (VLA) model for robotics that enables zero-shot cross-embodiment transfer. It extends the OpenPI framework (vendored in `third_party/openpi`) with a PaliGemma backbone (SigLIP vision + Gemma LLM) and a separate action expert.

## Commands

**Setup:**
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Training:**
```bash
# GPU training
JAX_PLATFORMS=cuda uv run --group cuda scripts/train.py lap_libero --exp-name=lap_libero --data.rlds_data_dir=<dir>

# TPU training
uv run scripts/train.py lap_libero --exp-name=lap_libero --data.rlds_data_dir=<dir>

# Custom dataset
JAX_PLATFORMS=cuda uv run --group cuda scripts/train.py lap --exp-name=lap_custom --data.data-mix=<datamix_name> --data.rlds_data_dir=<dir>
```

**Policy serving (inference):**
```bash
JAX_PLATFORMS=cuda uv run --group cuda scripts/serve_policy.py policy:checkpoint --policy.config=lap_libero --policy.dir=checkpoints/lap --policy.type=flow
```

**Evaluation:**
```bash
uv run scripts/eval.py <config_name> --data.rlds_data_dir=<dir>
```

**LIBERO simulator evaluation:**
```bash
source setting.sh  # Sets up Python 3.8 venv with LIBERO deps
python scripts/libero/main_custom.py
```

**Linting and formatting:**
```bash
uv run ruff check . --fix
uv run ruff format .
```

**Tests:**
```bash
uv run pytest
uv run pytest <test_file>  # single test file
```

## Architecture

### Core Model (`src/lap/models/`)

- **`lap.py`** — Main `LAP` class (inherits Pi0). Combines PaliGemma (vision + language backbone) with a smaller Gemma action expert. Supports four training modes: language-action CE loss, raw action diffusion, future-frame prediction, and VQA.
- **`lap_config.py`** — `LAPConfig` dataclass with model variants. Key params: `action_dim=7`, `action_horizon=16`, `max_token_len` (180–800 depending on variant).
- **`model_adapter.py`** — `CoTObservation` extends the base observation with reasoning tokens, language action tokens, and per-token masks.
- **`tokenizer.py`** — `PaligemmaTokenizer` encodes prompts, numerical language actions, and states into discrete tokens.
- **`backbones/`** — Gemma/Gemma3 language models and SigLIP vision encoder (Flax NNX).

### Data Layer (`src/lap/datasets/`)

- `OXEDatasets` loads RLDS-format robot datasets; `utils/mixtures.py` defines named data blends (e.g., `libero_finetune`, `oxe_magic_soup`).
- VQA datasets (COCO, LVIS, PACO, VQAv2, PixMo) are mixed in for auxiliary losses.
- Images are resized to 224×224. State/action normalization uses `bounds_q99` by default.

### Policy / Inference Layer (`src/lap/policies/`)

- **`lap_policy.py`** — Input/output transform wrappers used at deployment time.
- **`lang_action_formats.py`** — Converts model text output to motor commands (verbose, compact, JSON).
- **`transforms/`** — Image preprocessing, tokenization (input), and action extraction / language detokenization (output).
- **`question_types.py`** — VQA question generation (delta_motion, gripper, direction, etc.).

### Training Layer (`src/lap/training/`)

- **`config.py`** — ~10+ named `TrainConfig` presets (e.g., `lap`, `lap_libero`, `lap_gemma3_4b`). Pass the config name as the first positional arg to `scripts/train.py`.
- **`checkpoints.py`** / **`weight_loaders.py`** — Orbax checkpoint save/load; loading from Gemma3 upstream weights.
- **`mh_sharding.py`** — Multi-host FSDP sharding for TPU/GPU distributed training.
- **`metrics_logging.py`** — Per-dataset token accuracy, action accuracy, and VQA accuracy logged to W&B.

### Entry Points

| Script | Purpose |
|---|---|
| `scripts/train.py` | Main training loop (tyro CLI) |
| `scripts/eval.py` | Holdout-split evaluation |
| `scripts/serve_policy.py` | WebSocket policy server for real robots |
| `scripts/real_robot/droid_main.py` | DROID real-robot interface |
| `scripts/libero/main_custom.py` | LIBERO simulator evaluation |

### Training Data Flow

```
RLDS data → OXEDatasets (batching/shuffle)
         → Transforms (tokenize, augment, normalize)
         → CoTObservation (images + tokenized prompt + language actions + masks)
         → LAP model (embed → generate action/language tokens)
         → Multi-task loss (language CE + action diffusion + prediction + VQA)
         → Optax optimizer + EMA → updated params
```

## Key Conventions

**Multi-task loss weights:** Language CE (1.0), action diffusion (`enable_action_training`), prediction (`pred_prob=0.3`), VQA (`enable_vqa_training`, weight=0.1). Per-dataset VQA weights are configurable in the `TrainConfig`.

**Model variants:**
- `pi05=True`: Discrete state input, full-context action expert (PI0.5 style)
- `pi05=False`: Continuous proprioceptive only (PI0 style)
- `use_fast=True`: No separate action expert (FAST mode)
- Gemma3 configs extend `max_token_len` to 800

**Image keys:** `base_0_rgb`, `left_wrist_0_rgb`, optionally `right_wrist_0_rgb` (bimanual). Wrist images get RandomAffine/ColorJitter/GaussianBlur augmentation with 10% dropout.

**Distributed training:** `JAX_PLATFORMS=cuda` selects GPU; `fsdp_devices` controls sharding granularity; `jax.process_index()` gates W&B and per-process logging.

**Pre-commit hooks** enforce ruff lint/format and UV lockfile sync. `third_party/` is excluded from linting.

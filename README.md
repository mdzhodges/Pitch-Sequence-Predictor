# Pitch-Sequence-Predictor

Predict the next MLB pitch by blending hitter tendencies, pitcher repertoires, game context, and the recent pitch sequence. This project wires Statcast data collection, parquet fusion, PyTorch modeling, and evaluation into a single CLI-friendly package so you can iterate on pitch-sequence models without bespoke notebooks.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Workflow](#data-workflow)
6. [Dataset Details](#dataset-details)
7. [Model Architecture](#model-architecture)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation](#evaluation)
10. [Usage Examples](#usage-examples)
11. [Results](#results)
12. [Contributing](#contributing)
13. [Roadmap](#roadmap)
14. [Maintainers](#maintainers)

---

## Overview
The application ingests Statcast pitch-by-pitch data, enriches each event with hitter and pitcher leaderboards, and produces a fused dataset where every row contains:

- batter/pitcher IDs and handedness,
- contextual state (inning, count, base-out, alignments, scores),
- kinematic measurements (release speed/spin, plate crossing, break),
- and the **next pitch type** thrown in that plate appearance.

During training, each sample carries the history of pitches thrown earlier in the at-bat. The `PitchSequenceEncoder` attends over these sequences, masks logits by pitcher repertoire, and predicts the next pitch class. Macro/micro F1 are tracked to ensure rare offerings retain influence.

---

## Project Structure
```text
Pitch-Sequence-Predictor
├── controller/
│   ├── main.py                # CLI entry point
│   ├── controller.py          # Orchestrates data gen + pipeline
│   ├── cli_arguments.py       # argparse configuration
│   └── config.py              # Pydantic settings loader
├── data_collection/
│   ├── context_data_collection.py  # Statcast retrieval + cleaning
│   └── fusion_collection.py        # Merge hitter/pitcher assets, next-pitch labels
├── preprocessing/
│   └── fusion_dataset.py      # Torch Dataset with numeric/categorical tensors + histories
├── model/
│   ├── pitch_sequence_encoder/
│   │   └── pitch_sequence_encoder.py
│   ├── training/
│   │   └── pitch_sequence_trainer.py
│   ├── class_weights.py       # Per-pitcher class weighting utilities
│   └── pitch_sequence_pipeline.py
├── evaluation/
│   └── eval.py                # Held-out inference + metrics
├── utils/
│   ├── constants.py           # Column lists, pitch map, Statcast window
│   └── logger.py              # Color-coded console logger
├── data/                      # Local parquet outputs + pitcher_allowed.json
├── pyproject.toml             # Poetry dependencies
└── README.md
```

---

## Installation
This repository uses Poetry for dependency management.

```bash
poetry install
```

If you update `pyproject.toml`, run `poetry update` to regenerate `poetry.lock`.

---

## Configuration
Runtime values are centralized in `controller/config.py` (Pydantic). Create a `.env` in the repo root to override defaults:

```
SAMPLE=1000
NUM_EPOCHS=20
BATCH_SIZE=25
LR_PITCH_SEQ=1e-5
DROPOUT_PITCH_SEQ=0.3
HITTER_PARQUET_FILE_PATH=data/hitters_2025_full.parquet
PITCHER_PARQUET_FILE_PATH=data/pitchers_2025_full.parquet
CONTEXT_PARQUET_FILE_PATH=data/context_2025_full.parquet
FUSED_CONTEXT_DATASET_FILE_PATH=data/unified_context.parquet
PITCHER_ALLOWED_JSON=data/pitcher_allowed.json
```

CLI arguments (`controller/cli_arguments.py`) override these settings per run:
- `--sample`
- `--num_epochs`
- `--batch_size`
- `--lr_pitch_seq`
- `--dropout_pitch_seq`
- `--gen_data` (trigger a fresh Statcast + fusion pass)

---

## Data Workflow
1. **Statcast Retrieval (`ContextDataCollection`)**  
   - Calls `pybaseball.statcast` between `Constants.START_DATE_STR` and `Constants.END_DATE_STR`.  
   - Filters to `Constants.CONTEXT_COLUMNS_LIST` and adds helper columns (runner counts, xBA/xSLG/xwOBA aliases).  
   - Saves two parquet files: the raw Statcast slice and a cleaned `data/context_2025_full.parquet`.

2. **Fusion (`FusionCollection`)**  
   - Loads hitter and pitcher leaderboards (FanGraphs export).  
   - Resolves `IDfg → MLBAM` via `playerid_reverse_lookup`.  
   - Sorts context rows by `game_pk`, `at_bat_number`, `pitch_number`; shifts `pitch_type` to produce `next_pitch_type`.  
   - Encodes both current and next pitch types, merges hitter/pitcher stats with suffixes to avoid collisions, and exports `data/unified_context.parquet`.

3. **Repertoire Mask (`data/pitcher_allowed.json`)**  
   - JSON mapping pitcher IDs to legal pitch indices (aligned with `Constants.PITCH_TYPE_TO_IDX`).  
   - Used to hard-mask logits so the model cannot predict pitches a pitcher never throws.

Trigger the full pipeline (download + fusion + training) via:

```bash
poetry run python controller/main.py --gen_data
```

---

## Dataset Details
`preprocessing/FusionDataset.py` is the canonical dataset used by both the trainer and evaluator.

- **Categorical columns**: `stand`, `p_throws`, `inning_topbot`, `if_fielding_alignment`, `of_fielding_alignment`, `bb_type`, `pitch_name`, `type`, `home_team`.
- **Numeric columns**: release speeds/spins, movement (`pfx_x/z`), plate crossing, velocities (`vx0/vy0/vz0`), accelerations, release positions, strike-zone dimensions, count state, score differentials, runner contexts, and engineered features (`hyper_speed`, `attack_angle`, `attack_direction`).
- **Required columns**: `pitcher`, `mlbam_pitcher`, `next_pitch_type` plus sequencing keys.
- **Sampling**: pass `sample=N` to `FusionDataset(sample=N)` (wired via CLI) to downsample large parquets.
- **History construction**: for every row, capture all prior pitch indices in the same at-bat. Empty histories are discarded, so every training example has at least one previous pitch.
- **Normalization**: `normalize_using_indices(train_idx)` computes means/std using only the training split to avoid leakage; cached stats normalize all numeric tensors.
- **Output format** (`__getitem__`): dictionary with raw numeric/categorical tensors, label (pitch index), pitcher ID, and padded history tensors used for sequence modeling.

Dataset summary (available via `dataset.dataset_summary`) tracks sample count, feature sizes, and maximum sequence length.

---

## Model Architecture
`model/pitch_sequence_encoder/pitch_sequence_encoder.py` implements the core network:

1. **Embeddings**  
   Each categorical feature receives a dedicated embedding layer (`nn.Embedding`), stored in a `ModuleDict` for dynamic lookup.

2. **Sequence Assembly**  
   Numeric tensors and categorical embeddings are concatenated for each timestep in the pitch history. The helper `build_pitch_sequence_tensor` prepares padded batches for both training and evaluation.

3. **Transformer-style Encoder**  
   - Linear projection to `embed_dim` (default 256).  
   - Learnable positional encoding for up to `pitch_seq_max_len` timesteps.  
   - `nn.MultiheadAttention` (batch-first) with LayerNorm residuals.  
   - Feed-forward block (`Linear → GELU → Linear`) plus LayerNorm.

4. **Pooling + Head**  
   - Masked mean pooling over valid timesteps.  
   - Classification head (`Linear → GELU → Dropout → LayerNorm → Linear`) producing logits for every pitch class.

5. **Pitcher-aware Masking**  
   - During forward passes, logits are cloned and non-repertoire classes receive `-1e9` so softmax probability collapses to zero for illegal pitches.

Outputs: masked logits, probabilities, predicted class indices, and pooled states for downstream visualization.

---

## Training Pipeline
`model/pitch_sequence_pipeline.py` wires together the dataset, model, and trainer. Key training features (`model/training/pitch_sequence_trainer.py`):

- **Device selection**: CUDA → Apple MPS → CPU.
- **Splits**: Random 80/10/10, with normalization stats computed only on the training indices.
- **Custom collate**: Pads sequence tensors and generates boolean masks indicating valid timesteps.
- **Loss weighting**: `model/class_weights.py` precomputes per-pitcher class distributions, blends them with a global prior, and supplies per-sample weights to the cross-entropy loss.
- **Optimization**: Adam with weight decay (`1e-4`), gradient clipping (`5.0`), and configurable learning rate/dropout.
- **Early stopping**: Tracks validation macro F1 with configurable patience (default 10). The best checkpoint (highest val F1) is restored before final evaluation.
- **Logging**: Custom logger prints epoch loss summaries plus validation metrics (loss, macro/micro F1, accuracy).

---

## Evaluation
`evaluation/eval.py` provides `PitchSequenceEvaluator`:
- Moves each batch to the model’s device.
- Rebuilds sequence tensors via `build_pitch_sequence_tensor`.
- Computes logits/probs with repertoire masking active.
- Aggregates predictions/labels and reports macro F1, micro F1, and accuracy (also returned as a dict for logging).

The trainer automatically invokes `PitchSequenceEvaluator` on the held-out test loader once training completes.

---

## Usage Examples

### 1. Generate data + train with defaults
```bash
poetry run python controller/main.py --gen_data
```

### 2. Train on larger sample with custom hyperparameters
```bash
poetry run python controller/main.py \
  --sample 75000 \
  --num_epochs 35 \
  --batch_size 64 \
  --lr_pitch_seq 5e-5 \
  --dropout_pitch_seq 0.25
```

### 3. Run evaluation only (after training)
```python
from evaluation.eval import PitchSequenceEvaluator
from preprocessing.fusion_dataset import FusionDataset
from model.pitch_sequence_encoder.pitch_sequence_encoder import PitchSequenceEncoder

dataset = FusionDataset(sample=20000)
model = PitchSequenceEncoder(...)
# Load saved weights here
test_loader = ...
metrics = PitchSequenceEvaluator(model, test_loader).run()
```

---

## Results
| Experiment | Sample Size | Epochs | F1 Macro | F1 Micro | Accuracy | Notes |
|------------|-------------|--------|----------|----------|----------|-------|
| Baseline   | 50,000      | 20     | 0.3172   | 0.0438   | 0.3172   | Initial sanity check |
| Medium     | 100,000     | 30     | 0.3170   | 0.0373   | 0.3170   | Needs tuning |

*(Update this table with actual runs; keep both F1 metrics for regression tracking.)*

---

## Contributing
1. Fork the repo and create a feature branch.
2. Run formatting/linting if applicable.
3. Include unit tests or sample metrics when adding functionality.
4. Submit a PR describing:
   - data schema changes (new parquet columns, updated features),
   - model alterations (encoder architecture, trainer behavior),
   - evaluation/reporting updates.

For major structural tweaks, please open an issue first so we can coordinate data requirements.

---

## Roadmap
- **In Progress**: expand the encoder with multi-headed attention over longer histories and explore cross-attention between hitter/pitcher embeddings.
- **Planned**:
  1. GUI/visualizer for autoregressive pitch prediction by pitcher/hitter matchup.
  2. Integration with live Statcast feeds for near real-time inference.
  3. Automated hyperparameter sweeps + experiment tracking hooks.

---

## Maintainers
- Matthew Hodges (primary).  
- Henry Rothenberg
Reach out via GitHub issues for questions, bug reports, or ideas.

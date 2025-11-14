# Pitch-Sequence-Predictor
Predict the next MLB pitch by blending hitter tendencies, pitcher repertoires, in-game context, and recent pitch sequences through a PyTorch pipeline powered by pybaseball data and custom encoders.

## Table of Contents
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Model Training](#model-training)  
6. [Evaluation](#evaluation)  
7. [Results](#results)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Notes From Developer (“Matt’s Rambles”)](#notes-from-developer-matts-rambles)

## Overview
Pitch-Sequence-Predictor ingests Statcast data to forecast the next pitch type in real time. The system combines four learnable encoders—**HitterEncoder**, **PitcherEncoder**, **ContextEncoder**, and **PitchSequenceEncoder**—whose representations are fused via a `FusionDataset` before training a PyTorch classifier. Feature engineering leans on Statcast context (ball/strike state, run expectancy, field alignments, etc.), while repertoire-aware masking constrains predictions to the pitches each arm actually throws. Evaluation centers on macro/micro F1 so that rare offerings and power pitches both influence model health.

## Project Structure
```text
Pitch-Sequence-Predictor
├── controller/                 # CLI + orchestration for data gen and training
│   ├── cli_arguments.py
│   ├── config.py
│   ├── controller.py
│   └── main.py
├── data/                       # Local parquet assets + pitcher repertoire JSON
├── data_collection/            # Statcast + fusion ETL scripts
├── evaluation/                 # Post-training evaluation utilities
├── model/
│   ├── custom_types/           # Dataclasses describing encoder/trainer configs
│   ├── pitch_sequence_encoder/ # Fusion encoder → logits + masking
│   ├── pitch_sequence_pipeline.py
│   └── training/               # PitchSequenceTrainer loop
├── preprocessing/              # FusionDataset that merges hitter/pitcher/context
├── utils/                      # Logging + constants shared across modules
├── pyproject.toml              # Poetry project definition
└── README.md
```

## Installation

### 1. Poetry Environment
```bash
# Install dependencies
poetry install

# (Optional) upgrade lock file after editing pyproject.toml
poetry update
```

### 2. Environment Variables
Create `.env` at the repository root (values below match defaults in `controller/config.py`):
```text
SAMPLE=1000
BATCH_SIZE=25
NUM_EPOCHS=20
LR_PITCH_SEQ=1e-5
DROPOUT_PITCH_SEQ=0.3
HITTER_PARQUET_FILE_PATH=data/hitters_2025_full.parquet
PITCHER_PARQUET_FILE_PATH=data/pitchers_2025_full.parquet
CONTEXT_PARQUET_FILE_PATH=data/context_2025_full.parquet
PITCH_SEQUENCE_PARQUET_FILE_PATH=data/pitch_sequence_2025.parquet
FUSED_CONTEXT_DATASET_FILE_PATH=data/unified_context.parquet
PITCHER_ALLOWED_JSON=data/pitcher_allowed.json
```

### 3. Data Download & Preprocessing
1. Confirm your Statcast API access (`pybaseball` handles authentication).
2. Generate fresh context + fused datasets:
   ```bash
   poetry run python controller/main.py --gen_data
   ```
   - `ContextDataCollection` fetches Statcast data (`START_DATE_STR` → `END_DATE_STR`), selects relevant columns, and exports `context_2025_full.parquet`.
   - `FusionCollection` enriches context rows with hitter/pitcher summaries, resolves FanGraphs↔MLBAM IDs, adds `next_pitch_type`, and saves `unified_context.parquet`.
3. Manually drop new hitter/pitcher parquet snapshots into `data/` if you are iterating on projections.

## Usage
Kick off a training run (data prep optional if already done):
```bash
poetry run python controller/main.py \
  --sample 50000 \
  --num_epochs 30 \
  --batch_size 64 \
  --lr_pitch_seq 5e-5 \
  --dropout_pitch_seq 0.2
```
Flags override `.env` defaults, so you can sweep hyperparameters directly from the CLI or through shell scripts.

## Model Training
1. **Dataset assembly** – `FusionDataset` reads `unified_context.parquet`, filters categorical/numeric feature sets, normalizes counts, and aligns each row with a `next_pitch_type` index plus pitcher repertoire constraints.
2. **Encoder stack** – HitterEncoder, PitcherEncoder, ContextEncoder, and PitchSequenceEncoder transform their respective feature domains into embeddings which are concatenated and projected through GELU → LayerNorm blocks to form a shared latent vector.
3. **Trainer** – `PitchSequenceTrainer` builds stratified `DataLoader`s (train/val/test), optimizes the encoder with Adam + weight decay, and clips gradients for stability.
4. **In-flight masking** – During evaluation/inference, logits are hard-masked per pitcher (`pitcher_allowed.json`) so the model never predicts a pitch the current arm does not throw.
5. **Logging** – A lightweight `Logger` surfaces parameter settings and epoch losses so you can wire the run into experiment tracking later.

## Evaluation
- `evaluation/PitchSequenceEvaluator` executes on the held-out test loader automatically after each training run.
- Metrics:
  - **F1 Macro** – treats every pitch class equally; ideal for monitoring rarely thrown offerings.
  - **F1 Micro** – frequency-weighted view aligned with overall accuracy.
- Example standalone invocation (useful if you checkpoint encoders):
  ```bash
  poetry run python -c "from evaluation.eval import PitchSequenceEvaluator; ..."
  ```
  (Import your saved encoder, wrap the cached dataloader, and call `PitchSequenceEvaluator(model, loader).run()`.)

## Results
| Experiment | Sample Size | Epochs | F1 Macro | F1 Micro | Accuracy |
|------------|-------------|--------|----------|----------|----------|
| Base Run | 50k | 20 | .3172 | .0438 | .3172 | 
| Medium Run | 100k | 30 | .3170 | .0373 | .3170 | 


Populate the table with actual metrics after running the pipeline; keep both F1 variants for regression tracking.

## Contributing
Henry Rothenberg and Matthew Hodges currently maintain the project. Please open a discussion before submitting large PRs (data schema changes, new encoders, etc.) so we can coordinate preprocessing requirements.


## In Progress Enhancements

A potential enhancement to this project is adding multi-headed attention in the encoder in order to take into account the previous pitches in the sequence.

## Future Planned Enhancements

Create a GUI to auto-regressively predict the next pitch based off the pitcher and hitter.

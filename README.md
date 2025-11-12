# Pitch-Sequence-Predictor

[FanGraphs](https://community.fangraphs.com/no-pitch-is-an-island-pitch-prediction-with-sequence-to-sequence-deep-learning/)

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
    - [Dependency Management - Poetry](#dependency-management---poetry)
    - [Environment Variables](#environment-variables)
    - [Data Generation](#data-generation)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

This project analyzes baseball pitch sequences to predict the next pitch using deep learning models.  
It leverages the **pybaseball** library for data retrieval and **PyTorch** for model training.

---

## Project Structure

**ADD-CONTENT**

---

## Installation

**ADD-CONTENT**

### Dependency Management - Poetry

```bash
poetry init
poetry add <LIBRARY-NAME>
```

---

### Environment Variables

The following is an example `.env` file

```text
SAMPLE=1000
BATCH_SIZE=25
NUM_EPOCHS=20
LR_PITCH_SEQ=1E-5
DROPOUT_PITCH_SEQ=.3
HITTER_PARQUET_FILE_PATH="data/hitters_2025_full.parquet"
CONTEXT_PARQUET_FILE_PATH="data/context_2025_full.parquet"
PITCHER_PARQUET_FILE_PATH="data/pitchers_2025_full.parquet"
PITCH_SEQUENCE_PARQUET_FILE_PATH"=data/pitch_sequence_2025.parquet"
```

### Data Generation

**ADD-CONTENT**

---

## Usage

**ADD-CONTENT**

---

## Model Training

**ADD-CONTENT**

---

## Evaluation

**ADD-CONTENT**

---

## Results

**ADD-CONTENT**

---

## Contributing

**ADD-CONTENT**

---

## License

**ADD-CONTENT**

---
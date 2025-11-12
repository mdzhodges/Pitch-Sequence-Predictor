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


## Matt's Rambles

So, I believe the best thing for us to do is no longer use pitch_sequence_2025.parquet. THis is because every thing is in that is in the context parquet file too, so no need to overlap. Currently we have the hitter and pitcher parquet files too. What I think we should do is at preprocessing, we taking those ID"s (fangraph IDs) from the hitter and pitcher parquet, then move that information (we are gonna have to move the mapping to a MLBAM ID's (there is a statcast function for this))to the context file THEN we embed everything, categorical or not. That will allow us to use the "matchup" between hitter and pitcher. also we need to find a way to create a "Next_Pitch" field in the context file so we can have a target to classify. So essentially we need to completely change preprocessing into one huge (hand gesture) file. 
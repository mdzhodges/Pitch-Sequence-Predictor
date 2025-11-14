import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

from controller.config import Config
from utils.constants import Constants


class FusionDataset(Dataset):
    """
    Clean, minimal, production-ready dataset.
    """

    # ============================================================
    # OPTION A — Minimal, high-signal features
    # ============================================================

    CATEGORICAL_KEEP = [
        "stand",
        "p_throws",
        "inning_topbot",
        "if_fielding_alignment",
        "of_fielding_alignment",
        "bb_type",
        "pitch_name",
        "type",
        "home_team",
    ]

    NUMERIC_KEEP = [
        "release_speed",
        "release_spin_rate",
        "pfx_x", "pfx_z",
        "plate_x", "plate_z",
        "vx0", "vy0", "vz0",
        "ax", "ay", "az",
        "release_pos_x", "release_pos_y", "release_pos_z",
        "release_extension",
        "sz_top", "sz_bot",
        "balls", "strikes",
        "outs_when_up",
        "home_score", "away_score", "bat_score", "fld_score",
        "home_score_diff",
        "runners_on_base",
        "hyper_speed",
        "attack_angle",
        "attack_direction",
    ]

    REQUIRED = [
        "pitcher",
        "mlbam_pitcher",
        "next_pitch_type",
    ]

    def __init__(self, sample: int | None = None):
        self.config = Config()

        # ------------------------------------------------------------
        # Load fused parquet
        # ------------------------------------------------------------
        df: pd.DataFrame = pd.read_parquet(
            self.config.FUSED_CONTEXT_DATASET_FILE_PATH
        ).replace({None: np.nan})

        # Optional sampling
        if sample is not None and sample < len(df):
            df = df.sample(n=sample, random_state=1337).reset_index(drop=True)

        # ------------------------------------------------------------
        # Ensure required columns exist
        # ------------------------------------------------------------
        needed = set(self.CATEGORICAL_KEEP + self.NUMERIC_KEEP + self.REQUIRED)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # ------------------------------------------------------------
        # Normalize next_pitch_type → global mapping
        # ------------------------------------------------------------
        def norm_pitch(x):
            if x == "FF":
                return "FA"
            if x in ("ST", "SV"):
                return "SL"
            return x

        df["next_pitch_type"] = (
            df["next_pitch_type"]
            .astype("string").fillna("UN")
            .map(norm_pitch)
        )

        pitch_map = Constants.PITCH_TYPE_TO_IDX
        df["pitch_idx"] = df["next_pitch_type"].map(pitch_map)

        if df["pitch_idx"].isna().any():
            bad = df.loc[df["pitch_idx"].isna(), "next_pitch_type"].unique()
            raise ValueError(f"Unknown pitch types: {bad}")

        # Final labels
        self.y_labels = torch.tensor(
            df["pitch_idx"].astype(int).values,
            dtype=torch.long
        )

        # ------------------------------------------------------------
        # Numeric preprocessing
        # ------------------------------------------------------------
        num_df = df[self.NUMERIC_KEEP].apply(
            pd.to_numeric, errors="coerce").astype(np.float32)
        num_df = num_df.fillna(num_df.mean()).fillna(0)

        self.mean = num_df.mean()
        self.std = num_df.std().replace(0, 1)

        normalized = ((num_df - self.mean) / self.std).astype(np.float32)
        self.x_numeric = torch.tensor(normalized.values, dtype=torch.float32)

        # ------------------------------------------------------------
        # Categorical encoding
        # ------------------------------------------------------------
        self.vocab_maps = {}
        cat_map = {}

        for col in self.CATEGORICAL_KEEP:
            series = df[col].astype("string").fillna("UNK")
            vocab = sorted(series.unique().tolist())
            mapping = {v: i for i, v in enumerate(vocab)}

            encoded = series.map(mapping).astype(int)
            cat_map[col] = torch.tensor(encoded.values, dtype=torch.long)
            self.vocab_maps[col] = mapping

        self.x_categorical = cat_map

        # ------------------------------------------------------------
        # Store pitcher IDs
        # ------------------------------------------------------------
        self.pitcher_ids = df["pitcher"].astype(int).tolist()

        # ------------------------------------------------------------
        # Load allowed repertoire
        # ------------------------------------------------------------
        with open(self.config.PITCHER_ALLOWED_JSON, "r") as f:
            self.pitcher_to_allowed = {
                int(k): v for k, v in json.load(f).items()}

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        self.dataset_summary = {
            "samples": len(self),
            "numeric_dim": self.x_numeric.shape[1],
            "categorical_cols": len(self.x_categorical),
            "num_classes": len(pitch_map),
        }

    # ============================================================
    # PyTorch API
    # ============================================================

    def __len__(self):
        return len(self.y_labels)

    def __getitem__(self, idx: int):
        return {
            "numeric": self.x_numeric[idx],
            "categorical": {c: t[idx] for c, t in self.x_categorical.items()},
            "label": self.y_labels[idx],
            "pitcher_id": self.pitcher_ids[idx],
        }

    def get_vocab_sizes(self):
        return {col: len(v) for col, v in self.vocab_maps.items()}

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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

        seq_order_cols = ["game_pk", "at_bat_number", "pitch_number"]
        missing_seq_cols = [c for c in seq_order_cols if c not in df.columns]
        if missing_seq_cols:
            raise ValueError(
                f"Missing required sequencing columns in fused dataset: {missing_seq_cols}"
            )

        # Optional sampling
        if sample is not None and sample < len(df):
            df = df.sample(n=sample, random_state=1337).reset_index(drop=True)

        # Maintain chronological ordering so previous-pitch lookups are valid.
        df = df.sort_values(seq_order_cols).reset_index(drop=True)

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

        raw_numeric = torch.tensor(num_df.values, dtype=torch.float32)
        self._numeric_raw = raw_numeric
        self.norm_mean = None
        self.norm_std = None
        # Placeholder tensors populated once normalization stats are set
        self.x_numeric = raw_numeric.clone()

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
        # Build history indices for each pitch
        # ------------------------------------------------------------
        history_indices = self._build_history_indices(df[seq_order_cols])
        sample_indices = [idx for idx, hist in enumerate(history_indices) if hist]

        if not sample_indices:
            raise ValueError(
                "No pitches include prior history; cannot construct pitch sequences."
            )

        self.sample_indices = sample_indices
        self.history_indices = history_indices
        self.pitch_seq_max_len = max(
            len(history_indices[idx]) for idx in self.sample_indices
        )

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        self.dataset_summary = {
            "samples": len(self.sample_indices),
            "numeric_dim": self.x_numeric.shape[1],
            "categorical_cols": len(self.x_categorical),
            "num_classes": len(pitch_map),
            "pitch_seq_max_len": self.pitch_seq_max_len,
        }

        # Default normalization uses all available samples (maintains prior behavior)
        self.normalize_using_indices(list(range(len(self.sample_indices))))

    # ============================================================
    # PyTorch API
    # ============================================================

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        row_idx = self.sample_indices[idx]

        numeric = self.x_numeric[row_idx]
        categorical = {c: t[row_idx] for c, t in self.x_categorical.items()}
        label = self.y_labels[row_idx]
        pitcher_id = self.pitcher_ids[row_idx]

        history_rows = self.history_indices[row_idx]
        history_tensor = torch.tensor(history_rows, dtype=torch.long)

        pitch_seq_numeric = self.x_numeric[history_tensor]
        pitch_seq_categorical = {
            c: t[history_tensor] for c, t in self.x_categorical.items()
        }

        return {
            "numeric": numeric,
            "categorical": categorical,
            "label": label,
            "pitcher_id": pitcher_id,
            "pitch_seq_numeric": pitch_seq_numeric,
            "pitch_seq_categorical": pitch_seq_categorical,
        }

    def get_vocab_sizes(self):
        return {col: len(v) for col, v in self.vocab_maps.items()}

    def normalize_using_indices(self, dataset_indices: list[int]):
        """
        Normalize numeric tensors using only the provided dataset indices
        (indices are over the filtered dataset, not the raw dataframe).
        """
        if not dataset_indices:
            raise ValueError("Cannot normalize without dataset indices")

        row_indices = torch.tensor(
            [self.sample_indices[i] for i in dataset_indices],
            dtype=torch.long
        )

        base = self._numeric_raw[row_indices]
        mean = base.mean(dim=0)
        std = base.std(dim=0)
        std[std == 0] = 1

        self.norm_mean = mean
        self.norm_std = std
        self.x_numeric = (self._numeric_raw - mean) / std

    def _build_history_indices(self, seq_df: pd.DataFrame) -> list[list[int]]:
        histories: list[list[int]] = [[] for _ in range(len(seq_df))]
        at_bat_tracker: dict[tuple[int, int], list[int]] = defaultdict(list)

        for idx, row in enumerate(seq_df.itertuples(index=False)):
            key = (int(getattr(row, "game_pk")), int(getattr(row, "at_bat_number")))
            histories[idx] = at_bat_tracker[key].copy()
            at_bat_tracker[key].append(idx)

        return histories

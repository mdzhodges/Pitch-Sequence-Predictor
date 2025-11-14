import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from controller.config import Config
from utils.constants import Constants


class FusionDataset(Dataset):
    """
    Loads the unified context parquet and prepares it for model training.
    Produces:
        numeric: Tensor[F]
        categorical: {col: Tensor(int)}
        label: Tensor(int)
        pitcher_id: int
    """

    def __init__(self, sample: int | None = None):
        self.config = Config()

        # ------------------------------------------------------------
        # 1. Load parquet
        # ------------------------------------------------------------
        df: pd.DataFrame = pd.read_parquet(
            self.config.FUSED_CONTEXT_DATASET_FILE_PATH
        )

        # Optional sampling
        if sample is not None and sample < len(df):
            df = df.sample(n=sample, random_state=1337).reset_index(drop=True)

        # Convert None → NaN
        df = df.replace({None: np.nan})

        # Preserve pitcher IDs
        self.raw_pitcher_ids = df["pitcher"].astype(
            "int64").astype(int).tolist()

        # ------------------------------------------------------------
        # 2. Normalize / correct raw pitch labels BEFORE mapping
        # ------------------------------------------------------------
        def norm_pitch(x: str) -> str:
            if x == "FF":  # FF → FA fastball
                return "FA"
            if x == "ST":  # sweeper → slider
                return "SL"
            if x == "SV":  # slurve → slider
                return "SL"
            return x

        df["next_pitch_type"] = (
            df["next_pitch_type"]
            .astype("string")
            .fillna("UN")            # unknown
            .map(norm_pitch)
        )

        # ------------------------------------------------------------
        # 3. Create *correct labels* using global mapping
        # ------------------------------------------------------------
        pitch_map = Constants.PITCH_TYPE_TO_IDX
        df["correct_pitch_idx"] = df["next_pitch_type"].map(pitch_map)

        if df["correct_pitch_idx"].isna().any():
            missing_labels = df.loc[df["correct_pitch_idx"].isna(
            ), "next_pitch_type"].unique()
            raise ValueError(
                f"ERROR: Unknown pitch types encountered: {missing_labels}")

        self.y_labels = torch.tensor(
            df["correct_pitch_idx"].astype(np.int64).values,
            dtype=torch.long
        )

        # ------------------------------------------------------------
        # 4. Split numeric vs categorical
        # ------------------------------------------------------------
        exclude_cols = {
            "correct_pitch_idx",
            "next_pitch_type",
            "next_pitch_idx",    # ignore this completely
            "pitch_type",        # raw pitch type
            "events",            # non-numeric string
        }

        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            if col in exclude_cols:
                continue
            try:
                pd.to_numeric(df[col].dropna(), errors="raise")
                numeric_cols.append(col)
            except Exception:
                categorical_cols.append(col)

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # ------------------------------------------------------------
        # 5. Normalize numeric features
        # ------------------------------------------------------------
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.astype(np.float32)
        numeric_df = numeric_df.fillna(numeric_df.mean()).fillna(0.0)

        self.mean = numeric_df.mean()
        self.std = numeric_df.std().replace(0, 1)

        normalized = ((numeric_df - self.mean) / self.std).astype(np.float32)
        self.x_numeric = torch.tensor(normalized.values, dtype=torch.float32)

        # ------------------------------------------------------------
        # 6. Encode categorical columns (string → int)
        # ------------------------------------------------------------
        self.vocab_maps: dict[str, dict[str, int]] = {}
        cat_tensor_map: dict[str, Tensor] = {}

        for col in categorical_cols:
            series = df[col].astype("string").fillna("UNK")
            vocab = sorted(series.unique().tolist())
            mapping = {v: i for i, v in enumerate(vocab)}

            encoded = series.map(mapping).astype(np.int64)
            cat_tensor_map[col] = torch.tensor(
                encoded.values, dtype=torch.long)
            self.vocab_maps[col] = mapping

        self.x_categorical = cat_tensor_map

        # ------------------------------------------------------------
        # 7. Precompute allowed pitch indices for each pitcher
        # ------------------------------------------------------------
        grouped = df.groupby("pitcher")["pitch_type"].unique()
        self.pitcher_to_allowed: dict[int, list[int]] = {}

        for pitcher_id, raw_list in grouped.items():
            allowed = set()
            for raw in raw_list:
                norm = norm_pitch(str(raw))
                if norm in pitch_map:
                    allowed.add(pitch_map[norm])
            self.pitcher_to_allowed[int(pitcher_id)] = sorted(allowed)

        # ------------------------------------------------------------
        # 8. Summary
        # ------------------------------------------------------------
        self.dataset_summary = {
            "total_samples": len(self),
            "numeric_dim": self.x_numeric.shape[1],
            "num_categories": len(self.x_categorical),
            "num_classes": len(pitch_map),   # always 18
        }
        

    # ------------------------------------------------------------
    # PyTorch Dataset Interface
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.y_labels)

    def __getitem__(self, idx: int):
        return {
            "numeric": self.x_numeric[idx],
            "categorical": {col: tensor[idx] for col, tensor in self.x_categorical.items()},
            "label": self.y_labels[idx],
            "pitcher_id": self.raw_pitcher_ids[idx],
        }

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def get_vocab_sizes(self) -> dict[str, int]:
        return {col: len(vocab) for col, vocab in self.vocab_maps.items()}

    def get_example(self, idx: int = 0):
        cat_example = {
            col: next(k for k, v in vocab.items() if v ==
                      int(self.x_categorical[col][idx]))
            for col, vocab in self.vocab_maps.items()
        }
        return {
            "numeric": self.x_numeric[idx],
            "categorical": cat_example,
            "label": int(self.y_labels[idx]),
        }

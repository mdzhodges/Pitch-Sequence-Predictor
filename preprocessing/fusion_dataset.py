import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from controller.config import Config


class FusionDataset(Dataset):
    """
    Loads the unified context parquet (context + pitcher + hitter + next pitch target)
    and prepares it for model training. Handles numeric normalization, categorical
    encoding, and target tensorization.

    Each sample includes:
        {
            "numeric": Tensor([...]),
            "categorical": {col: Tensor(int), ...},
            "label": Tensor(int)   # next_pitch_idx
        }
    """

    def __init__(self, sample: int | None = None, debug: bool = False):
        self.config = Config()
        self.debug = debug
        self.debug_info = None

        # ------------------------------------------------------------
        # 1. Load unified parquet
        # ------------------------------------------------------------
        parquet_path = self.config.FUSED_CONTEXT_DATASET_FILE_PATH
        dataframe: pd.DataFrame = pd.read_parquet(parquet_path)

        if sample and sample < len(dataframe):
            dataframe = dataframe.sample(
                n=sample, random_state=42).reset_index(drop=True)

        # ------------------------------------------------------------
        # 2. Identify numeric + categorical columns robustly
        # ------------------------------------------------------------
        exclude = {"next_pitch_idx", "pitcher_fg", "batter_fg"}

        # Test convertibility for numeric columns instead of trusting dtype
        numeric_cols: list[str] = []
        for c in dataframe.columns:
            if c in exclude:
                continue
            try:
                pd.to_numeric(dataframe[c].dropna().sample(
                    n=min(500, len(dataframe))), errors="raise")
                numeric_cols.append(c)
            except Exception:
                # Skip if conversion fails (mixed or string column)
                continue

        categorical_cols = [
            c for c in dataframe.columns
            if c not in exclude and c not in numeric_cols
        ]

        if not numeric_cols and not categorical_cols:
            raise ValueError(
                "No usable columns found in unified context parquet.")

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # ------------------------------------------------------------
        # 3. Normalize numeric features safely
        # ------------------------------------------------------------
        numeric_df = dataframe[numeric_cols].apply(
            pd.to_numeric, errors="coerce")
        self.mean = numeric_df.mean()
        self.std = numeric_df.std().replace(0, 1)
        normalized = (numeric_df - self.mean) / self.std
        normalized = normalized.fillna(0.0)
        normalized = normalized.astype(np.float32)

        self.x_numeric = torch.tensor(normalized.values, dtype=torch.float32)
        self.x_numeric[torch.isnan(self.x_numeric)] = 0.0

        # ------------------------------------------------------------
        # 4. Encode categorical columns (string -> int)
        # ------------------------------------------------------------
        self.vocab_maps: dict[str, dict[str, int]] = {}
        cat_tensors: dict[str, Tensor] = {}

        for col in categorical_cols:
            series = dataframe[col].astype(str).fillna("UNK")
            vocab = sorted(series.unique().tolist())
            mapping = {v: i for i, v in enumerate(vocab)}
            encoded = series.map(mapping).astype(np.int64)
            cat_tensors[col] = torch.tensor(encoded.values, dtype=torch.long)
            self.vocab_maps[col] = mapping

        self.x_categorical = cat_tensors

        # ------------------------------------------------------------
        # 5. Target tensor (next pitch classification)
        # ------------------------------------------------------------
        if "next_pitch_idx" not in dataframe.columns:
            raise ValueError(
                "'next_pitch_idx' missing in unified context parquet.")

        labels = dataframe["next_pitch_idx"].fillna(-1).astype(np.int64)
        self.y_labels = torch.tensor(labels.values, dtype=torch.long)

        # ------------------------------------------------------------
        # 6. Dataset summary
        # ------------------------------------------------------------
        self.dataset_summary = {
            "total_samples": len(self),
            "numeric_dim": self.x_numeric.shape[1],
            "num_categories": len(self.x_categorical),
            "num_classes": int(self.y_labels.max().item() + 1),
        }

        if self.debug:
            self.debug_info = self.debug_summary()

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.x_numeric)

    def __getitem__(self, idx: int):
        numeric = self.x_numeric[idx]
        categorical = {col: tensor[idx]
                       for col, tensor in self.x_categorical.items()}
        label = self.y_labels[idx]
        return {"numeric": numeric, "categorical": categorical, "label": label}

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------
    def get_vocab_sizes(self) -> dict[str, int]:
        """Return {col: vocab_size} for each categorical column."""
        return {col: len(vocab) for col, vocab in self.vocab_maps.items()}

    def get_example(self, idx: int = 0):
        """Inspect decoded values for a single row (debug)."""
        cat_example = {
            col: list(vocab.keys())[list(vocab.values()).index(
                int(self.x_categorical[col][idx]))]
            for col, vocab in self.vocab_maps.items()
        }
        return {
            "numeric": self.x_numeric[idx],
            "categorical": cat_example,
            "label": int(self.y_labels[idx]),
        }

    def debug_summary(self) -> dict[str, object]:
        """Collect detailed tensor and column info for debugging."""
        vocab_sizes = self.get_vocab_sizes()
        categorical_preview = list(self.x_categorical.keys())[:10]
        numeric_preview = []
        if len(self) > 0:
            numeric_preview = self.x_numeric[0].tolist()[:10]

        summary = {
            "dataset_summary": self.dataset_summary,
            "categorical_preview": categorical_preview,
            "vocab_sizes": vocab_sizes,
            "sample_numeric_values": numeric_preview,
        }
        return summary

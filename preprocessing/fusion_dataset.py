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
            "label": Tensor(int)
        }
    """

    def __init__(self, sample: int | None = None):
        self.config = Config()

        # ------------------------------------------------------------
        # 1. Load unified parquet
        # ------------------------------------------------------------
        df: pd.DataFrame = pd.read_parquet(
            self.config.FUSED_CONTEXT_DATASET_FILE_PATH
        )

        if sample and sample < len(df):
            df = df.sample(n=sample, random_state=1337).reset_index(drop=True)

        df = df.replace({None: np.nan})

        # ------------------------------------------------------------
        # 2. Correct numeric detection (by convertibility)
        # ------------------------------------------------------------
        exclude = {"next_pitch_idx"}
        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            if col in exclude:
                continue
            try:
                pd.to_numeric(df[col].dropna(), errors="raise")
                numeric_cols.append(col)
            except Exception:
                categorical_cols.append(col)

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # ------------------------------------------------------------
        # 3. PERFECT numeric normalization pipeline
        # ------------------------------------------------------------

        # Convert everything numeric to float FIRST (fixes Int64 issues)
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.astype(np.float32)

        # Fill NaNs safely AFTER conversion
        numeric_df = numeric_df.fillna(numeric_df.mean()).fillna(0.0)

        # Compute mean/std AFTER filling
        self.mean = numeric_df.mean()
        self.std = numeric_df.std().replace(0, 1)

        normalized = (numeric_df - self.mean) / self.std
        normalized = normalized.astype(np.float32)

        self.x_numeric = torch.tensor(normalized.values, dtype=torch.float32)

        # ------------------------------------------------------------
        # 4. Encode categorical columns
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
        # 5. Label tensor
        # ------------------------------------------------------------
        if "next_pitch_idx" not in df.columns:
            raise ValueError("'next_pitch_idx' missing from fused dataset.")

        labels = df["next_pitch_idx"].fillna(-1).astype(np.int64)
        self.y_labels = torch.tensor(labels.values, dtype=torch.long)

        # ------------------------------------------------------------
        # 6. Summary
        # ------------------------------------------------------------
        self.dataset_summary = {
            "total_samples": len(self),
            "numeric_dim": self.x_numeric.shape[1],
            "num_categories": len(self.x_categorical),
            "num_classes": int(self.y_labels.max().item() + 1),
        }

    def __len__(self):
        return len(self.y_labels)

    def __getitem__(self, idx: int):
        numeric = self.x_numeric[idx]
        categorical = {col: tensor[idx]
                       for col, tensor in self.x_categorical.items()}
        label = self.y_labels[idx]
        return {"numeric": numeric, "categorical": categorical, "label": label}

    def get_vocab_sizes(self) -> dict[str, int]:
        return {col: len(vocab) for col, vocab in self.vocab_maps.items()}

    def get_example(self, idx: int = 0):
        cat_example = {
            col: next(
                k for k, v in vocab.items() if v == int(self.x_categorical[col][idx])
            )
            for col, vocab in self.vocab_maps.items()
        }
        return {
            "numeric": self.x_numeric[idx],
            "categorical": cat_example,
            "label": int(self.y_labels[idx]),
        }

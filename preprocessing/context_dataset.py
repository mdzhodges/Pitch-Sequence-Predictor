import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.logger import Logger


class ContextDataset(Dataset):
    """
    Loads Statcast context data from a parquet file and prepares both
    numeric and categorical columns as tensors suitable for modeling.
    Includes 'events' so it can be embedded as a categorical feature.
    """

    def __init__(self, parquet_file_path: str):
        # ------------------------------------------------------------
        # Load dataframe
        # ------------------------------------------------------------
        df = pd.read_parquet(parquet_file_path)

        # Metadata / non-feature columns that should NOT be used as input
        exclude = {
            "game_date",
            "description",
            "des",
            "umpire",
            "sv_id",
            "pitcher_name",
            "batter_name",
            "home_team",
            "away_team",
        }

        # ------------------------------------------------------------
        # Identify numeric and categorical columns
        # ------------------------------------------------------------
        numeric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_cols = [
            c for c in df.columns
            if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])
        ]

        if not numeric_cols and not categorical_cols:
            raise ValueError("No usable columns found in context parquet.")

        self.feature_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # ------------------------------------------------------------
        # Process numeric columns
        # ------------------------------------------------------------
# Force all numeric columns to float for safe fillna operations
        numeric_df = (
            df[self.feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float64")
        )
        # Drop all-NaN numeric columns
        if all_nan_cols := numeric_df.columns[numeric_df.isna().all()].tolist():
            numeric_df = numeric_df.drop(columns=all_nan_cols)

        # Fill NaNs with column mean, fallback to 0
        numeric_df = numeric_df.fillna(numeric_df.mean()).fillna(0)

        # Normalize numeric columns
        self.means = numeric_df.mean()
        self.stds = numeric_df.std().replace(0, 1)
        normalized = (numeric_df - self.means) / self.stds

        # Convert to float32 tensor
        X_numeric = torch.tensor(normalized.values, dtype=torch.float32)
        X_numeric[torch.isnan(X_numeric)] = 0.0
        self.X_numeric = X_numeric

        # ------------------------------------------------------------
        # Encode categorical columns (including 'events')
        # ------------------------------------------------------------
        self.cat_maps = {}
        cat_tensors = {}

        for col in self.categorical_cols:
            series = df[col].astype(str).fillna("UNK")
            vocab = sorted(series.unique().tolist())
            mapping = {v: i for i, v in enumerate(vocab)}
            encoded = series.map(mapping).astype(np.int64)
            cat_tensors[col] = torch.tensor(encoded.values, dtype=torch.long)
            self.cat_maps[col] = mapping

        self.X_categorical = cat_tensors

        # ------------------------------------------------------------
        # Logger
        # ------------------------------------------------------------
        self.logger = Logger(self.__class__.__name__)
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.X_numeric)

    def __getitem__(self, idx: int):
        """Return one sample as a dict of numeric + categorical tensors."""
        numeric = self.X_numeric[idx]
        categorical = {col: tensor[idx] for col, tensor in self.X_categorical.items()}
        return {"numeric": numeric, "categorical": categorical}

    # ------------------------------------------------------------
    def get_vocab_sizes(self):
        """Return dict mapping each categorical column → vocabulary size."""
        return {col: len(vocab) for col, vocab in self.cat_maps.items()}

    def get_example(self, idx: int = 0):
        """Inspect decoded categorical values for one example."""
        cat_example = {
            col: list(mapping.keys())[list(mapping.values()).index(int(self.X_categorical[col][idx]))]
            for col, mapping in self.cat_maps.items()
        }
        return {"numeric": self.X_numeric[idx], "categorical": cat_example}

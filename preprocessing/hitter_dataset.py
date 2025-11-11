import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HitterDataset(Dataset):
    def __init__(self, parquet_file_path: str):
        df = pd.read_parquet(parquet_file_path)

        exclude = {"IDfg", "Season", "Name", "Team", "Age", "G"}
        numeric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not numeric_cols:
            raise ValueError("No numeric columns found.")

        self.feature_cols = numeric_cols
        self.names = df["Name"].values if "Name" in df.columns else None

        # --- CLEANING ---
        features = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")

        if all_nan_cols := features.columns[features.isna().all()].tolist():
            features = features.drop(columns=all_nan_cols)

        # Fill remaining NaNs with column mean (use 0 if still all NaN after drop)
        features = features.fillna(features.mean()).fillna(0)

        # --- NORMALIZATION ---
        self.means = features.mean()
        self.stds = features.std().replace(0, 1)
        features = (features - self.means) / self.stds

        # --- CONVERT TO TENSOR ---
        X = torch.tensor(features.values, dtype=torch.float32)

        # Replace any residual NaNs (from weird numeric parsing) with 0
        X[torch.isnan(X)] = 0.0

        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_by_name(self, name: str):
        if self.names is None:
            raise ValueError("No Name column available.")
        mask = [name.lower() in n.lower() for n in self.names]
        if idx := [i for i, m in enumerate(mask) if m]:
            return self.X[idx[0]]
        else:
            raise ValueError(f"No player found matching '{name}'")
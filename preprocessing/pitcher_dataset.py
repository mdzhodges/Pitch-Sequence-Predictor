import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.logger import Logger


class PitcherDataset(Dataset):
    def __init__(self, parquet_file_path: str):
        df = pd.read_parquet(parquet_file_path)

        # Columns to exclude from numeric feature extraction
        exclude = {"IDfg", "Season", "Team", "Age"}
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

        # Fill remaining NaNs with column mean (fallback to 0 if all NaN)
        features = features.fillna(features.mean()).fillna(0)

        # --- NORMALIZATION ---
        self.means = features.mean()
        self.stds = features.std().replace(0, 1)
        features = (features - self.means) / self.stds

        # --- CONVERT TO TENSOR ---
        X = torch.tensor(features.values, dtype=torch.float32)

        # Replace any remaining NaNs with 0
        X[torch.isnan(X)] = 0.0

        self.X = X

        # Logger
        self.logger = Logger(self.__class__.__name__)
        self.logger.info("Pitcher Dataset Populated to Tensors")
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def get_by_name(self, name: str):
        """Retrieve a pitcher's feature vector by partial name match."""
        if self.names is None:
            raise ValueError("No Name column available.")
        mask = [name.lower() in n.lower() for n in self.names]
        if idx := [i for i, m in enumerate(mask) if m]:
            return self.X[idx[0]]
        else:
            raise ValueError(f"No pitcher found matching '{name}'")

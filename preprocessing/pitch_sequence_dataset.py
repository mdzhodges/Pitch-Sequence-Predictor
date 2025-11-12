import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.logger import Logger


class PitchSequenceDataset(Dataset):
    """
    Loads a pitch sequence dataset from a parquet file and converts both
    numeric and categorical columns into PyTorch tensors suitable for modeling.
    Includes 'pitch_type', 'next_pitch_type', and 'events' as categorical
    embedding features.

    Args:
        parquet_file_path (str): Path to the parquet dataset.
        sample (int): Number of rows to load (for memory or debugging).
    """

    def __init__(self, parquet_file_path: str, sample: int = 1000):
        self.logger = Logger(self.__class__.__name__)

        # ------------------------------------------------------------------
        # Load limited subset of dataset
        # ------------------------------------------------------------------
        df = pd.read_parquet(parquet_file_path)
        df = df.iloc[:sample]

        # ------------------------------------------------------------------
        # Identify numeric and categorical columns
        # ------------------------------------------------------------------
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        if not numeric_cols and not categorical_cols:
            raise ValueError("No usable columns found in pitch sequence data.")

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        # ------------------------------------------------------------------
        # Process numeric columns
        # ------------------------------------------------------------------
        numeric_df = df[self.numeric_cols].copy()

        # Force numeric coercion
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

        # Fill NaNs with mean, fallback to 0
        numeric_df = numeric_df.fillna(numeric_df.mean()).fillna(0)
        numeric_df = numeric_df.astype("float64")

        # Normalize numeric columns
        self.means = numeric_df.mean()
        self.stds = numeric_df.std().replace(0, 1)
        normalized = (numeric_df - self.means) / self.stds

        # Convert to tensor
        X_numeric = torch.tensor(normalized.to_numpy(dtype="float32"))
        X_numeric[torch.isnan(X_numeric)] = 0.0
        self.X_numeric = X_numeric

        # ------------------------------------------------------------------
        # Encode categorical columns
        # ------------------------------------------------------------------
        self.cat_maps = {}
        cat_tensors = {}

        for col in self.categorical_cols:
            series = df[col].astype(str).fillna("UNK")
            vocab = sorted(series.unique().tolist())
            mapping = {v: i for i, v in enumerate(vocab)}
            encoded = series.map(mapping).astype("int64")
            cat_tensors[col] = torch.tensor(encoded.values, dtype=torch.long)
            self.cat_maps[col] = mapping

        self.X_categorical = cat_tensors
        
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.X_numeric)

    def __getitem__(self, idx: int):
        """Return one sample as dict of numeric + categorical tensors."""
        numeric = self.X_numeric[idx]
        categorical = {col: tensor[idx] for col, tensor in self.X_categorical.items()}
        return {"numeric": numeric, "categorical": categorical}

    # ----------------------------------------------------------------------
    def get_vocab_sizes(self):
        """Return dict mapping categorical column -> vocabulary size."""
        return {col: len(vocab) for col, vocab in self.cat_maps.items()}

    def get_example(self, idx: int = 0):
        """Return decoded categorical values for a sample."""
        cat_example = {
            col: list(mapping.keys())[list(mapping.values()).index(int(self.X_categorical[col][idx]))]
            for col, mapping in self.cat_maps.items()
        }
        return {"numeric": self.X_numeric[idx], "categorical": cat_example}

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

        # Map pitch types → statcast pitch mix columns
        self.pitch_mix_columns = {
            "CH": "CH% (sc)",
            "CS": "CS% (sc)",
            "CU": "CU% (sc)",
            "EP": "EP% (sc)",
            "FA": "FA% (sc)",
            "FC": "FC% (sc)",
            "FF": None,
            "FO": "FO% (sc)",
            "FS": "FS% (sc)",
            "KC": "KC% (sc)",
            "KN": "KN% (sc)",
            "PO": "PO% (sc)",
            "SC": "SC% (sc)",
            "SI": "SI% (sc)",
            "SL": "SL% (sc)",
            "ST": None,
            "SV": None,
            "UN": "UN% (sc)",
        }

        # ------------------------------------------------------------
        # 1. Load unified parquet
        # ------------------------------------------------------------
        df: pd.DataFrame = pd.read_parquet(
            self.config.FUSED_CONTEXT_DATASET_FILE_PATH
        )

        # Convert pitcher ids → python int list
        self.raw_pitcher_ids = df["pitcher"].astype(
            "int64").astype(int).tolist()

        # Sample if needed
        if sample and sample < len(df):
            df = df.sample(n=sample, random_state=1337).reset_index(drop=True)

        df = df.replace({None: np.nan})

        # ------------------------------------------------------------
        # 1B. Build allowed pitch set per pitcher
        # ------------------------------------------------------------
        grouped = df.groupby("pitcher")["pitch_type"].unique()
        self.pitcher_to_allowed: dict[int, list[int]] = {}

        for pitcher_id, pitch_list in grouped.items():
            allowed: list[int] = []

            for raw_pitch in pitch_list:
                norm = self.normalize_pitch_label(str(raw_pitch))
                if norm in Constants.PITCH_TYPE_TO_IDX:
                    allowed.append(Constants.PITCH_TYPE_TO_IDX[norm])

            self.pitcher_to_allowed[int(pitcher_id)] = sorted(set(allowed))

        # ------------------------------------------------------------
        # 2. Detect numeric vs categorical
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
        # 3. Normalize numeric features
        # ------------------------------------------------------------
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.astype(np.float32)
        numeric_df = numeric_df.fillna(numeric_df.mean()).fillna(0.0)

        self.mean = numeric_df.mean()
        self.std = numeric_df.std().replace(0, 1)

        normalized = ((numeric_df - self.mean) / self.std).astype(np.float32)
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
        # 5. Labels
        # ------------------------------------------------------------
        if "next_pitch_idx" not in df.columns:
            raise ValueError("'next_pitch_idx' missing from fused dataset.")

        self.y_labels = torch.tensor(
            df["next_pitch_idx"].fillna(-1).astype(np.int64).values,
            dtype=torch.long,
        )

        # ------------------------------------------------------------
        # 6. Summary
        # ------------------------------------------------------------
        self.dataset_summary = {
            "total_samples": len(self),
            "numeric_dim": self.x_numeric.shape[1],
            "num_categories": len(self.x_categorical),
            "num_classes": int(self.y_labels.max().item() + 1),
        }

    # ------------------------------------------------------------
    # PyTorch Dataset Interface
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.y_labels)

    def __getitem__(self, idx: int):
        return {
            "numeric": self.x_numeric[idx],
            "categorical": {
                col: tensor[idx] for col, tensor in self.x_categorical.items()
            },
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

    # ------------------------------------------------------------
    # Pitch normalization logic
    # ------------------------------------------------------------
    def normalize_pitch_label(self, raw: str) -> str:
        """
        Normalizes statcast/raw pitch labels into official model-wide constants.
        """
        if raw == "FF":
            return "FA"   # FF → FA
        if raw == "ST":
            return "SL"   # sweeper → slider
        if raw == "SV":
            return "SL"   # slurve → slider
        return raw

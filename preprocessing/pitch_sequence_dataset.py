import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from controller.config import Config
from utils.constants import Constants


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

    def __init__(self, sample: int = 1000):

        # ------------------------------------------------------------------
        # Load limited subset of dataset
        # ------------------------------------------------------------------
        self.config = Config()
        dataframe = pd.read_parquet(self.config.PITCH_SEQUENCE_PARQUET_FILE_PATH)
        dataframe = dataframe.iloc[:sample]

        numeric_dataframe_column_list: list[str] = self._get_numeric_dataframe_columns_list(dataframe=dataframe)
        categorical_dataframe_column_list: list[str] = self._get_categorical_dataframe_columns_list(dataframe=dataframe)

        self._is_usable_columns_none(numeric_dataframe_column_list=numeric_dataframe_column_list,
                                     categorical_dataframe_column_list=categorical_dataframe_column_list)

        self.numeric_cols = numeric_dataframe_column_list
        self.categorical_columns = categorical_dataframe_column_list

        # ------------------------------------------------------------------
        # Process numeric columns
        # ------------------------------------------------------------------
        numeric_dataframe: pd.DataFrame = dataframe[self.numeric_cols].copy()

        # Force numeric coercion
        for col in numeric_dataframe.columns:
            numeric_dataframe[col] = pd.to_numeric(numeric_dataframe[col], errors="coerce")

        # Fill NaNs with mean, fallback to 0
        numeric_dataframe = numeric_dataframe.fillna(numeric_dataframe.mean()).fillna(0)
        numeric_dataframe = numeric_dataframe.astype("float64")

        # Normalize numeric columns
        self.mean_value = numeric_dataframe.mean()
        self.standard_deviation = numeric_dataframe.std().replace(0, 1)
        normalized = (numeric_dataframe - self.mean_value) / self.standard_deviation

        # Convert to tensor
        x_numeric = torch.tensor(normalized.to_numpy(dtype="float32"))
        x_numeric[torch.isnan(x_numeric)] = 0.0
        self.x_numeric = x_numeric

        # ------------------------------------------------------------------
        # Encode categorical columns
        # ------------------------------------------------------------------
        self.concatenated_tensor_dict: dict[str, dict[Tensor, int]] = {}
        concatenated_tensor_dict: dict[str, Tensor] = self._get_concatenated_tensor_dict(dataframe=dataframe)

        self.x_categorical = concatenated_tensor_dict

    def _get_categorical_dataframe_columns_list(self, dataframe: pd.DataFrame) -> list[str]:

        result_list: list[str] = []

        for column_str in dataframe.columns:
            if column_str not in Constants.COLUMN_EXCLUSION_SET and not pd.api.types.is_numeric_dtype(
                    dataframe[column_str]):
                result_list.append(column_str)

        return result_list

    def _get_numeric_dataframe_columns_list(self, dataframe: pd.DataFrame) -> list[str]:

        result_list: list[str] = []

        for column_str in dataframe.columns:
            if column_str not in Constants.COLUMN_EXCLUSION_SET and pd.api.types.is_numeric_dtype(
                    dataframe[column_str]):
                result_list.append(column_str)

        return result_list

    def _get_concatenated_tensor_dict(self, dataframe: pd.DataFrame) -> dict[str, Tensor]:
        concatenated_tensor_dict: dict[str, Tensor] = {}

        for col in self.categorical_columns:
            pandas_series: pd.Series = dataframe[col].astype(str).fillna("UNK")
            vocab_tensor_list: list[Tensor] = sorted(pandas_series.unique().tolist())
            mapping_dict: dict[Tensor, int] = {v: i for i, v in enumerate(vocab_tensor_list)}
            encoded_series = pandas_series.map(mapping_dict).astype(np.int64)
            concatenated_tensor_dict[col] = torch.tensor(encoded_series.values, dtype=torch.long)
            self.concatenated_tensor_dict[col] = mapping_dict

        return concatenated_tensor_dict

    def _is_usable_columns_none(self, numeric_dataframe_column_list: list[str],
                                categorical_dataframe_column_list: list[str]) -> None:

        if not numeric_dataframe_column_list and not categorical_dataframe_column_list:
            raise ValueError("No usable columns found in context parquet.")

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.x_numeric)

    def __getitem__(self, idx: int):
        """Return one sample as dict of numeric + categorical tensors."""
        numeric = self.x_numeric[idx]
        categorical = {col: tensor[idx] for col, tensor in self.x_categorical.items()}
        return {"numeric": numeric, "categorical": categorical}

    # ----------------------------------------------------------------------
    def get_vocab_sizes(self):
        """Return dict mapping categorical column -> vocabulary size."""
        return {col: len(vocab) for col, vocab in self.concatenated_tensor_dict.items()}

    def get_example(self, idx: int = 0):
        """Return decoded categorical values for a sample."""
        cat_example = {
            col: list(mapping.keys())[list(mapping.values()).index(int(self.x_categorical[col][idx]))]
            for col, mapping in self.concatenated_tensor_dict.items()
        }
        return {"numeric": self.x_numeric[idx], "categorical": cat_example}

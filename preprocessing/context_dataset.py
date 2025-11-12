import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.constants import Constants
from utils.logger import Logger


class ContextDataset(Dataset):
    """
    Loads Statcast context data from a parquet file and prepares both
    numeric and categorical columns as tensors suitable for modeling.
    Includes 'events' so it can be embedded as a categorical feature.
    """

    def __init__(self, parquet_file_path: str):

        dataframe: pd.DataFrame = pd.read_parquet(parquet_file_path)

        numeric_dataframe_column_list: list[str] = self._get_numeric_dataframe_columns_list(dataframe=dataframe)
        categorical_dataframe_column_list: list[str] = self._get_categorical_dataframe_columns_list(dataframe=dataframe)

        self._is_usable_columns_none(numeric_dataframe_column_list=numeric_dataframe_column_list,
                                     categorical_dataframe_column_list=categorical_dataframe_column_list)

        self.feature_columns = numeric_dataframe_column_list
        self.categorical_columns = categorical_dataframe_column_list

        numeric_dataframe: pd.DataFrame = self._get_numeric_dataframe_columns(dataframe=dataframe)

        # Normalize numeric columns
        self.mean_value = numeric_dataframe.mean()
        self.standard_deviation = numeric_dataframe.std().replace(0, 1)

        normalized = (numeric_dataframe - self.mean_value) / self.standard_deviation

        # Convert to float32 tensor
        x_numeric_tensor: Tensor = torch.tensor(normalized.values, dtype=torch.float32)
        x_numeric_tensor[torch.isnan(x_numeric_tensor)] = 0.0
        self.x_numeric = x_numeric_tensor

        # ------------------------------------------------------------
        # Encode categorical columns (including 'events')
        # ------------------------------------------------------------
        self.concatenate_map: dict[str, dict[Tensor, int]] = {}
        concatenated_tensor_dict: dict[str, Tensor] = self._get_concatenated_tensor_dict(dataframe=dataframe)

        self.x_categorical = concatenated_tensor_dict

        self.logger = Logger(self.__class__.__name__)

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

        for categorical_column in self.categorical_columns:
            pandas_series: pd.Series = dataframe[categorical_column].astype(str).fillna("UNK")
            vocab_tensor_list: list[Tensor] = sorted(pandas_series.unique().tolist())
            mapping_dict: dict[Tensor, int] = {v: i for i, v in enumerate(vocab_tensor_list)}
            encoded_series = pandas_series.map(mapping_dict).astype(np.int64)
            concatenated_tensor_dict[categorical_column] = torch.tensor(encoded_series.values, dtype=torch.long)
            self.concatenate_map[categorical_column] = mapping_dict

        return concatenated_tensor_dict

    def _get_numeric_dataframe_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_dataframe: pd.DataFrame = (
            dataframe[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float64")
        )

        # Remove all-NaN numeric columns
        all_nan_columns: list = numeric_dataframe.columns[numeric_dataframe.isna().all()].tolist()

        if all_nan_columns:
            numeric_dataframe = numeric_dataframe.drop(columns=all_nan_columns)

        # Fill NaNs with column mean, fallback to 0
        numeric_dataframe = numeric_dataframe.fillna(numeric_dataframe.mean()).fillna(0)

        return numeric_dataframe

    def _is_usable_columns_none(self, numeric_dataframe_column_list: list[str],
                                categorical_dataframe_column_list: list[str]) -> None:

        if not numeric_dataframe_column_list and not categorical_dataframe_column_list:
            raise ValueError("No usable columns found in context parquet.")

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.x_numeric)

    def __getitem__(self, idx: int):
        """Return one sample as a dict of numeric + categorical tensors."""
        numeric = self.x_numeric[idx]
        categorical = {col: tensor[idx] for col, tensor in self.x_categorical.items()}
        return {"numeric": numeric, "categorical": categorical}

    # ------------------------------------------------------------
    def get_vocab_sizes(self):
        """Return dict mapping each categorical column → vocabulary size."""
        return {col: len(vocab) for col, vocab in self.concatenate_map.items()}

    def get_example(self, idx: int = 0):
        """Inspect decoded categorical values for one example."""
        cat_example = {
            col: list(mapping.keys())[list(mapping.values()).index(int(self.x_categorical[col][idx]))]
            for col, mapping in self.concatenate_map.items()
        }
        return {"numeric": self.x_numeric[idx], "categorical": cat_example}

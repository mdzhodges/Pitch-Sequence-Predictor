import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.constants import Constants
from utils.logger import Logger


class PitcherDataset(Dataset):
    def __init__(self, parquet_file_path: str):
        dataframe: pd.DataFrame = pd.read_parquet(parquet_file_path)

        numeric_dataframe_column_list: list[str] = self._get_numeric_dataframe_columns_list(dataframe=dataframe)

        self.feature_columns = numeric_dataframe_column_list
        self.dataframe_column_names = dataframe["Name"].values if "Name" in dataframe.columns else None

        features_dataframe: pd.DataFrame = self._get_features_dataframe(dataframe=dataframe)

        # --- NORMALIZATION ---
        self.mean_value = features_dataframe.mean()
        self.standard_deviation = features_dataframe.std().replace(0, 1)
        features_dataframe = (features_dataframe - self.mean_value) / self.standard_deviation

        x_feature_tensor: Tensor = torch.tensor(features_dataframe.values, dtype=torch.float32)

        # Replace any remaining NaNs with 0
        x_feature_tensor[torch.isnan(x_feature_tensor)] = 0.0

        self.x_feature_tensor = x_feature_tensor

        self.logger = Logger(self.__class__.__name__)

    def _get_numeric_dataframe_columns_list(self, dataframe: pd.DataFrame) -> list[str]:

        result_list: list[str] = []

        for column_str in dataframe.columns:
            if column_str not in Constants.FEATURE_EXCLUSION_SET and pd.api.types.is_numeric_dtype(
                    dataframe[column_str]):
                result_list.append(column_str)

        if not result_list:
            raise ValueError("No numeric columns found.")

        return result_list

    def _get_features_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        features_dataframe: pd.DataFrame = dataframe[self.feature_columns].apply(pd.to_numeric, errors="coerce")

        all_nan_columns_list: list[str] = features_dataframe.columns[features_dataframe.isna().all()].tolist()

        if all_nan_columns_list:
            features_dataframe = features_dataframe.drop(columns=all_nan_columns_list)

        # Fill remaining NaNs with column mean (use 0 if still all NaN after drop)
        features_dataframe = features_dataframe.fillna(features_dataframe.mean()).fillna(0)

        return features_dataframe

    def __len__(self):
        return len(self.x_feature_tensor)

    def __getitem__(self, idx):
        return self.x_feature_tensor[idx]

    def get_by_name(self, name: str):
        """Retrieve a pitcher's feature vector by partial name match."""
        if self.dataframe_column_names is None:
            raise ValueError("No Name column available.")
        mask = [name.lower() in n.lower() for n in self.dataframe_column_names]
        if idx := [i for i, m in enumerate(mask) if m]:
            return self.x_feature_tensor[idx[0]]
        else:
            raise ValueError(f"No pitcher found matching '{name}'")

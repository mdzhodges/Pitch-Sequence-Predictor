from pathlib import Path

import pandas as pd
from pybaseball import statcast

from utils.constants import Constants


class ContextDataCollection:

    def __init__(self) -> None:
        self.export_context_file_path: Path = Path("data/context_2025_full.parquet")
        self.export_stat_cast_file_path: Path = Path("data/statcast_context_collection_2025.parquet")

    def _retrieve_stat_cast_dataframe(self) -> pd.DataFrame:

        try:

            statcast_dataframe: pd.DataFrame = statcast(start_dt=Constants.START_DATE_STR,
                                                        end_dt=Constants.END_DATE_STR)
            dataframe_length: int = len(statcast_dataframe)
            num_dataframe_columns: int = len(statcast_dataframe.columns)

            return statcast_dataframe

        except Exception as e:
            raise Exception(f"Exception thrown while retrieving statcast data {e}")

    def _get_cleaned_dataframe(self, statcast_dataframe: pd.DataFrame) -> pd.DataFrame:

        available_columns_list: list[str] = []

        for column_str in Constants.CONTEXT_COLUMNS_LIST:
            if column_str in statcast_dataframe.columns:
                available_columns_list.append(column_str)

        # Copy the subset of available columns
        context_dataframe: pd.DataFrame = statcast_dataframe[available_columns_list].copy()

        return context_dataframe

    def export_stat_cast_dataframe(self) -> None:

        statcast_dataframe: pd.DataFrame = self._retrieve_stat_cast_dataframe()
        cleaned_dataframe: pd.DataFrame = self._get_cleaned_dataframe(statcast_dataframe=statcast_dataframe)

        try:

            cleaned_dataframe.to_parquet(path=self.export_stat_cast_file_path, index=False)

        except Exception as e:
            raise Exception(f"Exception thrown while exporting statcast data {e}")

    # Columns to extract (based on 2025 Statcast schema)

    def _get_stat_cast_dataframe(self) -> pd.DataFrame:
        return pd.read_parquet(path=self.export_stat_cast_file_path)

    def _get_processed_dataframe(self) -> pd.DataFrame:
        statcast_dataframe: pd.DataFrame = self._get_stat_cast_dataframe()

        # Add base runner presence columns if missing
        for base_col in ["on_1b", "on_2b", "on_3b"]:
            if base_col not in statcast_dataframe.columns:
                statcast_dataframe[base_col] = pd.NA

        # Compute total runners on base
        statcast_dataframe["runners_on_base"] = statcast_dataframe[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)

        # Add convenience aliases for modeling
        if "estimated_ba_using_speedangle" in statcast_dataframe.columns:
            statcast_dataframe["xba"] = statcast_dataframe["estimated_ba_using_speedangle"]
        if "estimated_woba_using_speedangle" in statcast_dataframe.columns:
            statcast_dataframe["xwoba"] = statcast_dataframe["estimated_woba_using_speedangle"]
        if "estimated_slg_using_speedangle" in statcast_dataframe.columns:
            statcast_dataframe["xslg"] = statcast_dataframe["estimated_slg_using_speedangle"]

        return statcast_dataframe

    def export_dataframe_to_parquet_file(self) -> None:

        context_dataframe: pd.DataFrame = self._get_processed_dataframe()

        try:
            context_dataframe.to_parquet(path=self.export_context_file_path, index=False)

        except Exception as e:
            raise Exception(f"Exception thrown while exporting context data {e}")

from pathlib import Path

import pandas as pd
from pybaseball import statcast

from utils.constants import Constants
from utils.logger import Logger


class ContextDataCollection:

    def __init__(self) -> None:
        self.export_context_file_path: Path = Path(
            "data/context_2025_full.parquet")
        self.export_stat_cast_file_path: Path = Path(
            "data/statcast_context_collection_2025.parquet")
        self.logger = Logger(self.__class__.__name__)

    def _retrieve_stat_cast_dataframe(self) -> pd.DataFrame:

        try:

            self.logger.info("Retrieving StatCast data:")
            self.logger.info("=" * 100)

            statcast_dataframe: pd.DataFrame = statcast(start_dt=Constants.START_DATE_STR,
                                                        end_dt=Constants.END_DATE_STR)
            dataframe_length: int = len(statcast_dataframe)
            num_dataframe_columns: int = len(statcast_dataframe.columns)

            self.logger.info(
                f"Successfully retrieved {dataframe_length:,} records with {num_dataframe_columns:,}")
            self.logger.info("=" * 100)

            return statcast_dataframe

        except Exception as e:
            self.logger.error(
                f"Exception thrown while retrieving statcast data {e}")
            raise Exception(
                f"Exception thrown while retrieving statcast data {e}")

    def _get_cleaned_dataframe(self, statcast_dataframe: pd.DataFrame) -> pd.DataFrame:

        available_columns_list: list[str] = []

        for column_str in Constants.CONTEXT_COLUMNS_LIST:
            if column_str in statcast_dataframe.columns:
                available_columns_list.append(column_str)

        missing_columns_list: set[str] = set(
            Constants.CONTEXT_COLUMNS_LIST) - set(available_columns_list)
        missing_columns_list_length: int = len(missing_columns_list)
        sorted_missing_columns_list: list[str] = sorted(
            list(missing_columns_list))

        if missing_columns_list:
            self.logger.info(
                f"Missing columns skipped ({missing_columns_list_length}): {sorted_missing_columns_list}")

        # Copy the subset of available columns
        context_dataframe: pd.DataFrame = statcast_dataframe[available_columns_list].copy(
        )

        return context_dataframe

    def export_stat_cast_dataframe(self) -> None:

        statcast_dataframe: pd.DataFrame = self._retrieve_stat_cast_dataframe()
        cleaned_dataframe: pd.DataFrame = self._get_cleaned_dataframe(
            statcast_dataframe=statcast_dataframe)

        try:

            self.logger.info("Exporting StatCast Data:")
            self.logger.info("=" * 100)

            cleaned_dataframe.to_parquet(
                path=self.export_stat_cast_file_path, index=False)

            self.logger.info(f"Successfully exported StatCast Data")
            self.logger.info("=" * 100)

        except Exception as e:
            self.logger.error(
                f"Exception thrown while exporting statcast data {e}")
            raise Exception(
                f"Exception thrown while exporting statcast data {e}")

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
        statcast_dataframe["runners_on_base"] = statcast_dataframe[[
            "on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)

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
        context_dataframe_length: int = len(context_dataframe)
        context_dataframe_column_length: int = len(context_dataframe.columns)

        try:
            self.logger.info(
                f"Exporting data to: {self.export_context_file_path}")
            self.logger.info("=" * 100)

            context_dataframe.to_parquet(
                path=self.export_context_file_path, index=False)

            self.logger.info(
                f"Saved {context_dataframe_length:,} rows × {context_dataframe_column_length:,} columns to {self.export_context_file_path}")

            self.logger.info(
                f"Successfully exported data to: {self.export_context_file_path}")
            self.logger.info("=" * 100)

        except Exception as e:
            self.logger.error(
                f"Exception thrown while exporting context data {e}")
            raise Exception(
                f"Exception thrown while exporting context data {e}")

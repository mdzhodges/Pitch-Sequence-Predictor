from pathlib import Path

import pandas as pd
from pybaseball import statcast

from utils.constants import Constants


class PitchSequenceDataCollection:

    def __init__(self) -> None:
        self.export_pitch_sequence_file_path: Path = Path("data/pitch_sequence_2025.parquet")
        self.export_stat_cast_pitch_by_pitch_file_path: Path = Path("data/statcast_pitch_by_pitch_2025.parquet")

    def _retrieve_stat_cast_pitch_by_pitch_dataframe(self) -> pd.DataFrame:

        try:

            result_dataframe: pd.DataFrame = statcast(start_dt=Constants.START_DATE_STR, end_dt=Constants.END_DATE_STR)
            dataframe_length: int = len(result_dataframe)

            return result_dataframe

        except Exception as e:
            raise Exception(f"Exception thrown while retrieving statcast pitch-by-pitch data {e}")

    def export_stat_cast_pitch_by_pitch_dataframe(self) -> None:

        pitch_by_pitch_dataframe: pd.DataFrame = self._retrieve_stat_cast_pitch_by_pitch_dataframe()

        try:

            pitch_by_pitch_dataframe.to_parquet(path=self.export_stat_cast_pitch_by_pitch_file_path, index=False)

        except Exception as e:
            raise Exception(f"Exception thrown while exporting statcast pitch-by-pitch data {e}")

    def _get_stat_cast_pitch_by_pitch_dataframe(self) -> pd.DataFrame:
        return pd.read_parquet(path=self.export_stat_cast_pitch_by_pitch_file_path)

    def _get_valid_columns_list(self) -> list[str]:
        pitch_by_pitch_dataframe: pd.DataFrame = self._get_stat_cast_pitch_by_pitch_dataframe()

        valid_columns_list: list[str] = []

        for column_str in Constants.PITCH_SEQUENCE_COLUMNS_LIST:
            if column_str in pitch_by_pitch_dataframe.columns:
                valid_columns_list.append(column_str)

        return valid_columns_list

    def _get_pitch_by_pitch_subset_dataframe(self) -> pd.DataFrame:

        pitch_by_pitch_dataframe: pd.DataFrame = self._get_stat_cast_pitch_by_pitch_dataframe()
        valid_columns_list: list[str] = self._get_valid_columns_list()

        result_dataframe: pd.DataFrame = pitch_by_pitch_dataframe[valid_columns_list]

        return result_dataframe

    def _get_processed_dataframe(self) -> pd.DataFrame:

        pitch_by_pitch_dataframe: pd.DataFrame = self._get_pitch_by_pitch_subset_dataframe()

        # Sort by order of play
        order_of_play_columns_list: list[str] = ["game_pk", "at_bat_number", "pitch_number"]
        sorted_dataframe: pd.DataFrame = pitch_by_pitch_dataframe.sort_values(
            by=order_of_play_columns_list).reset_index(
            drop=True)

        # Create next pitch label
        next_pitch_columns_list: list[str] = ["game_pk", "pitcher", "batter"]
        sorted_dataframe["next_pitch_type"] = (
            sorted_dataframe.groupby(next_pitch_columns_list)["pitch_type"].shift(-1))

        # Drop final pitches of each at-bat (no next pitch)
        result_dataframe: pd.DataFrame = sorted_dataframe.dropna(subset=["next_pitch_type"])

        # Create numeric count feature
        result_dataframe["count_state"] = result_dataframe["balls"] * 10 + result_dataframe["strikes"]

        return result_dataframe

    def export_dataframe_to_parquet_file(self) -> None:

        pitch_by_pitch_dataframe: pd.DataFrame = self._get_processed_dataframe()
        pitch_by_pitch_dataframe_length: int = len(pitch_by_pitch_dataframe)
        pitch_by_pitch_dataframe_column_length: int = len(pitch_by_pitch_dataframe.columns)

        try:
            pitch_by_pitch_dataframe.to_parquet(path=self.export_pitch_sequence_file_path, index=False)

        except Exception as e:
            raise Exception(f"Exception thrown while exporting pitch sequence data {e}")

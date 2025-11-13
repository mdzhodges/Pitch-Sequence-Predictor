from pathlib import Path
from typing import Optional

import pandas as pd
from pybaseball import playerid_reverse_lookup

from controller.config import Config
from utils.logger import Logger


class FusionCollection:

    def __init__(self) -> None:
        self.logger = Logger(self.__class__.__name__)

    def preprocess_unified(self, debug: bool = True):
        """
        Creates a unified dataset combining context, hitter, and pitcher information.
        Generates a `next_pitch_type` and corresponding integer label `next_pitch_idx`
        for next-pitch prediction. Keeps only rows with valid next-pitch targets.
        """

        config: Config = Config()

        # ------------------------------------------------------------
        # 1. Load datasets
        # ------------------------------------------------------------
        context_dataframe: pd.DataFrame = pd.read_parquet(config.CONTEXT_PARQUET_FILE_PATH)
        hitter_dataframe: pd.DataFrame = pd.read_parquet(config.HITTER_PARQUET_FILE_PATH)
        pitcher_dataframe: pd.DataFrame = pd.read_parquet(config.PITCHER_PARQUET_FILE_PATH)

        # ------------------------------------------------------------
        # 2. Map FanGraphs -> MLBAM IDs
        # ------------------------------------------------------------
        fan_graph_id_list: list[str] = pd.concat([hitter_dataframe["IDfg"], pitcher_dataframe["IDfg"]]
                                                 ).dropna().unique().tolist()
        id_map_dataframe: pd.DataFrame = playerid_reverse_lookup(fan_graph_id_list, key_type="fangraphs")[
            ["key_fangraphs", "key_mlbam"]].dropna()
        id_map_dataframe = id_map_dataframe.rename(
            columns={"key_fangraphs": "IDfg", "key_mlbam": "mlbam"})

        hitter_dataframe = hitter_dataframe.merge(id_map_dataframe, on="IDfg", how="left")
        pitcher_dataframe = pitcher_dataframe.merge(id_map_dataframe, on="IDfg", how="left")

        # ------------------------------------------------------------
        # 3. Identify sequencing columns
        # ------------------------------------------------------------
        possible_at_bat_columns_list: list[str] = [column for column in context_dataframe.columns if
                                                   "at_bat" in column.lower()]

        possible_pitch_number_columns_list: list[str] = self._get_columns_list(context_dataframe=context_dataframe,
                                                                               column_name_str_1="pitch_number",
                                                                               column_name_str_2="pitch_no")
        possible_game_columns_list: list[str] = self._get_columns_list(context_dataframe=context_dataframe,
                                                                       column_name_str_1="game_pk",
                                                                       column_name_str_2="game_id")

        if not possible_at_bat_columns_list or not possible_pitch_number_columns_list:
            raise KeyError(
                f"Missing sequencing columns. Found: at_bat={possible_at_bat_columns_list}, pitch={possible_pitch_number_columns_list}"
            )

        at_bat_columns_str: str = possible_at_bat_columns_list[0]
        pitch_columns_str: str = possible_pitch_number_columns_list[0]
        game_columns_value: Optional[str] = possible_game_columns_list[0] if possible_game_columns_list else None

        # ------------------------------------------------------------
        # 4. Sort dataset by pitch order
        # ------------------------------------------------------------
        sorted_columns_list: list[str] = [column for column in [
            game_columns_value, at_bat_columns_str, pitch_columns_str] if column is not None]

        context_dataframe = context_dataframe.sort_values(sorted_columns_list).reset_index(drop=True)

        # ------------------------------------------------------------
        # 5. Generate next-pitch labels
        # ------------------------------------------------------------
        group_columns_list: list[str] = [column for column in [game_columns_value, at_bat_columns_str] if
                                         column is not None]
        context_dataframe["next_pitch_type"] = (context_dataframe.groupby(group_columns_list)["pitch_type"].shift(-1))

        context_dataframe = context_dataframe.dropna(subset=["next_pitch_type"]).reset_index(drop=True)
        # ------------------------------------------------------------
        # 6. Encode pitch type and next pitch type
        # ------------------------------------------------------------
        pitch_type_list: list[str] = sorted(context_dataframe["pitch_type"].dropna().unique().tolist())
        pitch_type_dict: dict[str, int] = {p: i for i, p in enumerate(pitch_type_list)}

        context_dataframe["pitch_type_idx"] = context_dataframe["pitch_type"].map(pitch_type_dict)
        context_dataframe["next_pitch_idx"] = context_dataframe["next_pitch_type"].map(pitch_type_dict)

        # ------------------------------------------------------------
        # 7. Check column overlaps before merging hitter/pitcher
        # ------------------------------------------------------------
        context_columns_set: set[str] = set(context_dataframe.columns)
        hitter_columns_set: set[str] = set(hitter_dataframe.columns)
        pitcher_columns_set: set[str] = set(pitcher_dataframe.columns)

        _overlap_hitter = context_columns_set & hitter_columns_set
        _overlap_pitcher = context_columns_set & pitcher_columns_set

        # ------------------------------------------------------------
        # 8. Merge hitter and pitcher info (collision-safe)
        # ------------------------------------------------------------
        context_dataframe = (
            context_dataframe.merge(
                hitter_dataframe.add_suffix("_hitter"),
                left_on="batter",
                right_on="mlbam_hitter",
                how="left",
                suffixes=("", "_dup_hitter")
            )
            .merge(
                pitcher_dataframe.add_suffix("_pitcher"),
                left_on="pitcher",
                right_on="mlbam_pitcher",
                how="left",
                suffixes=("", "_dup_pitcher")
            )
        )

        # ------------------------------------------------------------
        # 9. Save unified dataset
        # ------------------------------------------------------------
        file_save_path: Path = Path("data/unified_context.parquet")
        context_dataframe.to_parquet(file_save_path, index=False)

        self.logger.info(f"Generated data of size: {context_dataframe.shape} <= Should be (732976, 820)")

        return context_dataframe

    def _get_columns_list(self, context_dataframe: pd.DataFrame, column_name_str_1: str, column_name_str_2: str) -> \
            list[str]:
        result_list: list[str] = []

        for column_str in context_dataframe.columns:
            if column_name_str_1 in column_str.lower() or column_name_str_2 in column_str.lower():
                result_list.append(column_str)

        return result_list

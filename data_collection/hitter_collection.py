from pathlib import Path

import pandas as pd
from pybaseball import batting_stats

from utils.logger import Logger


class HitterDataCollection:

    def __init__(self) -> None:
        self.export_file_path: Path = Path("data/hitters_2025_full.parquet")
        self.logger = Logger(self.__class__.__name__)

    def _get_hitter_stats_dataframe(self) -> pd.DataFrame:

        try:

            self.logger.info("Retrieving hitter data:")
            self.logger.info("=" * 100)

            # Retrieve data from the 2025 MLB season
            hitter_stats_dataframe: pd.DataFrame = batting_stats(2025, qual=0)

            # Remove hitters with no plate appearances
            result_dataframe: pd.DataFrame = hitter_stats_dataframe[hitter_stats_dataframe["PA"] > 0].reset_index(
                drop=True)

            self.logger.info("Successfully retrieved hitter data:")
            self.logger.info("=" * 100)

            return result_dataframe


        except Exception as e:
            self.logger.error(f"Exception thrown while retrieving hitter data: {e}")
            raise Exception(f"Exception thrown while retrieving hitter data: {e}")

    def export_hitter_data_to_parquet_file(self) -> None:

        hitter_stats_dataframe: pd.DataFrame = self._get_hitter_stats_dataframe()

        try:

            self.logger.info(f"Exporting hitter data to: {self.export_file_path}")
            self.logger.info("=" * 100)

            hitter_stats_dataframe.to_parquet(path=self.export_file_path, index=False)

            self.logger.info(f"Successfully exported hitter data to: {self.export_file_path}")
            self.logger.info("=" * 100)

        except Exception as e:
            self.logger.error(f"Exception thrown while exporting hitter data: {e}")
            raise Exception(f"Exception thrown while exporting hitter data: {e}")

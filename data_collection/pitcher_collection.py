from pathlib import Path

import pandas as pd
from pybaseball import pitching_stats

from utils.logger import Logger


class PitcherDataCollection:

    def __init__(self) -> None:
        self.export_file_path: Path = Path("data/pitchers_2025_full.csv")
        self.logger = Logger(self.__class__.__name__)

    def get_pitcher_stats_dataframe(self) -> pd.DataFrame:

        try:

            self.logger.info("Retrieving pitcher data:")
            self.logger.info("=" * 100)

            # Retrieve data from the 2025 MLB season
            pitcher_stats_dataframe: pd.DataFrame = pitching_stats(2025, qual=0)

            # Remove pitchers with no innings pitched
            result_dataframe: pd.DataFrame = pitcher_stats_dataframe[pitcher_stats_dataframe["IP"] > 0].reset_index(
                drop=True)

            self.logger.info("Successfully retrieved pitcher data:")
            self.logger.info("=" * 100)

            return result_dataframe


        except Exception as e:
            self.logger.error(f"Exception thrown while retrieving pitching data: {e}")
            raise Exception(f"Exception thrown while retrieving pitching data: {e}")

    def export_pitcher_data_to_csv(self) -> None:

        pitcher_stats_dataframe: pd.DataFrame = self.get_pitcher_stats_dataframe()

        try:

            self.logger.info("Exporting pitcher data:")
            self.logger.info("=" * 100)

            pitcher_stats_dataframe.to_csv(path_or_buf=self.export_file_path, encoding="utf-8", index=False)

            self.logger.info("Successfully exported pitcher data:")
            self.logger.info("=" * 100)

        except Exception as e:
            self.logger.error(f"Exception thrown while exporting pitching data: {e}")
            raise Exception(f"Exception thrown while exporting pitching data: {e}")

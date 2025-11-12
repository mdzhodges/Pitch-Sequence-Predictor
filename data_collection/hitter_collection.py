from pathlib import Path

import pandas as pd
from pybaseball import batting_stats

class HitterDataCollection:

    def __init__(self) -> None:
        self.export_file_path: Path = Path("data/hitters_2025_full.parquet")

    def _get_hitter_stats_dataframe(self) -> pd.DataFrame:

        try:

            # Retrieve data from the 2025 MLB season
            hitter_stats_dataframe: pd.DataFrame = batting_stats(2025, qual=0)

            # Remove hitters with no plate appearances
            result_dataframe: pd.DataFrame = hitter_stats_dataframe[hitter_stats_dataframe["PA"] > 0].reset_index(
                drop=True)

            return result_dataframe


        except Exception as e:
            raise Exception(f"Exception thrown while retrieving hitter data: {e}")

    def export_hitter_data_to_parquet_file(self) -> None:

        hitter_stats_dataframe: pd.DataFrame = self._get_hitter_stats_dataframe()

        try:

            hitter_stats_dataframe.to_parquet(path=self.export_file_path, index=False)

        except Exception as e:
            raise Exception(f"Exception thrown while exporting hitter data: {e}")

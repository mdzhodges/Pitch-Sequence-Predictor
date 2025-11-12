from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    SAMPLE: int = 1000
    NUM_EPOCHS: int = 20
    LR_PITCH_SEQ: float = 1E-5
    DROPOUT_PITCH_SEQ: float = .3
    BATCH_SIZE: int = 25
    HITTER_PARQUET_FILE_PATH="data/hitters_2025_full.parquet"
    PITCHER_PARQUET_FILE_PATH="data/pitchers_2025_full.parquet"
    

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

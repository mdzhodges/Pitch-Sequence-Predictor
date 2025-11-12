from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    SAMPLE: int = 1000
    NUM_EPOCHS: int = 20
    LR_HITTER: float = 1E-5
    LR_PITCHER: float = 1E-5
    LR_CONTEXT: float = 1E-5
    LR_PITCH_SEQ: float = 1E-5
    DROPOUT_HITTER: float = .3
    DROPOUT_PITCHER: float = .3
    DROPOUT_CONTEXT: float = .3
    DROPOUT_PITCH_SEQ: float = .3
    HITTER_PARQUET_FILE_PATH: Path
    PITCHER_PARQUET_FILE_PATH: Path

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

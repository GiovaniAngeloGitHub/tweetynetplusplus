import logging
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class DataConfig(BaseSettings):
    raw_root: str = Field(..., alias="RAW_DATA_ROOT")
    processed_root: str = Field(..., alias="PROCESSED_ROOT")
    bird_ids: list[str] = Field(..., alias="BIRD_IDS")
    batch_size: int = Field(32, alias="BATCH_SIZE")
    target_width: int = Field(2048, alias="TARGET_WIDTH")


class SpectrogramConfig(BaseSettings):
    sample_rate: int = Field(48000, alias="SAMPLE_RATE")
    n_fft: int = Field(1024, alias="N_FFT")
    hop_length: int = Field(512, alias="HOP_LENGTH")
    target_height: int = Field(128, alias="TARGET_HEIGHT")

class TrainingConfig(BaseSettings):
    learning_rate: float = Field(0.001, alias="LEARNING_RATE")
    num_epochs: int = Field(50, alias="NUM_EPOCHS")
    pretrained_weights: str | None = Field(None, alias="PRETRAINED_WEIGHTS")

class LoggingConfig(BaseSettings):
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_dir: str = Field("logs", alias="LOG_DIR")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        v = v.upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v


class Settings(BaseSettings):
    data: DataConfig
    spectrogram: SpectrogramConfig
    training: TrainingConfig
    logging: LoggingConfig = LoggingConfig()

settings = Settings(
    data=DataConfig(),
    spectrogram=SpectrogramConfig(),
    training=TrainingConfig(),
)

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parents[2]


class EnvSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    log_level: str
    log_type: str

    model_path: Path
    model_device: str
    model_imgsz: int
    model_conf: float
    model_warmup: bool
    model_concurrent: int

    @field_validator("model_path", mode="before")
    @classmethod
    def _expand_model_path(cls, v: str) -> Path:
        return ROOT_DIR / v


env = EnvSettings()

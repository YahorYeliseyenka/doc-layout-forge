from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parents[1]


class EnvSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    log_level: str
    log_type: str


class Settings(BaseModel):
    root_dir: Path

    @classmethod
    def from_yaml(cls) -> "Settings":
        yaml_config = yaml.safe_load((ROOT_DIR / "config.yaml").open("r"))
        yaml_config["root_dir"] = ROOT_DIR
        return cls(**yaml_config)


env = EnvSettings()
settings = Settings.from_yaml()

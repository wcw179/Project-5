from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = BASE_DIR / "config" / "config.yml"


class BrokerConfig(BaseModel):
    api_key: str
    api_secret: str
    dry_run: bool


class DataConfig(BaseModel):
    symbols: List[str]
    rolling_window_hours: int = Field(..., gt=0)


class LoggingConfig(BaseModel):
    level: str
    format: str
    rotation: str
    retention: str


class AppConfig(BaseSettings):
    project_name: str
    version: str
    broker: BrokerConfig
    data: DataConfig
    logging: LoggingConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config_from_yaml(path: Path = CONFIG_FILE_PATH) -> AppConfig:
    """Loads configuration from a YAML file and validates it."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppConfig(**config_data)


# Load the main config instance that can be imported by other modules
config = load_config_from_yaml()

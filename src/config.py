from pathlib import Path
from typing import List

import keyring
import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = BASE_DIR / "config" / "config.yml"
SERVICE_NAME = "BlackSwanHunter"


class BrokerConfig(BaseModel):
    account_name: str
    dry_run: bool
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None

    @model_validator(mode="after")
    def get_secrets_from_keyring(self) -> "BrokerConfig":
        """Retrieve secrets from keyring if not provided directly."""
        if self.dry_run:
            self.api_key = SecretStr("dry_run_key")
            self.api_secret = SecretStr("dry_run_secret")
            return self

        if not self.api_key:
            key_from_keyring = keyring.get_password(
                SERVICE_NAME, f"{self.account_name}_api_key"
            )
            if key_from_keyring:
                self.api_key = SecretStr(key_from_keyring)

        if not self.api_secret:
            secret_from_keyring = keyring.get_password(
                SERVICE_NAME, f"{self.account_name}_api_secret"
            )
            if secret_from_keyring:
                self.api_secret = SecretStr(secret_from_keyring)

        if not self.api_key or not self.api_secret:
            raise ValueError(
                f"API key/secret for '{self.account_name}' not found. "
                f"Run 'scripts/set_secrets.py' to set them."
            )
        return self


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

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from src.core.logging import logger

DEFAULT_CONFIG_PATH = Path("config.yaml")

class AppConfig:
    """Manages application settings loaded from config.yaml."""
    
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self._raw_config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Using fallback defaults.")
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to parse config file: {e}. Using fallback defaults.")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        val = self._raw_config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def server(self) -> Dict[str, Any]:
        return self._raw_config.get("server", {})

    @property
    def thresholds(self) -> Dict[str, Any]:
        return self._raw_config.get("thresholds", {})

    @property
    def benchmarks(self) -> Dict[str, Any]:
        return self._raw_config.get("benchmarks", {})

    @property
    def engines(self) -> Dict[str, Any]:
        return self._raw_config.get("engines", {})

config = AppConfig()

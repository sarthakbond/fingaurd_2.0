import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from src.core.logging import logger

DEFAULT_CONFIG_PATH = Path("config.yaml")

class AppConfig:
    """Manages application settings loaded from config.yaml and environment variables."""
    
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
    def database_url(self) -> str:
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url
        return self.get("database.url", "sqlite:///./data/finguard.db")

    @property
    def cors_origins(self) -> List[str]:
        env_origins = os.getenv("ALLOWED_ORIGINS")
        if env_origins:
            return [o.strip() for o in env_origins.split(",") if o.strip()]
        cfg_origins = self.get("security.cors_origins")
        if cfg_origins and isinstance(cfg_origins, list):
            return cfg_origins
        return ["http://localhost:8000", "http://127.0.0.1:8000"]

    @property
    def max_upload_mb(self) -> int:
        env_max = os.getenv("MAX_UPLOAD_MB")
        if env_max:
            try:
                return int(env_max)
            except ValueError:
                pass
        return int(self.get("uploads.max_upload_mb", self.get("server.max_upload_mb", 50)))

    @property
    def rate_limit_per_minute(self) -> int:
        env_rate = os.getenv("RATE_LIMIT_PER_MINUTE")
        if env_rate:
            try:
                return int(env_rate)
            except ValueError:
                pass
        return int(self.get("security.rate_limit_per_minute", 60))

    @property
    def rate_limit_enabled(self) -> bool:
        return bool(self.get("security.rate_limit_enabled", True))

    @property
    def uploads(self) -> Dict[str, Any]:
        return self._raw_config.get("uploads", {})

    @property
    def security(self) -> Dict[str, Any]:
        return self._raw_config.get("security", {})

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

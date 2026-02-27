import yaml
from pathlib import Path


class ConfigLoader:
    """
    Industry-grade configuration loader.
    Supports nested YAML keys using dot notation.
    Example:
        config.get("model.path")
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"[ConfigLoader] Config file not found: {self.config_path}"
            )

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """
        Access nested config using dot notation.
        Example:
            get("model.path")
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value
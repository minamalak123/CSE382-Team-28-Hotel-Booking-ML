"""Project configuration helpers. config.yaml is the single source of truth."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_for(cfg: dict, key: str) -> Path:
    """Absolute path of an entry in cfg['paths'], independent of the working directory."""
    return ROOT / cfg["paths"][key]

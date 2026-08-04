"""Load the local WSI stream configuration without external side effects."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "datasets.yaml.example"


def _resolve(value: Any, data_root: str) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("{data_root}", data_root)


def load_dataset_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read YAML paths and expand ``{data_root}`` placeholders.

    Loading datasets must remain offline. In particular, an ``hf_token`` key is
    rejected so credentials cannot accidentally be committed or used as a
    dataset-loading side effect.
    """
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG
    if not config_path.exists():
        if path:
            raise FileNotFoundError(f"Dataset config does not exist: {config_path}")
        config_path = EXAMPLE_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "hf_token" in raw:
        raise ValueError(
            "Do not store hf_token in the dataset YAML. Remove it and expose "
            "HF_TOKEN in the environment only when a custom backbone needs it."
        )
    data_root = str(raw.get("data_root", ""))
    if not data_root:
        raise ValueError(f"Missing required 'data_root' in {config_path}")

    required_maps = ("annotations", "features", "split_dirs")
    for key in required_maps:
        if not isinstance(raw.get(key, {}), dict):
            raise ValueError(f"'{key}' must be a mapping in {config_path}")

    task_order = list(raw.get("task_order", []))
    if raw.get("reverse_task_order", False):
        task_order.reverse()
    if not task_order:
        raise ValueError(f"'task_order' cannot be empty in {config_path}")

    return {
        "config_path": str(config_path.resolve()),
        "data_root": data_root,
        "task_order": task_order,
        "annotations": {
            key: _resolve(value, data_root)
            for key, value in raw.get("annotations", {}).items()
        },
        "features": {
            key: _resolve(value, data_root)
            for key, value in raw.get("features", {}).items()
        },
        "split_dirs": {
            key: _resolve(value, data_root)
            for key, value in raw.get("split_dirs", {}).items()
        },
    }


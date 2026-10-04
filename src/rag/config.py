from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """configs/base.yaml 위에 실험별 config 를 덮어써서 반환."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if path is not None:
        cfg = _deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg

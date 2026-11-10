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


def parse_set(expr: str) -> tuple[list[str], Any]:
    """`--set a.b=value` -> (["a", "b"], value). value 는 YAML 로 해석 (숫자, bool, 리스트)."""
    key, sep, raw = expr.partition("=")
    if not sep:
        raise ValueError(f"--set 은 key=value 형식: {expr}")
    return key.split("."), yaml.safe_load(raw)


def load_layered_config(path: str | Path, overlays=(), sets=()) -> dict[str, Any]:
    """base.yaml + config + overlay 들 (순서대로 deep-merge) + --set key.sub=value."""
    cfg = load_config(path)
    for p in overlays:
        cfg = _deep_merge(cfg, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    for expr in sets:
        keys, value = parse_set(expr)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg

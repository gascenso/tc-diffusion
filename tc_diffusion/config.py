import yaml
from pathlib import Path
from copy import deepcopy


def load_config(path, overrides=None):
    """
    Load a YAML config and apply simple key=value overrides from CLI.

    overrides example:
        ["training.lr=5e-4", "data.batch_size=16"]
    """
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    return cfg


def apply_overrides(cfg, overrides):
    cfg = deepcopy(cfg)
    for ov in overrides:
        key, value = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # naive type inference
        v = parse_scalar(value)
        d[keys[-1]] = v
    return cfg


def parse_scalar(v: str):
    # Try bool, int, float; otherwise keep string
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

import yaml
from pathlib import Path
from copy import deepcopy


_DATA_PATH_KEYS = (
    ("data", "data_root"),
    ("data", "dataset_index"),
    ("data", "packed_manifest"),
    ("data", "split_dir"),
)


def load_config(path, overrides=None):
    """
    Load a YAML config, optionally inheriting from one or more base configs,
    and apply simple key=value overrides from CLI.

    overrides example:
        ["training.lr=5e-4", "data.batch_size=16"]
    """
    path = Path(path).resolve()
    cfg = _load_config_recursive(path)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    cfg = resolve_config_paths(cfg, base_dir=path.parent)
    return cfg


def _load_config_recursive(path: Path, seen=None):
    path = Path(path).resolve()
    if seen is None:
        seen = set()
    if path in seen:
        raise ValueError(f"Config inheritance cycle detected at: {path}")

    seen = set(seen)
    seen.add(path)

    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")

    cfg = resolve_config_paths(cfg, base_dir=path.parent)

    base_spec = cfg.pop("_base_", None)
    if base_spec is None:
        return cfg

    if isinstance(base_spec, (str, Path)):
        base_paths = [base_spec]
    elif isinstance(base_spec, list):
        base_paths = base_spec
    else:
        raise ValueError(
            f"_base_ must be a string path or list of paths in config: {path}"
        )

    merged = {}
    for base_entry in base_paths:
        base_path = (path.parent / Path(base_entry)).resolve()
        base_cfg = _load_config_recursive(base_path, seen=seen)
        merged = deep_merge_dicts(merged, base_cfg)

    return deep_merge_dicts(merged, cfg)


def deep_merge_dicts(base, override):
    base = deepcopy(base)
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = deep_merge_dicts(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def resolve_config_paths(cfg, base_dir: Path):
    cfg = deepcopy(cfg)
    base_dir = Path(base_dir).resolve()
    for path_keys in _DATA_PATH_KEYS:
        _resolve_one_path(cfg, path_keys, base_dir)
    return cfg


def _resolve_one_path(cfg, keys, base_dir: Path):
    d = cfg
    for key in keys[:-1]:
        if not isinstance(d, dict) or key not in d:
            return
        d = d[key]

    if not isinstance(d, dict):
        return

    leaf = keys[-1]
    value = d.get(leaf)
    if value is None or not isinstance(value, (str, Path)):
        return

    value_path = Path(value)
    if value_path.is_absolute():
        d[leaf] = str(value_path)
        return

    d[leaf] = str((base_dir / value_path).resolve())


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

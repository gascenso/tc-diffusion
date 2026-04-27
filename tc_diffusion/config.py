import yaml
from pathlib import Path
from copy import deepcopy
import os


_DATA_PATH_KEYS = (
    ("data", "data_root"),
    ("data", "dataset_index"),
    ("data", "packed_manifest"),
    ("data", "split_dir"),
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _absolute_preserving_symlinks(path: str | Path, cwd: Path | None = None) -> Path:
    """Return an absolute path without collapsing symlink components.

    This matters for Cassandra jobs that launch from `/work/...` while
    `configs/`, `scripts/`, and `tc_diffusion/` are symlinked back to
    `/users_home/...`. Relative data paths in configs must stay anchored to the
    invoked `/work` tree rather than the symlink target in `/users_home`.
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        base = Path.cwd() if cwd is None else Path(cwd)
        path = base / path
    return Path(os.path.abspath(os.fspath(path)))


def load_config(path, overrides=None):
    """
    Load a YAML config, optionally inheriting from one or more base configs,
    and apply simple key=value overrides from CLI.

    overrides example:
        ["training.lr=5e-4", "data.batch_size=16"]
    """
    path = _absolute_preserving_symlinks(path)
    cfg = _load_config_recursive(path)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    cfg = resolve_config_paths(cfg, base_dir=path.parent)
    return cfg


def _load_config_recursive(path: Path, seen=None):
    path = _absolute_preserving_symlinks(path)
    if seen is None:
        seen = set()
    cycle_key = path.resolve()
    if cycle_key in seen:
        raise ValueError(f"Config inheritance cycle detected at: {path}")

    seen = set(seen)
    seen.add(cycle_key)

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
        base_path = _absolute_preserving_symlinks(Path(path.parent) / Path(base_entry))
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
    base_dir = _absolute_preserving_symlinks(base_dir)
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

    resolved = _absolute_preserving_symlinks(base_dir / value_path)
    if resolved.exists() or not _looks_like_legacy_repo_relative_path(value_path):
        d[leaf] = str(resolved)
        return

    repo_relative = (_PROJECT_ROOT / value_path).resolve()
    if repo_relative.exists():
        d[leaf] = str(repo_relative)
        return

    d[leaf] = str(resolved)


def _looks_like_legacy_repo_relative_path(value_path: Path) -> bool:
    parts = value_path.parts
    if not parts:
        return False
    return parts[0] in {"data", "runs"}


def apply_overrides(cfg, overrides):
    cfg = deepcopy(cfg)
    for ov in overrides:
        key, value = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            if not isinstance(d[k], dict):
                raise ValueError(
                    f"Cannot apply override {key!r}: intermediate key {k!r} is not a mapping."
                )
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

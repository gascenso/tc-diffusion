from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tc_diffusion.evaluation.metrics import js_divergence, wasserstein1_from_hist
from tc_diffusion.sample_bank import model_outputs_root, model_paper_ready_root


DEFAULT_CLASS_LABELS = {
    0: "< Cat 1",
    1: "Cat 1",
    2: "Cat 2",
    3: "Cat 3",
    4: "Cat 4",
    5: "Cat 5",
}
FIGURE_PIXEL = "pixel_plausibility"
FIGURE_RADIAL = "radial_bt_profile"
FIGURE_PSD = "psd_radial"
FIGURE_DAV = "dav"
FIGURE_COLD = "cold_cloud_fraction"
FIGURE_MEMORIZATION = "memorization_pairs"
DEFAULT_FIGURES = (FIGURE_PIXEL, FIGURE_RADIAL, FIGURE_PSD, FIGURE_DAV, FIGURE_COLD)
ALL_FIGURES = (*DEFAULT_FIGURES, FIGURE_MEMORIZATION)

REAL_COLOR = "#111827"
MODEL_COLORS = ["#b65f2a", "#1f6f78", "#7b516d"]
MODEL_LINESTYLES = ["solid", (0, (6.0, 2.2)), (0, (2.5, 1.5))]
BT_CMAP = "gist_ncar"


@dataclass
class InputSpec:
    run_name: str
    eval_ref: str
    label: str


@dataclass
class ClassPanel:
    class_id: int
    label: str
    bins: np.ndarray
    real_mass: np.ndarray
    gen_mass: np.ndarray
    real_density: np.ndarray
    gen_density: np.ndarray
    js: float
    w1: float
    real_hist_counts: np.ndarray | None = None
    gen_hist_counts: np.ndarray | None = None


@dataclass
class ModelInference:
    gen_density_low: np.ndarray | None = None
    gen_density_high: np.ndarray | None = None
    js_ci: tuple[float, float] | None = None
    w1_ci: tuple[float, float] | None = None
    js_null_q95: float | None = None
    w1_null_q95: float | None = None
    js_pvalue: float | None = None
    w1_pvalue: float | None = None


@dataclass
class ClassInference:
    real_density_low: np.ndarray | None = None
    real_density_high: np.ndarray | None = None
    model: Dict[int, ModelInference] = field(default_factory=dict)
    supports_image_level_stats: bool = False


@dataclass
class LoadedReport:
    figure_kind: str
    run_name: str
    label: str
    eval_ref: str
    split: str
    tag: str
    n_per_class: int
    metrics_path: Path
    panels: Dict[int, Any]


@dataclass
class RadialPanel:
    class_id: int
    label: str
    radius: np.ndarray
    real_profile: np.ndarray
    gen_profile: np.ndarray
    mae_k: float
    real_profiles: np.ndarray | None = None
    gen_profiles: np.ndarray | None = None


@dataclass
class RadialModelInference:
    gen_profile_low: np.ndarray | None = None
    gen_profile_high: np.ndarray | None = None
    mae_ci: tuple[float, float] | None = None
    mae_null_q95: float | None = None
    mae_pvalue: float | None = None


@dataclass
class RadialClassInference:
    real_profile_low: np.ndarray | None = None
    real_profile_high: np.ndarray | None = None
    model: Dict[int, RadialModelInference] = field(default_factory=dict)
    supports_image_level_stats: bool = False


@dataclass
class PSDPanel:
    class_id: int
    label: str
    frequency: np.ndarray
    real_profile: np.ndarray
    gen_profile: np.ndarray
    l2: float
    real_profiles: np.ndarray | None = None
    gen_profiles: np.ndarray | None = None


@dataclass
class PSDModelInference:
    gen_profile_low: np.ndarray | None = None
    gen_profile_high: np.ndarray | None = None
    l2_ci: tuple[float, float] | None = None
    l2_null_q95: float | None = None
    l2_pvalue: float | None = None


@dataclass
class PSDClassInference:
    real_profile_low: np.ndarray | None = None
    real_profile_high: np.ndarray | None = None
    model: Dict[int, PSDModelInference] = field(default_factory=dict)
    supports_image_level_stats: bool = False


@dataclass
class DAVPanel:
    class_id: int
    label: str
    real_values: np.ndarray
    gen_values: np.ndarray
    real_mean: float
    gen_mean: float
    abs_gap_deg2: float


@dataclass
class DAVModelInference:
    gen_density_low: np.ndarray | None = None
    gen_density_high: np.ndarray | None = None
    gap_ci: tuple[float, float] | None = None
    gap_null_q95: float | None = None
    gap_pvalue: float | None = None


@dataclass
class DAVClassInference:
    real_density_low: np.ndarray | None = None
    real_density_high: np.ndarray | None = None
    model: Dict[int, DAVModelInference] = field(default_factory=dict)
    supports_image_level_stats: bool = False


@dataclass
class ColdCloudPanel:
    class_id: int
    label: str
    threshold_k: float
    radius_km: float | None
    pixel_size_km: float | None
    real_values: np.ndarray
    gen_values: np.ndarray
    real_mean_fraction: float
    gen_mean_fraction: float
    abs_gap_fraction: float


@dataclass
class ColdCloudModelInference:
    gen_density_low: np.ndarray | None = None
    gen_density_high: np.ndarray | None = None
    gap_ci: tuple[float, float] | None = None
    gap_null_q95: float | None = None
    gap_pvalue: float | None = None


@dataclass
class ColdCloudClassInference:
    real_density_low: np.ndarray | None = None
    real_density_high: np.ndarray | None = None
    model: Dict[int, ColdCloudModelInference] = field(default_factory=dict)
    supports_image_level_stats: bool = False


@dataclass
class MemorizationPair:
    class_id: int
    label: str
    rank_within_class: int
    distance: float
    generated_index: int
    train_index: int
    train_rel_path: str
    generated_bt: np.ndarray
    train_bt: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready evaluation figures from one to three saved evaluation reports. "
            "The script reads saved metrics.json summaries plus any linked sidecar artifacts; "
            "evaluation itself should be run ahead of time. By default, the standard quantitative "
            "paper-ready figures are emitted as separate files; qualitative memorization pairs are opt-in."
        )
    )
    parser.add_argument(
        "--plot_only",
        type=str,
        choices=list(ALL_FIGURES),
        default=None,
        help=(
            "Optional single-figure mode. If omitted, the script renders the standard quantitative plots. "
            f"{FIGURE_PIXEL!r} reproduces the BT histogram plot; "
            f"{FIGURE_RADIAL!r} renders the azimuthally averaged radial BT profiles; "
            f"{FIGURE_PSD!r} renders the radially averaged power spectral density profiles; "
            f"{FIGURE_DAV!r} renders the deviation-angle-variance distributions; "
            f"{FIGURE_COLD!r} renders the cold-cloud-fraction distributions; "
            f"{FIGURE_MEMORIZATION!r} renders qualitative closest generated/train pairs."
        ),
    )
    parser.add_argument(
        "--figure",
        type=str,
        choices=list(ALL_FIGURES),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Repeatable model input spec. Format: [Label=]RUN_NAME[:EVAL_REF]. "
            "If EVAL_REF is omitted, defaults to test/post_training. "
            "Examples: FINAL_BASELINE:test/post_training or "
            "PINN=baseline_pinn:test/guidance_0p5."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Legacy single-model alias for --input <name>:<split>/<eval_tag>.",
    )
    parser.add_argument(
        "--eval_tag",
        type=str,
        default="post_training",
        help="Legacy single-model eval tag used with --name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Legacy single-model split used with --name. Default is test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path. When rendering a single figure, this can be either a file path or a directory. "
            "When rendering all figures, this must be a directory. Defaults to outputs/<model>/paper_ready/ "
            "for a single input or outputs/paper_ready/ for multi-model comparisons."
        ),
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.6,
        help="Gaussian smoothing sigma in histogram bins for displayed pixel-histogram curves only.",
    )
    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=300,
        help="Image-level bootstrap repetitions for real ribbons and model-vs-real confidence intervals.",
    )
    parser.add_argument(
        "--null_reps",
        type=int,
        default=400,
        help="Image-level real-vs-real null repetitions used for matched p-values and null references.",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level for image-level bootstrap intervals. Default is 0.95.",
    )
    parser.add_argument(
        "--stats_seed",
        type=int,
        default=123,
        help="Random seed for image-level bootstrap and null calculations.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _first_present(mapping: Dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _ensure_1d_array(value: Any, *, field_name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required histogram field: {field_name}")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Field {field_name!r} could not be converted to a numeric array.") from exc
    if arr.ndim != 1:
        raise ValueError(f"Field {field_name!r} must be a 1D array, got shape {arr.shape}.")
    return arr


def _ensure_2d_array(value: Any, *, field_name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required array field: {field_name}")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Field {field_name!r} could not be converted to a numeric array.") from exc
    if arr.ndim != 2:
        raise ValueError(f"Field {field_name!r} must be a 2D array, got shape {arr.shape}.")
    return arr


def _ensure_3d_array(value: Any, *, field_name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required array field: {field_name}")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Field {field_name!r} could not be converted to a numeric array.") from exc
    if arr.ndim != 3:
        raise ValueError(f"Field {field_name!r} must be a 3D array, got shape {arr.shape}.")
    return arr


def _ensure_1d_string_array(value: Any, *, field_name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing required string array field: {field_name}")
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"Field {field_name!r} must be a 1D string array, got shape {arr.shape}.")
    return arr.astype(str)


def _ensure_offsets_array(value: Any, *, field_name: str) -> np.ndarray:
    arr = _ensure_1d_array(value, field_name=field_name).astype(np.int64)
    if arr.size < 2:
        raise ValueError(f"Field {field_name!r} must contain at least two offsets.")
    if arr[0] != 0:
        raise ValueError(f"Field {field_name!r} must start at 0, got {arr[0]}.")
    if np.any(np.diff(arr) < 0):
        raise ValueError(f"Field {field_name!r} must be monotonically nondecreasing.")
    return arr


def _infer_hist_representation(hist: np.ndarray, bins: np.ndarray, normalization: str | None) -> str:
    hint = (normalization or "").strip().lower()
    if hint in {"probability_mass", "prob_mass", "mass", "pmf", "probability"}:
        return "mass"
    if hint in {"density", "pdf"}:
        return "density"

    widths = np.diff(bins)
    mass_sum = float(np.sum(hist))
    density_integral = float(np.sum(hist * widths))
    if np.isclose(density_integral, 1.0, rtol=1e-2, atol=1e-3):
        return "density"
    if np.isclose(mass_sum, 1.0, rtol=1e-2, atol=1e-3):
        return "mass"
    return "counts"


def _hist_to_mass(hist: np.ndarray, bins: np.ndarray, normalization: str | None) -> np.ndarray:
    widths = np.diff(bins)
    if np.any(widths <= 0):
        raise ValueError("Histogram bins must be strictly increasing.")

    representation = _infer_hist_representation(hist, bins, normalization)
    if representation == "density":
        mass = hist * widths
    elif representation == "mass":
        mass = hist
    else:
        total = float(np.sum(hist))
        if total <= 0.0:
            raise ValueError("Histogram counts must sum to a positive value.")
        mass = hist / total

    total_mass = float(np.sum(mass))
    if total_mass <= 0.0:
        raise ValueError("Histogram mass must sum to a positive value.")
    return mass / total_mass


def _mass_to_density(mass: np.ndarray, bins: np.ndarray) -> np.ndarray:
    widths = np.diff(bins)
    if np.any(widths <= 0):
        raise ValueError("Histogram bins must be strictly increasing.")
    return mass / widths


def _normalize_mass(mass: np.ndarray, *, field_name: str) -> np.ndarray:
    arr = np.asarray(mass, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(f"Field {field_name!r} must sum to a positive value.")
    return arr / total


def _extract_inline_pixel_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        ("paper_ready", "pixel_plausibility"),
        ("paper_ready", "pixel_histogram_plausibility"),
        ("pixel_plausibility",),
        ("pixel_histograms",),
    ]
    for path in candidates:
        node: Any = report
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if ok and isinstance(node, dict):
            return node

    raise ValueError(
        "The metrics JSON does not contain saved histogram curves for paper-ready pixel plausibility "
        "plots. Expected report['paper_ready']['pixel_plausibility'] with 'bt_bins', "
        "'hist_real', and 'hist_gen'."
    )


def _resolve_artifact_path(metrics_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return metrics_path.parent / path


def _default_label(class_id: int) -> str:
    return DEFAULT_CLASS_LABELS.get(class_id, f"Class {class_id}")


def _build_panel_from_mass(
    *,
    class_id: int,
    label: str,
    bins: np.ndarray,
    real_mass: np.ndarray,
    gen_mass: np.ndarray,
    js: float | None = None,
    w1: float | None = None,
    real_hist_counts: np.ndarray | None = None,
    gen_hist_counts: np.ndarray | None = None,
) -> ClassPanel:
    bins = _ensure_1d_array(bins, field_name=f"per_class.{class_id}.bt_bins")
    real_mass = _normalize_mass(real_mass, field_name=f"per_class.{class_id}.real_mass")
    gen_mass = _normalize_mass(gen_mass, field_name=f"per_class.{class_id}.gen_mass")
    if bins.size != real_mass.size + 1 or bins.size != gen_mass.size + 1:
        raise ValueError(
            f"Histogram dimensions are inconsistent for class {class_id}: "
            f"len(bins)={bins.size}, len(real_mass)={real_mass.size}, len(gen_mass)={gen_mass.size}."
        )

    real_density = _mass_to_density(real_mass, bins)
    gen_density = _mass_to_density(gen_mass, bins)
    if js is None:
        js = js_divergence(real_mass, gen_mass)
    if w1 is None:
        w1 = wasserstein1_from_hist(real_mass, gen_mass, bin_edges=bins)

    return ClassPanel(
        class_id=class_id,
        label=label,
        bins=bins,
        real_mass=real_mass,
        gen_mass=gen_mass,
        real_density=real_density,
        gen_density=gen_density,
        js=float(js),
        w1=float(w1),
        real_hist_counts=None if real_hist_counts is None else np.asarray(real_hist_counts, dtype=np.float64),
        gen_hist_counts=None if gen_hist_counts is None else np.asarray(gen_hist_counts, dtype=np.float64),
    )


def _build_panel_from_inline_entry(
    entry: Dict[str, Any],
    *,
    class_id: int,
    label: str,
    normalization: str | None,
    fallback_metrics: Dict[str, Any] | None = None,
) -> ClassPanel:
    bins = _ensure_1d_array(
        _first_present(entry, ("bt_bins", "bin_edges_k", "bin_edges", "bins")),
        field_name=f"per_class.{class_id}.bt_bins",
    )
    hist_real = _ensure_1d_array(
        _first_present(entry, ("hist_real", "real_hist", "real")),
        field_name=f"per_class.{class_id}.hist_real",
    )
    hist_gen = _ensure_1d_array(
        _first_present(entry, ("hist_gen", "gen_hist", "generated_hist", "generated")),
        field_name=f"per_class.{class_id}.hist_gen",
    )

    if bins.size != hist_real.size + 1 or bins.size != hist_gen.size + 1:
        raise ValueError(
            f"Histogram dimensions are inconsistent for class {class_id}: "
            f"len(bins)={bins.size}, len(hist_real)={hist_real.size}, len(hist_gen)={hist_gen.size}."
        )

    norm_hint = normalization or entry.get("hist_normalization")
    real_mass = _hist_to_mass(hist_real, bins, norm_hint)
    gen_mass = _hist_to_mass(hist_gen, bins, norm_hint)

    metric_source = entry if not fallback_metrics else {**fallback_metrics, **entry}
    js = _first_present(metric_source, ("pixel_hist_js", "js_divergence", "js"))
    w1 = _first_present(metric_source, ("pixel_hist_w1", "wasserstein1", "w1"))
    if js is None:
        js = js_divergence(real_mass, gen_mass)
    if w1 is None:
        w1 = wasserstein1_from_hist(real_mass, gen_mass, bin_edges=bins)

    return _build_panel_from_mass(
        class_id=class_id,
        label=label,
        bins=bins,
        real_mass=real_mass,
        gen_mass=gen_mass,
        js=float(js),
        w1=float(w1),
    )


def _infer_split_from_eval_ref(eval_ref: str) -> str:
    rel = Path(eval_ref.strip())
    if rel.parts and rel.parts[0] in {"val", "test"}:
        return rel.parts[0]
    if rel.parts and len(rel.parts) >= 2 and rel.parts[0] == "eval" and rel.parts[1] in {"val", "test"}:
        return rel.parts[1]
    return "val"


def _resolve_metrics_path(repo_root: Path, run_name: str, eval_ref: str) -> Path:
    model_roots = [
        model_outputs_root(repo_root, run_name),
        repo_root / "runs" / run_name,
    ]
    existing_roots = [root for root in model_roots if root.exists()]
    if not existing_roots:
        tried_roots = "\n".join(f"  - {root}" for root in model_roots)
        raise FileNotFoundError(
            f"Could not find any model directory for run {run_name!r}. Tried:\n{tried_roots}"
        )

    eval_ref = eval_ref.strip() or "test/post_training"
    rel = Path(eval_ref)
    if rel.is_absolute() or str(rel).endswith(".json") or rel.name == "metrics.json":
        raise ValueError(
            "Input specs should point to a run-relative evaluation folder, not directly to metrics.json."
        )

    candidates = []
    for model_root in existing_roots:
        if rel.parts and rel.parts[0] == "eval":
            candidates.append(model_root / rel / "metrics.json")
        else:
            candidates.append(model_root / "eval" / rel / "metrics.json")
            if rel.parts and rel.parts[0] == "val" and len(rel.parts) >= 2:
                candidates.append(model_root / "eval" / Path(*rel.parts[1:]) / "metrics.json")
            candidates.append(model_root / rel / "metrics.json")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        f"Could not locate metrics.json for run {run_name!r} and eval ref {eval_ref!r}. Tried:\n{tried}"
    )


def _parse_input_spec(raw_spec: str) -> InputSpec:
    spec = raw_spec.strip()
    if not spec:
        raise ValueError("Empty --input spec.")

    label = None
    if "=" in spec:
        maybe_label, remainder = spec.split("=", 1)
        if maybe_label.strip() and remainder.strip():
            label = maybe_label.strip()
            spec = remainder.strip()

    if ":" in spec:
        run_name, eval_ref = spec.split(":", 1)
    else:
        run_name, eval_ref = spec, "test/post_training"

    run_name = run_name.strip()
    eval_ref = eval_ref.strip() or "test/post_training"
    if not run_name:
        raise ValueError(f"Invalid --input spec {raw_spec!r}: missing run name.")
    if label is None:
        label = run_name
    return InputSpec(run_name=run_name, eval_ref=eval_ref, label=label)


def _resolve_inputs(args: argparse.Namespace) -> list[InputSpec]:
    if args.input and args.name:
        raise ValueError("Use either repeatable --input specs or the legacy --name/--eval_tag flags, not both.")

    if args.input:
        raw_specs = args.input
    elif args.name:
        raw_specs = [f"{args.name}:{args.split}/{args.eval_tag}"]
    else:
        raw_specs = ["FINAL_BASELINE:test/post_training"]

    specs = [_parse_input_spec(spec) for spec in raw_specs]
    if not (1 <= len(specs) <= 3):
        raise ValueError(f"paper_ready_plots.py supports between 1 and 3 model inputs, got {len(specs)}.")
    return specs


def _load_panels_from_npz_artifact(
    *,
    metrics_path: Path,
    manifest: Dict[str, Any],
    per_class_metrics: Dict[str, Any],
) -> Dict[int, ClassPanel]:
    raw_path = manifest.get("path")
    if not raw_path:
        raise ValueError(f"Pixel plausibility artifact manifest in {metrics_path} is missing a path.")
    artifact_path = _resolve_artifact_path(metrics_path, str(raw_path))
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Pixel plausibility artifact not found for {metrics_path}: {artifact_path}"
        )

    with np.load(artifact_path, allow_pickle=False) as data:
        class_ids = _ensure_1d_array(data["class_ids"], field_name="class_ids").astype(int)
        bins = _ensure_1d_array(data["bt_bins"], field_name="bt_bins")
        real_mass_rows = _ensure_2d_array(data["real_probability_mass"], field_name="real_probability_mass")
        gen_mass_rows = _ensure_2d_array(data["gen_probability_mass"], field_name="gen_probability_mass")
        class_labels_arr = (
            _ensure_1d_string_array(data["class_labels"], field_name="class_labels")
            if "class_labels" in data.files
            else np.asarray([], dtype=str)
        )
        js_rows = (
            _ensure_1d_array(data["pixel_hist_js"], field_name="pixel_hist_js")
            if "pixel_hist_js" in data.files
            else None
        )
        w1_rows = (
            _ensure_1d_array(data["pixel_hist_w1_k"], field_name="pixel_hist_w1_k")
            if "pixel_hist_w1_k" in data.files
            else None
        )
        real_hist_counts_flat = (
            _ensure_2d_array(data["real_hist_counts_flat"], field_name="real_hist_counts_flat")
            if "real_hist_counts_flat" in data.files
            else None
        )
        gen_hist_counts_flat = (
            _ensure_2d_array(data["gen_hist_counts_flat"], field_name="gen_hist_counts_flat")
            if "gen_hist_counts_flat" in data.files
            else None
        )
        real_class_offsets = (
            _ensure_offsets_array(data["real_class_offsets"], field_name="real_class_offsets")
            if "real_class_offsets" in data.files
            else None
        )
        gen_class_offsets = (
            _ensure_offsets_array(data["gen_class_offsets"], field_name="gen_class_offsets")
            if "gen_class_offsets" in data.files
            else None
        )

    n_classes = class_ids.size
    if real_mass_rows.shape[0] != n_classes or gen_mass_rows.shape[0] != n_classes:
        raise ValueError(
            f"Artifact {artifact_path} has inconsistent class dimensions: "
            f"class_ids={n_classes}, real={real_mass_rows.shape}, gen={gen_mass_rows.shape}."
        )
    if real_mass_rows.shape[1] != bins.size - 1 or gen_mass_rows.shape[1] != bins.size - 1:
        raise ValueError(
            f"Artifact {artifact_path} has inconsistent bin dimensions: "
            f"len(bt_bins)={bins.size}, real={real_mass_rows.shape}, gen={gen_mass_rows.shape}."
        )
    if class_labels_arr.size not in {0, n_classes}:
        raise ValueError(
            f"Artifact {artifact_path} has {class_labels_arr.size} labels for {n_classes} classes."
        )
    if js_rows is not None and js_rows.size != n_classes:
        raise ValueError(f"Artifact {artifact_path} has {js_rows.size} JS values for {n_classes} classes.")
    if w1_rows is not None and w1_rows.size != n_classes:
        raise ValueError(f"Artifact {artifact_path} has {w1_rows.size} W1 values for {n_classes} classes.")
    if real_hist_counts_flat is not None:
        if real_class_offsets is None:
            raise ValueError(f"Artifact {artifact_path} includes real_hist_counts_flat without real_class_offsets.")
        if real_hist_counts_flat.shape[1] != bins.size - 1:
            raise ValueError(
                f"Artifact {artifact_path} real_hist_counts_flat has width {real_hist_counts_flat.shape[1]}, "
                f"expected {bins.size - 1}."
            )
        if real_class_offsets.size != n_classes + 1:
            raise ValueError(
                f"Artifact {artifact_path} has {real_class_offsets.size} real offsets for {n_classes} classes."
            )
        if int(real_class_offsets[-1]) != int(real_hist_counts_flat.shape[0]):
            raise ValueError(
                f"Artifact {artifact_path} real offsets end at {real_class_offsets[-1]}, "
                f"but real_hist_counts_flat has {real_hist_counts_flat.shape[0]} rows."
            )
    if gen_hist_counts_flat is not None:
        if gen_class_offsets is None:
            raise ValueError(f"Artifact {artifact_path} includes gen_hist_counts_flat without gen_class_offsets.")
        if gen_hist_counts_flat.shape[1] != bins.size - 1:
            raise ValueError(
                f"Artifact {artifact_path} gen_hist_counts_flat has width {gen_hist_counts_flat.shape[1]}, "
                f"expected {bins.size - 1}."
            )
        if gen_class_offsets.size != n_classes + 1:
            raise ValueError(
                f"Artifact {artifact_path} has {gen_class_offsets.size} generated offsets for {n_classes} classes."
            )
        if int(gen_class_offsets[-1]) != int(gen_hist_counts_flat.shape[0]):
            raise ValueError(
                f"Artifact {artifact_path} generated offsets end at {gen_class_offsets[-1]}, "
                f"but gen_hist_counts_flat has {gen_hist_counts_flat.shape[0]} rows."
            )

    manifest_labels = manifest.get("class_labels", {})
    if not isinstance(manifest_labels, dict):
        manifest_labels = {}

    panels: Dict[int, ClassPanel] = {}
    for idx, class_id in enumerate(class_ids.tolist()):
        fallback_metrics = per_class_metrics.get(str(class_id))
        fallback_metrics = fallback_metrics if isinstance(fallback_metrics, dict) else {}
        js = float(js_rows[idx]) if js_rows is not None else _first_present(fallback_metrics, ("pixel_hist_js",))
        w1 = float(w1_rows[idx]) if w1_rows is not None else _first_present(fallback_metrics, ("pixel_hist_w1",))
        label = str(
            manifest_labels.get(str(class_id))
            or (class_labels_arr[idx] if class_labels_arr.size else "")
            or _default_label(class_id)
        )
        real_hist_counts = None
        gen_hist_counts = None
        if real_hist_counts_flat is not None and real_class_offsets is not None:
            start = int(real_class_offsets[idx])
            stop = int(real_class_offsets[idx + 1])
            real_hist_counts = real_hist_counts_flat[start:stop]
        if gen_hist_counts_flat is not None and gen_class_offsets is not None:
            start = int(gen_class_offsets[idx])
            stop = int(gen_class_offsets[idx + 1])
            gen_hist_counts = gen_hist_counts_flat[start:stop]
        panels[class_id] = _build_panel_from_mass(
            class_id=class_id,
            label=label,
            bins=bins,
            real_mass=real_mass_rows[idx],
            gen_mass=gen_mass_rows[idx],
            js=js,
            w1=w1,
            real_hist_counts=real_hist_counts,
            gen_hist_counts=gen_hist_counts,
        )

    return panels


def _load_pixel_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    per_class_metrics = report.get("per_class", {})
    paper_ready = report.get("paper_ready", {})
    pixel_ref = paper_ready.get("pixel_plausibility") if isinstance(paper_ready, dict) else None

    if isinstance(pixel_ref, dict) and "path" in pixel_ref:
        panels = _load_panels_from_npz_artifact(
            metrics_path=metrics_path,
            manifest=pixel_ref,
            per_class_metrics=per_class_metrics if isinstance(per_class_metrics, dict) else {},
        )
    else:
        payload = _extract_inline_pixel_payload(report)
        normalization = payload.get("hist_normalization")

        class_labels_raw = payload.get("class_labels", {})
        class_labels = {}
        if isinstance(class_labels_raw, dict):
            for key, value in class_labels_raw.items():
                try:
                    class_labels[int(key)] = str(value)
                except Exception:
                    continue

        per_class_payload = payload.get("per_class")
        if not isinstance(per_class_payload, dict) or not per_class_payload:
            raise ValueError(f"Missing per-class histogram payload in {metrics_path}.")

        panels = {}
        for class_id in sorted(int(class_id) for class_id in per_class_payload.keys()):
            entry = per_class_payload[str(class_id)]
            if not isinstance(entry, dict):
                raise ValueError(f"per_class[{class_id!r}] must be a JSON object in {metrics_path}.")
            fallback_metrics = per_class_metrics.get(str(class_id))
            label = str(entry.get("label") or class_labels.get(class_id) or _default_label(class_id))
            panels[class_id] = _build_panel_from_inline_entry(
                entry,
                class_id=class_id,
                label=label,
                normalization=normalization,
                fallback_metrics=fallback_metrics if isinstance(fallback_metrics, dict) else None,
            )

    class_ids = sorted(panels.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(f"Expected classes 0..5 in {metrics_path}, found {class_ids}.")

    split = str(report.get("split") or _infer_split_from_eval_ref(spec.eval_ref)).strip().lower()
    return LoadedReport(
        figure_kind=FIGURE_PIXEL,
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _build_radial_panel(
    *,
    class_id: int,
    label: str,
    radius: np.ndarray,
    real_profile: np.ndarray,
    gen_profile: np.ndarray,
    mae_k: float | None = None,
    real_profiles: np.ndarray | None = None,
    gen_profiles: np.ndarray | None = None,
) -> RadialPanel:
    radius = _ensure_1d_array(radius, field_name=f"per_class.{class_id}.radius_normalized")
    real_profile = _ensure_1d_array(real_profile, field_name=f"per_class.{class_id}.radial_mean_real_mu")
    gen_profile = _ensure_1d_array(gen_profile, field_name=f"per_class.{class_id}.radial_mean_gen_mu")
    if radius.size != real_profile.size or radius.size != gen_profile.size:
        raise ValueError(
            f"Radial profile dimensions are inconsistent for class {class_id}: "
            f"len(radius)={radius.size}, len(real)={real_profile.size}, len(gen)={gen_profile.size}."
        )
    if mae_k is None:
        mae_k = float(np.mean(np.abs(real_profile - gen_profile)))
    return RadialPanel(
        class_id=class_id,
        label=label,
        radius=radius,
        real_profile=np.asarray(real_profile, dtype=np.float64),
        gen_profile=np.asarray(gen_profile, dtype=np.float64),
        mae_k=float(mae_k),
        real_profiles=None if real_profiles is None else np.asarray(real_profiles, dtype=np.float64),
        gen_profiles=None if gen_profiles is None else np.asarray(gen_profiles, dtype=np.float64),
    )


def _load_radial_panels_from_npz_artifact(
    *,
    metrics_path: Path,
    manifest: Dict[str, Any],
) -> Dict[int, RadialPanel]:
    raw_path = manifest.get("path")
    if not raw_path:
        raise ValueError(f"Radial BT profile artifact manifest in {metrics_path} is missing a path.")
    artifact_path = _resolve_artifact_path(metrics_path, str(raw_path))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Radial BT profile artifact not found for {metrics_path}: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        class_ids = _ensure_1d_array(data["class_ids"], field_name="class_ids").astype(int)
        radius = _ensure_1d_array(data["radius_normalized"], field_name="radius_normalized")
        real_mean_rows = _ensure_2d_array(data["real_mean_profile_k"], field_name="real_mean_profile_k")
        gen_mean_rows = _ensure_2d_array(data["gen_mean_profile_k"], field_name="gen_mean_profile_k")
        class_labels_arr = (
            _ensure_1d_string_array(data["class_labels"], field_name="class_labels")
            if "class_labels" in data.files
            else np.asarray([], dtype=str)
        )
        mae_rows = (
            _ensure_1d_array(data["radial_profile_mae_k"], field_name="radial_profile_mae_k")
            if "radial_profile_mae_k" in data.files
            else None
        )
        real_profiles_flat = (
            _ensure_2d_array(data["real_mean_profiles_flat"], field_name="real_mean_profiles_flat")
            if "real_mean_profiles_flat" in data.files
            else None
        )
        gen_profiles_flat = (
            _ensure_2d_array(data["gen_mean_profiles_flat"], field_name="gen_mean_profiles_flat")
            if "gen_mean_profiles_flat" in data.files
            else None
        )
        real_class_offsets = (
            _ensure_offsets_array(data["real_class_offsets"], field_name="real_class_offsets")
            if "real_class_offsets" in data.files
            else None
        )
        gen_class_offsets = (
            _ensure_offsets_array(data["gen_class_offsets"], field_name="gen_class_offsets")
            if "gen_class_offsets" in data.files
            else None
        )

    n_classes = class_ids.size
    if real_mean_rows.shape != gen_mean_rows.shape or real_mean_rows.shape[0] != n_classes:
        raise ValueError(
            f"Artifact {artifact_path} has inconsistent radial profile dimensions: "
            f"class_ids={n_classes}, real={real_mean_rows.shape}, gen={gen_mean_rows.shape}."
        )
    if real_mean_rows.shape[1] != radius.size:
        raise ValueError(
            f"Artifact {artifact_path} has profile width {real_mean_rows.shape[1]} for radius size {radius.size}."
        )
    if class_labels_arr.size not in {0, n_classes}:
        raise ValueError(f"Artifact {artifact_path} has {class_labels_arr.size} labels for {n_classes} classes.")
    if mae_rows is not None and mae_rows.size != n_classes:
        raise ValueError(f"Artifact {artifact_path} has {mae_rows.size} MAE values for {n_classes} classes.")
    if real_profiles_flat is not None:
        if real_class_offsets is None:
            raise ValueError(f"Artifact {artifact_path} includes real_mean_profiles_flat without offsets.")
        if real_profiles_flat.shape[1] != radius.size:
            raise ValueError(
                f"Artifact {artifact_path} real_mean_profiles_flat has width {real_profiles_flat.shape[1]}, "
                f"expected {radius.size}."
            )
        if real_class_offsets.size != n_classes + 1 or int(real_class_offsets[-1]) != int(real_profiles_flat.shape[0]):
            raise ValueError(f"Artifact {artifact_path} has inconsistent real profile offsets.")
    if gen_profiles_flat is not None:
        if gen_class_offsets is None:
            raise ValueError(f"Artifact {artifact_path} includes gen_mean_profiles_flat without offsets.")
        if gen_profiles_flat.shape[1] != radius.size:
            raise ValueError(
                f"Artifact {artifact_path} gen_mean_profiles_flat has width {gen_profiles_flat.shape[1]}, "
                f"expected {radius.size}."
            )
        if gen_class_offsets.size != n_classes + 1 or int(gen_class_offsets[-1]) != int(gen_profiles_flat.shape[0]):
            raise ValueError(f"Artifact {artifact_path} has inconsistent generated profile offsets.")

    manifest_labels = manifest.get("class_labels", {})
    if not isinstance(manifest_labels, dict):
        manifest_labels = {}

    panels: Dict[int, RadialPanel] = {}
    for idx, class_id in enumerate(class_ids.tolist()):
        label = str(
            manifest_labels.get(str(class_id))
            or (class_labels_arr[idx] if class_labels_arr.size else "")
            or _default_label(class_id)
        )
        real_profiles = None
        gen_profiles = None
        if real_profiles_flat is not None and real_class_offsets is not None:
            start = int(real_class_offsets[idx])
            stop = int(real_class_offsets[idx + 1])
            real_profiles = real_profiles_flat[start:stop]
        if gen_profiles_flat is not None and gen_class_offsets is not None:
            start = int(gen_class_offsets[idx])
            stop = int(gen_class_offsets[idx + 1])
            gen_profiles = gen_profiles_flat[start:stop]
        panels[class_id] = _build_radial_panel(
            class_id=class_id,
            label=label,
            radius=radius,
            real_profile=real_mean_rows[idx],
            gen_profile=gen_mean_rows[idx],
            mae_k=float(mae_rows[idx]) if mae_rows is not None else None,
            real_profiles=real_profiles,
            gen_profiles=gen_profiles,
        )
    return panels


def _load_radial_panels_from_report_metrics(metrics_path: Path, per_class_metrics: Dict[str, Any]) -> Dict[int, RadialPanel]:
    if not isinstance(per_class_metrics, dict) or not per_class_metrics:
        raise ValueError(f"Missing per-class metrics payload in {metrics_path}.")

    first_entry = per_class_metrics.get("0")
    if not isinstance(first_entry, dict) or "radial_mean_real_mu" not in first_entry or "radial_mean_gen_mu" not in first_entry:
        raise ValueError(
            "The metrics JSON does not contain saved radial BT profile curves for paper-ready plotting. "
            "Re-run eval with the updated code so it writes paper_ready.radial_bt_profile."
        )

    panels: Dict[int, RadialPanel] = {}
    radius_ref: np.ndarray | None = None
    for class_id in range(6):
        entry = per_class_metrics.get(str(class_id))
        if not isinstance(entry, dict):
            raise ValueError(f"Missing per_class[{class_id!r}] metrics in {metrics_path}.")
        real_profile = _ensure_1d_array(
            entry.get("radial_mean_real_mu"),
            field_name=f"per_class.{class_id}.radial_mean_real_mu",
        )
        gen_profile = _ensure_1d_array(
            entry.get("radial_mean_gen_mu"),
            field_name=f"per_class.{class_id}.radial_mean_gen_mu",
        )
        radius = np.linspace(0.0, 1.0, real_profile.size, dtype=np.float64)
        if radius_ref is None:
            radius_ref = radius
        elif radius.shape != radius_ref.shape or not np.allclose(radius, radius_ref):
            raise ValueError(f"Radial profile widths differ across classes in {metrics_path}.")
        panels[class_id] = _build_radial_panel(
            class_id=class_id,
            label=_default_label(class_id),
            radius=radius,
            real_profile=real_profile,
            gen_profile=gen_profile,
        )
    return panels


def _load_radial_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    per_class_metrics = report.get("per_class", {})
    paper_ready = report.get("paper_ready", {})
    radial_ref = paper_ready.get("radial_bt_profile") if isinstance(paper_ready, dict) else None

    if isinstance(radial_ref, dict) and "path" in radial_ref:
        panels = _load_radial_panels_from_npz_artifact(metrics_path=metrics_path, manifest=radial_ref)
    else:
        panels = _load_radial_panels_from_report_metrics(
            metrics_path,
            per_class_metrics if isinstance(per_class_metrics, dict) else {},
        )

    class_ids = sorted(panels.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(f"Expected classes 0..5 in {metrics_path}, found {class_ids}.")

    split = str(report.get("split") or _infer_split_from_eval_ref(spec.eval_ref)).strip().lower()
    return LoadedReport(
        figure_kind=FIGURE_RADIAL,
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _build_psd_panel(
    *,
    class_id: int,
    label: str,
    frequency: np.ndarray,
    real_profile: np.ndarray,
    gen_profile: np.ndarray,
    l2: float | None = None,
    real_profiles: np.ndarray | None = None,
    gen_profiles: np.ndarray | None = None,
) -> PSDPanel:
    frequency = _ensure_1d_array(frequency, field_name=f"per_class.{class_id}.frequency_normalized")
    real_profile = _ensure_1d_array(real_profile, field_name=f"per_class.{class_id}.real_psd_profile")
    gen_profile = _ensure_1d_array(gen_profile, field_name=f"per_class.{class_id}.gen_psd_profile")
    if frequency.size != real_profile.size or frequency.size != gen_profile.size:
        raise ValueError(
            f"PSD profile dimensions are inconsistent for class {class_id}: "
            f"len(frequency)={frequency.size}, len(real)={real_profile.size}, len(gen)={gen_profile.size}."
        )
    if l2 is None:
        l2 = float(np.mean(np.square(real_profile - gen_profile)))
    return PSDPanel(
        class_id=class_id,
        label=label,
        frequency=frequency,
        real_profile=np.asarray(real_profile, dtype=np.float64),
        gen_profile=np.asarray(gen_profile, dtype=np.float64),
        l2=float(l2),
        real_profiles=None if real_profiles is None else np.asarray(real_profiles, dtype=np.float64),
        gen_profiles=None if gen_profiles is None else np.asarray(gen_profiles, dtype=np.float64),
    )


def _load_psd_panels_from_npz_artifact(
    *,
    metrics_path: Path,
    manifest: Dict[str, Any],
) -> Dict[int, PSDPanel]:
    raw_path = manifest.get("path")
    if not raw_path:
        raise ValueError(f"PSD artifact manifest in {metrics_path} is missing a path.")
    artifact_path = _resolve_artifact_path(metrics_path, str(raw_path))
    if not artifact_path.exists():
        raise FileNotFoundError(f"PSD artifact not found for {metrics_path}: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        class_ids = _ensure_1d_array(data["class_ids"], field_name="class_ids").astype(int)
        frequency = _ensure_1d_array(data["frequency_normalized"], field_name="frequency_normalized")
        real_mean_rows = _ensure_2d_array(data["real_mean_psd_log10"], field_name="real_mean_psd_log10")
        gen_mean_rows = _ensure_2d_array(data["gen_mean_psd_log10"], field_name="gen_mean_psd_log10")
        class_labels_arr = (
            _ensure_1d_string_array(data["class_labels"], field_name="class_labels")
            if "class_labels" in data.files
            else np.asarray([], dtype=str)
        )
        l2_rows = (
            _ensure_1d_array(data["psd_l2"], field_name="psd_l2")
            if "psd_l2" in data.files
            else None
        )
        real_profiles_flat = (
            _ensure_2d_array(data["real_psd_profiles_flat"], field_name="real_psd_profiles_flat")
            if "real_psd_profiles_flat" in data.files
            else None
        )
        gen_profiles_flat = (
            _ensure_2d_array(data["gen_psd_profiles_flat"], field_name="gen_psd_profiles_flat")
            if "gen_psd_profiles_flat" in data.files
            else None
        )
        real_class_offsets = (
            _ensure_offsets_array(data["real_class_offsets"], field_name="real_class_offsets")
            if "real_class_offsets" in data.files
            else None
        )
        gen_class_offsets = (
            _ensure_offsets_array(data["gen_class_offsets"], field_name="gen_class_offsets")
            if "gen_class_offsets" in data.files
            else None
        )

    n_classes = class_ids.size
    if real_mean_rows.shape != gen_mean_rows.shape or real_mean_rows.shape[0] != n_classes:
        raise ValueError(
            f"Artifact {artifact_path} has inconsistent PSD profile dimensions: "
            f"class_ids={n_classes}, real={real_mean_rows.shape}, gen={gen_mean_rows.shape}."
        )
    if real_mean_rows.shape[1] != frequency.size:
        raise ValueError(
            f"Artifact {artifact_path} has profile width {real_mean_rows.shape[1]} "
            f"for frequency size {frequency.size}."
        )
    if class_labels_arr.size not in {0, n_classes}:
        raise ValueError(f"Artifact {artifact_path} has {class_labels_arr.size} labels for {n_classes} classes.")
    if l2_rows is not None and l2_rows.size != n_classes:
        raise ValueError(f"Artifact {artifact_path} has {l2_rows.size} PSD L2 values for {n_classes} classes.")
    if real_profiles_flat is not None:
        if real_class_offsets is None:
            raise ValueError(f"Artifact {artifact_path} includes real_psd_profiles_flat without offsets.")
        if real_profiles_flat.shape[1] != frequency.size:
            raise ValueError(
                f"Artifact {artifact_path} real_psd_profiles_flat has width {real_profiles_flat.shape[1]}, "
                f"expected {frequency.size}."
            )
        if real_class_offsets.size != n_classes + 1 or int(real_class_offsets[-1]) != int(real_profiles_flat.shape[0]):
            raise ValueError(f"Artifact {artifact_path} has inconsistent real PSD offsets.")
    if gen_profiles_flat is not None:
        if gen_class_offsets is None:
            raise ValueError(f"Artifact {artifact_path} includes gen_psd_profiles_flat without offsets.")
        if gen_profiles_flat.shape[1] != frequency.size:
            raise ValueError(
                f"Artifact {artifact_path} gen_psd_profiles_flat has width {gen_profiles_flat.shape[1]}, "
                f"expected {frequency.size}."
            )
        if gen_class_offsets.size != n_classes + 1 or int(gen_class_offsets[-1]) != int(gen_profiles_flat.shape[0]):
            raise ValueError(f"Artifact {artifact_path} has inconsistent generated PSD offsets.")

    manifest_labels = manifest.get("class_labels", {})
    if not isinstance(manifest_labels, dict):
        manifest_labels = {}

    panels: Dict[int, PSDPanel] = {}
    for idx, class_id in enumerate(class_ids.tolist()):
        label = str(
            manifest_labels.get(str(class_id))
            or (class_labels_arr[idx] if class_labels_arr.size else "")
            or _default_label(class_id)
        )
        real_profiles = None
        gen_profiles = None
        if real_profiles_flat is not None and real_class_offsets is not None:
            start = int(real_class_offsets[idx])
            stop = int(real_class_offsets[idx + 1])
            real_profiles = real_profiles_flat[start:stop]
        if gen_profiles_flat is not None and gen_class_offsets is not None:
            start = int(gen_class_offsets[idx])
            stop = int(gen_class_offsets[idx + 1])
            gen_profiles = gen_profiles_flat[start:stop]
        panels[class_id] = _build_psd_panel(
            class_id=class_id,
            label=label,
            frequency=frequency,
            real_profile=real_mean_rows[idx],
            gen_profile=gen_mean_rows[idx],
            l2=float(l2_rows[idx]) if l2_rows is not None else None,
            real_profiles=real_profiles,
            gen_profiles=gen_profiles,
        )
    return panels


def _load_psd_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    paper_ready = report.get("paper_ready", {})
    psd_ref = paper_ready.get("psd_radial") if isinstance(paper_ready, dict) else None

    if not (isinstance(psd_ref, dict) and "path" in psd_ref):
        raise ValueError(
            "The metrics JSON does not contain saved radial PSD profile curves for paper-ready plotting. "
            "Re-run eval with the updated code so it writes paper_ready.psd_radial."
        )
    panels = _load_psd_panels_from_npz_artifact(metrics_path=metrics_path, manifest=psd_ref)

    class_ids = sorted(panels.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(f"Expected classes 0..5 in {metrics_path}, found {class_ids}.")

    split = str(report.get("split") or _infer_split_from_eval_ref(spec.eval_ref)).strip().lower()
    return LoadedReport(
        figure_kind=FIGURE_PSD,
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _build_dav_panel(
    *,
    class_id: int,
    label: str,
    real_values: np.ndarray,
    gen_values: np.ndarray,
    abs_gap_deg2: float | None = None,
) -> DAVPanel:
    real_values = _ensure_1d_array(real_values, field_name=f"per_class.{class_id}.real_dav_values")
    gen_values = _ensure_1d_array(gen_values, field_name=f"per_class.{class_id}.gen_dav_values")
    if abs_gap_deg2 is None:
        abs_gap_deg2 = float(abs(np.mean(real_values) - np.mean(gen_values)))
    return DAVPanel(
        class_id=class_id,
        label=label,
        real_values=np.asarray(real_values, dtype=np.float64),
        gen_values=np.asarray(gen_values, dtype=np.float64),
        real_mean=float(np.mean(real_values)),
        gen_mean=float(np.mean(gen_values)),
        abs_gap_deg2=float(abs_gap_deg2),
    )


def _load_dav_panels_from_npz_artifact(
    *,
    metrics_path: Path,
    manifest: Dict[str, Any],
) -> Dict[int, DAVPanel]:
    raw_path = manifest.get("path")
    if not raw_path:
        raise ValueError(f"DAV artifact manifest in {metrics_path} is missing a path.")
    artifact_path = _resolve_artifact_path(metrics_path, str(raw_path))
    if not artifact_path.exists():
        raise FileNotFoundError(f"DAV artifact not found for {metrics_path}: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        class_ids = _ensure_1d_array(data["class_ids"], field_name="class_ids").astype(int)
        class_labels_arr = (
            _ensure_1d_string_array(data["class_labels"], field_name="class_labels")
            if "class_labels" in data.files
            else np.asarray([], dtype=str)
        )
        real_mean_rows = _ensure_1d_array(data["real_mean_dav_deg2"], field_name="real_mean_dav_deg2")
        gen_mean_rows = _ensure_1d_array(data["gen_mean_dav_deg2"], field_name="gen_mean_dav_deg2")
        gap_rows = (
            _ensure_1d_array(data["dav_abs_gap_deg2"], field_name="dav_abs_gap_deg2")
            if "dav_abs_gap_deg2" in data.files
            else None
        )
        real_flat = _ensure_1d_array(data["real_dav_flat"], field_name="real_dav_flat")
        gen_flat = _ensure_1d_array(data["gen_dav_flat"], field_name="gen_dav_flat")
        real_class_offsets = _ensure_offsets_array(data["real_class_offsets"], field_name="real_class_offsets")
        gen_class_offsets = _ensure_offsets_array(data["gen_class_offsets"], field_name="gen_class_offsets")

    n_classes = class_ids.size
    if class_labels_arr.size not in {0, n_classes}:
        raise ValueError(f"Artifact {artifact_path} has {class_labels_arr.size} labels for {n_classes} classes.")
    if real_mean_rows.size != n_classes or gen_mean_rows.size != n_classes:
        raise ValueError(f"Artifact {artifact_path} has inconsistent DAV mean dimensions for {n_classes} classes.")
    if gap_rows is not None and gap_rows.size != n_classes:
        raise ValueError(f"Artifact {artifact_path} has {gap_rows.size} DAV gap values for {n_classes} classes.")
    if real_class_offsets.size != n_classes + 1 or int(real_class_offsets[-1]) != int(real_flat.size):
        raise ValueError(f"Artifact {artifact_path} has inconsistent real DAV offsets.")
    if gen_class_offsets.size != n_classes + 1 or int(gen_class_offsets[-1]) != int(gen_flat.size):
        raise ValueError(f"Artifact {artifact_path} has inconsistent generated DAV offsets.")

    manifest_labels = manifest.get("class_labels", {})
    if not isinstance(manifest_labels, dict):
        manifest_labels = {}

    panels: Dict[int, DAVPanel] = {}
    for idx, class_id in enumerate(class_ids.tolist()):
        label = str(
            manifest_labels.get(str(class_id))
            or (class_labels_arr[idx] if class_labels_arr.size else "")
            or _default_label(class_id)
        )
        real_start = int(real_class_offsets[idx])
        real_stop = int(real_class_offsets[idx + 1])
        gen_start = int(gen_class_offsets[idx])
        gen_stop = int(gen_class_offsets[idx + 1])
        panels[class_id] = _build_dav_panel(
            class_id=class_id,
            label=label,
            real_values=real_flat[real_start:real_stop],
            gen_values=gen_flat[gen_start:gen_stop],
            abs_gap_deg2=float(gap_rows[idx]) if gap_rows is not None else None,
        )
    return panels


def _load_dav_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    paper_ready = report.get("paper_ready", {})
    dav_ref = paper_ready.get("dav") if isinstance(paper_ready, dict) else None
    if not isinstance(dav_ref, dict) or "path" not in dav_ref:
        raise ValueError(
            "The metrics JSON does not contain saved DAV curves for paper-ready plotting. "
            "Re-run eval with the updated code so it writes paper_ready.dav."
        )

    panels = _load_dav_panels_from_npz_artifact(metrics_path=metrics_path, manifest=dav_ref)
    class_ids = sorted(panels.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(f"Expected classes 0..5 in {metrics_path}, found {class_ids}.")

    split = str(report.get("split") or _infer_split_from_eval_ref(spec.eval_ref)).strip().lower()
    return LoadedReport(
        figure_kind=FIGURE_DAV,
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _build_cold_cloud_panel(
    *,
    class_id: int,
    label: str,
    threshold_k: float,
    real_values: np.ndarray,
    gen_values: np.ndarray,
    radius_km: float | None = None,
    pixel_size_km: float | None = None,
    abs_gap_fraction: float | None = None,
) -> ColdCloudPanel:
    real_values = _ensure_1d_array(real_values, field_name=f"per_class.{class_id}.real_cold_fraction_values")
    gen_values = _ensure_1d_array(gen_values, field_name=f"per_class.{class_id}.gen_cold_fraction_values")
    if abs_gap_fraction is None:
        abs_gap_fraction = float(abs(np.mean(real_values) - np.mean(gen_values)))
    return ColdCloudPanel(
        class_id=class_id,
        label=label,
        threshold_k=float(threshold_k),
        radius_km=None if radius_km is None else float(radius_km),
        pixel_size_km=None if pixel_size_km is None else float(pixel_size_km),
        real_values=np.asarray(real_values, dtype=np.float64),
        gen_values=np.asarray(gen_values, dtype=np.float64),
        real_mean_fraction=float(np.mean(real_values)),
        gen_mean_fraction=float(np.mean(gen_values)),
        abs_gap_fraction=float(abs_gap_fraction),
    )


def _load_cold_cloud_panels_from_npz_artifact(
    *,
    metrics_path: Path,
    manifest: Dict[str, Any],
) -> Dict[int, ColdCloudPanel]:
    raw_path = manifest.get("path")
    if not raw_path:
        raise ValueError(f"Cold-cloud-fraction artifact manifest in {metrics_path} is missing a path.")
    artifact_path = _resolve_artifact_path(metrics_path, str(raw_path))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Cold-cloud-fraction artifact not found for {metrics_path}: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        class_ids = _ensure_1d_array(data["class_ids"], field_name="class_ids").astype(int)
        class_labels_arr = (
            _ensure_1d_string_array(data["class_labels"], field_name="class_labels")
            if "class_labels" in data.files
            else np.asarray([], dtype=str)
        )
        threshold_arr = _ensure_1d_array(data["cold_threshold_k"], field_name="cold_threshold_k")
        radius_arr = (
            _ensure_1d_array(data["cold_radius_km"], field_name="cold_radius_km")
            if "cold_radius_km" in data.files
            else None
        )
        pixel_size_arr = (
            _ensure_1d_array(data["cold_pixel_size_km"], field_name="cold_pixel_size_km")
            if "cold_pixel_size_km" in data.files
            else None
        )
        real_mean_rows = _ensure_1d_array(data["real_mean_fraction"], field_name="real_mean_fraction")
        gen_mean_rows = _ensure_1d_array(data["gen_mean_fraction"], field_name="gen_mean_fraction")
        gap_rows = (
            _ensure_1d_array(data["cold_abs_gap_fraction"], field_name="cold_abs_gap_fraction")
            if "cold_abs_gap_fraction" in data.files
            else None
        )
        real_flat = _ensure_1d_array(data["real_fraction_flat"], field_name="real_fraction_flat")
        gen_flat = _ensure_1d_array(data["gen_fraction_flat"], field_name="gen_fraction_flat")
        real_class_offsets = _ensure_offsets_array(data["real_class_offsets"], field_name="real_class_offsets")
        gen_class_offsets = _ensure_offsets_array(data["gen_class_offsets"], field_name="gen_class_offsets")

    threshold_k = float(threshold_arr[0])
    radius_km = (
        float(radius_arr[0])
        if radius_arr is not None
        else (float(manifest["radius_km"]) if manifest.get("radius_km") is not None else None)
    )
    pixel_size_km = (
        float(pixel_size_arr[0])
        if pixel_size_arr is not None
        else (float(manifest["pixel_size_km"]) if manifest.get("pixel_size_km") is not None else None)
    )
    n_classes = class_ids.size
    if class_labels_arr.size not in {0, n_classes}:
        raise ValueError(f"Artifact {artifact_path} has {class_labels_arr.size} labels for {n_classes} classes.")
    if real_mean_rows.size != n_classes or gen_mean_rows.size != n_classes:
        raise ValueError(
            f"Artifact {artifact_path} has inconsistent cold-cloud-fraction mean dimensions for {n_classes} classes."
        )
    if gap_rows is not None and gap_rows.size != n_classes:
        raise ValueError(
            f"Artifact {artifact_path} has {gap_rows.size} cold-cloud-fraction gap values for {n_classes} classes."
        )
    if real_class_offsets.size != n_classes + 1 or int(real_class_offsets[-1]) != int(real_flat.size):
        raise ValueError(f"Artifact {artifact_path} has inconsistent real cold-cloud-fraction offsets.")
    if gen_class_offsets.size != n_classes + 1 or int(gen_class_offsets[-1]) != int(gen_flat.size):
        raise ValueError(f"Artifact {artifact_path} has inconsistent generated cold-cloud-fraction offsets.")

    manifest_labels = manifest.get("class_labels", {})
    if not isinstance(manifest_labels, dict):
        manifest_labels = {}

    panels: Dict[int, ColdCloudPanel] = {}
    for idx, class_id in enumerate(class_ids.tolist()):
        label = str(
            manifest_labels.get(str(class_id))
            or (class_labels_arr[idx] if class_labels_arr.size else "")
            or _default_label(class_id)
        )
        real_start = int(real_class_offsets[idx])
        real_stop = int(real_class_offsets[idx + 1])
        gen_start = int(gen_class_offsets[idx])
        gen_stop = int(gen_class_offsets[idx + 1])
        panels[class_id] = _build_cold_cloud_panel(
            class_id=class_id,
            label=label,
            threshold_k=threshold_k,
            real_values=real_flat[real_start:real_stop],
            gen_values=gen_flat[gen_start:gen_stop],
            radius_km=radius_km,
            pixel_size_km=pixel_size_km,
            abs_gap_fraction=float(gap_rows[idx]) if gap_rows is not None else None,
        )
    return panels


def _load_cold_cloud_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    paper_ready = report.get("paper_ready", {})
    cold_ref = paper_ready.get("cold_cloud_fraction") if isinstance(paper_ready, dict) else None
    if not isinstance(cold_ref, dict) or "path" not in cold_ref:
        raise ValueError(
            "The metrics JSON does not contain saved cold-cloud-fraction curves for paper-ready plotting. "
            "Re-run eval with the updated code so it writes paper_ready.cold_cloud_fraction."
        )

    panels = _load_cold_cloud_panels_from_npz_artifact(metrics_path=metrics_path, manifest=cold_ref)
    class_ids = sorted(panels.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(f"Expected classes 0..5 in {metrics_path}, found {class_ids}.")

    split = str(report.get("split") or _infer_split_from_eval_ref(spec.eval_ref)).strip().lower()
    return LoadedReport(
        figure_kind=FIGURE_COLD,
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _load_memorization_pairs_from_npz_artifact(
    *,
    metrics_path: Path,
    manifest: Dict[str, Any],
) -> Dict[int, list[MemorizationPair]]:
    raw_path = manifest.get("path")
    if not raw_path:
        raise ValueError(f"Memorization-pairs artifact manifest in {metrics_path} is missing a path.")
    artifact_path = _resolve_artifact_path(metrics_path, str(raw_path))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Memorization-pairs artifact not found for {metrics_path}: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        class_ids = _ensure_1d_array(data["class_ids"], field_name="class_ids").astype(int)
        class_labels_arr = (
            _ensure_1d_string_array(data["class_labels"], field_name="class_labels")
            if "class_labels" in data.files
            else np.asarray([], dtype=str)
        )
        class_offsets = _ensure_offsets_array(data["class_offsets"], field_name="class_offsets")
        pair_class_ids = _ensure_1d_array(data["pair_class_ids"], field_name="pair_class_ids").astype(int)
        ranks = _ensure_1d_array(data["rank_within_class"], field_name="rank_within_class").astype(int)
        distances = _ensure_1d_array(data["nearest_train_distance"], field_name="nearest_train_distance")
        generated_indices = _ensure_1d_array(data["generated_index"], field_name="generated_index").astype(int)
        train_indices = _ensure_1d_array(data["train_index"], field_name="train_index").astype(int)
        train_rel_paths = _ensure_1d_string_array(data["train_rel_path"], field_name="train_rel_path")
        generated_bt = _ensure_3d_array(data["generated_bt_k"], field_name="generated_bt_k")
        train_bt = _ensure_3d_array(data["train_bt_k"], field_name="train_bt_k")

    n_classes = class_ids.size
    n_pairs = distances.size
    if class_offsets.size != n_classes + 1 or int(class_offsets[-1]) != n_pairs:
        raise ValueError(f"Artifact {artifact_path} has inconsistent class offsets for {n_pairs} pairs.")
    if class_labels_arr.size not in {0, n_classes}:
        raise ValueError(f"Artifact {artifact_path} has {class_labels_arr.size} labels for {n_classes} classes.")
    for name, arr in {
        "pair_class_ids": pair_class_ids,
        "rank_within_class": ranks,
        "generated_index": generated_indices,
        "train_index": train_indices,
        "train_rel_path": train_rel_paths,
    }.items():
        if arr.size != n_pairs:
            raise ValueError(f"Artifact {artifact_path} field {name!r} has {arr.size} rows, expected {n_pairs}.")
    if generated_bt.shape[0] != n_pairs or train_bt.shape[0] != n_pairs:
        raise ValueError(
            f"Artifact {artifact_path} has image rows generated={generated_bt.shape[0]}, "
            f"train={train_bt.shape[0]}, expected {n_pairs}."
        )
    if generated_bt.shape[1:] != train_bt.shape[1:]:
        raise ValueError(
            f"Artifact {artifact_path} generated/train image shapes differ: "
            f"{generated_bt.shape[1:]} vs {train_bt.shape[1:]}."
        )

    manifest_labels = manifest.get("class_labels", {})
    if not isinstance(manifest_labels, dict):
        manifest_labels = {}

    panels: Dict[int, list[MemorizationPair]] = {}
    for class_pos, class_id in enumerate(class_ids.tolist()):
        label = str(
            manifest_labels.get(str(class_id))
            or (class_labels_arr[class_pos] if class_labels_arr.size else "")
            or _default_label(class_id)
        )
        start = int(class_offsets[class_pos])
        stop = int(class_offsets[class_pos + 1])
        pairs = []
        for row in range(start, stop):
            if int(pair_class_ids[row]) != int(class_id):
                raise ValueError(
                    f"Artifact {artifact_path} pair row {row} belongs to class {pair_class_ids[row]}, "
                    f"expected {class_id} from class_offsets."
                )
            pairs.append(
                MemorizationPair(
                    class_id=int(class_id),
                    label=label,
                    rank_within_class=int(ranks[row]),
                    distance=float(distances[row]),
                    generated_index=int(generated_indices[row]),
                    train_index=int(train_indices[row]),
                    train_rel_path=str(train_rel_paths[row]),
                    generated_bt=np.asarray(generated_bt[row], dtype=np.float32),
                    train_bt=np.asarray(train_bt[row], dtype=np.float32),
                )
            )
        panels[int(class_id)] = sorted(pairs, key=lambda p: (p.rank_within_class, p.distance))
    return panels


def _load_memorization_pairs_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    paper_ready = report.get("paper_ready", {})
    pair_ref = paper_ready.get("memorization_pairs") if isinstance(paper_ready, dict) else None
    if not isinstance(pair_ref, dict) or "path" not in pair_ref:
        raise ValueError(
            "The metrics JSON does not contain saved nearest-train memorization pairs for paper-ready plotting. "
            "Re-run eval with evaluator/distributional metrics enabled so it writes paper_ready.memorization_pairs."
        )

    panels = _load_memorization_pairs_from_npz_artifact(metrics_path=metrics_path, manifest=pair_ref)
    class_ids = sorted(panels.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(f"Expected classes 0..5 in {metrics_path}, found {class_ids}.")

    split = str(report.get("split") or _infer_split_from_eval_ref(spec.eval_ref)).strip().lower()
    return LoadedReport(
        figure_kind=FIGURE_MEMORIZATION,
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _load_report(repo_root: Path, spec: InputSpec, *, figure_kind: str) -> LoadedReport:
    if figure_kind == FIGURE_PIXEL:
        return _load_pixel_report(repo_root, spec)
    if figure_kind == FIGURE_RADIAL:
        return _load_radial_report(repo_root, spec)
    if figure_kind == FIGURE_PSD:
        return _load_psd_report(repo_root, spec)
    if figure_kind == FIGURE_DAV:
        return _load_dav_report(repo_root, spec)
    if figure_kind == FIGURE_COLD:
        return _load_cold_cloud_report(repo_root, spec)
    if figure_kind == FIGURE_MEMORIZATION:
        return _load_memorization_pairs_report(repo_root, spec)
    raise ValueError(f"Unsupported figure kind: {figure_kind}")


def _validate_pixel_reports(reports: list[LoadedReport]) -> None:
    if not reports:
        raise ValueError("No reports were loaded.")

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    for other in reports[1:]:
        if other.split != ref.split:
            raise ValueError(
                "All plotted reports must use the same split. "
                f"Found {ref.run_name!r} on {ref.split!r} and {other.run_name!r} on {other.split!r}."
            )
        if other.n_per_class != ref.n_per_class:
            raise ValueError(
                "All plotted reports must use the same n_per_class for a fair comparison. "
                f"Found {ref.run_name!r}={ref.n_per_class} and {other.run_name!r}={other.n_per_class}."
            )
        if sorted(other.panels.keys()) != class_ids:
            raise ValueError("All plotted reports must contain the same class panels.")

        for class_id in class_ids:
            ref_panel = ref.panels[class_id]
            other_panel = other.panels[class_id]
            if ref_panel.bins.shape != other_panel.bins.shape or not np.allclose(ref_panel.bins, other_panel.bins):
                raise ValueError(
                    f"Histogram bins differ for class {class_id} between {ref.run_name!r} and {other.run_name!r}."
                )
            if not np.allclose(ref_panel.real_density, other_panel.real_density, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "Real reference histograms differ across reports. "
                    "Re-run eval with the same split, n_per_class, and real-data seed before plotting together."
                )


def _validate_radial_reports(reports: list[LoadedReport]) -> None:
    if not reports:
        raise ValueError("No reports were loaded.")

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    for other in reports[1:]:
        if other.split != ref.split:
            raise ValueError(
                "All plotted reports must use the same split. "
                f"Found {ref.run_name!r} on {ref.split!r} and {other.run_name!r} on {other.split!r}."
            )
        if other.n_per_class != ref.n_per_class:
            raise ValueError(
                "All plotted reports must use the same n_per_class for a fair comparison. "
                f"Found {ref.run_name!r}={ref.n_per_class} and {other.run_name!r}={other.n_per_class}."
            )
        if sorted(other.panels.keys()) != class_ids:
            raise ValueError("All plotted reports must contain the same class panels.")

        for class_id in class_ids:
            ref_panel = ref.panels[class_id]
            other_panel = other.panels[class_id]
            if ref_panel.radius.shape != other_panel.radius.shape or not np.allclose(ref_panel.radius, other_panel.radius):
                raise ValueError(
                    f"Normalized radius coordinates differ for class {class_id} between "
                    f"{ref.run_name!r} and {other.run_name!r}."
                )
            if not np.allclose(ref_panel.real_profile, other_panel.real_profile, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "Real reference radial profiles differ across reports. "
                    "Re-run eval with the same split, n_per_class, and real-data seed before plotting together."
                )


def _validate_psd_reports(reports: list[LoadedReport]) -> None:
    if not reports:
        raise ValueError("No reports were loaded.")

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    for other in reports[1:]:
        if other.split != ref.split:
            raise ValueError(
                "All plotted reports must use the same split. "
                f"Found {ref.run_name!r} on {ref.split!r} and {other.run_name!r} on {other.split!r}."
            )
        if other.n_per_class != ref.n_per_class:
            raise ValueError(
                "All plotted reports must use the same n_per_class for a fair comparison. "
                f"Found {ref.run_name!r}={ref.n_per_class} and {other.run_name!r}={other.n_per_class}."
            )
        if sorted(other.panels.keys()) != class_ids:
            raise ValueError("All plotted reports must contain the same class panels.")

        for class_id in class_ids:
            ref_panel = ref.panels[class_id]
            other_panel = other.panels[class_id]
            if ref_panel.frequency.shape != other_panel.frequency.shape or not np.allclose(
                ref_panel.frequency,
                other_panel.frequency,
            ):
                raise ValueError(
                    f"Normalized PSD frequency coordinates differ for class {class_id} between "
                    f"{ref.run_name!r} and {other.run_name!r}."
                )
            if not np.allclose(ref_panel.real_profile, other_panel.real_profile, atol=1e-12, rtol=0.0):
                raise ValueError(
                    "Real reference PSD profiles differ across reports. "
                    "Re-run eval with the same split, n_per_class, and real-data seed before plotting together."
                )


def _validate_dav_reports(reports: list[LoadedReport]) -> None:
    if not reports:
        raise ValueError("No reports were loaded.")

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    for other in reports[1:]:
        if other.split != ref.split:
            raise ValueError(
                "All plotted reports must use the same split. "
                f"Found {ref.run_name!r} on {ref.split!r} and {other.run_name!r} on {other.split!r}."
            )
        if other.n_per_class != ref.n_per_class:
            raise ValueError(
                "All plotted reports must use the same n_per_class for a fair comparison. "
                f"Found {ref.run_name!r}={ref.n_per_class} and {other.run_name!r}={other.n_per_class}."
            )
        if sorted(other.panels.keys()) != class_ids:
            raise ValueError("All plotted reports must contain the same class panels.")

        for class_id in class_ids:
            ref_panel = ref.panels[class_id]
            other_panel = other.panels[class_id]
            if ref_panel.real_values.shape != other_panel.real_values.shape or not np.allclose(
                np.sort(ref_panel.real_values),
                np.sort(other_panel.real_values),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "Real reference DAV values differ across reports. "
                    "Re-run eval with the same split, n_per_class, and real-data seed before plotting together."
                )


def _validate_cold_cloud_reports(reports: list[LoadedReport]) -> None:
    if not reports:
        raise ValueError("No reports were loaded.")

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    ref_threshold = float(ref.panels[class_ids[0]].threshold_k)
    ref_radius = ref.panels[class_ids[0]].radius_km
    for other in reports[1:]:
        if other.split != ref.split:
            raise ValueError(
                "All plotted reports must use the same split. "
                f"Found {ref.run_name!r} on {ref.split!r} and {other.run_name!r} on {other.split!r}."
            )
        if other.n_per_class != ref.n_per_class:
            raise ValueError(
                "All plotted reports must use the same n_per_class for a fair comparison. "
                f"Found {ref.run_name!r}={ref.n_per_class} and {other.run_name!r}={other.n_per_class}."
            )
        if sorted(other.panels.keys()) != class_ids:
            raise ValueError("All plotted reports must contain the same class panels.")
        other_threshold = float(other.panels[class_ids[0]].threshold_k)
        if not np.isclose(other_threshold, ref_threshold):
            raise ValueError(
                "All plotted reports must use the same cold-cloud threshold. "
                f"Found {ref_threshold} K and {other_threshold} K."
            )
        other_radius = other.panels[class_ids[0]].radius_km
        if (ref_radius is None) != (other_radius is None):
            raise ValueError("All plotted reports must either include or omit cold-cloud radius metadata.")
        if ref_radius is not None and other_radius is not None and not np.isclose(other_radius, ref_radius):
            raise ValueError(
                "All plotted reports must use the same cold-cloud radius. "
                f"Found {ref_radius} km and {other_radius} km."
            )
        for class_id in class_ids:
            ref_panel = ref.panels[class_id]
            other_panel = other.panels[class_id]
            if ref_panel.real_values.shape != other_panel.real_values.shape or not np.allclose(
                np.sort(ref_panel.real_values),
                np.sort(other_panel.real_values),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "Real reference cold-cloud-fraction values differ across reports. "
                    "Re-run eval with the same split, n_per_class, and real-data seed before plotting together."
                )


def _validate_memorization_pair_reports(reports: list[LoadedReport]) -> None:
    if not reports:
        raise ValueError("No reports were loaded.")

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    for other in reports[1:]:
        if other.split != ref.split:
            raise ValueError(
                "All plotted reports must use the same split. "
                f"Found {ref.run_name!r} on {ref.split!r} and {other.run_name!r} on {other.split!r}."
            )
        if other.n_per_class != ref.n_per_class:
            raise ValueError(
                "All plotted reports must use the same n_per_class for a fair comparison. "
                f"Found {ref.run_name!r}={ref.n_per_class} and {other.run_name!r}={other.n_per_class}."
            )
        if sorted(other.panels.keys()) != class_ids:
            raise ValueError("All plotted reports must contain the same class panels.")

    for report in reports:
        empty = [class_id for class_id in class_ids if not report.panels[class_id]]
        if empty:
            raise ValueError(f"Report {report.metrics_path} has no memorization pairs for classes: {empty}.")


def _validate_reports(reports: list[LoadedReport], *, figure_kind: str) -> None:
    if figure_kind == FIGURE_PIXEL:
        _validate_pixel_reports(reports)
        return
    if figure_kind == FIGURE_RADIAL:
        _validate_radial_reports(reports)
        return
    if figure_kind == FIGURE_PSD:
        _validate_psd_reports(reports)
        return
    if figure_kind == FIGURE_DAV:
        _validate_dav_reports(reports)
        return
    if figure_kind == FIGURE_COLD:
        _validate_cold_cloud_reports(reports)
        return
    if figure_kind == FIGURE_MEMORIZATION:
        _validate_memorization_pair_reports(reports)
        return
    raise ValueError(f"Unsupported figure kind: {figure_kind}")


def _slugify_token(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "model"


def _default_output_filename(reports: list[LoadedReport], *, figure_kind: str) -> str:
    split_slug = reports[0].split
    model_slug = "_vs_".join(_slugify_token(report.label) for report in reports)
    if figure_kind == FIGURE_PIXEL:
        stem = "paper_ready_pixel_plausibility"
    elif figure_kind == FIGURE_RADIAL:
        stem = "paper_ready_radial_bt_profile"
    elif figure_kind == FIGURE_PSD:
        stem = "paper_ready_radial_psd_profile"
    elif figure_kind == FIGURE_DAV:
        stem = "paper_ready_dav"
    elif figure_kind == FIGURE_COLD:
        stem = "paper_ready_cold_cloud_fraction"
    elif figure_kind == FIGURE_MEMORIZATION:
        stem = "paper_ready_memorization_pairs"
    else:
        raise ValueError(f"Unsupported figure kind: {figure_kind}")
    return f"{stem}_{split_slug}_{model_slug}.png"


def _default_output_root(repo_root: Path, reports: list[LoadedReport]) -> Path:
    if len(reports) == 1:
        return model_paper_ready_root(repo_root, reports[0].run_name)
    return repo_root / "outputs" / "paper_ready"


def _resolve_output_path(
    repo_root: Path,
    reports: list[LoadedReport],
    output: str | None,
    *,
    figure_kind: str,
    multiple_figures: bool,
) -> Path:
    default_name = _default_output_filename(reports, figure_kind=figure_kind)
    if output is None:
        return _default_output_root(repo_root, reports) / default_name

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    if out_path.suffix:
        if multiple_figures:
            raise ValueError(
                "--output must point to a directory when rendering multiple figures. "
                f"Received file-like path: {out_path}"
            )
        return out_path
    return out_path / default_name


def _gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def _smooth_density_for_display(density: np.ndarray, bins: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return np.asarray(density, dtype=np.float64)

    widths = np.diff(bins)
    mass = np.asarray(density, dtype=np.float64) * widths
    smoothed_mass = np.convolve(mass, _gaussian_kernel1d(sigma_bins), mode="same")
    total = float(np.sum(smoothed_mass))
    if total > 0.0:
        smoothed_mass /= total
    return smoothed_mass / widths


def _display_curve(density: np.ndarray, bins: np.ndarray, sigma_bins: float) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[:-1] + bins[1:])
    smooth_density = _smooth_density_for_display(density, bins, sigma_bins)
    x_nodes = np.concatenate(([bins[0]], centers, [bins[-1]]))
    y_nodes = np.concatenate(([0.0], smooth_density, [0.0]))
    x_dense = np.linspace(float(bins[0]), float(bins[-1]), max(800, centers.size * 10))
    y_dense = np.interp(x_dense, x_nodes, y_nodes)
    return x_dense, y_dense


def _display_profile_curve(radius: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _ensure_1d_array(radius, field_name="radius")
    y = _ensure_1d_array(values, field_name="profile_values")
    if x.size != y.size:
        raise ValueError(f"Radius and profile arrays must have matching lengths, got {x.size} and {y.size}.")
    x_dense = np.linspace(float(x[0]), float(x[-1]), max(800, x.size * 10))
    y_dense = np.interp(x_dense, x, y)
    return x_dense, y_dense


def _normalize_rows_to_mass(hist_rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(hist_rows, dtype=np.float64)
    totals = rows.sum(axis=1, keepdims=True)
    totals[totals <= 0.0] = 1.0
    return rows / totals


def _smooth_mass_rows_for_display(mass_rows: np.ndarray, sigma_bins: float) -> np.ndarray:
    mass_rows = np.asarray(mass_rows, dtype=np.float64)
    if sigma_bins <= 0.0 or mass_rows.size == 0:
        return mass_rows

    kernel = _gaussian_kernel1d(sigma_bins)
    smoothed = np.empty_like(mass_rows)
    for idx in range(mass_rows.shape[0]):
        row = np.convolve(mass_rows[idx], kernel, mode="same")
        total = float(np.sum(row))
        smoothed[idx] = row / total if total > 0.0 else mass_rows[idx]
    return smoothed


def _scalar_hist_counts_rows(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    vals = _ensure_1d_array(values, field_name="scalar_values")
    edges = _ensure_1d_array(bins, field_name="scalar_bins")
    if edges.size < 2:
        raise ValueError("Scalar histogram bins must contain at least two edges.")
    nbins = edges.size - 1
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    out = np.zeros((vals.size, nbins), dtype=np.float64)
    out[np.arange(vals.size), idx] = 1.0
    return out


def _scalar_density_from_values(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=np.asarray(bins, dtype=np.float64))
    mass = _normalize_mass(counts, field_name="scalar_hist_counts")
    return _mass_to_density(mass, np.asarray(bins, dtype=np.float64))


def _build_scalar_bins(reports: list[LoadedReport], *, bins_per_panel: int = 56) -> np.ndarray:
    values = []
    for report in reports:
        for class_id in sorted(report.panels.keys()):
            panel = report.panels[class_id]
            values.append(np.asarray(panel.real_values, dtype=np.float64))
            values.append(np.asarray(panel.gen_values, dtype=np.float64))
    pooled = np.concatenate(values, axis=0)
    lo = float(np.min(pooled))
    hi = float(np.max(pooled))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("DAV values must be finite.")
    if hi <= lo:
        span = max(abs(lo), 1.0)
        lo -= 0.05 * span
        hi += 0.05 * span
    else:
        pad = 0.035 * (hi - lo)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, int(bins_per_panel) + 1, dtype=np.float64)


def _quantile_interval(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = 0.5 * (1.0 - float(ci_level))
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _iter_bootstrap_hist_sums(
    hist_counts: np.ndarray,
    *,
    sample_size: int,
    reps: int,
    seed: int,
    batch_reps: int = 64,
):
    counts = np.asarray(hist_counts, dtype=np.float64)
    n_images = counts.shape[0]
    if reps <= 0 or n_images <= 0 or sample_size <= 0:
        return

    probs = np.full(n_images, 1.0 / n_images, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, reps, batch_reps):
        chunk = min(batch_reps, reps - start)
        weights = rng.multinomial(sample_size, probs, size=chunk).astype(np.float64, copy=False)
        yield weights @ counts


def _bootstrap_density_band(
    hist_counts: np.ndarray | None,
    bins: np.ndarray,
    *,
    reps: int,
    ci_level: float,
    sigma_bins: float,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hist_counts is None:
        return None, None

    counts = np.asarray(hist_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] < 2:
        return None, None

    widths = np.diff(bins).astype(np.float64)
    density_batches = []
    for hist_sums in _iter_bootstrap_hist_sums(
        counts,
        sample_size=counts.shape[0],
        reps=reps,
        seed=seed,
    ):
        mass_rows = _normalize_rows_to_mass(hist_sums)
        mass_rows = _smooth_mass_rows_for_display(mass_rows, sigma_bins)
        density_batches.append(mass_rows / widths[None, :])

    if not density_batches:
        return None, None

    densities = np.concatenate(density_batches, axis=0)
    alpha = 0.5 * (1.0 - float(ci_level))
    low = np.quantile(densities, alpha, axis=0)
    high = np.quantile(densities, 1.0 - alpha, axis=0)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def _bootstrap_model_metric_distribution(
    real_hist_counts: np.ndarray | None,
    gen_hist_counts: np.ndarray | None,
    bins: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if real_hist_counts is None or gen_hist_counts is None:
        return None, None

    real_counts = np.asarray(real_hist_counts, dtype=np.float64)
    gen_counts = np.asarray(gen_hist_counts, dtype=np.float64)
    if real_counts.ndim != 2 or gen_counts.ndim != 2:
        return None, None
    if real_counts.shape[0] < 2 or gen_counts.shape[0] < 2 or reps <= 0:
        return None, None

    js_batches = []
    w1_batches = []
    gen_iter = _iter_bootstrap_hist_sums(
        gen_counts,
        sample_size=gen_counts.shape[0],
        reps=reps,
        seed=seed + 1,
    )
    for real_hist_sums, gen_hist_sums in zip(
        _iter_bootstrap_hist_sums(
            real_counts,
            sample_size=real_counts.shape[0],
            reps=reps,
            seed=seed,
        ),
        gen_iter,
    ):
        real_mass = _normalize_rows_to_mass(real_hist_sums)
        gen_mass = _normalize_rows_to_mass(gen_hist_sums)
        js_batches.append(np.asarray([js_divergence(r, g) for r, g in zip(real_mass, gen_mass)], dtype=np.float64))
        w1_batches.append(
            np.asarray(
                [wasserstein1_from_hist(r, g, bin_edges=bins) for r, g in zip(real_mass, gen_mass)],
                dtype=np.float64,
            )
        )

    if not js_batches or not w1_batches:
        return None, None
    return np.concatenate(js_batches), np.concatenate(w1_batches)


def _bootstrap_real_vs_real_null(
    real_hist_counts: np.ndarray | None,
    bins: np.ndarray,
    *,
    other_sample_size: int,
    reps: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if real_hist_counts is None:
        return None, None

    real_counts = np.asarray(real_hist_counts, dtype=np.float64)
    if real_counts.ndim != 2 or real_counts.shape[0] < 2 or other_sample_size <= 0 or reps <= 0:
        return None, None

    js_batches = []
    w1_batches = []
    other_iter = _iter_bootstrap_hist_sums(
        real_counts,
        sample_size=int(other_sample_size),
        reps=reps,
        seed=seed + 1,
    )
    for left_sums, right_sums in zip(
        _iter_bootstrap_hist_sums(
            real_counts,
            sample_size=real_counts.shape[0],
            reps=reps,
            seed=seed,
        ),
        other_iter,
    ):
        left_mass = _normalize_rows_to_mass(left_sums)
        right_mass = _normalize_rows_to_mass(right_sums)
        js_batches.append(
            np.asarray([js_divergence(left, right) for left, right in zip(left_mass, right_mass)], dtype=np.float64)
        )
        w1_batches.append(
            np.asarray(
                [wasserstein1_from_hist(left, right, bin_edges=bins) for left, right in zip(left_mass, right_mass)],
                dtype=np.float64,
            )
        )

    if not js_batches or not w1_batches:
        return None, None
    return np.concatenate(js_batches), np.concatenate(w1_batches)


def _empirical_pvalue(observed: float, null_values: np.ndarray | None) -> float | None:
    if null_values is None or null_values.size == 0:
        return None
    return float((np.count_nonzero(null_values >= float(observed)) + 1.0) / (null_values.size + 1.0))


def _bootstrap_scalar_gap_distribution(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    *,
    reps: int,
    seed: int,
    batch_reps: int = 128,
) -> np.ndarray | None:
    real_arr = np.asarray(real_values, dtype=np.float64).reshape(-1)
    gen_arr = np.asarray(gen_values, dtype=np.float64).reshape(-1)
    if real_arr.size < 2 or gen_arr.size < 2 or reps <= 0:
        return None

    rng = np.random.default_rng(int(seed))
    gap_batches = []
    for start in range(0, reps, batch_reps):
        chunk = min(batch_reps, reps - start)
        real_idx = rng.integers(real_arr.size, size=(chunk, real_arr.size))
        gen_idx = rng.integers(gen_arr.size, size=(chunk, gen_arr.size))
        real_means = real_arr[real_idx].mean(axis=1)
        gen_means = gen_arr[gen_idx].mean(axis=1)
        gap_batches.append(np.abs(real_means - gen_means))
    return np.concatenate(gap_batches, axis=0) if gap_batches else None


def _bootstrap_scalar_real_null(
    real_values: np.ndarray,
    *,
    other_sample_size: int,
    reps: int,
    seed: int,
    batch_reps: int = 128,
) -> np.ndarray | None:
    real_arr = np.asarray(real_values, dtype=np.float64).reshape(-1)
    if real_arr.size < 2 or other_sample_size <= 0 or reps <= 0:
        return None

    rng = np.random.default_rng(int(seed))
    gap_batches = []
    for start in range(0, reps, batch_reps):
        chunk = min(batch_reps, reps - start)
        left_idx = rng.integers(real_arr.size, size=(chunk, real_arr.size))
        right_idx = rng.integers(real_arr.size, size=(chunk, int(other_sample_size)))
        left_means = real_arr[left_idx].mean(axis=1)
        right_means = real_arr[right_idx].mean(axis=1)
        gap_batches.append(np.abs(left_means - right_means))
    return np.concatenate(gap_batches, axis=0) if gap_batches else None


def _compute_class_inference(
    reports: list[LoadedReport],
    class_id: int,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    seed: int,
) -> ClassInference:
    ref_panel = reports[0].panels[class_id]
    if ref_panel.real_hist_counts is None:
        return ClassInference()

    out = ClassInference(supports_image_level_stats=True)
    out.real_density_low, out.real_density_high = _bootstrap_density_band(
        ref_panel.real_hist_counts,
        ref_panel.bins,
        reps=bootstrap_reps,
        ci_level=ci_level,
        sigma_bins=smooth_sigma,
        seed=seed + 101 * class_id,
    )

    null_js = None
    null_w1 = None
    if reports and reports[0].panels[class_id].gen_hist_counts is not None:
        null_js, null_w1 = _bootstrap_real_vs_real_null(
            ref_panel.real_hist_counts,
            ref_panel.bins,
            other_sample_size=int(reports[0].panels[class_id].gen_hist_counts.shape[0]),
            reps=null_reps,
            seed=seed + 1000 + 101 * class_id,
        )

    for report_idx, report in enumerate(reports):
        panel = report.panels[class_id]
        model_stats = ModelInference()
        model_stats.gen_density_low, model_stats.gen_density_high = _bootstrap_density_band(
            panel.gen_hist_counts,
            panel.bins,
            reps=bootstrap_reps,
            ci_level=ci_level,
            sigma_bins=smooth_sigma,
            seed=seed + 2000 + 211 * class_id + report_idx,
        )
        js_dist, w1_dist = _bootstrap_model_metric_distribution(
            ref_panel.real_hist_counts,
            panel.gen_hist_counts,
            panel.bins,
            reps=bootstrap_reps,
            seed=seed + 3000 + 211 * class_id + report_idx,
        )
        if js_dist is not None and js_dist.size:
            model_stats.js_ci = _quantile_interval(js_dist, ci_level)
        if w1_dist is not None and w1_dist.size:
            model_stats.w1_ci = _quantile_interval(w1_dist, ci_level)
        if null_js is not None and null_js.size:
            model_stats.js_null_q95 = float(np.quantile(null_js, 0.95))
            model_stats.js_pvalue = _empirical_pvalue(panel.js, null_js)
        if null_w1 is not None and null_w1.size:
            model_stats.w1_null_q95 = float(np.quantile(null_w1, 0.95))
            model_stats.w1_pvalue = _empirical_pvalue(panel.w1, null_w1)
        out.model[report_idx] = model_stats
    return out


def _iter_bootstrap_profile_means(
    profile_rows: np.ndarray,
    *,
    sample_size: int,
    reps: int,
    seed: int,
    batch_reps: int = 64,
):
    rows = np.asarray(profile_rows, dtype=np.float64)
    n_images = rows.shape[0]
    if reps <= 0 or n_images <= 0 or sample_size <= 0:
        return

    probs = np.full(n_images, 1.0 / n_images, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, reps, batch_reps):
        chunk = min(batch_reps, reps - start)
        weights = rng.multinomial(sample_size, probs, size=chunk).astype(np.float64, copy=False)
        yield (weights @ rows) / float(sample_size)


def _bootstrap_profile_band(
    profile_rows: np.ndarray | None,
    *,
    reps: int,
    ci_level: float,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if profile_rows is None:
        return None, None

    rows = np.asarray(profile_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return None, None

    mean_batches = list(
        _iter_bootstrap_profile_means(
            rows,
            sample_size=rows.shape[0],
            reps=reps,
            seed=seed,
        )
    )
    if not mean_batches:
        return None, None

    means = np.concatenate(mean_batches, axis=0)
    alpha = 0.5 * (1.0 - float(ci_level))
    low = np.quantile(means, alpha, axis=0)
    high = np.quantile(means, 1.0 - alpha, axis=0)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def _profile_mae(real_profiles: np.ndarray, gen_profiles: np.ndarray) -> np.ndarray:
    real_arr = np.asarray(real_profiles, dtype=np.float64)
    gen_arr = np.asarray(gen_profiles, dtype=np.float64)
    return np.mean(np.abs(real_arr - gen_arr), axis=1)


def _profile_l2(real_profiles: np.ndarray, gen_profiles: np.ndarray) -> np.ndarray:
    real_arr = np.asarray(real_profiles, dtype=np.float64)
    gen_arr = np.asarray(gen_profiles, dtype=np.float64)
    return np.mean(np.square(real_arr - gen_arr), axis=1)


def _bootstrap_profile_mae_distribution(
    real_profile_rows: np.ndarray | None,
    gen_profile_rows: np.ndarray | None,
    *,
    reps: int,
    seed: int,
) -> np.ndarray | None:
    if real_profile_rows is None or gen_profile_rows is None:
        return None

    real_rows = np.asarray(real_profile_rows, dtype=np.float64)
    gen_rows = np.asarray(gen_profile_rows, dtype=np.float64)
    if real_rows.ndim != 2 or gen_rows.ndim != 2:
        return None
    if real_rows.shape[0] < 2 or gen_rows.shape[0] < 2 or reps <= 0:
        return None

    mae_batches = []
    gen_iter = _iter_bootstrap_profile_means(
        gen_rows,
        sample_size=gen_rows.shape[0],
        reps=reps,
        seed=seed + 1,
    )
    for real_means, gen_means in zip(
        _iter_bootstrap_profile_means(
            real_rows,
            sample_size=real_rows.shape[0],
            reps=reps,
            seed=seed,
        ),
        gen_iter,
    ):
        mae_batches.append(_profile_mae(real_means, gen_means))

    if not mae_batches:
        return None
    return np.concatenate(mae_batches)


def _bootstrap_profile_l2_distribution(
    real_profile_rows: np.ndarray | None,
    gen_profile_rows: np.ndarray | None,
    *,
    reps: int,
    seed: int,
) -> np.ndarray | None:
    if real_profile_rows is None or gen_profile_rows is None:
        return None

    real_rows = np.asarray(real_profile_rows, dtype=np.float64)
    gen_rows = np.asarray(gen_profile_rows, dtype=np.float64)
    if real_rows.ndim != 2 or gen_rows.ndim != 2:
        return None
    if real_rows.shape[0] < 2 or gen_rows.shape[0] < 2 or reps <= 0:
        return None

    l2_batches = []
    gen_iter = _iter_bootstrap_profile_means(
        gen_rows,
        sample_size=gen_rows.shape[0],
        reps=reps,
        seed=seed + 1,
    )
    for real_means, gen_means in zip(
        _iter_bootstrap_profile_means(
            real_rows,
            sample_size=real_rows.shape[0],
            reps=reps,
            seed=seed,
        ),
        gen_iter,
    ):
        l2_batches.append(_profile_l2(real_means, gen_means))

    if not l2_batches:
        return None
    return np.concatenate(l2_batches)


def _bootstrap_real_profile_null(
    real_profile_rows: np.ndarray | None,
    *,
    other_sample_size: int,
    reps: int,
    seed: int,
) -> np.ndarray | None:
    if real_profile_rows is None:
        return None

    real_rows = np.asarray(real_profile_rows, dtype=np.float64)
    if real_rows.ndim != 2 or real_rows.shape[0] < 2 or other_sample_size <= 0 or reps <= 0:
        return None

    mae_batches = []
    other_iter = _iter_bootstrap_profile_means(
        real_rows,
        sample_size=int(other_sample_size),
        reps=reps,
        seed=seed + 1,
    )
    for left_means, right_means in zip(
        _iter_bootstrap_profile_means(
            real_rows,
            sample_size=real_rows.shape[0],
            reps=reps,
            seed=seed,
        ),
        other_iter,
    ):
        mae_batches.append(_profile_mae(left_means, right_means))

    if not mae_batches:
        return None
    return np.concatenate(mae_batches)


def _bootstrap_real_profile_l2_null(
    real_profile_rows: np.ndarray | None,
    *,
    other_sample_size: int,
    reps: int,
    seed: int,
) -> np.ndarray | None:
    if real_profile_rows is None:
        return None

    real_rows = np.asarray(real_profile_rows, dtype=np.float64)
    if real_rows.ndim != 2 or real_rows.shape[0] < 2 or other_sample_size <= 0 or reps <= 0:
        return None

    l2_batches = []
    other_iter = _iter_bootstrap_profile_means(
        real_rows,
        sample_size=int(other_sample_size),
        reps=reps,
        seed=seed + 1,
    )
    for left_means, right_means in zip(
        _iter_bootstrap_profile_means(
            real_rows,
            sample_size=real_rows.shape[0],
            reps=reps,
            seed=seed,
        ),
        other_iter,
    ):
        l2_batches.append(_profile_l2(left_means, right_means))

    if not l2_batches:
        return None
    return np.concatenate(l2_batches)


def _compute_radial_class_inference(
    reports: list[LoadedReport],
    class_id: int,
    *,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    seed: int,
) -> RadialClassInference:
    ref_panel = reports[0].panels[class_id]
    if ref_panel.real_profiles is None:
        return RadialClassInference()

    out = RadialClassInference(supports_image_level_stats=True)
    out.real_profile_low, out.real_profile_high = _bootstrap_profile_band(
        ref_panel.real_profiles,
        reps=bootstrap_reps,
        ci_level=ci_level,
        seed=seed + 101 * class_id,
    )

    null_mae = None
    if reports and reports[0].panels[class_id].gen_profiles is not None:
        null_mae = _bootstrap_real_profile_null(
            ref_panel.real_profiles,
            other_sample_size=int(reports[0].panels[class_id].gen_profiles.shape[0]),
            reps=null_reps,
            seed=seed + 1000 + 101 * class_id,
        )

    for report_idx, report in enumerate(reports):
        panel = report.panels[class_id]
        model_stats = RadialModelInference()
        model_stats.gen_profile_low, model_stats.gen_profile_high = _bootstrap_profile_band(
            panel.gen_profiles,
            reps=bootstrap_reps,
            ci_level=ci_level,
            seed=seed + 2000 + 211 * class_id + report_idx,
        )
        mae_dist = _bootstrap_profile_mae_distribution(
            ref_panel.real_profiles,
            panel.gen_profiles,
            reps=bootstrap_reps,
            seed=seed + 3000 + 211 * class_id + report_idx,
        )
        if mae_dist is not None and mae_dist.size:
            model_stats.mae_ci = _quantile_interval(mae_dist, ci_level)
        if null_mae is not None and null_mae.size:
            model_stats.mae_null_q95 = float(np.quantile(null_mae, 0.95))
            model_stats.mae_pvalue = _empirical_pvalue(panel.mae_k, null_mae)
        out.model[report_idx] = model_stats
    return out


def _compute_psd_class_inference(
    reports: list[LoadedReport],
    class_id: int,
    *,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    seed: int,
) -> PSDClassInference:
    ref_panel = reports[0].panels[class_id]
    if ref_panel.real_profiles is None:
        return PSDClassInference()

    out = PSDClassInference(supports_image_level_stats=True)
    out.real_profile_low, out.real_profile_high = _bootstrap_profile_band(
        ref_panel.real_profiles,
        reps=bootstrap_reps,
        ci_level=ci_level,
        seed=seed + 101 * class_id,
    )

    null_l2 = None
    if reports and reports[0].panels[class_id].gen_profiles is not None:
        null_l2 = _bootstrap_real_profile_l2_null(
            ref_panel.real_profiles,
            other_sample_size=int(reports[0].panels[class_id].gen_profiles.shape[0]),
            reps=null_reps,
            seed=seed + 1000 + 101 * class_id,
        )

    for report_idx, report in enumerate(reports):
        panel = report.panels[class_id]
        model_stats = PSDModelInference()
        model_stats.gen_profile_low, model_stats.gen_profile_high = _bootstrap_profile_band(
            panel.gen_profiles,
            reps=bootstrap_reps,
            ci_level=ci_level,
            seed=seed + 2000 + 211 * class_id + report_idx,
        )
        l2_dist = _bootstrap_profile_l2_distribution(
            ref_panel.real_profiles,
            panel.gen_profiles,
            reps=bootstrap_reps,
            seed=seed + 3000 + 211 * class_id + report_idx,
        )
        if l2_dist is not None and l2_dist.size:
            model_stats.l2_ci = _quantile_interval(l2_dist, ci_level)
        if null_l2 is not None and null_l2.size:
            model_stats.l2_null_q95 = float(np.quantile(null_l2, 0.95))
            model_stats.l2_pvalue = _empirical_pvalue(panel.l2, null_l2)
        out.model[report_idx] = model_stats
    return out


def _compute_dav_class_inference(
    reports: list[LoadedReport],
    class_id: int,
    bins: np.ndarray,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    seed: int,
) -> DAVClassInference:
    ref_panel = reports[0].panels[class_id]
    real_hist_counts = _scalar_hist_counts_rows(ref_panel.real_values, bins)
    if real_hist_counts.shape[0] < 2:
        return DAVClassInference()

    out = DAVClassInference(supports_image_level_stats=True)
    out.real_density_low, out.real_density_high = _bootstrap_density_band(
        real_hist_counts,
        bins,
        reps=bootstrap_reps,
        ci_level=ci_level,
        sigma_bins=smooth_sigma,
        seed=seed + 101 * class_id,
    )

    null_gap = _bootstrap_scalar_real_null(
        ref_panel.real_values,
        other_sample_size=int(ref_panel.gen_values.shape[0]),
        reps=null_reps,
        seed=seed + 1000 + 101 * class_id,
    )

    for report_idx, report in enumerate(reports):
        panel = report.panels[class_id]
        model_stats = DAVModelInference()
        gen_hist_counts = _scalar_hist_counts_rows(panel.gen_values, bins)
        model_stats.gen_density_low, model_stats.gen_density_high = _bootstrap_density_band(
            gen_hist_counts,
            bins,
            reps=bootstrap_reps,
            ci_level=ci_level,
            sigma_bins=smooth_sigma,
            seed=seed + 2000 + 211 * class_id + report_idx,
        )
        gap_dist = _bootstrap_scalar_gap_distribution(
            ref_panel.real_values,
            panel.gen_values,
            reps=bootstrap_reps,
            seed=seed + 3000 + 211 * class_id + report_idx,
        )
        if gap_dist is not None and gap_dist.size:
            model_stats.gap_ci = _quantile_interval(gap_dist, ci_level)
        if null_gap is not None and null_gap.size:
            model_stats.gap_null_q95 = float(np.quantile(null_gap, 0.95))
            model_stats.gap_pvalue = _empirical_pvalue(panel.abs_gap_deg2, null_gap)
        out.model[report_idx] = model_stats
    return out


def _format_dav_value(value: float) -> str:
    value = float(value)
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _compute_cold_cloud_class_inference(
    reports: list[LoadedReport],
    class_id: int,
    bins: np.ndarray,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    seed: int,
) -> ColdCloudClassInference:
    ref_panel = reports[0].panels[class_id]
    real_values = np.asarray(ref_panel.real_values, dtype=np.float64) * 100.0
    real_hist_counts = _scalar_hist_counts_rows(real_values, bins)
    if real_hist_counts.shape[0] < 2:
        return ColdCloudClassInference()

    out = ColdCloudClassInference(supports_image_level_stats=True)
    out.real_density_low, out.real_density_high = _bootstrap_density_band(
        real_hist_counts,
        bins,
        reps=bootstrap_reps,
        ci_level=ci_level,
        sigma_bins=smooth_sigma,
        seed=seed + 101 * class_id,
    )

    null_gap = _bootstrap_scalar_real_null(
        real_values,
        other_sample_size=int(ref_panel.gen_values.shape[0]),
        reps=null_reps,
        seed=seed + 1000 + 101 * class_id,
    )

    for report_idx, report in enumerate(reports):
        panel = report.panels[class_id]
        gen_values = np.asarray(panel.gen_values, dtype=np.float64) * 100.0
        model_stats = ColdCloudModelInference()
        gen_hist_counts = _scalar_hist_counts_rows(gen_values, bins)
        model_stats.gen_density_low, model_stats.gen_density_high = _bootstrap_density_band(
            gen_hist_counts,
            bins,
            reps=bootstrap_reps,
            ci_level=ci_level,
            sigma_bins=smooth_sigma,
            seed=seed + 2000 + 211 * class_id + report_idx,
        )
        gap_dist = _bootstrap_scalar_gap_distribution(
            real_values,
            gen_values,
            reps=bootstrap_reps,
            seed=seed + 3000 + 211 * class_id + report_idx,
        )
        if gap_dist is not None and gap_dist.size:
            model_stats.gap_ci = _quantile_interval(gap_dist, ci_level)
        if null_gap is not None and null_gap.size:
            model_stats.gap_null_q95 = float(np.quantile(null_gap, 0.95))
            model_stats.gap_pvalue = _empirical_pvalue(panel.abs_gap_fraction * 100.0, null_gap)
        out.model[report_idx] = model_stats
    return out


def _format_pct_points(value: float) -> str:
    value = float(value)
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _truncate_label(text: str, max_len: int = 18) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#4b5563",
            "axes.linewidth": 0.85,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "font.size": 10.3,
            "axes.titlesize": 12.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.2,
            "legend.fontsize": 10.0,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#ffffff",
        }
    )


def _format_pvalue(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1e-3:
        return "<0.001"
    return f"={value:.3f}"


def _draw_metrics_text(
    ax,
    reports: list[LoadedReport],
    inference: ClassInference,
    class_id: int,
) -> None:
    lines = []
    for idx, report in enumerate(reports):
        panel = report.panels[class_id]
        stats = inference.model.get(idx)
        label = _truncate_label(report.label, max_len=14)
        if stats is not None and stats.w1_ci is not None:
            lines.append(
                (
                    f"{label}  W1 {panel.w1:.2f} [{stats.w1_ci[0]:.2f}, {stats.w1_ci[1]:.2f}] K, "
                    f"p{_format_pvalue(stats.w1_pvalue)}",
                    MODEL_COLORS[idx],
                )
            )
        else:
            lines.append((f"{label}  W1 {panel.w1:.2f} K", MODEL_COLORS[idx]))

    text = "\n".join(line for line, _ in lines)
    ax.text(
        0.03,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        linespacing=1.38,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.06",
            "facecolor": (1.0, 1.0, 1.0, 0.86),
            "edgecolor": (1.0, 1.0, 1.0, 0.0),
        },
        zorder=8,
    )

    for idx, (_, color) in enumerate(lines):
        y = 0.946 - idx * 0.084
        ax.plot(
            [0.038, 0.055],
            [y - 0.012, y - 0.012],
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=9,
        )


def _draw_radial_metrics_text(
    ax,
    reports: list[LoadedReport],
    inference: RadialClassInference,
    class_id: int,
) -> None:
    lines = []
    for idx, report in enumerate(reports):
        panel = report.panels[class_id]
        stats = inference.model.get(idx)
        label = _truncate_label(report.label, max_len=14)
        if stats is not None and stats.mae_ci is not None:
            lines.append(
                (
                    f"{label}  MAE {panel.mae_k:.2f} [{stats.mae_ci[0]:.2f}, {stats.mae_ci[1]:.2f}] K, "
                    f"p{_format_pvalue(stats.mae_pvalue)}",
                    MODEL_COLORS[idx],
                )
            )
        else:
            lines.append((f"{label}  MAE {panel.mae_k:.2f} K", MODEL_COLORS[idx]))

    text = "\n".join(line for line, _ in lines)
    ax.text(
        0.03,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        linespacing=1.38,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.06",
            "facecolor": (1.0, 1.0, 1.0, 0.86),
            "edgecolor": (1.0, 1.0, 1.0, 0.0),
        },
        zorder=8,
    )

    for idx, (_, color) in enumerate(lines):
        y = 0.946 - idx * 0.084
        ax.plot(
            [0.038, 0.055],
            [y - 0.012, y - 0.012],
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=9,
        )


def _draw_psd_metrics_text(
    ax,
    reports: list[LoadedReport],
    inference: PSDClassInference,
    class_id: int,
) -> None:
    lines = []
    for idx, report in enumerate(reports):
        panel = report.panels[class_id]
        stats = inference.model.get(idx)
        label = _truncate_label(report.label, max_len=14)
        if stats is not None and stats.l2_ci is not None:
            lines.append(
                (
                    f"{label}  L2 {panel.l2:.3f} [{stats.l2_ci[0]:.3f}, {stats.l2_ci[1]:.3f}], "
                    f"p{_format_pvalue(stats.l2_pvalue)}",
                    MODEL_COLORS[idx],
                )
            )
        else:
            lines.append((f"{label}  L2 {panel.l2:.3f}", MODEL_COLORS[idx]))

    text = "\n".join(line for line, _ in lines)
    ax.text(
        0.03,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        linespacing=1.38,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.06",
            "facecolor": (1.0, 1.0, 1.0, 0.86),
            "edgecolor": (1.0, 1.0, 1.0, 0.0),
        },
        zorder=8,
    )

    for idx, (_, color) in enumerate(lines):
        y = 0.946 - idx * 0.084
        ax.plot(
            [0.038, 0.055],
            [y - 0.012, y - 0.012],
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=9,
        )


def _draw_dav_metrics_text(
    ax,
    reports: list[LoadedReport],
    inference: DAVClassInference,
    class_id: int,
) -> None:
    lines = []
    for idx, report in enumerate(reports):
        panel = report.panels[class_id]
        stats = inference.model.get(idx)
        label = _truncate_label(report.label, max_len=14)
        if stats is not None and stats.gap_ci is not None:
            lines.append(
                (
                    f"{label}  |Δμ| {_format_dav_value(panel.abs_gap_deg2)} "
                    f"[{_format_dav_value(stats.gap_ci[0])}, {_format_dav_value(stats.gap_ci[1])}] deg², "
                    f"p{_format_pvalue(stats.gap_pvalue)}",
                    MODEL_COLORS[idx],
                )
            )
        else:
            lines.append((f"{label}  |Δμ| {_format_dav_value(panel.abs_gap_deg2)} deg²", MODEL_COLORS[idx]))

    text = "\n".join(line for line, _ in lines)
    ax.text(
        0.03,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        linespacing=1.38,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.06",
            "facecolor": (1.0, 1.0, 1.0, 0.86),
            "edgecolor": (1.0, 1.0, 1.0, 0.0),
        },
        zorder=8,
    )

    for idx, (_, color) in enumerate(lines):
        y = 0.946 - idx * 0.084
        ax.plot(
            [0.038, 0.055],
            [y - 0.012, y - 0.012],
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=9,
        )


def _draw_cold_cloud_metrics_text(
    ax,
    reports: list[LoadedReport],
    inference: ColdCloudClassInference,
    class_id: int,
) -> None:
    lines = []
    for idx, report in enumerate(reports):
        panel = report.panels[class_id]
        stats = inference.model.get(idx)
        label = _truncate_label(report.label, max_len=14)
        gap_pp = panel.abs_gap_fraction * 100.0
        if stats is not None and stats.gap_ci is not None:
            lines.append(
                (
                    f"{label}  |Δμ| {_format_pct_points(gap_pp)} "
                    f"[{_format_pct_points(stats.gap_ci[0])}, {_format_pct_points(stats.gap_ci[1])}] pp, "
                    f"p{_format_pvalue(stats.gap_pvalue)}",
                    MODEL_COLORS[idx],
                )
            )
        else:
            lines.append((f"{label}  |Δμ| {_format_pct_points(gap_pp)} pp", MODEL_COLORS[idx]))

    text = "\n".join(line for line, _ in lines)
    ax.text(
        0.03,
        0.965,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        linespacing=1.38,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.34,rounding_size=0.06",
            "facecolor": (1.0, 1.0, 1.0, 0.86),
            "edgecolor": (1.0, 1.0, 1.0, 0.0),
        },
        zorder=8,
    )

    for idx, (_, color) in enumerate(lines):
        y = 0.946 - idx * 0.084
        ax.plot(
            [0.038, 0.055],
            [y - 0.012, y - 0.012],
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=9,
        )


def _plot_pixel_reports(
    reports: list[LoadedReport],
    out_path: Path,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    stats_seed: int,
) -> None:
    _validate_reports(reports, figure_kind=FIGURE_PIXEL)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    inference_by_class = {
        class_id: _compute_class_inference(
            reports,
            class_id,
            smooth_sigma=smooth_sigma,
            bootstrap_reps=int(bootstrap_reps),
            null_reps=int(null_reps),
            ci_level=float(ci_level),
            seed=int(stats_seed),
        )
        for class_id in class_ids
    }
    x_min = min(float(ref.panels[c].bins[0]) for c in class_ids)
    x_max = max(float(ref.panels[c].bins[-1]) for c in class_ids)

    display_cache: Dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    y_max = 0.0
    for class_id in class_ids:
        panel = ref.panels[class_id]
        display_cache[(0, class_id, "real")] = _display_curve(
            panel.real_density,
            panel.bins,
            smooth_sigma,
        )
        y_max = max(y_max, float(np.max(display_cache[(0, class_id, "real")][1])))
        inference = inference_by_class[class_id]
        if inference.real_density_low is not None and inference.real_density_high is not None:
            y_max = max(
                y_max,
                float(np.max(inference.real_density_low)),
                float(np.max(inference.real_density_high)),
            )
        for report_idx, report in enumerate(reports):
            curve = _display_curve(report.panels[class_id].gen_density, report.panels[class_id].bins, smooth_sigma)
            display_cache[(report_idx, class_id, "gen")] = curve
            y_max = max(y_max, float(np.max(curve[1])))
            model_inference = inference.model.get(report_idx)
            if model_inference is not None:
                if model_inference.gen_density_low is not None:
                    y_max = max(y_max, float(np.max(model_inference.gen_density_low)))
                if model_inference.gen_density_high is not None:
                    y_max = max(y_max, float(np.max(model_inference.gen_density_high)))

    y_top = y_max * 1.14 if y_max > 0.0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    panel_letters = "ABCDEF"

    for panel_idx, (ax, class_id) in enumerate(zip(flat_axes, class_ids)):
        panel = ref.panels[class_id]
        inference = inference_by_class[class_id]
        x_real, y_real = display_cache[(0, class_id, "real")]
        if inference.real_density_low is not None and inference.real_density_high is not None:
            low_x, low_y = _display_curve(inference.real_density_low, panel.bins, 0.0)
            high_x, high_y = _display_curve(inference.real_density_high, panel.bins, 0.0)
            ax.fill_between(
                low_x,
                low_y,
                high_y,
                color="#cbd5e1",
                alpha=0.42,
                zorder=1,
                linewidth=0.0,
            )
        ax.fill_between(x_real, 0.0, y_real, color="#dbe4ee", alpha=0.24, zorder=2)
        ax.plot(
            x_real,
            y_real,
            color=REAL_COLOR,
            linewidth=2.6,
            solid_capstyle="round",
            label="Real",
            zorder=5,
        )

        for idx, report in enumerate(reports):
            x_gen, y_gen = display_cache[(idx, class_id, "gen")]
            model_inference = inference.model.get(idx)
            if model_inference is not None and model_inference.gen_density_low is not None and model_inference.gen_density_high is not None:
                low_x, low_y = _display_curve(model_inference.gen_density_low, report.panels[class_id].bins, 0.0)
                high_x, high_y = _display_curve(model_inference.gen_density_high, report.panels[class_id].bins, 0.0)
                ax.fill_between(
                    low_x,
                    low_y,
                    high_y,
                    color=MODEL_COLORS[idx],
                    alpha=0.10,
                    zorder=3 + idx,
                    linewidth=0.0,
                )
            ax.plot(
                x_gen,
                y_gen,
                color=MODEL_COLORS[idx],
                linewidth=2.2,
                linestyle=MODEL_LINESTYLES[idx],
                solid_capstyle="round",
                label=report.label,
                zorder=6 + idx,
            )

        ax.set_title(f"{panel_letters[panel_idx]}   {panel.label}", loc="left", pad=8.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_top)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.75, alpha=0.42)
        ax.tick_params(length=3.2, width=0.8, color="#6b7280")
        ax.margins(x=0.0)
        _draw_metrics_text(ax, reports, inference, class_id)

    legend_handles = [Line2D([0], [0], color=REAL_COLOR, lw=2.8, label="Real")]
    for idx, report in enumerate(reports):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[idx],
                lw=2.15,
                linestyle=MODEL_LINESTYLES[idx],
                label=report.label,
            )
        )

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    fig.supxlabel("Brightness temperature [K]", y=0.055)
    fig.supylabel("Density [1/K]", x=0.04)
    fig.text(
        0.5,
        0.017,
        (
            f"Shaded bands: {int(round(ci_level * 100))}% image-level bootstrap intervals for the real and "
            "generated histograms. Reported p-values compare observed W1 against a matched real-vs-real null."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_radial_reports(
    reports: list[LoadedReport],
    out_path: Path,
    *,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    stats_seed: int,
) -> None:
    _validate_reports(reports, figure_kind=FIGURE_RADIAL)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    inference_by_class = {
        class_id: _compute_radial_class_inference(
            reports,
            class_id,
            bootstrap_reps=int(bootstrap_reps),
            null_reps=int(null_reps),
            ci_level=float(ci_level),
            seed=int(stats_seed),
        )
        for class_id in class_ids
    }

    x_min = min(float(ref.panels[c].radius[0]) for c in class_ids)
    x_max = max(float(ref.panels[c].radius[-1]) for c in class_ids)
    display_cache: Dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    y_min = float("inf")
    y_max = float("-inf")

    for class_id in class_ids:
        panel = ref.panels[class_id]
        display_cache[(0, class_id, "real")] = _display_profile_curve(panel.radius, panel.real_profile)
        y_vals = display_cache[(0, class_id, "real")][1]
        y_min = min(y_min, float(np.min(y_vals)))
        y_max = max(y_max, float(np.max(y_vals)))
        inference = inference_by_class[class_id]
        if inference.real_profile_low is not None and inference.real_profile_high is not None:
            y_min = min(y_min, float(np.min(inference.real_profile_low)), float(np.min(inference.real_profile_high)))
            y_max = max(y_max, float(np.max(inference.real_profile_low)), float(np.max(inference.real_profile_high)))
        for report_idx, report in enumerate(reports):
            curve = _display_profile_curve(report.panels[class_id].radius, report.panels[class_id].gen_profile)
            display_cache[(report_idx, class_id, "gen")] = curve
            y_min = min(y_min, float(np.min(curve[1])))
            y_max = max(y_max, float(np.max(curve[1])))
            model_inference = inference.model.get(report_idx)
            if model_inference is not None:
                if model_inference.gen_profile_low is not None:
                    y_min = min(y_min, float(np.min(model_inference.gen_profile_low)))
                    y_max = max(y_max, float(np.max(model_inference.gen_profile_low)))
                if model_inference.gen_profile_high is not None:
                    y_min = min(y_min, float(np.min(model_inference.gen_profile_high)))
                    y_max = max(y_max, float(np.max(model_inference.gen_profile_high)))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    pad = max(2.0, 0.06 * (y_max - y_min))
    y_lo = y_min - pad
    y_hi = y_max + pad

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    panel_letters = "ABCDEF"

    for panel_idx, (ax, class_id) in enumerate(zip(flat_axes, class_ids)):
        panel = ref.panels[class_id]
        inference = inference_by_class[class_id]
        x_real, y_real = display_cache[(0, class_id, "real")]
        if inference.real_profile_low is not None and inference.real_profile_high is not None:
            low_x, low_y = _display_profile_curve(panel.radius, inference.real_profile_low)
            high_x, high_y = _display_profile_curve(panel.radius, inference.real_profile_high)
            ax.fill_between(
                low_x,
                low_y,
                high_y,
                color="#cbd5e1",
                alpha=0.42,
                zorder=1,
                linewidth=0.0,
            )
        ax.plot(
            x_real,
            y_real,
            color=REAL_COLOR,
            linewidth=2.6,
            solid_capstyle="round",
            label="Real",
            zorder=5,
        )

        for idx, report in enumerate(reports):
            x_gen, y_gen = display_cache[(idx, class_id, "gen")]
            model_inference = inference.model.get(idx)
            if (
                model_inference is not None
                and model_inference.gen_profile_low is not None
                and model_inference.gen_profile_high is not None
            ):
                low_x, low_y = _display_profile_curve(report.panels[class_id].radius, model_inference.gen_profile_low)
                high_x, high_y = _display_profile_curve(report.panels[class_id].radius, model_inference.gen_profile_high)
                ax.fill_between(
                    low_x,
                    low_y,
                    high_y,
                    color=MODEL_COLORS[idx],
                    alpha=0.10,
                    zorder=3 + idx,
                    linewidth=0.0,
                )
            ax.plot(
                x_gen,
                y_gen,
                color=MODEL_COLORS[idx],
                linewidth=2.2,
                linestyle=MODEL_LINESTYLES[idx],
                solid_capstyle="round",
                label=report.label,
                zorder=6 + idx,
            )

        ax.set_title(f"{panel_letters[panel_idx]}   {panel.label}", loc="left", pad=8.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(np.linspace(0.0, 1.0, 5))
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.75, alpha=0.42)
        ax.tick_params(length=3.2, width=0.8, color="#6b7280")
        ax.margins(x=0.0)
        _draw_radial_metrics_text(ax, reports, inference, class_id)

    legend_handles = [Line2D([0], [0], color=REAL_COLOR, lw=2.8, label="Real")]
    for idx, report in enumerate(reports):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[idx],
                lw=2.15,
                linestyle=MODEL_LINESTYLES[idx],
                label=report.label,
            )
        )

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    fig.supxlabel("Normalized radius", y=0.055)
    fig.supylabel("Brightness temperature [K]", x=0.04)
    fig.text(
        0.5,
        0.017,
        (
            f"Shaded bands: {int(round(ci_level * 100))}% image-level bootstrap intervals for the real and "
            "generated mean radial BT profiles. Reported p-values compare observed profile MAE against a "
            "matched real-vs-real null."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_psd_reports(
    reports: list[LoadedReport],
    out_path: Path,
    *,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    stats_seed: int,
) -> None:
    _validate_reports(reports, figure_kind=FIGURE_PSD)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    inference_by_class = {
        class_id: _compute_psd_class_inference(
            reports,
            class_id,
            bootstrap_reps=int(bootstrap_reps),
            null_reps=int(null_reps),
            ci_level=float(ci_level),
            seed=int(stats_seed),
        )
        for class_id in class_ids
    }

    x_min = min(float(ref.panels[c].frequency[0]) for c in class_ids)
    x_max = max(float(ref.panels[c].frequency[-1]) for c in class_ids)
    display_cache: Dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    y_min = float("inf")
    y_max = float("-inf")

    for class_id in class_ids:
        panel = ref.panels[class_id]
        display_cache[(0, class_id, "real")] = _display_profile_curve(panel.frequency, panel.real_profile)
        y_vals = display_cache[(0, class_id, "real")][1]
        y_min = min(y_min, float(np.min(y_vals)))
        y_max = max(y_max, float(np.max(y_vals)))
        inference = inference_by_class[class_id]
        if inference.real_profile_low is not None and inference.real_profile_high is not None:
            y_min = min(y_min, float(np.min(inference.real_profile_low)), float(np.min(inference.real_profile_high)))
            y_max = max(y_max, float(np.max(inference.real_profile_low)), float(np.max(inference.real_profile_high)))
        for report_idx, report in enumerate(reports):
            curve = _display_profile_curve(report.panels[class_id].frequency, report.panels[class_id].gen_profile)
            display_cache[(report_idx, class_id, "gen")] = curve
            y_min = min(y_min, float(np.min(curve[1])))
            y_max = max(y_max, float(np.max(curve[1])))
            model_inference = inference.model.get(report_idx)
            if model_inference is not None:
                if model_inference.gen_profile_low is not None:
                    y_min = min(y_min, float(np.min(model_inference.gen_profile_low)))
                    y_max = max(y_max, float(np.max(model_inference.gen_profile_low)))
                if model_inference.gen_profile_high is not None:
                    y_min = min(y_min, float(np.min(model_inference.gen_profile_high)))
                    y_max = max(y_max, float(np.max(model_inference.gen_profile_high)))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    pad = max(0.05, 0.06 * (y_max - y_min))
    y_lo = y_min - pad
    y_hi = y_max + pad

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    panel_letters = "ABCDEF"

    for panel_idx, (ax, class_id) in enumerate(zip(flat_axes, class_ids)):
        panel = ref.panels[class_id]
        inference = inference_by_class[class_id]
        x_real, y_real = display_cache[(0, class_id, "real")]
        if inference.real_profile_low is not None and inference.real_profile_high is not None:
            low_x, low_y = _display_profile_curve(panel.frequency, inference.real_profile_low)
            high_x, high_y = _display_profile_curve(panel.frequency, inference.real_profile_high)
            ax.fill_between(
                low_x,
                low_y,
                high_y,
                color="#cbd5e1",
                alpha=0.42,
                zorder=1,
                linewidth=0.0,
            )
        ax.plot(
            x_real,
            y_real,
            color=REAL_COLOR,
            linewidth=2.6,
            solid_capstyle="round",
            label="Real",
            zorder=5,
        )

        for idx, report in enumerate(reports):
            x_gen, y_gen = display_cache[(idx, class_id, "gen")]
            model_inference = inference.model.get(idx)
            if (
                model_inference is not None
                and model_inference.gen_profile_low is not None
                and model_inference.gen_profile_high is not None
            ):
                low_x, low_y = _display_profile_curve(report.panels[class_id].frequency, model_inference.gen_profile_low)
                high_x, high_y = _display_profile_curve(report.panels[class_id].frequency, model_inference.gen_profile_high)
                ax.fill_between(
                    low_x,
                    low_y,
                    high_y,
                    color=MODEL_COLORS[idx],
                    alpha=0.10,
                    zorder=3 + idx,
                    linewidth=0.0,
                )
            ax.plot(
                x_gen,
                y_gen,
                color=MODEL_COLORS[idx],
                linewidth=2.2,
                linestyle=MODEL_LINESTYLES[idx],
                solid_capstyle="round",
                label=report.label,
                zorder=6 + idx,
            )

        ax.set_title(f"{panel_letters[panel_idx]}   {panel.label}", loc="left", pad=8.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(np.linspace(0.0, 1.0, 5))
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.75, alpha=0.42)
        ax.tick_params(length=3.2, width=0.8, color="#6b7280")
        ax.margins(x=0.0)
        _draw_psd_metrics_text(ax, reports, inference, class_id)

    legend_handles = [Line2D([0], [0], color=REAL_COLOR, lw=2.8, label="Real")]
    for idx, report in enumerate(reports):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[idx],
                lw=2.15,
                linestyle=MODEL_LINESTYLES[idx],
                label=report.label,
            )
        )

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    fig.supxlabel("Normalized radial frequency", y=0.055)
    fig.supylabel("Radial PSD log10 power", x=0.04)
    fig.text(
        0.5,
        0.017,
        (
            f"Shaded bands: {int(round(ci_level * 100))}% image-level bootstrap intervals for real and "
            "generated mean radial PSD profiles. Reported p-values compare observed PSD L2 against a "
            "matched real-vs-real null."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dav_reports(
    reports: list[LoadedReport],
    out_path: Path,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    stats_seed: int,
) -> None:
    _validate_reports(reports, figure_kind=FIGURE_DAV)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    bins = _build_scalar_bins(reports)
    widths = np.diff(bins)
    inference_by_class = {
        class_id: _compute_dav_class_inference(
            reports,
            class_id,
            bins,
            smooth_sigma=smooth_sigma,
            bootstrap_reps=int(bootstrap_reps),
            null_reps=int(null_reps),
            ci_level=float(ci_level),
            seed=int(stats_seed),
        )
        for class_id in class_ids
    }

    x_min = float(bins[0])
    x_max = float(bins[-1])
    display_cache: Dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    y_max = 0.0
    for class_id in class_ids:
        panel = ref.panels[class_id]
        real_density = _scalar_density_from_values(panel.real_values, bins)
        display_cache[(0, class_id, "real")] = _display_curve(real_density, bins, smooth_sigma)
        y_max = max(y_max, float(np.max(display_cache[(0, class_id, "real")][1])))
        inference = inference_by_class[class_id]
        if inference.real_density_low is not None and inference.real_density_high is not None:
            y_max = max(y_max, float(np.max(inference.real_density_low)), float(np.max(inference.real_density_high)))
        for report_idx, report in enumerate(reports):
            gen_density = _scalar_density_from_values(report.panels[class_id].gen_values, bins)
            curve = _display_curve(gen_density, bins, smooth_sigma)
            display_cache[(report_idx, class_id, "gen")] = curve
            y_max = max(y_max, float(np.max(curve[1])))
            model_inference = inference.model.get(report_idx)
            if model_inference is not None:
                if model_inference.gen_density_low is not None:
                    y_max = max(y_max, float(np.max(model_inference.gen_density_low)))
                if model_inference.gen_density_high is not None:
                    y_max = max(y_max, float(np.max(model_inference.gen_density_high)))

    y_top = y_max * 1.14 if y_max > 0.0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    panel_letters = "ABCDEF"

    for panel_idx, (ax, class_id) in enumerate(zip(flat_axes, class_ids)):
        panel = ref.panels[class_id]
        inference = inference_by_class[class_id]
        x_real, y_real = display_cache[(0, class_id, "real")]
        if inference.real_density_low is not None and inference.real_density_high is not None:
            low_x, low_y = _display_curve(inference.real_density_low, bins, 0.0)
            high_x, high_y = _display_curve(inference.real_density_high, bins, 0.0)
            ax.fill_between(
                low_x,
                low_y,
                high_y,
                color="#cbd5e1",
                alpha=0.42,
                zorder=1,
                linewidth=0.0,
            )
        ax.fill_between(x_real, 0.0, y_real, color="#dbe4ee", alpha=0.24, zorder=2)
        ax.plot(
            x_real,
            y_real,
            color=REAL_COLOR,
            linewidth=2.6,
            solid_capstyle="round",
            label="Real",
            zorder=5,
        )

        for idx, report in enumerate(reports):
            x_gen, y_gen = display_cache[(idx, class_id, "gen")]
            model_inference = inference.model.get(idx)
            if (
                model_inference is not None
                and model_inference.gen_density_low is not None
                and model_inference.gen_density_high is not None
            ):
                low_x, low_y = _display_curve(model_inference.gen_density_low, bins, 0.0)
                high_x, high_y = _display_curve(model_inference.gen_density_high, bins, 0.0)
                ax.fill_between(
                    low_x,
                    low_y,
                    high_y,
                    color=MODEL_COLORS[idx],
                    alpha=0.10,
                    zorder=3 + idx,
                    linewidth=0.0,
                )
            ax.plot(
                x_gen,
                y_gen,
                color=MODEL_COLORS[idx],
                linewidth=2.2,
                linestyle=MODEL_LINESTYLES[idx],
                solid_capstyle="round",
                label=report.label,
                zorder=6 + idx,
            )

        ax.set_title(f"{panel_letters[panel_idx]}   {panel.label}", loc="left", pad=8.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_top)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.75, alpha=0.42)
        ax.tick_params(length=3.2, width=0.8, color="#6b7280")
        ax.margins(x=0.0)
        _draw_dav_metrics_text(ax, reports, inference, class_id)

    legend_handles = [Line2D([0], [0], color=REAL_COLOR, lw=2.8, label="Real")]
    for idx, report in enumerate(reports):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[idx],
                lw=2.15,
                linestyle=MODEL_LINESTYLES[idx],
                label=report.label,
            )
        )

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    fig.supxlabel("Deviation angle variance [deg²]", y=0.055)
    fig.supylabel("Density [1/deg²]", x=0.04)
    fig.text(
        0.5,
        0.017,
        (
            f"Lower DAV indicates more axisymmetric cloud organization. Shaded bands: "
            f"{int(round(ci_level * 100))}% image-level bootstrap intervals for the real and generated "
            "DAV distributions. Reported p-values compare observed |Δμ_DAV| against a matched real-vs-real null."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cold_cloud_reports(
    reports: list[LoadedReport],
    out_path: Path,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    stats_seed: int,
) -> None:
    _validate_reports(reports, figure_kind=FIGURE_COLD)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    transformed_reports = []
    for report in reports:
        transformed_panels = {}
        for class_id in class_ids:
            panel = report.panels[class_id]
            transformed_panels[class_id] = (
                np.asarray(panel.real_values, dtype=np.float64) * 100.0,
                np.asarray(panel.gen_values, dtype=np.float64) * 100.0,
            )
        transformed_reports.append(transformed_panels)

    bins = _build_scalar_bins(
        [
            LoadedReport(
                figure_kind=FIGURE_COLD,
                run_name=report.run_name,
                label=report.label,
                eval_ref=report.eval_ref,
                split=report.split,
                tag=report.tag,
                n_per_class=report.n_per_class,
                metrics_path=report.metrics_path,
                panels={
                    class_id: DAVPanel(
                        class_id=class_id,
                        label=report.panels[class_id].label,
                        real_values=transformed_reports[idx][class_id][0],
                        gen_values=transformed_reports[idx][class_id][1],
                        real_mean=float(np.mean(transformed_reports[idx][class_id][0])),
                        gen_mean=float(np.mean(transformed_reports[idx][class_id][1])),
                        abs_gap_deg2=float(abs(np.mean(transformed_reports[idx][class_id][0]) - np.mean(transformed_reports[idx][class_id][1]))),
                    )
                    for class_id in class_ids
                },
            )
            for idx, report in enumerate(reports)
        ],
        bins_per_panel=56,
    )

    inference_by_class = {
        class_id: _compute_cold_cloud_class_inference(
            reports,
            class_id,
            bins,
            smooth_sigma=smooth_sigma,
            bootstrap_reps=int(bootstrap_reps),
            null_reps=int(null_reps),
            ci_level=float(ci_level),
            seed=int(stats_seed),
        )
        for class_id in class_ids
    }

    x_min = float(bins[0])
    x_max = float(bins[-1])
    display_cache: Dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    y_max = 0.0
    for class_id in class_ids:
        real_values = transformed_reports[0][class_id][0]
        real_density = _scalar_density_from_values(real_values, bins)
        display_cache[(0, class_id, "real")] = _display_curve(real_density, bins, smooth_sigma)
        y_max = max(y_max, float(np.max(display_cache[(0, class_id, "real")][1])))
        inference = inference_by_class[class_id]
        if inference.real_density_low is not None and inference.real_density_high is not None:
            y_max = max(y_max, float(np.max(inference.real_density_low)), float(np.max(inference.real_density_high)))
        for report_idx, _report in enumerate(reports):
            gen_values = transformed_reports[report_idx][class_id][1]
            gen_density = _scalar_density_from_values(gen_values, bins)
            curve = _display_curve(gen_density, bins, smooth_sigma)
            display_cache[(report_idx, class_id, "gen")] = curve
            y_max = max(y_max, float(np.max(curve[1])))
            model_inference = inference.model.get(report_idx)
            if model_inference is not None:
                if model_inference.gen_density_low is not None:
                    y_max = max(y_max, float(np.max(model_inference.gen_density_low)))
                if model_inference.gen_density_high is not None:
                    y_max = max(y_max, float(np.max(model_inference.gen_density_high)))

    y_top = y_max * 1.14 if y_max > 0.0 else 1.0
    threshold_k = float(ref.panels[class_ids[0]].threshold_k)
    radius_km = ref.panels[class_ids[0]].radius_km

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    panel_letters = "ABCDEF"

    for panel_idx, (ax, class_id) in enumerate(zip(flat_axes, class_ids)):
        panel = ref.panels[class_id]
        inference = inference_by_class[class_id]
        x_real, y_real = display_cache[(0, class_id, "real")]
        if inference.real_density_low is not None and inference.real_density_high is not None:
            low_x, low_y = _display_curve(inference.real_density_low, bins, 0.0)
            high_x, high_y = _display_curve(inference.real_density_high, bins, 0.0)
            ax.fill_between(
                low_x,
                low_y,
                high_y,
                color="#cbd5e1",
                alpha=0.42,
                zorder=1,
                linewidth=0.0,
            )
        ax.fill_between(x_real, 0.0, y_real, color="#dbe4ee", alpha=0.24, zorder=2)
        ax.plot(
            x_real,
            y_real,
            color=REAL_COLOR,
            linewidth=2.6,
            solid_capstyle="round",
            label="Real",
            zorder=5,
        )

        for idx, report in enumerate(reports):
            x_gen, y_gen = display_cache[(idx, class_id, "gen")]
            model_inference = inference.model.get(idx)
            if (
                model_inference is not None
                and model_inference.gen_density_low is not None
                and model_inference.gen_density_high is not None
            ):
                low_x, low_y = _display_curve(model_inference.gen_density_low, bins, 0.0)
                high_x, high_y = _display_curve(model_inference.gen_density_high, bins, 0.0)
                ax.fill_between(
                    low_x,
                    low_y,
                    high_y,
                    color=MODEL_COLORS[idx],
                    alpha=0.10,
                    zorder=3 + idx,
                    linewidth=0.0,
                )
            ax.plot(
                x_gen,
                y_gen,
                color=MODEL_COLORS[idx],
                linewidth=2.2,
                linestyle=MODEL_LINESTYLES[idx],
                solid_capstyle="round",
                label=report.label,
                zorder=6 + idx,
            )

        ax.set_title(f"{panel_letters[panel_idx]}   {panel.label}", loc="left", pad=8.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_top)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.75, alpha=0.42)
        ax.tick_params(length=3.2, width=0.8, color="#6b7280")
        ax.margins(x=0.0)
        _draw_cold_cloud_metrics_text(ax, reports, inference, class_id)

    legend_handles = [Line2D([0], [0], color=REAL_COLOR, lw=2.8, label="Real")]
    for idx, report in enumerate(reports):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[idx],
                lw=2.15,
                linestyle=MODEL_LINESTYLES[idx],
                label=report.label,
            )
        )

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.986),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    region_label = (
        f" within {radius_km:.0f} km"
        if radius_km is not None and np.isfinite(float(radius_km))
        else ""
    )
    fig.supxlabel(f"Cold-cloud fraction [% pixels < {threshold_k:.0f} K{region_label}]", y=0.055)
    fig.supylabel("Density [1/pp]", x=0.04)
    fig.text(
        0.5,
        0.017,
        (
            f"Shaded bands: {int(round(ci_level * 100))}% image-level bootstrap intervals for the real and "
            "generated cold-cloud-fraction distributions. Reported p-values compare observed |Δμ| in "
            "percentage points against a matched real-vs-real null."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_image_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = []
    for image in images:
        arr = np.asarray(image, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite.append(arr)
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite, axis=0)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        pad = max(0.5, abs(vmin) * 0.01)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _plot_memorization_pair_reports(
    reports: list[LoadedReport],
    out_path: Path,
) -> None:
    _validate_reports(reports, figure_kind=FIGURE_MEMORIZATION)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
    n_models = len(reports)
    nrows = len(class_ids)
    ncols = 2 * n_models
    displayed_pairs: Dict[tuple[int, int], MemorizationPair] = {}
    displayed_images = []

    for report_idx, report in enumerate(reports):
        for class_id in class_ids:
            pair = sorted(report.panels[class_id], key=lambda p: (p.rank_within_class, p.distance))[0]
            displayed_pairs[(report_idx, class_id)] = pair
            displayed_images.extend([pair.generated_bt, pair.train_bt])

    vmin, vmax = _resolve_image_limits(displayed_images)
    fig_width = max(8.6, 2.15 * ncols + 1.2)
    fig_height = max(10.2, 1.92 * nrows + 1.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    im_plot = None

    for row_idx, class_id in enumerate(class_ids):
        for report_idx, report in enumerate(reports):
            pair = displayed_pairs[(report_idx, class_id)]
            gen_ax = axes[row_idx, 2 * report_idx]
            train_ax = axes[row_idx, 2 * report_idx + 1]
            for ax in (gen_ax, train_ax):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_edgecolor("#e5e7eb")

            im_plot = gen_ax.imshow(
                pair.generated_bt,
                origin="lower",
                cmap=BT_CMAP,
                vmin=vmin,
                vmax=vmax,
            )
            train_ax.imshow(
                pair.train_bt,
                origin="lower",
                cmap=BT_CMAP,
                vmin=vmin,
                vmax=vmax,
            )

            if row_idx == 0:
                gen_ax.set_title(f"{_truncate_label(report.label, 18)}\nGenerated", fontsize=10.2, pad=7.0)
                train_ax.set_title("Nearest train", fontsize=10.2, pad=7.0)
            gen_ax.text(
                -0.08,
                0.5,
                f"{pair.label}\nd={pair.distance:.3f}",
                transform=gen_ax.transAxes,
                ha="right",
                va="center",
                fontsize=8.2,
                color="#111827",
                linespacing=1.35,
            )
            train_ax.text(
                0.02,
                0.02,
                f"g#{pair.generated_index}  t#{pair.train_index}",
                transform=train_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.8,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18,rounding_size=0.04",
                    "facecolor": (1.0, 1.0, 1.0, 0.75),
                    "edgecolor": (1.0, 1.0, 1.0, 0.0),
                },
            )

    if im_plot is not None:
        cbar = fig.colorbar(
            im_plot,
            ax=axes.ravel().tolist(),
            fraction=0.014,
            pad=0.012,
        )
        cbar.set_label("Brightness temperature [K]")

    fig.text(
        0.5,
        0.018,
        (
            "Each row shows the generated sample with the smallest same-class nearest-train distance in the "
            "evaluation feature space, paired with its nearest training image. Lower distances indicate higher "
            "memorization risk; this figure is qualitative context for the scalar nearest-train metric."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4b5563",
        wrap=True,
    )
    fig.tight_layout(rect=(0.045, 0.055, 0.97, 0.98), w_pad=0.08, h_pad=0.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_figure_kinds(args: argparse.Namespace) -> list[str]:
    if args.plot_only and args.figure and args.plot_only != args.figure:
        raise ValueError("Use either --plot_only or --figure, not both.")
    figure_kind = args.plot_only or args.figure
    if figure_kind is not None:
        return [str(figure_kind)]
    return list(DEFAULT_FIGURES)


def _figure_label(figure_kind: str) -> str:
    if figure_kind == FIGURE_PIXEL:
        return "pixel plausibility"
    if figure_kind == FIGURE_RADIAL:
        return "radial BT profile"
    if figure_kind == FIGURE_PSD:
        return "radial PSD profile"
    if figure_kind == FIGURE_DAV:
        return "DAV"
    if figure_kind == FIGURE_COLD:
        return "cold-cloud fraction"
    if figure_kind == FIGURE_MEMORIZATION:
        return "nearest-train memorization pairs"
    raise ValueError(f"Unsupported figure kind: {figure_kind}")


def _missing_stats_field(figure_kind: str) -> str:
    if figure_kind == FIGURE_PIXEL:
        return "histogram"
    if figure_kind == FIGURE_RADIAL:
        return "radial-profile"
    if figure_kind == FIGURE_PSD:
        return "radial-PSD-profile"
    if figure_kind == FIGURE_DAV:
        return "DAV"
    if figure_kind == FIGURE_COLD:
        return "cold-cloud-fraction"
    if figure_kind == FIGURE_MEMORIZATION:
        return "memorization-pair"
    raise ValueError(f"Unsupported figure kind: {figure_kind}")


def _report_missing_image_level_stats(reports: list[LoadedReport], *, figure_kind: str) -> bool:
    if figure_kind == FIGURE_PIXEL:
        return any(report.panels[0].real_hist_counts is None for report in reports)
    if figure_kind == FIGURE_RADIAL:
        return any(report.panels[0].real_profiles is None for report in reports)
    if figure_kind == FIGURE_PSD:
        return any(report.panels[0].real_profiles is None for report in reports)
    if figure_kind == FIGURE_DAV:
        return any(report.panels[0].real_values.size < 2 for report in reports)
    if figure_kind == FIGURE_COLD:
        return any(report.panels[0].real_values.size < 2 for report in reports)
    if figure_kind == FIGURE_MEMORIZATION:
        return False
    raise ValueError(f"Unsupported figure kind: {figure_kind}")


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap_reps must be >= 0.")
    if args.null_reps < 0:
        raise ValueError("--null_reps must be >= 0.")
    if not (0.0 < float(args.ci_level) < 1.0):
        raise ValueError("--ci_level must lie strictly between 0 and 1.")

    repo_root = _repo_root()
    specs = _resolve_inputs(args)
    figure_kinds = _resolve_figure_kinds(args)
    multiple_figures = len(figure_kinds) > 1

    loaded_by_figure: dict[str, list[LoadedReport]] = {}
    output_by_figure: dict[str, Path] = {}
    for figure_kind in figure_kinds:
        reports = [_load_report(repo_root, spec, figure_kind=figure_kind) for spec in specs]
        loaded_by_figure[figure_kind] = reports
        output_by_figure[figure_kind] = _resolve_output_path(
            repo_root,
            reports,
            args.output,
            figure_kind=figure_kind,
            multiple_figures=multiple_figures,
        )

    for figure_kind in figure_kinds:
        reports = loaded_by_figure[figure_kind]
        output_path = output_by_figure[figure_kind]
        if figure_kind == FIGURE_PIXEL:
            _plot_pixel_reports(
                reports,
                output_path,
                smooth_sigma=float(args.smooth_sigma),
                bootstrap_reps=int(args.bootstrap_reps),
                null_reps=int(args.null_reps),
                ci_level=float(args.ci_level),
                stats_seed=int(args.stats_seed),
            )
        elif figure_kind == FIGURE_RADIAL:
            _plot_radial_reports(
                reports,
                output_path,
                bootstrap_reps=int(args.bootstrap_reps),
                null_reps=int(args.null_reps),
                ci_level=float(args.ci_level),
                stats_seed=int(args.stats_seed),
            )
        elif figure_kind == FIGURE_PSD:
            _plot_psd_reports(
                reports,
                output_path,
                bootstrap_reps=int(args.bootstrap_reps),
                null_reps=int(args.null_reps),
                ci_level=float(args.ci_level),
                stats_seed=int(args.stats_seed),
            )
        elif figure_kind == FIGURE_DAV:
            _plot_dav_reports(
                reports,
                output_path,
                smooth_sigma=float(args.smooth_sigma),
                bootstrap_reps=int(args.bootstrap_reps),
                null_reps=int(args.null_reps),
                ci_level=float(args.ci_level),
                stats_seed=int(args.stats_seed),
            )
        elif figure_kind == FIGURE_COLD:
            _plot_cold_cloud_reports(
                reports,
                output_path,
                smooth_sigma=float(args.smooth_sigma),
                bootstrap_reps=int(args.bootstrap_reps),
                null_reps=int(args.null_reps),
                ci_level=float(args.ci_level),
                stats_seed=int(args.stats_seed),
            )
        elif figure_kind == FIGURE_MEMORIZATION:
            _plot_memorization_pair_reports(
                reports,
                output_path,
            )
        else:
            raise ValueError(f"Unsupported figure kind: {figure_kind}")

        source_list = ", ".join(f"{report.label} -> {report.metrics_path}" for report in reports)
        print(f"Saved paper-ready {_figure_label(figure_kind)} figure to: {output_path}")
        print(f"Loaded reports: {source_list}")
        if _report_missing_image_level_stats(reports, figure_kind=figure_kind):
            print(
                f"Note: loaded report artifacts do not include per-image {_missing_stats_field(figure_kind)} "
                "blocks, so the figure omits bootstrap ribbons and p-values. Re-run eval with the updated code "
                "to enable them."
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

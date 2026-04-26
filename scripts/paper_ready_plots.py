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

REAL_COLOR = "#111827"
MODEL_COLORS = ["#b65f2a", "#1f6f78", "#7b516d"]
MODEL_LINESTYLES = ["solid", (0, (6.0, 2.2)), (0, (2.5, 1.5))]


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
    run_name: str
    label: str
    eval_ref: str
    split: str
    tag: str
    n_per_class: int
    metrics_path: Path
    panels: Dict[int, ClassPanel]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready pixel plausibility figures from one to three saved evaluation reports. "
            "The script reads saved metrics.json summaries plus any linked sidecar artifacts; "
            "evaluation itself should be run ahead of time."
        )
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
            "Output figure path. Defaults to outputs/<model>/paper_ready/... for a single input "
            "or outputs/paper_ready/... for multi-model comparisons."
        ),
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.6,
        help="Gaussian smoothing sigma in histogram bins for displayed curves only.",
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


def _load_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
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
        run_name=spec.run_name,
        label=spec.label,
        eval_ref=spec.eval_ref,
        split=split,
        tag=str(report.get("tag", Path(spec.eval_ref).name)),
        n_per_class=int(report.get("n_per_class", -1)),
        metrics_path=metrics_path,
        panels=panels,
    )


def _validate_reports(reports: list[LoadedReport]) -> None:
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


def _slugify_token(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "model"


def _default_output_filename(reports: list[LoadedReport]) -> str:
    split_slug = reports[0].split
    model_slug = "_vs_".join(_slugify_token(report.label) for report in reports)
    return f"paper_ready_pixel_plausibility_{split_slug}_{model_slug}.png"


def _resolve_output_path(repo_root: Path, reports: list[LoadedReport], output: str | None) -> Path:
    default_name = _default_output_filename(reports)
    if output is None:
        if len(reports) == 1:
            return model_paper_ready_root(repo_root, reports[0].run_name) / default_name
        return repo_root / "outputs" / "paper_ready" / default_name

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    if out_path.suffix:
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


def _plot_reports(
    reports: list[LoadedReport],
    out_path: Path,
    *,
    smooth_sigma: float,
    bootstrap_reps: int,
    null_reps: int,
    ci_level: float,
    stats_seed: int,
) -> None:
    _validate_reports(reports)
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
    reports = [_load_report(repo_root, spec) for spec in specs]
    output_path = _resolve_output_path(repo_root, reports, args.output)
    _plot_reports(
        reports,
        output_path,
        smooth_sigma=float(args.smooth_sigma),
        bootstrap_reps=int(args.bootstrap_reps),
        null_reps=int(args.null_reps),
        ci_level=float(args.ci_level),
        stats_seed=int(args.stats_seed),
    )

    source_list = ", ".join(f"{report.label} -> {report.metrics_path}" for report in reports)
    print(f"Saved paper-ready pixel plausibility figure to: {output_path}")
    print(f"Loaded reports: {source_list}")
    if any(report.panels[0].real_hist_counts is None for report in reports):
        print(
            "Note: loaded report artifacts do not include per-image histogram blocks, "
            "so the figure omits bootstrap ribbons and p-values. Re-run eval with the updated code to enable them."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

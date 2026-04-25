from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from tc_diffusion.evaluation.metrics import js_divergence, wasserstein1_from_hist


DEFAULT_CLASS_LABELS = {
    0: "< Cat 1",
    1: "Cat 1",
    2: "Cat 2",
    3: "Cat 3",
    4: "Cat 4",
    5: "Cat 5",
}

REAL_COLOR = "#1f2937"
MODEL_COLORS = ["#c66a2b", "#2f7e79", "#8a4d74"]
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
            "The script only reads saved metrics.json files; evaluation itself should be run ahead of time."
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
        help="Output figure path. Defaults to outputs/paper_ready_pixel_plausibility_<split>_<models>.png.",
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.35,
        help="Gaussian smoothing sigma in histogram bins for displayed curves only.",
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


def _extract_pixel_payload(report: Dict[str, Any]) -> Dict[str, Any]:
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


def _default_label(class_id: int) -> str:
    return DEFAULT_CLASS_LABELS.get(class_id, f"Class {class_id}")


def _build_panel(
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
    real_density = _mass_to_density(real_mass, bins)
    gen_density = _mass_to_density(gen_mass, bins)

    metric_source = entry if not fallback_metrics else {**fallback_metrics, **entry}
    js = _first_present(metric_source, ("pixel_hist_js", "js_divergence", "js"))
    w1 = _first_present(metric_source, ("pixel_hist_w1", "wasserstein1", "w1"))
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
    )


def _infer_split_from_eval_ref(eval_ref: str) -> str:
    rel = Path(eval_ref.strip())
    if rel.parts and rel.parts[0] in {"val", "test"}:
        return rel.parts[0]
    if rel.parts and len(rel.parts) >= 2 and rel.parts[0] == "eval" and rel.parts[1] in {"val", "test"}:
        return rel.parts[1]
    return "val"


def _resolve_metrics_path(repo_root: Path, run_name: str, eval_ref: str) -> Path:
    run_dir = repo_root / "runs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    eval_ref = eval_ref.strip() or "test/post_training"
    rel = Path(eval_ref)
    if rel.is_absolute() or str(rel).endswith(".json") or rel.name == "metrics.json":
        raise ValueError(
            "Input specs should point to a run-relative evaluation folder, not directly to metrics.json."
        )

    candidates = []
    if rel.parts and rel.parts[0] == "eval":
        candidates.append(run_dir / rel / "metrics.json")
    else:
        candidates.append(run_dir / "eval" / rel / "metrics.json")
        if rel.parts and rel.parts[0] == "val" and len(rel.parts) >= 2:
            candidates.append(run_dir / "eval" / Path(*rel.parts[1:]) / "metrics.json")
        candidates.append(run_dir / rel / "metrics.json")

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


def _load_report(repo_root: Path, spec: InputSpec) -> LoadedReport:
    metrics_path = _resolve_metrics_path(repo_root, spec.run_name, spec.eval_ref)
    report = _load_json(metrics_path)
    payload = _extract_pixel_payload(report)
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

    per_class_metrics = report.get("per_class", {})
    class_ids = sorted(int(class_id) for class_id in per_class_payload.keys())
    expected_class_ids = list(range(6))
    if class_ids != expected_class_ids:
        raise ValueError(
            f"Expected classes 0..5 in {metrics_path}, found {class_ids}."
        )

    panels = {}
    for class_id in class_ids:
        entry = per_class_payload[str(class_id)]
        if not isinstance(entry, dict):
            raise ValueError(f"per_class[{class_id!r}] must be a JSON object in {metrics_path}.")
        fallback_metrics = per_class_metrics.get(str(class_id))
        label = str(entry.get("label") or class_labels.get(class_id) or _default_label(class_id))
        panels[class_id] = _build_panel(
            entry,
            class_id=class_id,
            label=label,
            normalization=normalization,
            fallback_metrics=fallback_metrics if isinstance(fallback_metrics, dict) else None,
        )

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
        return repo_root / "outputs" / default_name

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


def _truncate_label(text: str, max_len: int = 18) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f4ef",
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": "#b7b2a8",
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "savefig.facecolor": "#f6f4ef",
        }
    )


def _draw_metrics_box(ax, reports: list[LoadedReport], class_id: int) -> None:
    line_height = 0.082
    top = 0.965
    left = 0.03
    width = 0.60
    height = 0.052 + line_height * len(reports)
    patch = FancyBboxPatch(
        (left, top - height),
        width,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor=(1.0, 1.0, 1.0, 0.92),
        edgecolor="#d8d3c9",
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(patch)

    for idx, report in enumerate(reports):
        panel = report.panels[class_id]
        y = top - 0.028 - idx * line_height
        ax.text(
            left + 0.02,
            y,
            f"{_truncate_label(report.label)}: JS {panel.js:.4f} | W1 {panel.w1:.2f} K",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color=MODEL_COLORS[idx],
            zorder=3,
        )


def _plot_reports(reports: list[LoadedReport], out_path: Path, smooth_sigma: float) -> None:
    _validate_reports(reports)
    _configure_style()

    ref = reports[0]
    class_ids = sorted(ref.panels.keys())
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
        for report_idx, report in enumerate(reports):
            curve = _display_curve(report.panels[class_id].gen_density, report.panels[class_id].bins, smooth_sigma)
            display_cache[(report_idx, class_id, "gen")] = curve
            y_max = max(y_max, float(np.max(curve[1])))

    y_top = y_max * 1.10 if y_max > 0.0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), sharex=True, sharey=True)
    flat_axes = axes.ravel()

    for ax, class_id in zip(flat_axes, class_ids):
        panel = ref.panels[class_id]
        x_real, y_real = display_cache[(0, class_id, "real")]
        ax.fill_between(x_real, 0.0, y_real, color=REAL_COLOR, alpha=0.07, zorder=1)
        ax.plot(
            x_real,
            y_real,
            color=REAL_COLOR,
            linewidth=2.8,
            solid_capstyle="round",
            label="Real",
            zorder=4,
        )

        for idx, report in enumerate(reports):
            x_gen, y_gen = display_cache[(idx, class_id, "gen")]
            ax.plot(
                x_gen,
                y_gen,
                color=MODEL_COLORS[idx],
                linewidth=2.15,
                linestyle=MODEL_LINESTYLES[idx],
                solid_capstyle="round",
                label=report.label,
                zorder=5 + idx,
            )

        ax.set_title(panel.label, loc="left", pad=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_top)
        ax.grid(axis="y", color="#e5e1d8", linewidth=0.8, alpha=0.9)
        ax.tick_params(length=3.5, color="#8e877b")
        ax.margins(x=0.0)
        _draw_metrics_box(ax, reports, class_id)

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
        bbox_to_anchor=(0.5, 0.988),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.supxlabel("Brightness temperature [K]", y=0.04)
    fig.supylabel("Density [1/K]", x=0.045)
    fig.tight_layout(rect=(0.045, 0.055, 0.995, 0.93))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    specs = _resolve_inputs(args)
    reports = [_load_report(repo_root, spec) for spec in specs]
    output_path = _resolve_output_path(repo_root, reports, args.output)
    _plot_reports(reports, output_path, smooth_sigma=float(args.smooth_sigma))

    source_list = ", ".join(f"{report.label} -> {report.metrics_path}" for report in reports)
    print(f"Saved paper-ready pixel plausibility figure to: {output_path}")
    print(f"Loaded reports: {source_list}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

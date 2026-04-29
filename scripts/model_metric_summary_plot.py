from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tc_diffusion.sample_bank import model_outputs_root


MODEL_COLORS = ["#b65f2a", "#1f6f78", "#7b516d", "#4b5563"]
GROUP_COLORS = {
    "physical": "#f8fafc",
    "distributional": "#eef6f8",
    "memorization": "#fff7ed",
}


@dataclass
class InputSpec:
    run_name: str
    eval_ref: str
    label: str


@dataclass
class MetricSpec:
    key: str
    label: str
    group: str
    direction: str
    raw_format: str
    radar: bool = True


METRICS = [
    MetricSpec("hist_w1_k", "BT histogram W1", "physical", "lower", "{:.2f} K"),
    MetricSpec("dav_gap_deg2", "DAV gap", "physical", "lower", "{:.1f} deg$^2$"),
    MetricSpec("radial_mae_k", "Radial profile MAE", "physical", "lower", "{:.2f} K"),
    MetricSpec("cold_gap_pct", "Cold-cloud gap", "physical", "lower", "{:.2f} pp"),
    MetricSpec("psd_l2", "Radial PSD L2", "physical", "lower", "{:.3f}"),
    MetricSpec("eye_gap_k", "Eye contrast gap", "physical", "lower", "{:.2f} K"),
    MetricSpec("evaluator_fd", "Evaluator FD", "distributional", "lower_empirical", "{:.3f}"),
    MetricSpec("diversity_closeness", "Diversity match", "distributional", "higher", "{:.3f}"),
    MetricSpec("coverage", "Coverage", "distributional", "higher", "{:.3f}"),
    MetricSpec("memorization_q01", "Train NN q01", "memorization", "higher_relative", "{:.3f}", radar=False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a compact model-comparison summary for physical and distributional TC-diffusion metrics. "
            "The primary panel is a normalized higher-is-better dot plot with raw values annotated; "
            "the secondary radar panel gives a compact relative overview excluding memorization."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Repeatable model input spec: [Label=]RUN_NAME[:SPLIT/EVAL_TAG]. "
            "Use one --input per model; any number of models is supported."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to outputs/paper_ready/model_metric_summary.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TC Diffusion Model Metric Summary",
        help="Figure title.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_input_spec(raw: str) -> InputSpec:
    text = str(raw).strip()
    if not text:
        raise ValueError("--input cannot be empty.")

    if "=" in text:
        label, text = text.split("=", 1)
        label = label.strip()
    else:
        label = ""

    if ":" in text:
        run_name, eval_ref = text.split(":", 1)
    else:
        run_name, eval_ref = text, "test/final_eval_500_distributional"

    run_name = run_name.strip()
    eval_ref = eval_ref.strip().strip("/")
    if not run_name:
        raise ValueError(f"Invalid input spec {raw!r}: run name is empty.")
    if "/" not in eval_ref:
        raise ValueError(
            f"Invalid eval ref {eval_ref!r} in input {raw!r}; expected SPLIT/EVAL_TAG."
        )
    if not label:
        label = run_name
    return InputSpec(run_name=run_name, eval_ref=eval_ref, label=label)


def _eval_ref_suffix(eval_ref: str) -> str:
    rel = Path(str(eval_ref).strip())
    if rel.name:
        return rel.name
    text = str(eval_ref).strip().replace("/", ":")
    return text or "eval"


def _with_unique_display_labels(specs: list[InputSpec]) -> list[InputSpec]:
    base_counts: Dict[str, int] = {}
    for spec in specs:
        base_counts[spec.label] = base_counts.get(spec.label, 0) + 1

    used: set[str] = set()
    out: list[InputSpec] = []
    for spec in specs:
        if base_counts.get(spec.label, 0) <= 1:
            candidate = spec.label
        else:
            candidate = f"{spec.label} [{_eval_ref_suffix(spec.eval_ref)}]"
            if candidate in used:
                candidate = f"{spec.label} [{spec.eval_ref.replace('/', ':')}]"

        if candidate in used:
            stem = candidate
            suffix = 2
            while f"{stem} #{suffix}" in used:
                suffix += 1
            candidate = f"{stem} #{suffix}"

        used.add(candidate)
        out.append(InputSpec(run_name=spec.run_name, eval_ref=spec.eval_ref, label=candidate))
    return out


def _metrics_path(repo: Path, spec: InputSpec) -> Path:
    split, tag = spec.eval_ref.split("/", 1)
    return model_outputs_root(repo, spec.run_name) / "eval" / split / tag / "metrics.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _nested_get(node: Dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = node
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_radial_mae(metrics_path: Path, report: Dict[str, Any]) -> float:
    radial_info = _nested_get(report, ("paper_ready", "radial_bt_profile"))
    if not isinstance(radial_info, dict):
        raise ValueError(f"{metrics_path} is missing paper_ready.radial_bt_profile.")
    raw_path = radial_info.get("path")
    if not raw_path:
        raise ValueError(f"{metrics_path} radial artifact is missing a path.")
    artifact_path = Path(str(raw_path))
    if not artifact_path.is_absolute():
        artifact_path = metrics_path.parent / artifact_path
    with np.load(artifact_path) as data:
        return float(np.mean(np.asarray(data["radial_profile_mae_k"], dtype=np.float64)))


def _fd_bad_anchor(report: Dict[str, Any]) -> float | None:
    controls = _nested_get(report, ("evaluator_feature_metric", "negative_controls", "controls"), {})
    if not isinstance(controls, dict):
        return None
    values = []
    for name in ("gaussian_blur", "pixel_shuffle"):
        node = controls.get(name)
        if isinstance(node, dict) and node.get("value") is not None:
            values.append(float(node["value"]))
    if not values:
        return None
    return float(min(values))


def _extract_metrics(metrics_path: Path) -> tuple[Dict[str, float], Dict[str, Any]]:
    report = _load_json(metrics_path)
    agg = report.get("aggregate_primary_raw", {})
    dist = report.get("distributional_feature_metric", {})

    if not isinstance(agg, dict):
        raise ValueError(f"{metrics_path} is missing aggregate_primary_raw.")

    required_aggregate_keys = {
        "pixel_hist_w1": "BT histogram W1",
        "dav_abs_gap_deg2": "DAV gap",
        "cold_cloud_fraction_200K_abs_gap": "cold-cloud fraction gap",
        "psd_l2": "radial PSD L2",
        "eye_contrast_proxy_abs_gap": "eye contrast proxy gap",
        "evaluator_fd": "evaluator FD",
        "evaluator_embedding_diversity_ratio": "evaluator-embedding diversity ratio",
        "evaluator_embedding_coverage": "evaluator-embedding coverage",
        "nearest_train_q01_distance": "nearest-train memorization q01 distance",
    }
    missing = [key for key in required_aggregate_keys if key not in agg]
    if missing:
        names = ", ".join(f"{key} ({required_aggregate_keys[key]})" for key in missing)
        fd_report = report.get("evaluator_feature_metric", {})
        dist_report = report.get("distributional_feature_metric", {})
        fd_reason = fd_report.get("reason") if isinstance(fd_report, dict) else None
        dist_reason = dist_report.get("reason") if isinstance(dist_report, dict) else None
        hint_parts = [
            f"{metrics_path} is missing required comparison metric(s): {names}.",
            "This usually means the evaluation was run without evaluator/distributional feature metrics.",
        ]
        if fd_reason:
            hint_parts.append(f"evaluator_feature_metric reason: {fd_reason}")
        if dist_reason:
            hint_parts.append(f"distributional_feature_metric reason: {dist_reason}")
        hint_parts.append(
            "Re-run eval with --split test --full_test --sample_bank <bank> and "
            "--override evaluation.evaluator_feature_metric.enabled=true."
        )
        raise ValueError(" ".join(hint_parts))

    values = {
        "hist_w1_k": float(agg["pixel_hist_w1"]),
        "dav_gap_deg2": float(agg["dav_abs_gap_deg2"]),
        "radial_mae_k": _load_radial_mae(metrics_path, report),
        "cold_gap_pct": 100.0 * float(agg["cold_cloud_fraction_200K_abs_gap"]),
        "psd_l2": float(agg["psd_l2"]),
        "eye_gap_k": float(agg["eye_contrast_proxy_abs_gap"]),
        "evaluator_fd": float(agg["evaluator_fd"]),
        "diversity_closeness": float(
            np.exp(-abs(np.log(float(agg["evaluator_embedding_diversity_ratio"]))))
        ),
        "coverage": float(agg["evaluator_embedding_coverage"]),
        "memorization_q01": float(agg["nearest_train_q01_distance"]),
    }
    meta = {
        "metrics_path": str(metrics_path),
        "fd_bad_anchor": _fd_bad_anchor(report),
        "diversity_ratio": float(agg["evaluator_embedding_diversity_ratio"]),
        "memorization_mean": _nested_get(
            dist,
            ("macro", "nearest_train_memorization", "mean_generated_to_train_nn"),
        ),
    }
    return values, meta


def _score_metric(
    *,
    spec: MetricSpec,
    value: float,
    all_values: np.ndarray,
    fd_bad_anchor: float | None,
) -> float:
    value = float(value)
    all_values = np.asarray(all_values, dtype=np.float64)
    if spec.direction == "higher":
        if spec.key in {"coverage", "diversity_closeness"}:
            return float(np.clip(value, 0.0, 1.0))
        best = float(np.max(all_values))
        return float(value / best) if best > 0.0 else 0.0

    if spec.direction == "higher_relative":
        best = float(np.max(all_values))
        return float(value / best) if best > 0.0 else 0.0

    if spec.direction == "lower_empirical" and fd_bad_anchor is not None and fd_bad_anchor > 0.0:
        return float(np.clip(1.0 - value / fd_bad_anchor, 0.0, 1.0))

    best = float(np.min(all_values))
    if value <= 0.0:
        return 1.0
    return float(np.clip(best / value, 0.0, 1.0))


def _build_score_matrix(
    values_by_label: Dict[str, Dict[str, float]],
    meta_by_label: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {label: {} for label in values_by_label}
    for spec in METRICS:
        all_values = np.asarray(
            [values_by_label[label][spec.key] for label in values_by_label],
            dtype=np.float64,
        )
        shared_fd_anchor = None
        if spec.direction == "lower_empirical":
            anchors = [
                meta_by_label[label].get("fd_bad_anchor")
                for label in values_by_label
                if meta_by_label[label].get("fd_bad_anchor") is not None
            ]
            if anchors:
                shared_fd_anchor = float(np.mean(anchors))
        for label in values_by_label:
            out[label][spec.key] = _score_metric(
                spec=spec,
                value=values_by_label[label][spec.key],
                all_values=all_values,
                fd_bad_anchor=shared_fd_anchor,
            )
    return out


def _format_raw(spec: MetricSpec, value: float, meta: Dict[str, Any] | None = None) -> str:
    if spec.key == "diversity_closeness" and meta is not None:
        return f"{value:.3f} (ratio {meta['diversity_ratio']:.2f})"
    return spec.raw_format.format(float(value))


def _plot_summary(
    *,
    values_by_label: Dict[str, Dict[str, float]],
    scores_by_label: Dict[str, Dict[str, float]],
    meta_by_label: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    labels = list(values_by_label.keys())
    cmap = plt.get_cmap("tab20")
    colors = {
        label: (MODEL_COLORS[i] if i < len(MODEL_COLORS) else cmap(i % cmap.N))
        for i, label in enumerate(labels)
    }
    metric_specs = METRICS
    y = np.arange(len(metric_specs), dtype=np.float64)
    n_models = len(labels)
    if n_models == 1:
        y_offsets = np.zeros((1,), dtype=np.float64)
    else:
        spread = min(0.34, 0.055 * max(n_models - 1, 1))
        y_offsets = np.linspace(-spread, spread, n_models, dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "savefig.facecolor": "#ffffff",
        }
    )
    fig_width = max(13.5, 11.5 + 0.45 * n_models)
    fig = plt.figure(figsize=(fig_width, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_radar = fig.add_subplot(gs[0, 1], projection="polar")

    group_spans = []
    start = 0
    for idx, spec in enumerate(metric_specs + [None]):
        if spec is None or spec.group != metric_specs[start].group:
            group = metric_specs[start].group
            group_spans.append((group, start, idx - 1))
            start = idx
    for group, lo, hi in group_spans:
        ax.axhspan(lo - 0.5, hi + 0.5, color=GROUP_COLORS.get(group, "#ffffff"), zorder=0)
        ax.text(
            -0.08,
            (lo + hi) / 2.0,
            group.upper(),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color="#64748b",
            rotation=90,
        )

    for idx, spec in enumerate(metric_specs):
        xs = [scores_by_label[label][spec.key] for label in labels]
        if len(xs) >= 2:
            ax.plot([min(xs), max(xs)], [idx, idx], color="#cbd5e1", linewidth=1.4, zorder=1)
        for label_idx, label in enumerate(labels):
            x = scores_by_label[label][spec.key]
            y_pos = idx + float(y_offsets[label_idx])
            ax.scatter(x, y_pos, s=58, color=colors[label], edgecolor="white", linewidth=0.8, zorder=3)
            raw = _format_raw(spec, values_by_label[label][spec.key], meta_by_label[label])
            if n_models <= 4:
                dx = 0.018 if x < 0.82 else -0.018
                ha = "left" if x < 0.82 else "right"
                ax.text(
                    x + dx,
                    y_pos,
                    raw,
                    color=colors[label],
                    ha=ha,
                    va="center",
                    fontsize=7.4,
                )

    ax.set_yticks(y)
    ax.set_yticklabels([spec.label for spec in metric_specs])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel("Normalized display score (higher is better)")
    title_suffix = "with raw values" if n_models <= 4 else "(raw values omitted to avoid label crowding)"
    ax.set_title(f"Metric-by-metric comparison {title_suffix}")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[label], label=label, markersize=7)
        for label in labels
    ]
    legend_cols = min(max(1, n_models), 4)
    ax.legend(handles=handles, loc="lower right", frameon=False, ncols=legend_cols)

    radar_specs = [spec for spec in metric_specs if spec.radar]
    theta = np.linspace(0.0, 2.0 * np.pi, len(radar_specs), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    for label in labels:
        vals = np.asarray([scores_by_label[label][spec.key] for spec in radar_specs], dtype=np.float64)
        vals_closed = np.concatenate([vals, vals[:1]])
        ax_radar.plot(theta_closed, vals_closed, color=colors[label], linewidth=2.0, label=label)
        ax_radar.fill(theta_closed, vals_closed, color=colors[label], alpha=0.12)
    ax_radar.set_xticks(theta)
    ax_radar.set_xticklabels([spec.label.replace(" ", "\n") for spec in radar_specs], fontsize=8)
    ax_radar.set_ylim(0.0, 1.0)
    ax_radar.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#64748b")
    ax_radar.grid(color="#dbe3ea", linewidth=0.8)
    ax_radar.spines["polar"].set_color("#cbd5e1")
    ax_radar.set_title("Compact score profile\n(memorization shown only at left)", pad=22)

    if n_models > 4:
        raw_lines = []
        for spec in metric_specs:
            entries = [
                f"{label}: {_format_raw(spec, values_by_label[label][spec.key], meta_by_label[label])}"
                for label in labels
            ]
            raw_lines.append(f"{spec.label}: " + "; ".join(entries))
        raw_path = output_path.with_suffix(".raw_values.txt")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("\n".join(raw_lines) + "\n")

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.text(
        0.01,
        0.01,
        (
            "Normalization: lower-is-better metrics use best/value unless an empirical FD bad-control anchor is "
            "available; coverage and diversity-match are already bounded in [0,1]; train NN q01 is relative "
            "and should be read as memorization-risk context, not model quality."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#475569",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    specs = _with_unique_display_labels([_parse_input_spec(raw) for raw in args.input])
    if len(specs) < 2:
        raise ValueError("Pass at least two --input specs to compare models.")

    values_by_label: Dict[str, Dict[str, float]] = {}
    meta_by_label: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        path = _metrics_path(repo, spec)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found for input {spec}: {path}")
        values, meta = _extract_metrics(path)
        values_by_label[spec.label] = values
        meta_by_label[spec.label] = meta

    scores_by_label = _build_score_matrix(values_by_label, meta_by_label)
    output_path = Path(args.output) if args.output else repo / "outputs" / "paper_ready" / "model_metric_summary.png"
    if not output_path.is_absolute():
        output_path = repo / output_path
    _plot_summary(
        values_by_label=values_by_label,
        scores_by_label=scores_by_label,
        meta_by_label=meta_by_label,
        output_path=output_path,
        title=str(args.title),
    )
    print(f"Wrote model metric summary figure to: {output_path}")


if __name__ == "__main__":
    main()

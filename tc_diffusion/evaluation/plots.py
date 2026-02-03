# tc_diffusion/evaluation/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_radial_profiles(
    out_path: Path,
    r: np.ndarray,
    real_mean: np.ndarray,
    real_std: np.ndarray,
    gen_mean: np.ndarray,
    gen_std: np.ndarray,
    title: str,
    ylabel: str,
):
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    plt.plot(r, real_mean, label="Real")
    plt.fill_between(r, real_mean - real_std, real_mean + real_std, alpha=0.2)
    plt.plot(r, gen_mean, label="Generated")
    plt.fill_between(r, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2)
    plt.xlabel("Radius bin (normalized)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_psd(
    out_path: Path,
    k: np.ndarray,
    real_mean: np.ndarray,
    real_std: np.ndarray,
    gen_mean: np.ndarray,
    gen_std: np.ndarray,
    title: str,
):
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    plt.plot(k, real_mean, label="Real")
    plt.fill_between(k, real_mean - real_std, real_mean + real_std, alpha=0.2)
    plt.plot(k, gen_mean, label="Generated")
    plt.fill_between(k, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2)
    plt.xlabel("Radial frequency bin (normalized)")
    plt.ylabel("log10 Power")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist_overlay(
    out_path: Path,
    bins: np.ndarray,
    real_hist: np.ndarray,
    gen_hist: np.ndarray,
    title: str,
    xlabel: str,
):
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.plot(centers, real_hist, label="Real")
    plt.plot(centers, gen_hist, label="Generated")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

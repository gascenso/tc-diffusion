# tc_diffusion/plotting.py
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_image_grid(x, path, bt_min_k, bt_max_k, ncols=4):
    """
    x: np.array or tf.Tensor of shape (B, H, W, 1) in [-1, 1].
    Save a grid of images to 'path' (PNG).
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    x = np.asarray(x)
    B, H, W, C = x.shape
    assert C == 1

    # Map back from [-1, 1] to [bt_min, bt_max] for nicer display
    bt_norm01 = (x + 1.0) / 2.0
    bt_k = bt_norm01 * (bt_max_k - bt_min_k) + bt_min_k

    ncols = min(ncols, B)
    nrows = int(np.ceil(B / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        ax.axis("off")

        if i < B:
            im = bt_k[i, ..., 0]
            im_plot = ax.imshow(im, origin="lower", cmap='gist_ncar')
        else:
            ax.imshow(np.zeros((H, W)), origin="lower")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _resolve_shared_limits(*arrays):
    finite_values = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            finite_values.append(flat)

    if not finite_values:
        return 0.0, 1.0

    finite = np.concatenate(finite_values, axis=0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))

    if vmin == vmax:
        pad = 0.5 if vmin == 0.0 else max(0.5, abs(vmin) * 0.01)
        vmin -= pad
        vmax += pad

    return vmin, vmax


def save_real_generated_comparison_grid(
    *,
    real_k,
    gen_k,
    path,
    n_show=25,
    ncols=5,
    cmap="gist_ncar",
    real_title="Real",
    gen_title="Generated",
    colorbar_label="Brightness temperature [K]",
    suptitle=None,
):
    """
    real_k, gen_k: np.array-like of shape (B, H, W) in Kelvin.
    Saves side-by-side real/generated grids with one shared color scale and colorbar.
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    real_k = np.asarray(real_k, dtype=np.float32)
    gen_k = np.asarray(gen_k, dtype=np.float32)

    if real_k.ndim != 3 or gen_k.ndim != 3:
        raise ValueError(
            "save_real_generated_comparison_grid expects real_k and gen_k with shape (B, H, W)."
        )
    if real_k.shape[0] <= 0 or gen_k.shape[0] <= 0:
        raise ValueError("save_real_generated_comparison_grid requires at least one real and one generated sample.")

    n_show = int(n_show)
    if n_show <= 0:
        raise ValueError(f"n_show must be > 0, got {n_show}")

    n_show = min(n_show, real_k.shape[0], gen_k.shape[0])
    ncols = max(1, min(int(ncols), n_show))
    nrows = int(np.ceil(n_show / ncols))

    real_show = real_k[:n_show]
    gen_show = gen_k[:n_show]
    vmin, vmax = _resolve_shared_limits(real_show, gen_show)

    fig = plt.figure(figsize=(2.6 * ncols * 2 + 1.2, 2.6 * nrows + 1.4))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.05)
    real_grid = outer[0, 0].subgridspec(nrows, ncols, wspace=0.02, hspace=0.04)
    gen_grid = outer[0, 1].subgridspec(nrows, ncols, wspace=0.02, hspace=0.04)

    real_axes = []
    gen_axes = []
    im_plot = None

    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax_real = fig.add_subplot(real_grid[r, c])
        ax_gen = fig.add_subplot(gen_grid[r, c])

        for ax in (ax_real, ax_gen):
            ax.set_xticks([])
            ax.set_yticks([])

        if i < n_show:
            im_plot = ax_real.imshow(real_show[i], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_gen.imshow(gen_show[i], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax_real.axis("off")
            ax_gen.axis("off")

        real_axes.append(ax_real)
        gen_axes.append(ax_gen)

    title_col = min(ncols - 1, max(0, ncols // 2))
    real_axes[title_col].set_title(real_title, fontsize=12, pad=10)
    gen_axes[title_col].set_title(gen_title, fontsize=12, pad=10)

    cax = fig.add_subplot(outer[0, 2])
    cbar = fig.colorbar(im_plot, cax=cax)
    cbar.set_label(colorbar_label)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)
        fig.subplots_adjust(top=0.90)
    else:
        fig.subplots_adjust(top=0.94)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(x, loss, path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(x, loss, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (epsilon MSE)")
    plt.title("DDPM Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

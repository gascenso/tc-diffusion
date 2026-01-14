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
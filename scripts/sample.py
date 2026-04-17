# scripts/sample.py

# Usage:
# (unconditional sampling)      python -m scripts.sample --name <RUN_NAME> --uncond
# (conditional sampling, Cat3)  python -m scripts.sample --name <RUN_NAME> --ss_cat 3
# (conditional with CFG)        python -m scripts.sample --name <RUN_NAME> --ss_cat 3 --guidance_scale 2.0
# (DDIM sampling)               python -m scripts.sample --name <RUN_NAME> --sampler ddim --ddim_steps 50

import argparse
from pathlib import Path

import tensorflow as tf

from tc_diffusion.config import load_config
from tc_diffusion.model_unet import build_unet
from tc_diffusion.diffusion import Diffusion
from tc_diffusion.plotting import save_image_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path. Defaults to runs/<name>/config.yaml saved at training time.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    p.add_argument(
        "--ss_cat",
        type=int,
        default=5,
        help=(
            "Saffir–Simpson category index to condition on:\n"
            "0=TS (35–63 kt), 1=Cat1, 2=Cat2, 3=Cat3, 4=Cat4, 5=Cat5."
        ),
    )
    p.add_argument(
        "--wind_kt",
        type=float,
        default=None,
        help="Continuous wind conditioning value [kt]. If omitted, class midpoint is used.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="baseline_ddpm_tc",
        help="Run name under runs/ to load weights from.",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. 0 disables CFG. Typical: 1-5."
    )
    p.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["ddpm", "ddim"],
        help="Sampler to use. Defaults to evaluation.sampler from the config.",
    )
    p.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="Number of DDIM steps. Defaults to evaluation.ddim_steps or the full schedule.",
    )
    p.add_argument(
        "--ddim_eta",
        type=float,
        default=None,
        help="DDIM stochasticity. 0.0 gives deterministic DDIM. Defaults to evaluation.ddim_eta.",
    )
    p.add_argument(
        "--uncond",
        action="store_true",
        help="Unconditional sampling (ignore --ss_cat and use null label)."
    )
    p.add_argument(
        "--use_ema",
        action="store_true",
        help="If set, load EMA weights (weights_ema_best_val.weights.h5) instead of raw best weights.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = Path("runs") / args.name
    config_path = Path(args.config) if args.config else (run_dir / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Pass --config explicitly or ensure the run has a saved config.yaml."
        )
    cfg = load_config(str(config_path), overrides=[])

    # GPU memory growth (optional, same as train)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    image_size = int(cfg["data"]["image_size"])

    model = build_unet(cfg)
    diffusion = Diffusion(cfg)

    def resolve_weights_path(use_ema: bool) -> Path:
        # Prefer the newer naming used by train_loop.py, but fall back to legacy names if present.
        if use_ema:
            candidates = [
                run_dir / "weights_ema_best_val.weights.h5",
                run_dir / "weights_ema_best.weights.h5",
            ]
        else:
            candidates = [
                run_dir / "weights_best_val.weights.h5",
                run_dir / "weights_best.weights.h5",
            ]

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Could not find weights. Tried: {[str(p) for p in candidates]}"
        )

    weights_path = resolve_weights_path(args.use_ema)
    print(f"Loading weights from {weights_path}")
    model.load_weights(str(weights_path))

    cond_value = None if args.uncond else args.ss_cat
    wind_value_kt = None if args.uncond else args.wind_kt
    eval_cfg = cfg.get("evaluation", {})
    sampler = args.sampler or str(eval_cfg.get("sampler", "ddpm"))
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else eval_cfg.get("ddim_steps", None)
    ddim_eta = (
        float(args.ddim_eta)
        if args.ddim_eta is not None
        else float(eval_cfg.get("ddim_eta", 0.0))
    )

    print("Sampling...")
    sample_outputs = diffusion.sample(
        model=model,
        batch_size=args.batch_size,
        image_size=image_size,
        cond_value=cond_value,
        wind_value_kt=wind_value_kt,
        guidance_scale=args.guidance_scale,
        sampler=sampler,
        num_sampling_steps=ddim_steps,
        ddim_eta=ddim_eta,
        return_both=True,
    )
    x_samples = sample_outputs["raw_final"]

    out_path = f"runs/{args.name}/sample.png"
    save_image_grid(
        x_samples,
        path=str(out_path),
        bt_min_k=float(cfg["data"]["bt_min_k"]),
        bt_max_k=float(cfg["data"]["bt_max_k"]),
    )
    print(f"Saved sample grid to {out_path}")

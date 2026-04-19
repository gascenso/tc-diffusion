import numpy as np

from tc_diffusion.diffusion import Diffusion


def _make_cfg(
    *,
    num_steps: int = 100,
    beta_schedule: str = "linear",
    rescale_zero_terminal_snr: bool = False,
    timestep_schedule: str = "linear",
):
    return {
        "diffusion": {
            "num_steps": int(num_steps),
            "beta_schedule": beta_schedule,
            "rescale_zero_terminal_snr": bool(rescale_zero_terminal_snr),
            "loss_type": "v_mse",
            "dynamic_threshold": True,
            "dynamic_threshold_p": 0.995,
        },
        "conditioning": {
            "num_ss_classes": 6,
            "use_wind_speed": True,
            "wind_min_kt": 35.0,
            "wind_max_kt": 170.0,
            "wind_null_kt": 0.0,
        },
        "evaluation": {
            "timestep_schedule": timestep_schedule,
        },
    }


def _assert_valid_beta_buffers(diffusion: Diffusion, *, expect_terminal_drop: bool = False):
    betas = diffusion.betas.numpy()
    alpha_bar = diffusion.alphas_cumprod.numpy()

    assert betas.shape == (diffusion.num_steps,)
    assert np.all(np.isfinite(betas))
    assert np.all((betas > 0.0) & (betas < 1.0))
    assert alpha_bar.shape == (diffusion.num_steps,)
    assert np.all(np.isfinite(alpha_bar))
    assert np.all((alpha_bar > 0.0) & (alpha_bar <= 1.0))
    assert np.all(np.diff(alpha_bar) < 0.0)
    if expect_terminal_drop:
        assert alpha_bar[-1] < 1e-6


def _assert_valid_sampling_timesteps(diffusion: Diffusion, schedule_name: str):
    timesteps = diffusion._build_sampling_timesteps(
        25,
        sampler_name="ddim",
        timestep_schedule=schedule_name,
    )
    arr = np.asarray(timesteps, dtype=np.int32)
    assert arr.shape == (25,)
    assert np.all(arr[:-1] > arr[1:])
    assert arr.min() >= 0
    assert arr.max() < diffusion.num_steps


def main():
    linear = Diffusion(_make_cfg(beta_schedule="linear"))
    cosine = Diffusion(_make_cfg(beta_schedule="cosine"))
    linear_ztsnr = Diffusion(
        _make_cfg(
            beta_schedule="linear",
            rescale_zero_terminal_snr=True,
        )
    )

    _assert_valid_beta_buffers(linear)
    _assert_valid_beta_buffers(cosine)
    _assert_valid_beta_buffers(linear_ztsnr, expect_terminal_drop=True)

    for schedule_name in ("linear", "leading", "trailing"):
        diffusion = Diffusion(_make_cfg(beta_schedule="cosine", timestep_schedule=schedule_name))
        _assert_valid_sampling_timesteps(diffusion, schedule_name)

        full = diffusion._build_sampling_timesteps(
            diffusion.num_steps,
            sampler_name="dpmpp_2m",
            timestep_schedule=schedule_name,
        )
        assert full == list(range(diffusion.num_steps - 1, -1, -1))

        one_step = diffusion._build_sampling_timesteps(
            1,
            sampler_name="ddim",
            timestep_schedule=schedule_name,
        )
        assert one_step == [diffusion.num_steps - 1]

    print("Diffusion schedule sanity checks passed.")


if __name__ == "__main__":
    main()

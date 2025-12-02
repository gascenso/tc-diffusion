# tc_diffusion/model_unet.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def sinusoidal_time_embedding(t, dim):
    half_dim = dim // 2
    t = tf.cast(t, tf.float32)[:, None]          # (B, 1)
    freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(10000.0), half_dim))[None, :]
    args = t * freqs                             # (B, half_dim)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    return emb


def make_res_block(x, emb, out_channels, name_prefix):
    """
    Simple ResNet-ish block with time/cond embedding added.
    x: (B, H, W, C)
    emb: (B, E)
    """
    in_channels = x.shape[-1]
    h = x

    # project embedding to channels and add
    h_emb = layers.Dense(out_channels, name=f"{name_prefix}_emb_dense")(emb)
    h_emb = layers.Activation("swish", name=f"{name_prefix}_emb_swish")(h_emb)
    h_emb = layers.Reshape((1, 1, out_channels), name=f"{name_prefix}_emb_reshape")(h_emb)

    # first conv
    h = layers.Conv2D(out_channels, 3, padding="same", name=f"{name_prefix}_conv1")(h)
    h = layers.Activation("swish", name=f"{name_prefix}_act1")(h)
    h = layers.Add(name=f"{name_prefix}_add_emb")([h, h_emb])
    h = layers.Conv2D(out_channels, 3, padding="same", name=f"{name_prefix}_conv2")(h)
    h = layers.Activation("swish", name=f"{name_prefix}_act2")(h)

    # skip connection if channels differ
    if in_channels != out_channels:
        x = layers.Conv2D(out_channels, 1, padding="same", name=f"{name_prefix}_skip")(x)

    return layers.Add(name=f"{name_prefix}_residual")([x, h])


def build_unet(cfg):
    image_size = int(cfg["data"]["image_size"])
    base_channels = int(cfg["model"]["base_channels"])
    channel_mults = cfg["model"].get("channel_mults", [1, 2, 4])
    num_res_blocks = int(cfg["model"].get("num_res_blocks", 2))

    # Inputs
    x_in = keras.Input(shape=(image_size, image_size, 1), name="x_t")
    t_in = keras.Input(shape=(), dtype=tf.int32, name="t")
    cond_in = keras.Input(shape=(), dtype=tf.float32, name="cond")  # scalar for now

    # ---- time embedding ----
    t_emb_dim = base_channels * 4
    t_emb = layers.Lambda(
        lambda t: sinusoidal_time_embedding(t, t_emb_dim),
        name="t_sinusoidal_emb",
    )(t_in)
    t_emb = layers.Dense(t_emb_dim, activation="swish", name="t_emb_dense1")(t_emb)
    t_emb = layers.Dense(t_emb_dim, activation="swish", name="t_emb_dense2")(t_emb)

    # ---- cond embedding (still scalar for now) ----
    cond_vec = layers.Lambda(lambda c: tf.expand_dims(c, -1), name="cond_expand")(cond_in)
    c_emb = layers.Dense(t_emb_dim, activation="swish", name="c_emb_dense1")(cond_vec)
    c_emb = layers.Dense(t_emb_dim, activation="swish", name="c_emb_dense2")(c_emb)

    # fuse time + cond
    tc_emb = layers.Add(name="tc_emb")([t_emb, c_emb])  # (B, t_emb_dim)

    # ---- Downsampling path ----
    hs = []  # skip connections

    h = layers.Conv2D(base_channels, 3, padding="same", name="in_conv")(x_in)

    # levels 0..(L-1)
    in_ch = base_channels
    for level, mult in enumerate(channel_mults):
        out_ch = base_channels * mult
        for b in range(num_res_blocks):
            h = make_res_block(
                h,
                tc_emb,
                out_ch,
                name_prefix=f"down_l{level}_b{b}",
            )
        hs.append(h)  # store skip at this resolution

        # don’t downsample after last level
        if level != len(channel_mults) - 1:
            h = layers.AveragePooling2D(pool_size=2, name=f"down_l{level}_pool")(h)
        in_ch = out_ch

    # ---- Bottleneck ----
    h = make_res_block(h, tc_emb, in_ch, name_prefix="bottleneck_0")
    h = make_res_block(h, tc_emb, in_ch, name_prefix="bottleneck_1")

    # ---- Upsampling path ----
    # walk back through channel_mults in reverse, popping skips
    for level, mult in reversed(list(enumerate(channel_mults))):
        skip = hs[level]  # this has the same H,W as h *after* upsampling
        # upsample except at the highest-res level (level 0) – but we’ve stored skip before pool,
        # so we first upsample then concat
        if level != len(channel_mults) - 1:
            h = layers.UpSampling2D(size=2, interpolation="nearest", name=f"up_l{level}_up")(h)

        # After upsampling, h should match skip spatial dims
        h = layers.Concatenate(name=f"up_l{level}_concat")([h, skip])
        out_ch = base_channels * mult

        for b in range(num_res_blocks):
            h = make_res_block(
                h,
                tc_emb,
                out_ch,
                name_prefix=f"up_l{level}_b{b}",
            )

    # ---- Output ----
    eps_out = layers.Conv2D(1, 3, padding="same", name="eps_out")(h)

    model = keras.Model(inputs=[x_in, t_in, cond_in], outputs=eps_out, name="unet_ddpm")
    return model

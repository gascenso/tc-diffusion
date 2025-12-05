# tc_diffusion/model_unet.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ----------------- Time embedding ----------------- #

def sinusoidal_time_embedding(t, dim):
    """
    t: (B,) int32 timesteps
    returns: (B, dim) float32 sinusoidal embedding
    """
    half_dim = dim // 2
    t = tf.cast(t, tf.float32)[:, None]  # (B, 1)
    freqs = tf.exp(
        tf.linspace(tf.math.log(1.0), tf.math.log(10000.0), half_dim)
    )[None, :]  # (1, half_dim)
    args = t * freqs  # (B, half_dim)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    return emb


# ----------------- ResBlock ----------------- #

def make_res_block(x, emb, out_channels, name_prefix):
    """
    Simple ResNet-ish block with time/cond embedding added.

    x:   (B, H, W, C)
    emb: (B, E) global conditioning embedding (time + cond)
    """
    in_channels = x.shape[-1]
    h = x

    # project embedding to out_channels and add as bias
    h_emb = layers.Dense(
        out_channels, name=f"{name_prefix}_emb_dense"
    )(emb)
    h_emb = layers.Activation(
        "swish", name=f"{name_prefix}_emb_swish"
    )(h_emb)
    h_emb = layers.Reshape(
        (1, 1, out_channels), name=f"{name_prefix}_emb_reshape"
    )(h_emb)

    # first conv
    h = layers.Conv2D(
        out_channels, 3, padding="same", name=f"{name_prefix}_conv1"
    )(h)
    h = layers.Activation("swish", name=f"{name_prefix}_act1")(h)
    h = layers.Add(name=f"{name_prefix}_add_emb")([h, h_emb])

    # second conv
    h = layers.Conv2D(
        out_channels, 3, padding="same", name=f"{name_prefix}_conv2"
    )(h)
    h = layers.Activation("swish", name=f"{name_prefix}_act2")(h)

    # skip connection if channels differ
    if in_channels != out_channels:
        x = layers.Conv2D(
            out_channels, 1, padding="same", name=f"{name_prefix}_skip"
        )(x)

    return layers.Add(name=f"{name_prefix}_residual")([x, h])


# ----------------- Attention block (optional) ----------------- #

def attention_block(x, num_heads, name_prefix):
    """
    Simple self-attention over spatial locations.

    x: (B, H, W, C)
    returns: (B, H, W, C)
    """
    b, h, w, c = x.shape
    if h is None or w is None:
        # Fallback: compute spatial dims at runtime if needed
        spatial_shape = tf.shape(x)[1:3]
        h_dyn = spatial_shape[0]
        w_dyn = spatial_shape[1]
    else:
        h_dyn = h
        w_dyn = w

    # flatten spatial dims: (B, H*W, C)
    x_flat = layers.Reshape((-1, c), name=f"{name_prefix}_reshape_to_seq")(x)

    # layer norm + MHA
    h_flat = layers.LayerNormalization(name=f"{name_prefix}_ln")(x_flat)
    h_flat = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=c // num_heads,
        name=f"{name_prefix}_mha",
    )(h_flat, h_flat)
    # residual
    x_flat = layers.Add(name=f"{name_prefix}_attn_residual")([x_flat, h_flat])

    # back to (B, H, W, C)
    x_out = layers.Reshape(
        (h_dyn, w_dyn, c), name=f"{name_prefix}_reshape_to_spatial"
    )(x_flat)
    return x_out


# ----------------- U-Net builder ----------------- #

def build_unet(cfg):
    """
    Build a U-Net backbone for DDPM with:

    - time embedding (sinusoidal + MLP)
    - SS-category conditioning via embedding (discrete classes 0..N-1)
    - multi-resolution ResNet blocks
    - optional self-attention at chosen resolutions

    Scales:
      - base_channels
      - channel_mults      (e.g. [1, 2, 4, 4])
      - num_res_blocks     (per level)
      - attn_resolutions   (e.g. [16, 8])
      - attn_num_heads
    """
    image_size = int(cfg["data"]["image_size"])
    base_channels = int(cfg["model"]["base_channels"])
    channel_mults = cfg["model"].get("channel_mults", [1, 2, 4])
    num_res_blocks = int(cfg["model"].get("num_res_blocks", 2))

    # Optional attention
    attn_res = cfg["model"].get("attn_resolutions", [])
    attn_num_heads = int(cfg["model"].get("attn_num_heads", 4))

    # Conditioning: number of SS classes (default 6: TS + Cat1..5)
    num_ss_classes = int(cfg.get("conditioning", {}).get("num_ss_classes", 6))
    cond_dim = base_channels * 4  # dimension of cond embedding

    # ---- Inputs ---- #
    x_in = keras.Input(shape=(image_size, image_size, 1), name="x_t")
    t_in = keras.Input(shape=(), dtype=tf.int32, name="t")
    # SS category as int32 scalar, in [0, num_ss_classes-1]
    cond_in = keras.Input(shape=(), dtype=tf.int32, name="ss_category")

    # ---- time embedding ---- #
    t_emb_dim = base_channels * 4
    t_emb = layers.Lambda(
        lambda t: sinusoidal_time_embedding(t, t_emb_dim),
        name="t_sinusoidal_emb",
    )(t_in)
    t_emb = layers.Dense(
        t_emb_dim, activation="swish", name="t_emb_dense1"
    )(t_emb)
    t_emb = layers.Dense(
        t_emb_dim, activation="swish", name="t_emb_dense2"
    )(t_emb)

    # ---- SS category embedding ---- #
    # Embedding: (B,) -> (B, cond_dim)
    c_emb = layers.Embedding(
        input_dim=num_ss_classes,
        output_dim=cond_dim,
        name="ss_category_embedding",
    )(cond_in)  # (B, cond_dim)
    c_emb = layers.Dense(
        t_emb_dim, activation="swish", name="c_emb_dense"
    )(c_emb)

    # ---- fuse time + cond ---- #
    tc_emb = layers.Add(name="tc_emb")([t_emb, c_emb])  # (B, t_emb_dim)

    # ---- Downsampling path ---- #
    hs = []  # skip connections
    h = layers.Conv2D(
        base_channels, 3, padding="same", name="in_conv"
    )(x_in)

    # Track resolution to know where to apply attention
    current_res = image_size

    for level, mult in enumerate(channel_mults):
        out_ch = base_channels * mult

        for b in range(num_res_blocks):
            h = make_res_block(
                h,
                tc_emb,
                out_ch,
                name_prefix=f"down_l{level}_b{b}",
            )

            # optional attention at this resolution
            if current_res in attn_res:
                h = attention_block(
                    h,
                    num_heads=attn_num_heads,
                    name_prefix=f"down_l{level}_b{b}_attn",
                )

        hs.append(h)  # store skip at this resolution

        if level != len(channel_mults) - 1:
            h = layers.AveragePooling2D(
                pool_size=2, name=f"down_l{level}_pool"
            )(h)
            current_res //= 2

    # ---- Bottleneck ---- #
    h = make_res_block(
        h, tc_emb, h.shape[-1], name_prefix="bottleneck_0"
    )
    if current_res in attn_res:
        h = attention_block(
            h, num_heads=attn_num_heads, name_prefix="bottleneck_attn"
        )
    h = make_res_block(
        h, tc_emb, h.shape[-1], name_prefix="bottleneck_1"
    )

    # ---- Upsampling path ---- #
    # Walk back channel_mults in reverse
    for level, mult in reversed(list(enumerate(channel_mults))):
        skip = hs[level]
        out_ch = base_channels * mult

        # We upsample except at the deepest level (already matching skip)
        if level != len(channel_mults) - 1:
            h = layers.UpSampling2D(
                size=2,
                interpolation="nearest",
                name=f"up_l{level}_up",
            )(h)
            current_res *= 2  # reverse of downsampling

        # concat skip
        h = layers.Concatenate(
            name=f"up_l{level}_concat"
        )([h, skip])

        for b in range(num_res_blocks):
            h = make_res_block(
                h,
                tc_emb,
                out_ch,
                name_prefix=f"up_l{level}_b{b}",
            )

            if current_res in attn_res:
                h = attention_block(
                    h,
                    num_heads=attn_num_heads,
                    name_prefix=f"up_l{level}_b{b}_attn",
                )

    # ---- Output ---- #
    eps_out = layers.Conv2D(
        1, 3, padding="same", name="eps_out"
    )(h)

    model = keras.Model(
        inputs=[x_in, t_in, cond_in],
        outputs=eps_out,
        name="unet_ddpm_ss_cond",
    )
    model.summary()
    return model

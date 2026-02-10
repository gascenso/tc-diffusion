# tc_diffusion/model_unet.py
import tensorflow as tf
from tensorflow import keras
from keras import layers


class SinusoidalTimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)

    def call(self, t):
        # t: (B,) int32/float32
        t = tf.cast(t, tf.float32)
        half = self.dim // 2

        i = tf.cast(tf.range(half), tf.float32)
        inv_freq = tf.exp(
            -tf.math.log(10000.0) * i / tf.maximum(tf.cast(half - 1, tf.float32), 1.0)
        )  # (half,)

        args = t[:, None] * inv_freq[None, :]              # (B, half)
        emb = tf.concat([tf.sin(args), tf.cos(args)], -1)  # (B, 2*half)

        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])

        return emb

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

class GroupNorm(layers.Layer):
    def __init__(self, groups=32, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.requested_groups = int(groups)
        self.groups = int(groups)
        self.eps = float(eps)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if channels is None:
            raise ValueError("GroupNorm requires a known channel dimension (input_shape[-1]).")

        # Pick the largest G <= requested_groups such that channels % G == 0
        g = min(self.requested_groups, channels)
        while g > 1 and (channels % g) != 0:
            g -= 1
        self.groups = max(g, 1)

        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B,H,W,C)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        G = self.groups

        # C_per_group is guaranteed integer because we enforce divisibility in build()
        C_per_group = tf.math.floordiv(C, G)

        x = tf.reshape(x, [B, H, W, G, C_per_group])
        mean, var = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [B, H, W, C])

        return x * self.gamma[None, None, None, :] + self.beta[None, None, None, :]



class ClassifierFreeCondDropout(layers.Layer):
    """CFG label dropout.

    Replaces a fraction of conditioning labels with a special "null" label id.
    This enables classifier-free guidance at inference time (two-pass guidance).
    """

    def __init__(self, drop_prob: float, null_label: int, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)
        self.null_label = int(null_label)

    def call(self, labels, training=None):
        labels = tf.cast(labels, tf.int32)
        if not training or self.drop_prob <= 0.0:
            return labels

        # Randomly drop conditioning per-sample.
        rnd = tf.random.uniform(tf.shape(labels), 0.0, 1.0)
        dropped = tf.where(rnd < self.drop_prob, tf.fill(tf.shape(labels), self.null_label), labels)
        return dropped

def make_res_block(x, emb, out_channels, name_prefix, gn_groups=32):
    """
    ResBlock with GroupNorm + FiLM (scale/shift) conditioning from emb.

    x:   (B,H,W,C)
    emb: (B,E)  (time+cond fused embedding)
    """
    in_channels = x.shape[-1]
    h = x

    # Project emb -> (gamma1, beta1, gamma2, beta2) for two GN layers
    film = layers.Dense(
        4 * out_channels,
        kernel_initializer="zeros",   # start close to "no conditioning"
        bias_initializer="zeros",
        name=f"{name_prefix}_film",
    )(emb)
    film = layers.Reshape((1, 1, 4 * out_channels), name=f"{name_prefix}_film_reshape")(film)
    
    # film: (B,1,1,4*out_channels)
    gamma1 = layers.Lambda(lambda z: z[..., 0*out_channels:1*out_channels],
                        name=f"{name_prefix}_gamma1")(film)
    beta1  = layers.Lambda(lambda z: z[..., 1*out_channels:2*out_channels],
                        name=f"{name_prefix}_beta1")(film)
    gamma2 = layers.Lambda(lambda z: z[..., 2*out_channels:3*out_channels],
                        name=f"{name_prefix}_gamma2")(film)
    beta2  = layers.Lambda(lambda z: z[..., 3*out_channels:4*out_channels],
                        name=f"{name_prefix}_beta2")(film)

    # --- Conv 1 ---
    h = layers.Conv2D(out_channels, 3, padding="same", name=f"{name_prefix}_conv1")(h)
    h = GroupNorm(groups=gn_groups, name=f"{name_prefix}_gn1")(h)
    h = layers.Lambda(lambda z: z[0] * (1.0 + z[1]) + z[2], name=f"{name_prefix}_film1")([h, gamma1, beta1])
    h = layers.Activation("swish", name=f"{name_prefix}_act1")(h)

    # --- Conv 2 ---
    h = layers.Conv2D(out_channels, 3, padding="same", name=f"{name_prefix}_conv2")(h)
    h = GroupNorm(groups=gn_groups, name=f"{name_prefix}_gn2")(h)
    h = layers.Lambda(lambda z: z[0] * (1.0 + z[1]) + z[2], name=f"{name_prefix}_film2")([h, gamma2, beta2])
    h = layers.Activation("swish", name=f"{name_prefix}_act2")(h)

    # Skip connection if channels differ
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
    # SS category conditioning (0..num_ss_classes-1). We reserve one extra id = num_ss_classes
    # as the CFG "null" label used when conditioning is dropped during training.
    num_ss_classes = int(cfg.get("conditioning", {}).get("num_ss_classes", 6))
    cfg_drop_prob = float(cfg.get("conditioning", {}).get("cfg_drop_prob", 0.0))

    cond_in = keras.Input(shape=(), dtype=tf.int32, name="ss_cat")

    # ---- time embedding ----
    t_emb_dim = base_channels * 4
    t_emb = SinusoidalTimeEmbedding(t_emb_dim, name="t_sin_emb")(t_in)

    t_emb = layers.Dense(t_emb_dim, activation="swish", name="t_emb_dense1")(t_emb)
    t_emb = layers.Dense(t_emb_dim, activation="swish", name="t_emb_dense2")(t_emb)

    # ---- cond embedding (categorical SS class) + CFG dropout ----
    null_label = num_ss_classes

    # Safety: clip to valid range [0, num_ss_classes-1] before applying CFG.
    cond_clipped = layers.Lambda(
        lambda c: tf.where(
            tf.equal(tf.cast(c, tf.int32), tf.cast(null_label, tf.int32)),
            tf.cast(c, tf.int32),  # keep null label as-is
            tf.clip_by_value(tf.cast(c, tf.int32), 0, num_ss_classes - 1),
        ),
        name="ss_cat_clip_preserve_null",
    )(cond_in)

    cond_dropped = ClassifierFreeCondDropout(
        drop_prob=cfg_drop_prob,
        null_label=null_label,
        name="cfg_cond_dropout",
    )(cond_clipped)

    # Embedding includes the extra "null" label.
    c_emb = layers.Embedding(
        input_dim=num_ss_classes + 1,
        output_dim=t_emb_dim,
        name="ss_cat_emb",
    )(cond_dropped)

    # Optional MLP on top of the embedding (helps expressivity without much cost).
    c_emb = layers.Dense(t_emb_dim, activation="swish", name="c_emb_dense1")(c_emb)
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
            h = layers.UpSampling2D(size=2, interpolation="bilinear", name=f"up_l{level}_up")(h)

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

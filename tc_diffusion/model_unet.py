import tensorflow as tf
from tensorflow import keras
from keras import layers


def sinusoidal_time_embedding(t, dim):
    """
    t: (batch,) integer timesteps
    Return: (batch, dim) sinusoidal embedding
    """
    half_dim = dim // 2
    # [batch, 1]
    t = tf.cast(t, tf.float32)[:, None]
    freqs = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(10000.0),
            half_dim,
        )
    )[None, :]  # (1, half_dim)
    args = t * freqs  # (batch, half_dim)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    return emb  # (batch, dim)


def build_unet(cfg):
    image_size = int(cfg["data"]["image_size"])
    base_channels = int(cfg["model"]["base_channels"])

    # Inputs
    x_in = keras.Input(shape=(image_size, image_size, 1), name="x_t")
    t_in = keras.Input(shape=(), dtype=tf.int32, name="t")
    cond_in = keras.Input(shape=(), dtype=tf.float32, name="cond")  # scalar per sample

    # Time embedding
    t_emb_dim = base_channels * 4
    t_emb = layers.Lambda(
        lambda t: sinusoidal_time_embedding(t, t_emb_dim), name="t_sinusoidal_emb"
    )(t_in)
    t_emb = layers.Dense(t_emb_dim, activation="swish")(t_emb)
    t_emb = layers.Dense(t_emb_dim, activation="swish")(t_emb)

    # ----- FIXED: cond embedding with a Lambda, no raw tf op on KerasTensor -----
    cond_vec = layers.Lambda(
        lambda c: tf.expand_dims(c, -1), name="cond_expand"
    )(cond_in)  # (batch, 1)
    c_emb = layers.Dense(t_emb_dim, activation="swish")(cond_vec)
    c_emb = layers.Dense(t_emb_dim, activation="swish")(c_emb)
    # ---------------------------------------------------------------------------

    # Fuse time + cond
    tc_emb = layers.Add(name="tc_emb")([t_emb, c_emb])  # (batch, t_emb_dim)

    def add_time_cond(x, emb):
        ch = x.shape[-1]
        h = layers.Dense(ch)(emb)
        h = layers.Activation("swish")(h)
        h = layers.Reshape((1, 1, ch))(h)
        return layers.Add()([x, h])

    # Simple encoder
    x = x_in
    # Down 1
    x = layers.Conv2D(base_channels, 3, padding="same")(x)
    x = add_time_cond(x, tc_emb)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(base_channels, 3, padding="same")(x)
    x = layers.Activation("swish")(x)
    d1 = x
    x = layers.MaxPool2D()(x)

    # Down 2
    x = layers.Conv2D(base_channels * 2, 3, padding="same")(x)
    x = add_time_cond(x, tc_emb)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(base_channels * 2, 3, padding="same")(x)
    x = layers.Activation("swish")(x)
    d2 = x
    x = layers.MaxPool2D()(x)

    # Bottleneck
    x = layers.Conv2D(base_channels * 4, 3, padding="same")(x)
    x = add_time_cond(x, tc_emb)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(base_channels * 4, 3, padding="same")(x)
    x = layers.Activation("swish")(x)

    # Up 2
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, d2])
    x = layers.Conv2D(base_channels * 2, 3, padding="same")(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(base_channels * 2, 3, padding="same")(x)
    x = layers.Activation("swish")(x)

    # Up 1
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, d1])
    x = layers.Conv2D(base_channels, 3, padding="same")(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(base_channels, 3, padding="same")(x)
    x = layers.Activation("swish")(x)

    eps_out = layers.Conv2D(1, 3, padding="same", name="eps_out")(x)

    model = keras.Model(inputs=[x_in, t_in, cond_in], outputs=eps_out, name="unet_ddpm")
    return model

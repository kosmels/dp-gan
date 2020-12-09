from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Input, LeakyReLU, Reshape)
from tensorflow.keras.models import Model

from models.custom_layers import upsample_block


def get_generator_model(noise_dim: int):
    noise = Input(shape=(noise_dim,))
    x = Dense(7 * 7 * 256, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Reshape((7, 7, 256))(x)
    x = upsample_block(
        x,
        128,
        LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        32,
        LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        16,
        LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(x, 3, Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True)

    g_model = Model(noise, x, name="generator")
    return g_model

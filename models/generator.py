import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Embedding,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    Conv2DTranspose,
    concatenate,
    multiply,
)
from tensorflow.keras.models import Model, Sequential

from models.custom_layers import upsample_block


def get_generator_model_v3(noise_dim: int, class_dim: int, architecture: str = "acgan"):
    # https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py

    noise = Input(shape=(noise_dim,), dtype=tf.float32)
    labels = Input(shape=(1,), dtype=tf.int32)
    input_embedding = Flatten()(Embedding(class_dim, noise_dim)(labels))

    mul_input = multiply([noise, input_embedding])
    x = Dense(7 * 7 * 256, activation="relu", input_dim=noise_dim)(mul_input)
    x = Reshape((7, 7, 256))(x)
    if architecture == "acgan":
        x = upsample_block(
            x,
            128,
            LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )   # 6 x 6 x 128
        # x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)  # 7 x 7 x 128
        x = upsample_block(
            x,
            64,
            LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )   # 14 x 14 x 64
        x = upsample_block(
            x,
            32,
            LeakyReLU(0.2),
            strides=(1, 1),
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )   # 28 x 28 x 32
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
        output = upsample_block(x, 3, Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True)
        # output = Conv2D(1, (3, 3), activation="tanh", strides=(1, 1), use_bias=False, padding="same")(x)

        return Model([noise, labels], output)



def get_generator_model_v2(noise_dim: int, class_dim: int, architecture: str = "acgan"):
    # https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py

    noise = Input(shape=(noise_dim,), dtype=tf.float32)
    labels = Input(shape=(1,), dtype=tf.int32)
    input_embedding = Flatten()(Embedding(class_dim, noise_dim)(labels))

    mul_input = concatenate([noise, input_embedding])
    x = Dense(7 * 7 * 256, activation="relu", input_dim=noise_dim)(mul_input)
    x = Reshape((7, 7, 256))(x)
    if architecture == "acgan":
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        x = Conv2D(16, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = UpSampling2D()(x)
        output = Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)

        return Model([noise, labels], output)
    else:
        assert architecture == "wgan", (
            f"You passed wrong 'architecture' argument: {architecture}." f"Available architectures: 'acgan', 'wgan'"
        )
        # TODO: Add generator architecture from WGAN below


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

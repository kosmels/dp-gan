"""
## Create the discriminator (aka critic in the original WGAN)
The samples in the dataset have shape `(28, 28, 1)`. As we will be
using strided convolutions, this can result in a shape with odd dimensions.
For example,
`(28, 28) -> Conv_s2 -> (14, 14) -> Conv_s2 -> (7, 7) -> Conv_s2 ->(3, 3)`.
While doing upsampling in the generator, we won't get the same input shape
as the original images if we aren't careful. To avoid this, we will do
something much simpler. In the discriminator, we will "zero pad" the input
to make the shape `(32, 32, 1)` for each sample, while in the generator we will
crop the final output to match the shape with input shape.
"""
from typing import Tuple

from tensorflow.keras.layers import (Dense, Dropout, Flatten, Input, LeakyReLU,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model

from models.custom_layers import conv_block


def get_discriminator_model(img_shape: Tuple[int, int, int]):
    img_input = Input(shape=img_shape)
    x = ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    )

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)

    d_model = Model(img_input, x, name="discriminator")
    return d_model
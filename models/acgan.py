import tensorflow as tf
from tensorflow.keras.models import Model


class ACGAN(Model):
    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3):
        super(ACGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, **kwargs):
        super(ACGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

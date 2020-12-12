import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class ACGAN(Model):
    def __init__(self, discriminator, generator, latent_dim, n_classes, discriminator_extra_steps=3):
        super(ACGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.d_steps = discriminator_extra_steps

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, d_loss_cls_fn, g_loss_fn, g_loss_cls_fn, **kwargs):
        super(ACGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.d_loss_cls_fn = d_loss_cls_fn
        self.g_loss_fn = g_loss_fn
        self.g_loss_cls_fn = g_loss_cls_fn

    def train_step(self, inputs):
        # https://keras.io/guides/customizing_what_happens_in_fit/

        # Get the batch size
        real_images, target_labels = inputs[0], inputs[1]
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            sampled_labels = np.random.randint(0, self.n_classes, batch_size)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, sampled_labels.reshape((-1, 1))], training=True)
                # Get the logits for the fake images
                fake_logits, fake_cls_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits, real_cls_logits = self.discriminator(real_images, training=True)

                # Calculate discriminator loss using fake and real logits
                d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                d_loss_cls = self.d_loss_cls_fn(real_cls_logits=real_cls_logits, fake_cls_logits=fake_cls_logits)
                d_loss += d_loss_cls

                # gradient penalty scope
                # gp = self.gradient_penalty(batch_size, inputs, fake_images)
                # d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(2 * batch_size, self.latent_dim))
        sampled_labels = np.random.randint(0, self.n_classes, 2 * batch_size)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, sampled_labels.reshape((-1, 1))], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits, gen_cls_logits = self.discriminator(generated_images, training=False)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(img_logits=gen_img_logits)
            g_loss_cls = self.g_loss_cls_fn(real_cls_logits=real_cls_logits, fake_cls_logits=fake_cls_logits)
            g_loss = g_loss_cls - g_loss

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

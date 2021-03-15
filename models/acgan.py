"""
Main parts of code from https://keras.io/examples/generative/wgan_gp/
Reimplemented and adjusted for ACGAN purposes.
"""
import tensorflow as tf
from tensorflow.keras.models import Model


class ACGAN(Model):
    def __init__(self, discriminator, generator, latent_dim, n_classes, discriminator_extra_steps=3, gp_weight=10.0):
        super(ACGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, d_loss_cls_fn, g_loss_fn, g_loss_cls_fn, **kwargs):
        super(ACGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.d_loss_cls_fn = d_loss_cls_fn
        self.g_loss_fn = g_loss_fn
        self.g_loss_cls_fn = g_loss_cls_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, inputs):
        # https://keras.io/guides/customizing_what_happens_in_fit/

        # Get the batch size
        inputs = inputs[0]
        real_images, target_labels = inputs[0], inputs[1]
        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            sampled_labels = tf.random.uniform(shape=(batch_size,), minval=0, maxval=self.n_classes, dtype=tf.int32)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, sampled_labels], training=True)
                # Get the logits for the fake images
                fake_logits, fake_cls_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits, real_cls_logits = self.discriminator(real_images, training=True)

                # Calculate discriminator loss using fake and real logits
                d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                d_loss_cls = self.d_loss_cls_fn(
                    real_cls_logits=real_cls_logits,
                    fake_cls_logits=fake_cls_logits,
                    real_labels=target_labels,
                    fake_labels=sampled_labels,
                )
                # d_loss = d_loss + 5 * d_loss_cls

                # gradient penalty scope
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_loss + gp * self.gp_weight + 5 * d_loss_cls

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(2 * batch_size, self.latent_dim))
        sampled_labels = tf.random.uniform(shape=(2 * batch_size,), minval=0, maxval=self.n_classes, dtype=tf.int32)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, sampled_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits, gen_cls_logits = self.discriminator(generated_images, training=False)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(fake_img=gen_img_logits)
            g_loss_cls = self.g_loss_cls_fn(cls_logits=gen_cls_logits, labels=sampled_labels)
            g_loss = g_loss + 5 * g_loss_cls

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

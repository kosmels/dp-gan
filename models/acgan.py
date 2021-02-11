"""
Main parts of code from https://keras.io/examples/generative/wgan_gp/
Reimplemented and adjusted for ACGAN purposes.
"""
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
        inputs = inputs[0]
        real_images, target_labels = inputs[0], inputs[1]
        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            sampled_labels = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int32)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, sampled_labels], training=True)
                # Get the logits for the fake images
                fake_logits, fake_cls_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits, real_cls_logits = self.discriminator(real_images, training=True)

                # Calculate discriminator loss using fake and real logits
                d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                print(f"Debug")
                d_loss_cls = self.d_loss_cls_fn(
                    real_cls_logits=real_cls_logits,
                    fake_cls_logits=fake_cls_logits,
                    real_labels=target_labels,
                    fake_labels=sampled_labels,
                )
                d_loss += 5*d_loss_cls

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
        sampled_labels = tf.random.uniform(shape=(2 * batch_size, 1), minval=0, maxval=self.n_classes, dtype=tf.int32)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, sampled_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits, gen_cls_logits = self.discriminator(generated_images, training=False)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(fake_img=gen_img_logits)
            g_loss_cls = self.g_loss_cls_fn(cls_logits=gen_cls_logits, labels=sampled_labels)
            print(g_loss_cls)
            g_loss = 5*g_loss_cls - g_loss

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        print(f"d_loss: {d_loss}, g_loss: {g_loss}")
        return {"d_loss": d_loss, "g_loss": g_loss}

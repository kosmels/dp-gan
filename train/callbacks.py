"""
## Create a callback that periodically saves generated images
"""
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class GANMonitor(Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        if (epoch + 1) % 10 == 0:
            for i in range(self.num_img):
                img = generated_images[i].numpy()
                cv2.imwrite(f"outputs/generated_img_{epoch}_{i}.png", img)

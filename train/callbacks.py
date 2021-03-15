"""
## Create a callback that periodically saves generated images
"""
import os

import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class CustomModelCheckpoint(Callback):
    def __init__(self, output_dir: str, best_only: bool = False):
        super().__init__()
        self.output_dir = output_dir
        self.save_best_only = best_only

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}, valid_acc: {logs['val_acc']}")
        if logs["val_acc"] > logs["acc"]:  # your custom condition
            self.model.generator.save(f"generator_model_epoch_{epoch}.hdf5")


class WGAN_Visual_Monitor(Callback):
    def __init__(self, output_dir, num_img: int, latent_dim: int, visual_frequency: int):
        super().__init__()
        self.output_dir = output_dir
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.visual_frequency = visual_frequency

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        if (epoch + 1) % self.visual_frequency == 0:
            print(f"Finished epoch number: {epoch}")
            for i in range(self.num_img):
                img = generated_images[i].numpy()
                cv2.imwrite(os.path.join(self.output_dir, f"img_{epoch}_{i}.png"), img)


class ACGAN_Visual_Monitor(Callback):
    def __init__(self, output_dir, n_classes, num_img=6, latent_dim=128):
        super().__init__()
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.n_classes, self.latent_dim))
        sampled_labels = tf.reshape(tf.range(0, self.n_classes, dtype=tf.int32), shape=(-1, 1))
        generated_images = self.model.generator([random_latent_vectors, sampled_labels])
        generated_images = (generated_images * 127.5) + 127.5

        if (epoch + 1) % 25 == 0:
            print(f"Finished epoch number: {epoch}")
            for i in range(self.num_img):
                img = generated_images[i].numpy()
                cv2.imwrite(os.path.join(self.output_dir, f"img_{epoch}_{i}.png"), img)

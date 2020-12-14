"""
## Create a callback that periodically saves generated images
"""
import datetime
import os

import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from train.utils import create_clean_dir


class GANMonitor(Callback):
    def __init__(self, output_dir, model_name, n_classes, num_img=6, latent_dim=128):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.output_dir = create_clean_dir(os.path.join(output_dir, f"{model_name}_{current_time}"))
        self.model_name = model_name
        self.n_classes = n_classes
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        sampled_labels = tf.random.uniform(shape=(self.num_img, 1), minval=0, maxval=self.n_classes, dtype=tf.int32)
        generated_images = self.model.generator([random_latent_vectors, sampled_labels])
        generated_images = (generated_images * 127.5) + 127.5

        if (epoch + 1) % 10 == 0:
            print(f"Finished epoch number: {epoch}")
            for i in range(self.num_img):
                img = generated_images[i].numpy()
                cv2.imwrite(os.path.join(self.output_dir, f"img_{epoch}_{i}.png"), img)

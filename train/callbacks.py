"""
## Create a callback that periodically saves generated images
"""
import os

import cv2
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import Callback

from datasets.preprocessing import get_wgan_original_images, parse_config
from metrics.evaluate import calculate_fid


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


class WGAN_FID_Monitor(Callback):
    def __init__(self, latent_dim: int, calculate_fid_frequency: int, model_dir: str):
        super().__init__()
        self.inception_model = InceptionV3(include_top=False, pooling="avg", input_shape=(224, 224, 3))
        self.latent_dim = latent_dim
        self.calculate_fid_frequency = calculate_fid_frequency
        self.model_dir = model_dir

        # Prepare test images that will be used to calculate FID
        yaml_path = "configs/wpgan_config_default.yml"
        parsed_config = parse_config(yaml_path)
        dataset_config = parsed_config["dataset"]
        train_images = get_wgan_original_images(dataset_config)
        self.target_images = preprocess_input(
            train_images.reshape(train_images.shape[0], *dataset_config["image_shape"]).astype("float32")
        )
        self.num_img = self.target_images.shape[0]
        self.min_fid = 500.0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.calculate_fid_frequency == 0:
            print(f"Finished epoch number: {epoch + 1} and calculating FID for {self.num_img} images.")
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5
            generated_images = preprocess_input(generated_images)
            fid = calculate_fid(self.inception_model, generated_images, self.target_images)
            if fid < self.min_fid:
                best_model_path = os.path.join(self.model_dir, "best_model.h5")
                print(f"FID has improved from {self.min_fid} to {fid:.3f}. Saving model to {best_model_path}.")
                self.min_fid = fid
                self.model.generator.save(best_model_path)
            else:
                print(f"Calculated FID is {fid:.3f} which is lower than last checkpoint with FID {self.min_fid}")


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

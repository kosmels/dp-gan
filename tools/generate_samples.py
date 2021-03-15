import datetime
import os

import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import get_acgan_train_games, parse_config
from models.acgan import ACGAN
from models.discriminator import get_discriminator_model, get_discriminator_model_v2, get_discriminator_model_v3
from models.generator import get_generator_model, get_generator_model_v2, get_generator_model_v3
from train.callbacks import ACGAN_Visual_Monitor
from train.losses import (
    acgan_disc_cls_loss,
    acgan_gen_cls_loss,
    discriminator_bce_loss,
    discriminator_loss,
    generator_bce_loss,
    generator_loss,
)
from train.utils import create_clean_dir

if __name__ == "__main__":
    yaml_path = "configs/acgan_config_default.yml"
    parsed_config = parse_config(yaml_path)
    dataset_config = parsed_config["dataset"]
    train_images, train_labels = get_acgan_train_games(dataset_config)
    train_config = parsed_config["train"]

    class_dim = len(dataset_config["class_root_path"])
    d_model = get_discriminator_model_v3(dataset_config["image_shape"], class_dim)
    g_model = get_generator_model_v3(dataset_config["noise_dim"], class_dim)

    g_beta_1, g_beta_2 = train_config["generator_betas"][0], train_config["generator_betas"][1]
    generator_optimizer = Adam(learning_rate=train_config["generator_lr"], beta_1=g_beta_1)
    d_beta_1, d_beta_2 = train_config["discriminator_betas"][0], train_config["discriminator_betas"][1]
    discriminator_optimizer = Adam(learning_rate=train_config["discriminator_lr"], beta_1=d_beta_1)

    acgan = ACGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=dataset_config["noise_dim"],
        n_classes=class_dim,
        discriminator_extra_steps=3,
    )
    acgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        d_loss_cls_fn=acgan_disc_cls_loss,
        g_loss_fn=generator_loss,
        g_loss_cls_fn=acgan_gen_cls_loss,
        # run_eagerly=True
    )
    acgan.built = True
    acgan.load_weights("outputs/ACWGAN_2021-03-14_18:37:25/checkpoints/saved-model-1600.hdf5")

    n = 40
    random_latent_vectors = tf.random.normal(shape=(n, dataset_config["noise_dim"]))
    sampled_labels = tf.ones((n, 1), dtype=tf.int32)
    sampled_labels *= 0
    # sampled_labels = tf.reshape(tf.range(0, 1, dtype=tf.int32), shape=(-1, 1))
    generated_images = acgan.generator([random_latent_vectors, sampled_labels])
    generated_images = (generated_images * 127.5) + 127.5

    for i in range(n):
        img = generated_images[i].numpy()
        cv2.imwrite(f"visual-test/generated_{i}.jpg", img)

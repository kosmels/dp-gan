import datetime
import os
import shutil

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import get_train_images, parse_config
from models.discriminator import get_discriminator_model, get_wgan_discriminator_model_v5
from models.generator import get_generator_model, get_wgan_generator_model_v5
from models.wgan import WGAN
from train.callbacks import WGAN_FID_Monitor, WGAN_Visual_Monitor
from train.losses import discriminator_loss, generator_loss
from train.utils import create_clean_dir


def train_wgan():
    # we will reshape each sample to (28, 28, 1) and normalize the pixel values in [-1, 1].
    yaml_path = "configs/wpgan_config_default.yml"
    parsed_config = parse_config(yaml_path)
    dataset_config = parsed_config["dataset"]
    train_config = parsed_config["train"]
    train_images = get_train_images(dataset_config)
    train_images = train_images.reshape(train_images.shape[0], *dataset_config["image_shape"]).astype("float32")
    train_images = (train_images - 127.5) / 127.5
    print(f"Loaded {train_images.shape[0]} images from {dataset_config['class_root_path']}")

    d_model = get_wgan_discriminator_model_v5(dataset_config["image_shape"])
    d_model.summary()

    g_model = get_wgan_generator_model_v5(dataset_config["noise_dim"])
    g_model.summary()

    # Optimizer for both the networks
    # learning_rate=0.0002, beta_1=0.5 are recommened
    g_beta_1, g_beta_2 = train_config["generator_betas"][0], train_config["generator_betas"][1]
    generator_optimizer = Adam(learning_rate=train_config["generator_lr"], beta_1=g_beta_1, beta_2=g_beta_2)
    d_beta_1, d_beta_2 = train_config["discriminator_betas"][0], train_config["discriminator_betas"][1]
    discriminator_optimizer = Adam(learning_rate=train_config["discriminator_lr"], beta_1=d_beta_1, beta_2=d_beta_2)

    # Create log dir for config, visual outputs, model checkpoints
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = create_clean_dir(
        os.path.join(
            train_config["train_log_root"],
            f"{parsed_config['model']['type']}_{dataset_config['class_root_path'].split('/')[-1]}_{current_time}",
        )
    )

    shutil.copy(yaml_path, os.path.join(output_dir, "config.yaml"))

    # Create callbacks
    # Visualize samples after x epochs
    images_dir = create_clean_dir(os.path.join(output_dir, "images"))
    wgan_visualizer = WGAN_Visual_Monitor(
        output_dir=images_dir,
        num_img=train_config["image_visual_num"],
        latent_dim=dataset_config["noise_dim"],
        visual_frequency=train_config["image_visual_frequency"],
    )

    # Prepare directory for model checkpoints
    checkpoint_dir = create_clean_dir(os.path.join(output_dir, train_config["train_checkpoints"]))

    """
    WE HAVE SUPPRESSED ModeCheckpoint AS WE ARE NOW CALCULATING FID SCORE OVER TIME AND
    BASED ON THIS METRIC WE ARE SAVING ONLY 1 BEST MODEL (April 5, 2021)
    """
    # checkpoint_name = os.path.join(checkpoint_dir, "epoch_{epoch:04d}.hdf5")
    # print(f"Checkpoint saved after 20 epochs. Step size: {np.ceil(len(train_images) / train_config['batch_size'])}")
    # wgan_model_checkpointer = ModelCheckpoint(
    #     filepath=checkpoint_name,
    #     monitor="d_loss",
    #     save_freq=int(np.ceil(len(train_images) / train_config["batch_size"]) * train_config["checkpoint_freq"]),
    #     save_best_only=False,
    #     save_weights_only=True,
    #     verbose=1,
    # )

    wgan_fid_callback = WGAN_FID_Monitor(
        latent_dim=dataset_config["noise_dim"],
        calculate_fid_frequency=train_config["calculate_fid_frequency"],
        model_dir=checkpoint_dir,
    )

    tensorboard_dir = os.path.join(train_config["tensorboard_root"], f"{output_dir.split('/')[-1]}")
    wgan_tensorboard = TensorBoard(log_dir=tensorboard_dir)

    # Get the wgan model
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=dataset_config["noise_dim"],
        discriminator_extra_steps=train_config["discriminator_extra_steps"],
    )

    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss,
    )

    # Start training
    wgan.fit(
        train_images,
        batch_size=train_config["batch_size"],
        epochs=train_config["epochs"],
        callbacks=[wgan_visualizer, wgan_fid_callback, wgan_tensorboard],
    )


if __name__ == "__main__":
    train_wgan()

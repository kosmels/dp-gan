"""
Training script for the end-to-end AC(W)GAN model.
Author: Silvester Kosmel
"""
import datetime
import logging
import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import get_acgan_train_games, parse_config
from models.acgan import ACGAN
from models.discriminator import (get_discriminator_model,
                                  get_discriminator_model_v2,
                                  get_discriminator_model_v3)
from models.generator import (get_generator_model, get_generator_model_v2,
                              get_generator_model_v3)
from train.callbacks import ACGAN_Visual_Monitor
from train.losses import (acgan_disc_cls_loss, acgan_gen_cls_loss,
                          discriminator_bce_loss, discriminator_loss,
                          generator_bce_loss, generator_loss)
from train.utils import create_clean_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_acgan():
    # we will reshape each sample to (28, 28, 1) and normalize the pixel values in [-1, 1].
    yaml_path = "configs/acgan_config_default.yml"
    parsed_config = parse_config(yaml_path)
    dataset_config = parsed_config["dataset"]
    train_config = parsed_config["train"]
    train_images, train_labels = get_acgan_train_games(dataset_config)
    train_images = (train_images - 127.5) / 127.5

    class_dim = len(dataset_config["class_root_path"])
    logger.info(f"Loaded {len(train_images)} images for {class_dim} classes.")
    logger.info(f"Loaded classes: {dataset_config['class_root_path']}")
    d_model = get_discriminator_model_v3(dataset_config["image_shape"], class_dim)
    d_model.summary()

    g_model = get_generator_model_v3(dataset_config["noise_dim"], class_dim)
    g_model.summary()

    g_beta_1, g_beta_2 = train_config["generator_betas"][0], train_config["generator_betas"][1]
    generator_optimizer = Adam(learning_rate=train_config["generator_lr"], beta_1=g_beta_1)
    d_beta_1, d_beta_2 = train_config["discriminator_betas"][0], train_config["discriminator_betas"][1]
    discriminator_optimizer = Adam(learning_rate=train_config["discriminator_lr"], beta_1=d_beta_1)

    # Callbacks
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = create_clean_dir(
        os.path.join(train_config["sampled_output_dir"], f"{parsed_config['model']['type']}_{current_time}")
    )
    cbk = ACGAN_Visual_Monitor(
        output_dir=output_dir,
        n_classes=class_dim,
        num_img=8,
        latent_dim=dataset_config["noise_dim"],
    )

    checkpoints_dir = create_clean_dir(os.path.join(output_dir, "checkpoints"))
    checkpointer = ModelCheckpoint(
        os.path.join(checkpoints_dir, "saved-model-{epoch:02d}.hdf5"),
        monitor="d_loss",
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        save_freq=100 * (train_images.shape[0] // train_config["batch_size"]),
    )

    tensorboard = TensorBoard(output_dir)

    # Get the wgan model
    acgan = ACGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=dataset_config["noise_dim"],
        n_classes=class_dim,
        discriminator_extra_steps=3,
    )

    # Compile the wgan model
    acgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        d_loss_cls_fn=acgan_disc_cls_loss,
        g_loss_fn=generator_loss,
        g_loss_cls_fn=acgan_gen_cls_loss,
        # run_eagerly=True
    )

    # Start training
    acgan.fit(
        (train_images, train_labels),
        batch_size=train_config["batch_size"],
        epochs=train_config["epochs"],
        shuffle=True,
        verbose=1,
        callbacks=[cbk, checkpointer, tensorboard],
    )


if __name__ == "__main__":
    train_acgan()

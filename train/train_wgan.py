import datetime
import os
import shutil

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import get_train_images, parse_config
from models.discriminator import get_discriminator_model
from models.generator import get_generator_model
from models.wgan import WGAN
from train.callbacks import WGAN_Visual_Monitor
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

    d_model = get_discriminator_model(dataset_config["image_shape"])
    d_model.summary()

    g_model = get_generator_model(dataset_config["noise_dim"])
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
            "images",
        )
    )

    shutil.copy(yaml_path, os.path.join(output_dir, "config.yaml"))

    # Create callbacks
    # Visualize samples after x epochs
    wgan_visualizer = WGAN_Visual_Monitor(
        output_dir=output_dir,
        num_img=3,
        latent_dim=dataset_config["noise_dim"],
    )

    # Save model after each epoch, only better models based on discriminator loss
    checkpoint_dir = os.path.join(output_dir, train_config["train_checkpoints"])
    wgan_model_checkpointer = ModelCheckpoint(
        filepath=checkpoint_dir, monitor="d_loss", mode="max", save_best_only=True
    )

    tensorboard_dir = os.path.join(output_dir, "logs")
    wgan_tensorboard = TensorBoard(log_dir=tensorboard_dir)

    # Get the wgan model
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=dataset_config["noise_dim"],
        discriminator_extra_steps=3,
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
        callbacks=[wgan_visualizer, wgan_model_checkpointer, wgan_tensorboard],
    )


if __name__ == "__main__":
    train_wgan()

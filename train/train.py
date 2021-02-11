"""
## Train the end-to-end model
"""
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import (get_acgan_train_games, get_train_images,
                                    parse_config)
from models.acgan import ACGAN
from models.discriminator import (get_discriminator_model,
                                  get_discriminator_model_v2)
from models.generator import get_generator_model, get_generator_model_v2
from models.wgan import WGAN
from train.callbacks import GANMonitor
from train.losses import (acgan_disc_cls_loss, acgan_disc_loss,
                          acgan_gen_cls_loss, acgan_gen_loss,
                          discriminator_loss, generator_loss)


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
    discriminator_optimizer = Adam(learning_rate=train_config["discriminator_lr"], beta_1=0.5, beta_2=0.9)

    # Callbacks
    cbk = GANMonitor(
        output_dir=train_config["sampled_output_dir"],
        model_name=parsed_config["model"]["type"],
        num_img=3,
        latent_dim=dataset_config["noise_dim"],
    )

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
    wgan.fit(train_images, batch_size=train_config["batch_size"], epochs=train_config["epochs"], callbacks=[cbk])


def train_acgan():
    # we will reshape each sample to (28, 28, 1) and normalize the pixel values in [-1, 1].
    yaml_path = "configs/acgan_config_default.yml"
    parsed_config = parse_config(yaml_path)
    dataset_config = parsed_config["dataset"]
    train_config = parsed_config["train"]
    print(dataset_config)
    train_images, train_labels = get_acgan_train_games(dataset_config)
    train_images = (train_images - 127.5) / 127.5

    class_dim = len(dataset_config["class_root_path"])
    d_model = get_discriminator_model_v2(dataset_config["image_shape"], class_dim)
    d_model.summary()

    g_model = get_generator_model_v2(dataset_config["noise_dim"], class_dim)
    g_model.summary()

    # Optimizer for both the networks
    # learning_rate=0.0002, beta_1=0.5 are recommened
    g_beta_1, g_beta_2 = train_config["generator_betas"][0], train_config["generator_betas"][1]
    generator_optimizer = Adam(learning_rate=train_config["generator_lr"], beta_1=g_beta_1, beta_2=g_beta_2)
    d_beta_1, d_beta_2 = train_config["discriminator_betas"][0], train_config["discriminator_betas"][1]
    discriminator_optimizer = Adam(learning_rate=train_config["discriminator_lr"], beta_1=d_beta_1, beta_2=d_beta_2)

    # Callbacks
    cbk = GANMonitor(
        output_dir=train_config["sampled_output_dir"],
        model_name=parsed_config["model"]["type"],
        n_classes=class_dim,
        num_img=3,
        latent_dim=dataset_config["noise_dim"],
    )

    # Get the wgan model
    wgan = ACGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=dataset_config["noise_dim"],
        n_classes=class_dim,
        discriminator_extra_steps=3,
    )

    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        d_loss_cls_fn=acgan_disc_cls_loss,
        g_loss_fn=generator_loss,
        g_loss_cls_fn=acgan_gen_cls_loss,
        # run_eagerly=True
    )

    # Start training
    wgan.fit(
        (train_images, train_labels),
        batch_size=train_config["batch_size"],
        epochs=train_config["epochs"],
        verbose=0,
        callbacks=[cbk],
    )


if __name__ == "__main__":
    train_acgan()

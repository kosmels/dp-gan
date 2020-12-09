"""
## Train the end-to-end model
"""
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import get_train_images, parse_config
from models.discriminator import get_discriminator_model
from models.generator import get_generator_model
from models.wgan import WGAN
from train.callbacks import GANMonitor
from train.losses import discriminator_loss, generator_loss

if __name__ == "__main__":
    # we will reshape each sample to (28, 28, 1) and normalize the pixel values in [-1, 1].
    yaml_path = "configs/wpgan_config_default.yml"
    dataset_config = parse_config(yaml_path)["dataset"]
    train_config = parse_config(yaml_path)["train"]
    train_images = get_train_images(dataset_config)
    train_images = train_images.reshape(train_images.shape[0], *dataset_config["image_shape"]).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    d_model = get_discriminator_model(dataset_config["image_shape"])
    d_model.summary()

    g_model = get_generator_model(dataset_config["noise_dim"])
    g_model.summary()

    # Optimizer for both the networks
    # learning_rate=0.0002, beta_1=0.5 are recommened
    generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    # Epochs to train
    epochs = 500

    # Callbacks
    cbk = GANMonitor(num_img=3, latent_dim=dataset_config["noise_dim"])

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
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    # Start training
    wgan.fit(train_images, batch_size=train_config["batch_size"], epochs=epochs, callbacks=[cbk])
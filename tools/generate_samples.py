import cv2
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam

from datasets.preprocessing import get_acgan_train_games, get_train_images, parse_config
from metrics.evaluate import calculate_fid
from models.acgan import ACGAN
from models.discriminator import get_discriminator_model, get_discriminator_model_v2, get_discriminator_model_v3
from models.generator import get_generator_model, get_generator_model_v2, get_generator_model_v3
from models.wgan import WGAN
from train.losses import acgan_disc_cls_loss, acgan_gen_cls_loss, discriminator_loss, generator_loss


def get_train_and_dataset_configs(config_path: str):
    parsed_config = parse_config(config_path)
    dataset_config = parsed_config["dataset"]
    train_config = parsed_config["train"]

    return train_config, dataset_config


def generate_acgan_samples():
    train_config, dataset_config = get_train_and_dataset_configs("configs/acgan_config_default.yml")

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

    n = 500
    random_latent_vectors = tf.random.normal(shape=(n, dataset_config["noise_dim"]))
    sampled_labels = tf.ones((n, 1), dtype=tf.int32)
    sampled_labels *= 0
    # sampled_labels = tf.reshape(tf.range(0, 1, dtype=tf.int32), shape=(-1, 1))
    generated_images = acgan.generator([random_latent_vectors, sampled_labels])
    generated_images = (generated_images * 127.5) + 127.5

    for i in range(n):
        img = generated_images[i].numpy()
        cv2.imwrite(f"visual-test/generated_{i+501}.jpg", img)


def generate_wgan_samples(calculate_stats: bool = False):
    train_config, dataset_config = get_train_and_dataset_configs("configs/wpgan_config_default.yml")
    train_images = get_train_images(dataset_config)
    train_images = train_images.reshape(train_images.shape[0], *dataset_config["image_shape"]).astype("float32")

    d_model = get_discriminator_model(dataset_config["image_shape"])
    g_model = get_generator_model(dataset_config["noise_dim"])

    g_beta_1, g_beta_2 = train_config["generator_betas"][0], train_config["generator_betas"][1]
    generator_optimizer = Adam(learning_rate=train_config["generator_lr"], beta_1=g_beta_1)
    d_beta_1, d_beta_2 = train_config["discriminator_betas"][0], train_config["discriminator_betas"][1]
    discriminator_optimizer = Adam(learning_rate=train_config["discriminator_lr"], beta_1=d_beta_1)

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=dataset_config["noise_dim"],
        discriminator_extra_steps=3,
    )
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss,
    )
    wgan.built = True
    if calculate_stats:

        model = InceptionV3(include_top=False, pooling="avg", input_shape=(224, 224, 3))
        train_images = preprocess_input(train_images)
        for x in range(1500, 3000, 20):
            wgan.load_weights(f"outputs/WGAN_cierneFlaky_2021-03-13_21:47:07/checkpoints/epoch_{str(x)}.hdf5")

            n = 224
            random_latent_vectors = tf.random.normal(shape=(n, dataset_config["noise_dim"]))
            generated_images = wgan.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5

            test_images = preprocess_input(generated_images)
            fid = calculate_fid(model, test_images, train_images)
            print(f"FID for epoch {x} (different): %.3f" % fid)
    else:
        wgan.load_weights(f"outputs/WGAN_cierneFlaky_2021-03-13_21:47:07/checkpoints/epoch_1920.hdf5")
        n = 656
        random_latent_vectors = tf.random.normal(shape=(n, dataset_config["noise_dim"]))
        generated_images = wgan.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(n):
            img = generated_images[i].numpy()
            cv2.imwrite(f"visual-test/generated_{i+151}.jpg", img)


if __name__ == "__main__":
    generate_wgan_samples()

import tensorflow as tf


def acgan_disc_cls_loss(real_cls_logits, fake_cls_logits, real_labels, fake_labels):
    real_loss = tf.keras.losses.sparse_categorical_crossentropy(real_labels, real_cls_logits)
    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_cls_logits)
    return fake_loss + real_loss


def acgan_gen_cls_loss(cls_logits, labels):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, cls_logits)


def discriminator_bce_loss(real_img, fake_img):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_img), real_img)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_img), fake_img)
    return real_loss + fake_loss


def generator_bce_loss(fake_img):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_img), fake_img)


# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

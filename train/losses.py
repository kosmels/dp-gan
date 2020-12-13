import tensorflow as tf


def acgan_disc_loss(real_img_logits, fake_img_logits):
    # print(real_img_logits)
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones(tf.shape(real_img_logits)), real_img_logits)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros(tf.shape(fake_img_logits)), fake_img_logits)
    return fake_loss + real_loss


def acgan_disc_cls_loss(real_cls_logits, fake_cls_logits, real_labels, fake_labels):
    real_loss = tf.nn.softmax_cross_entropy_with_logits(real_labels, real_cls_logits)
    fake_loss = tf.nn.softmax_cross_entropy_with_logits(fake_labels, fake_cls_logits)
    return fake_loss + real_loss


def acgan_gen_loss(img_logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(tf.ones(tf.shape(img_logits)), img_logits)


def acgan_gen_cls_loss(cls_logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(labels, cls_logits)


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

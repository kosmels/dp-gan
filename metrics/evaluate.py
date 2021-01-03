import os

import cv2
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)

from datasets.preprocessing import (get_acgan_train_games, get_train_images,
                                    parse_config)


def calculate_fid(incept_model, input_images, target_images):
    """
    Code from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

    :param incept_model:
    :param input_images:
    :param target_images:
    :return:
    """
    # calculate activations
    act1 = incept_model.predict(input_images)
    act2 = incept_model.predict(target_images)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    yaml_path = "configs/wpgan_config_default.yml"
    parsed_config = parse_config(yaml_path)
    dataset_config = parsed_config["dataset"]
    train_config = parsed_config["train"]
    train_images = get_train_images(dataset_config)
    train_images = train_images.reshape(train_images.shape[0], *dataset_config["image_shape"]).astype("float32")

    image_paths = os.listdir("outputs/wgan_23_11_2020_cierneFlaky_outputs")[1000:1224]
    test_images = []
    for image_path in image_paths:
        test_images.append(cv2.imread(os.path.join("outputs/wgan_23_11_2020_cierneFlaky_outputs", image_path)))
    test_images = np.array(test_images)
    print(test_images.shape, train_images.shape)
    train_images = preprocess_input(train_images)
    test_images = preprocess_input(test_images)

    model = InceptionV3(include_top=False, pooling="avg", input_shape=(224, 224, 3))
    fid = calculate_fid(model, test_images, train_images)
    print("FID (different): %.3f" % fid)

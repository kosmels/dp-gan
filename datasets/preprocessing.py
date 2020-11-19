import logging
import os
from typing import Any, Dict, Union

import cv2
import numpy as np
import yaml


def setup_logger():
    # create logger with 'spam_application'
    logger = logging.getLogger('spam_application')
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)


def parse_config(yaml_path: str) -> Union[Dict[str, Any], None]:
    """
    #TODO: Add docstring

    :param yaml_path:
    :return:
    """
    with open(yaml_path, "r") as yaml_file:
        loaded_yaml = yaml.safe_load(yaml_file)
    return loaded_yaml


def get_train_images(config: Dict[str, Any]) -> np.ndarray:
    data_root = os.path.join(config.get("data_path"), config.get("class_root_path"))
    image_paths = os.listdir(data_root)
    train_images = []
    for image_path in image_paths:
        train_images.append(cv2.imread(os.path.join(data_root, image_path)))

    return np.array(train_images)


if __name__ == "__main__":
    yaml_path = "configs/wpgan_config_default.yml"
    dataset_config = parse_config(yaml_path)['dataset']
    train_images = get_train_images(dataset_config)
    print(train_images.shape)

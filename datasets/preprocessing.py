import logging

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


def parse_config(yaml_path: str):
    with open(yaml_path, "r") as yaml_file:
        loaded_yaml = yaml.safe_load(yaml_file)
    return loaded_yaml


def get_train_images():
    pass


if __name__ == "__main__":
    pass

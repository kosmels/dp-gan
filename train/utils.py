import os


def create_clean_dir(path, overwrite=False):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            return path
    os.makedirs(path)
    return path

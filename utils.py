import os


def ensure_dir(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

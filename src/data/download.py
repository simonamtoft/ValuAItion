import os
import wget
import logging
from pathlib import Path
from .constants import DATA_DIR, BASE_FILENAME, BASE_DATASET_URL


def download_data():
    for split in ['TRAIN', 'TEST']:
        filename = f'{BASE_FILENAME}_{split}.csv'
        filepath = os.path.join(DATA_DIR, filename)
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(filepath):
            logging.info(
                'Downloading dataset file: %s from %s',
                filename, BASE_DATASET_URL)
            filename = wget.download(
                f'{BASE_DATASET_URL}/{filename}', out=filepath)
        else:
            logging.debug(
                'Skipping download of already existing dataset file: %s',
                filepath)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    download_data()

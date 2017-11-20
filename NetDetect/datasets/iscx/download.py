from . import config
from ..utils.network_utils import download_file


if __name__ == "__main__":
  download_file(config.TRAIN_FEATURES_URL,
                config.DUMPS_DIR + config.PROCESSED_TRAIN_NAME + "_features")
  download_file(config.TRAIN_LABELS_URL,
                config.DUMPS_DIR + config.PROCESSED_TRAIN_NAME + "_labels")

  download_file(config.TEST_FEATURES_URL,
                config.DUMPS_DIR + config.PROCESSED_TEST_NAME + "_features")
  download_file(config.TEST_LABELS_URL,
                config.DUMPS_DIR + config.PROCESSED_TEST_NAME + "_labels")


from ..utils.network_utils import download_file
from . import config


if __name__ == "__main__":
  # Download training set
  download_file("iscx_train_raw", RAW_TRAINING_DATASET_PATH)

  # Download testing set
  download_file("iscx_test_raw", RAW_TESTING_DATASET_PATH)


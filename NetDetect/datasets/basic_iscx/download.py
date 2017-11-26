from ..utils.network_utils import download_file
from . import config


if __name__ == "__main__":
  # Download training set to:
  download_file("datasets", "iscx_train_X_basic",
                config.DUMPS_DIR + "train_X_basic" + ".np")
  download_file("datasets", "iscx_train_Y_basic",
                config.DUMPS_DIR + "train_Y_basic" + ".np")

  # Download testing set
  download_file("datasets", "iscx_test_X_basic",
                config.DUMPS_DIR + "test_X_basic" + ".np")
  download_file("datasets", "iscx_test_Y_basic",
                config.DUMPS_DIR + "test_Y_basic" + ".np")


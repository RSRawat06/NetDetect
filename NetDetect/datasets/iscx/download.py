from ..utils.network_utils import download_file
from . import config
import argparse


if __name__ == "__main__":
  # Set up n_steps
  parser = argparse.ArgumentParser()
  parser.add_argument("-steps", "--steps",
                      help="Steps in sequence.",
                      type=int, required=True)
  n_steps = str(parser.parse_args().steps)

  # Download training set to:
  # /iscx_train_X_24 -> /train_X_24.np
  # /iscx_train_Y_24 -> /train_Y_24.np
  download_file("datasets", "iscx_train_X_" + n_steps,
                config.DUMPS_DIR + "train_X_" + n_steps + ".np")
  download_file("datasets", "iscx_train_Y_" + n_steps,
                config.DUMPS_DIR + "train_Y_" + n_steps + ".np")

  # Download testing set
  # /iscx_test_X_24 -> /test_X_24.np
  # /iscx_test_Y_24 -> /test_Y_24.np
  download_file("datasets", "iscx_test_X_" + n_steps,
                config.DUMPS_DIR + "test_X_" + n_steps + ".np")
  download_file("datasets", "iscx_test_Y_" + n_steps,
                config.DUMPS_DIR + "test_Y_" + n_steps + ".np")


from ..utils.network_utils import upload_file
from . import config, preprocess_file
from .logger import set_logger
import numpy as np


def main():
  '''
  Generates dataset out of ISCX.
  '''

  set_logger.info("Beginning ISCX dataset generation")

  ##############################
  ### Generate training dataset

  # Preprocess file
  train_X, train_Y = preprocess_file(
      config.RAW_TRAINING_DATASET_PATH)
  set_logger.info("Training dataset preprocessed.")

  # Dumping training features
  with open(config.DUMPS_DIR + "train_X_basic" + ".np",
            'wb') as f:
    np.save(f, np.array(train_X, dtype=np.float32))
    set_logger.info("Training features dumped.")
  del(train_X)

  # Dumping training labels
  with open(config.DUMPS_DIR + "train_Y_basic" + ".np",
            'wb') as f:
    np.save(f, np.array(train_Y))
    set_logger.info("Training labels dumped.")
  del(train_Y)

  ##############################


  ##############################
  ### Generate testing dataset

  # Preprocess file
  test_X, test_Y = preprocess_file(
      config.RAW_TESTING_DATASET_PATH)
  set_logger.info("Testing dataset preprocessed.")

  # Dumping testing features
  with open(config.DUMPS_DIR + "test_X_basic" + ".np",
            'wb') as f:
    np.save(f, np.array(test_X, dtype=np.float32))
    set_logger.info("Testing features dumped.")
  del(test_X)

  # Dumping testing labels
  with open(config.DUMPS_DIR + "test_Y_basic" + ".np",
            'wb') as f:
    np.save(f, np.array(test_Y))
    set_logger.info("Testing labels dumped.")
  del(test_Y)

  ##############################


  ##############################
  ### Upload files

  upload_file("datasets", "iscx_train_X_basic",
              config.DUMPS_DIR + "train_X_basic" + ".np")
  upload_file("datasets", "iscx_train_Y_basic",
              config.DUMPS_DIR + "train_Y_basic" + ".np")

  upload_file("datasets", "iscx_test_X_basic",
              config.DUMPS_DIR + "test_X_basic" + ".np")
  upload_file("datasets", "iscx_test_Y_basic",
              config.DUMPS_DIR + "test_Y_basic" + ".np")

  ##############################

  return None


if __name__ == "__main__":
  main()


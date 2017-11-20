from . import config, preprocess_file
import numpy as np
from .logger import set_logger


def main():
  '''
  Attempts to load label and feature components of the dataset.
  Triggers preprocessing of raw file if loading fails.
  '''

  set_logger.info("Creating and writing new dataset...")

  train_X, train_Y = preprocess_file(config.DUMPS_DIR + config.RAW_TRAIN_NAME)
  set_logger.info("Training dataset preprocessed.")

  with open(config.DUMPS_DIR + config.PROCESSED_TRAIN_NAME + "_features",
            'wb') as f:
    np.save(f, np.array(train_X, dtype=np.float32))
    set_logger.info("Training features dumped.")

  del(train_X)

  with open(config.DUMPS_DIR + config.PROCESSED_TRAIN_NAME + "_labels",
            'wb') as f:
    np.save(f, np.array(train_Y))
    set_logger.info("Training labels dumped.")

  del(train_Y)

  test_X, test_Y = preprocess_file(config.DUMPS_DIR + config.RAW_TEST_NAME)
  set_logger.info("Testing dataset preprocessed.")

  with open(config.DUMPS_DIR + config.PROCESSED_TEST_NAME + "_features",
            'wb') as f:
    np.save(f, np.array(test_X, dtype=np.float32))
    set_logger.info("Testing features dumped.")

  del(test_X)

  with open(config.DUMPS_DIR + config.PROCESSED_TEST_NAME + "_labels",
            'wb') as f:
    np.save(f, np.array(test_Y))
    set_logger.info("Testing labels dumped.")

  del(test_Y)

  return None


if __name__ == "__main__":
  set_logger.info("Beginning dataset generation.")
  main()


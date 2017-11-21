from . import config, preprocess_file
import numpy as np
from .logger import set_logger
import argparse


def main(n_steps):
  '''
  Generates dataset out of ISCX.
  '''

  set_logger.info("Beginning ISCX dataset generation")
  set_logger.info("Number of steps: " + str(n_steps))

  ##############################
  ### Generate training dataset

  # Preprocess file
  train_X, train_Y = preprocess_file(config.DUMPS_DIR + config.RAW_TRAIN_NAME,
                                     n_steps)
  set_logger.info("Training dataset preprocessed.")

  # Dumping training features
  with open(config.DUMPS_DIR + config.PROCESSED_TRAIN_NAME + "_X",
            'wb') as f:
    np.save(f, np.array(train_X, dtype=np.float32))
    set_logger.info("Training features dumped.")
  del(train_X)

  # Dumping training labels
  with open(config.DUMPS_DIR + config.PROCESSED_TRAIN_NAME + "_Y",
            'wb') as f:
    np.save(f, np.array(train_Y))
    set_logger.info("Training labels dumped.")
  del(train_Y)

  ##############################


  ##############################
  ### Generate testing dataset

  # Preprocess file
  test_X, test_Y = preprocess_file(config.DUMPS_DIR + config.RAW_TEST_NAME,
                                   n_steps)
  set_logger.info("Testing dataset preprocessed.")

  # Dumping testing features
  with open(config.DUMPS_DIR + config.PROCESSED_TEST_NAME + "_X",
            'wb') as f:
    np.save(f, np.array(test_X, dtype=np.float32))
    set_logger.info("Testing features dumped.")
  del(test_X)

  # Dumping testing labels
  with open(config.DUMPS_DIR + config.PROCESSED_TEST_NAME + "_Y",
            'wb') as f:
    np.save(f, np.array(test_Y))
    set_logger.info("Testing labels dumped.")
  del(test_Y)

  ##############################

  return None


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-steps", "--steps", help="Steps in sequence", type=int, required=True)

  main(n_steps=parser.parse_args().steps)


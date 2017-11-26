from . import config
from .logger import set_logger
from ..utils import shaping_utils
import numpy as np


def load(test_size):
  '''
  Loads preprocessed data dump if possible.
  '''

  try:
    #######################################
    ### Testing dataset load.

    # Load testing features
    with open(config.DUMPS_DIR + "test_X_basic" + ".np",
              'rb') as f_x:
      set_logger.info("Testing features exist. Processing...")
      full_test_X = np.load(f_x)
    # Load testing labels
    with open(config.DUMPS_DIR + "test_Y_basic" + ".np",
              'rb') as f_y:
      set_logger.info("Testing labels exist. Processing...")
      full_test_Y = np.load(f_y)

    # Shuffle testing dataset
    full_test_X, full_test_Y = shaping_utils.shuffle_twins(full_test_X,
                                                           full_test_Y)
    # Cut testing features
    test_X = full_test_X[:test_size]
    del(full_test_X)
    # Cut testing labels
    test_Y = full_test_Y[:test_size]
    del(full_test_Y)

    #######################################


    #######################################
    ### Training dataset load.

    # Load training features
    with open(config.DUMPS_DIR + "train_X_basic" + ".np",
              'rb') as f_x:
      set_logger.info("Training features exist. Processing...")
      train_X = np.load(f_x)
    # Load training labels
    with open(config.DUMPS_DIR + "train_Y_basic" + ".np",
              'rb') as f_y:
      set_logger.info("Training labels exist. Processing...")
      train_Y = np.load(f_y)
    # Shuffle training dataset
    train_X, train_Y = shaping_utils.shuffle_twins(train_X, train_Y)

    #######################################

    return (train_X, train_Y), (test_X, test_Y)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


def load_full_test(n_steps):
  '''
  Loads preprocessed data dump for test if possible.
  '''

  try:
    # Load testing features
    with open(config.DUMPS_DIR + "test_X_basic" + ".np",
              'rb') as f_x:
      set_logger.info("Testing features exist. Processing...")
      full_test_X = np.load(f_x)
    # Load testing labels
    with open(config.DUMPS_DIR + "test_Y_basic" + ".np",
              'rb') as f_y:
      set_logger.info("Testing labels exist. Processing...")
      full_test_Y = np.load(f_y)

    # Shuffle testing dataset
    full_test_X, full_test_Y = shaping_utils.shuffle_twins(full_test_X,
                                                           full_test_Y)

    return full_test_X, full_test_Y

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


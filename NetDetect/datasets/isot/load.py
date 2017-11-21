from . import config
from .logger import set_logger
from ..utils import shaping_utils
import pickle


def load(test_size):
  '''
  Loads preprocessed data dump if possible.
  '''

  try:
    #######################################
    ### Dataset load.

    with open(config.DUMPS_DIR + config.PROCESSED_NAME, "rb") as f:
      set_logger.info("Dataset exists. Processing...")
      X, Y = pickle.load(f)

    # Shuffle dataset
    X, Y = shaping_utils.shuffle_twins((X, Y))

    # Cut testing features
    test_X = X[:test_size]
    train_X = X[test_size:]
    del(X)

    # Cut testing labels
    test_Y = Y[:test_size]
    train_Y = Y[test_size:]
    del(Y)

    #######################################

    return (train_X, train_Y), (test_X, test_Y)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


def load_full_test():
  '''
  Loads preprocessed data dump for test if possible.
  '''

  try:
    with open(config.DUMPS_DIR + config.PROCESSED_NAME, "rb") as f:
      set_logger.info("Dataset exists. Processing...")
      X, Y = pickle.load(f)
    X, Y = shaping_utils.shuffle_twins((X, Y))
    return X, Y

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


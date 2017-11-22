from . import config
from .logger import set_logger
from ..utils import shaping_utils
import pickle


def load(test_size, n_steps):
  '''
  Loads preprocessed data dump if possible.
  '''

  try:
    #######################################
    ### Dataset load.

    with open(config.DUMPS_DIR + "dataset_" + str(n_steps) + ".p", "rb") as f:
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


def load_full_test(n_steps):
  '''
  Loads preprocessed data dump for test if possible.
  '''

  try:
    with open(config.DUMPS_DIR + "dataset_" + str(n_steps) + ".p", "rb") as f:
      set_logger.info("Dataset exists. Processing...")
      X, Y = pickle.load(f)
    X, Y = shaping_utils.shuffle_twins((X, Y))
    return X, Y

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


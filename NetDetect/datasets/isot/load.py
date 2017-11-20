from . import config
from .logger import set_logger
from ..utils import shaping_utils
import pickle


def load(test_size):
  '''
  Loads preprocessed data dump if possible.
  '''

  try:
    with open(config.DUMPS_DIR + config.PROCESSED_NAME, "rb") as f:
      set_logger.info("Dataset exists. Processing...")
      training_dataset, testing_dataset = pickle.load(f)

    #######################################
    ### Testing dataset load.

    # Shuffle testing dataset
    full_test_X, full_test_Y = shaping_utils.shuffle_twins(testing_dataset)
    del(testing_dataset)

    # Cut testing features
    test_X = full_test_X[:test_size]
    del(full_test_X)
    # Cut testing labels
    test_Y = full_test_Y[:test_size]
    del(full_test_Y)

    #######################################


    #######################################
    ### Training dataset load.

    train_X, train_Y = shaping_utils.shuffle_twins(training_dataset)
    del(training_dataset)

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
      training_dataset, testing_dataset = pickle.load(f)
      del(training_dataset)

    # Shuffle testing dataset
    full_test_X, full_test_Y = shaping_utils.shuffle_twins(testing_dataset)
    del(testing_dataset)

    return full_test_X, full_test_Y

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


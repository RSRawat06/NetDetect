from . import config
from .logger import set_logger
from ..utils import shaping_utils
import numpy as np


def load(n_test, n_val):
  '''
  Loads preprocessed data dump if possible.
  '''
  try:
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_labels",
              'rb') as f:
      Y = np.load(f)
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_features",
              'rb') as f:
      X = np.load(f)
    set_logger.info("Dataset exists. Processing...")
    X, Y = shaping_utils.shuffle_twins(X, Y)
    return shaping_utils.partition_dataset(X, Y, n_test, n_val)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


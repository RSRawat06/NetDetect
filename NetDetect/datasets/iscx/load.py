from . import config
from .logger import set_logger
from ..utils import shaping_utils
import numpy as np


def load(n_test, n_val):
  '''
  Loads preprocessed data dump if possible.
  '''
  try:
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_features",
              'rb') as f_x:
      with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_labels",
                'rb') as f_y:
        set_logger.info("Dataset exists. Processing...")
        return shaping_utils.partition_dataset(
            *shaping_utils.shuffle_twins(np.array(np.load(f_x)), np.load(f_y)), 
            n_test, n_val)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


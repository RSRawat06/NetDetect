from . import config
from .logger import set_logger
from ..utils import shaping_utils
import pickle


def load(n_test, n_val):
  '''
  Loads preprocessed data dump if possible.
  '''
  try:
    set_logger.info("Dataset exists. Attempting pickle load...")
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, "rb") as f:
      X, Y = shaping_utils.shuffle_twins(pickle.load(f))
      return shaping_utils.partition_dataset(X, Y, n_test, n_val)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None


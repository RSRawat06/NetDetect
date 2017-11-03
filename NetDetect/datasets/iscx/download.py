from . import config, preprocess_file
import numpy as np
from .logger import set_logger


def main():
  '''
  Main function of this download module.
  Attempts to load existing dataset and if fails, downloads raw dataset and
  saves a preprocessed copy.
  '''

  try:
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_labels",
              'rb') as f:
      Y = np.load(f)
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_features",
              'rb') as f:
      X = np.load(f)
    set_logger.info("Dataset loaded")

  except (EOFError, OSError, IOError) as e:
    set_logger.info("No dataset yet. Creating and writing new dataset...")
    X, Y = preprocess_file(config.DUMPS_DIR + config.RAW_SAVE_NAME)
    set_logger.info("Dataset preprocessed.")

    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_labels",
              'wb') as f:
      np.save(f, Y)
      set_logger.info("Labels dumped.")
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME + "_features",
              'wb') as f:
      np.save(f, X)
      set_logger.info("Features dumped.")

  return None


if __name__ == "__main__":
  set_logger.info("Beginning dataset download")
  main()

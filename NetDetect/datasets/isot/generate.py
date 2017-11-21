from . import config, preprocess_file
import pickle
from .logger import set_logger
import argparse


def main(n_steps):
  '''
  Generates dataset out of ISOT.
  '''

  set_logger.info("Beginning ISOT dataset generation")
  set_logger.info("Number of steps: " + str(n_steps))

  # Preprocess file
  X, Y = preprocess_file(
      config.DUMPS_DIR + config.RAW_NAME, n_steps)
  set_logger.info("Dataset preprocessed.")

  with open(config.DUMPS_DIR + config.PROCESSED_NAME, 'wb') as f:
    pickle.dump((X, Y), f)
    set_logger.info("Dataset pickle loaded and dumped.")

  return None


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-steps", "--steps", help="Steps in sequence",
                      type=int, required=True)

  main(n_steps=parser.parse_args().steps)


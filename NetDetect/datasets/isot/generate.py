from ..utils.network_utils import upload_file
from . import config, preprocess_file
from .logger import set_logger
import pickle
import argparse


def main(n_steps):
  '''
  Generates dataset out of ISOT.
  '''

  set_logger.info("Beginning ISOT dataset generation")
  set_logger.info("Number of steps: " + str(n_steps))

  # Preprocess file
  X, Y = preprocess_file(
      config.RAW_DATASET_PATH, n_steps)
  set_logger.info("Dataset preprocessed.")

  with open(config.DUMPS_DIR + "dataset_" + str(n_steps) + ".p", 'wb') as f:
    pickle.dump((X, Y), f)
    set_logger.info("Dataset pickle loaded and dumped.")

  # Upload file
  upload_file("datasets", "isot_" + str(n_steps),
              config.DUMPS_DIR + "dataset_" + str(n_steps) + ".p")

  return None


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-steps", "--steps", help="Steps in sequence",
                      type=int, required=True)

  main(n_steps=parser.parse_args().steps)


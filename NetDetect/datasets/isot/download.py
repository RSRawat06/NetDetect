from ..utils.network_utils import download_file
from . import config
import argparse


if __name__ == "__main__":
  # Set up n_steps
  parser = argparse.ArgumentParser()
  parser.add_argument("-steps", "--steps",
                      help="Steps in sequence.",
                      type=int, required=True)
  n_steps = str(parser.parse_args().steps)

  download_file("isot_" + n_steps,
                config.DUMPS_DIR + "dataset_" + n_steps + ".p")


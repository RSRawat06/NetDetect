from ..utils.network_utils import download_file
from . import config


if __name__ == "__main__":
  # Download set
  download_file("datasets", "isot_raw", config.RAW_DATASET_PATH)


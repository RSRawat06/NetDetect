from . import config
from ..utils.network_utils import download_file


if __name__ == "__main__":
  download_file(config.DATASET_URL,
                config.DUMPS_DIR + config.PROCESSED_NAME)


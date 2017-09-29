"""
iscx.download

Download the ISCX dataset locally.
Run: `python3 -m botnet_attention.iscx.download`
"""

from . import config
from ..utils import network
from ..preprocessing import main
import pickle


if __name__ == "__main__":
  try:
    preprocessed = pickle.load(open(config.DATA_DIR + config.TRAIN_SAVE, "rb"))
  except (OSError, IOError) as e:
    # network.download_file(config.TRAIN_URL, config.DATA_DIR + config.TRAIN_SAVE_RAW)
    preprocessed = main.preprocess(config.DATA_DIR + config.TRAIN_SAVE_RAW, config)
    print("File preprocessed")
    with open(config.DATA_DIR + config.TRAIN_SAVE, 'wb') as f:
      pickle.dump(preprocessed, f)
    print("File dumped")

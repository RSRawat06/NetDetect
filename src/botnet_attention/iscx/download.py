"""
iscx.download

Download the ISCX dataset locally.
Run: `python3 -m botnet_attention.iscx.download`
"""

from . import config
from ..utils import network

if __name__ == "__main__":
  # network.download_file(config.TEST_DATA_URL, config.DATA_DIR + config.TEST_DATA_NAME)
  network.download_file(config.TRAIN_DATA_URL, config.DATA_DIR + config.TRAIN_DATA_NAME)

  # fout = open(config.DATA_DIR + config.DATA_NAME, "a")
  # for line in open(config.DATA_DIR + config.TEST_DATA_NAME):
    # fout.write(line)
  # f = open(config.DATA_DIR + config.TRAIN_DATA_NAME)
  # f.next()
  # for line in f:
     # fout.write(line)
  # f.close()
  # fout.close()

"""
iscx.download

Download the ISCX dataset locally.
Run: `python3 -m botnet_attention.iscx.download`
"""

from . import config
from ..utils import network

if __name__ == "__main__":
  network.download_file(config.DATA_URL, config.DATA_DIR + config.DATA_NAME)

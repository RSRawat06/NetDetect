"""
isot.download

Download the ISOT dataset locally.
Run: `python3 -m botnet_attention.isot.download`
"""

from . import config
from ..utils import network

if __name__ == "__main__":
  network.download_file(config.DATA_URL, config.DATA_DIR + config.DATA_NAME)

from ...utils import network
from . import test_config as config


def test_download_file():
  network.download_file("http://x.com/", config.DATA_DIR + config.NETWORK_DATA_NAME)
  with open(config.DATA_DIR + config.NETWORK_DATA_NAME, 'rb') as myfile:
    data = str(myfile.read().decode())
  assert(data == "x")

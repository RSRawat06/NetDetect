from ...utils import network
from . import test_config as config


def test_download_file():
  network.download_file("https://dropbox.com/temp", "test_dump/temp.download")
  with open(config.DATA_DIR + config.NETWORK_DATA_NAME, 'rb') as myfile:
    data = myfile.read().replace('\n', '')
  assert(data == "The Test Worked")

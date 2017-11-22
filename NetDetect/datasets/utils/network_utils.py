"""
This module offers miscellanious networking functions.
"""

import requests
from azure.storage.blob import ContentSettings, BlockBlobService


block_blob_service = BlockBlobService(
    account_name='netdetect',
    account_key='REDACTED'
)


def download_file(container_name, blob_name, file_path):
  '''
  Download file from Azure blob.
  '''

  assert(blob_name in [x.name for x in
                       block_blob_service.list_blobs(container_name)])
  block_blob_service.get_blob_to_path(container_name, blob_name, file_path)


def upload_file(container_name, blob_name, file_path):
  '''
  Upload file from Azure blob.
  '''

  block_blob_service.create_blob_from_path(
      container_name,
      blob_name,
      file_path,
      content_settings=ContentSettings()
  )


def standard_download_file(url, file_path):
  '''
  Generic function to stream-download a large file.
  Args:
    - url (str): url to download from.
    - file_path (str): path to save file to.
  Returns:
    - file_path (str): original file_path where saved is.
  '''
  r = requests.get(url, stream=True)
  with open(file_path, 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)
  return file_path


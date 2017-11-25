from ..utils import csv_utils, shaping_utils, analysis_utils
from . import config
import numpy as np
import csv
from .logger import set_logger
from sklearn import preprocessing


def preprocess_file(file_path):
  '''
  Return preprocessed dataset from raw data file.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
    - n_steps (int)
  '''

  set_logger.info("Starting preprocessing...")
  return preprocess_features(*load_data(file_path))


def load_data(file_path):
  '''
  Load in a CSV and parse for data.
  Args:
    - file_path (str): path to raw dataset file.
  Returns:
    - X: np.array([[0.3, 0.3...]...], dtype=float32)
    - ips: [[ip1, ip2], [ip1, ip3], [ip1, ip2]...]
           where ip is string of ip address.
  '''

  X = []
  Y = []

  with open(file_path, 'r') as f:
    set_logger.debug("Opened dataset csv...")
    for i, row in enumerate(csv.reader(f)):
      # We assume the first row is a header.
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        set_logger.debug("Headers key generated: " + str(headers_key))
        continue

      # We collect relevant features and corresponding IPs.
      X.append(csv_utils.featurize_row(
          row, headers_key, config.numerical_fields))
      X.append(csv_utils.featurize_row(
          row, headers_key, config.numerical_fields))

      for j, ip in enumerate(identify_participants(row, headers_key)):
        assert(j != 2)
        Y.append(shaping_utils.build_one_hot(
            1 if ip in config.malicious_ips else 0,
            [0, 1]
        ))

  set_logger.debug("Basic data loading complete.")
  class_counts = analysis_utils.count_classes(Y)
  set_logger.debug("Class distribution is malignant: '" +
                   str(class_counts['1']) +
                   "', benign: '" + str(class_counts['0']) + "'.")

  return np.array(X, dtype=np.float32), \
      np.array(Y, dtype=np.uint8)


def preprocess_features(X, Y):
  '''
  Scale the feature vectors using scikit preprocessing.
  '''

  assert(len(X.shape) == 2)  # Double check that X is 2d.
  X = preprocessing.maxabs_scale(X, copy=False)
  return X, Y


def identify_participants(row, headers_key):
  '''
  Return participants for a given row.
  '''

  participants = []
  for i, value in enumerate(row):
    if headers_key[i] in config.participant_fields:
      participants.append(str(value))
  return participants


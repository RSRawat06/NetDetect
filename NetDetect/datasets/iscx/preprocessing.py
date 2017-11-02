from . import config
from ..utils import csv_utils, shaping_utils, analysis_utils
import numpy as np
import csv
from .logger import set_logger


def preprocess_file(file_path):
  '''
  Return preprocessed dataset from raw data file.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
  '''

  set_logger.info("Starting preprocessing...")
  return label_data(*segment_histories(*separate_ips(*load_data(file_path))))


def identify_participants(row, headers_key):
  '''
  Return participants for a given row.
  '''

  participants = []
  for i, value in enumerate(row):
    if headers_key[i] in config.participant_fields:
      participants.append(str(value))
  return participants


def load_data(file_path):
  '''
  Load in a CSV and parse for data.
  Args:
    - file_path (str): path to raw dataset file.
  Returns:
    - X: np.array([[0.3, 0.3...]...], dtype=float32)
    - metadata: [{info:a, label:x}... (n_classes)]
  '''

  feature_vectors = []
  participating_ips = []

  with open(file_path, 'r') as f:
    set_logger.debug("Opened dataset csv...")
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        set_logger.debug("Headers key generated: " + str(headers_key))
        continue

      set_logger.debug('Loading row: ' + str(i))
      feature_vectors.append(shaping_utils.featurize_row(
          row, headers_key, config.numerical_fields))
      participating_ips.append(identify_participants(row, headers_key))

  set_logger.debug("Basic data loading complete.")

  return np.array(feature_vectors, dtype=np.float32), participating_ips


def separate_ips(feature_vectors, participating_ips):
  '''
  Segment feature_vectors into their IPs.
  Essentially creates array of feature vectors that
  a given IP was involved in for each IP.
  Args:
    - feature_vectors (np.array)
    - participating_ips (list(str))
  Returns:
    - X (np.array): array of feature vectors
    - ip_addresses: corresponding IP addresses for entries in X
  '''

  X = []
  ip_addresses = []
  encountered_ip_count = 0

  # Maps a given IP address to its history's index in X.
  ip_history_map = {}

  set_logger.debug("Mapping history for each IP...")
  for i in range(feature_vectors.shape[0]):
    for ip in participating_ips[i]:
      if ip not in ip_history_map:
        ip_history_map[ip] = encountered_ip_count
        X.append([])
        ip_addresses.append(ip)
        encountered_ip_count += 1
      X[ip_history_map[ip]].append(feature_vectors[i])

  set_logger.debug("Separation by IP is complete.")
  set_logger.debug("Average history length for each ip: " +
                   str(len(X) / encountered_ip_count))

  return X, ip_addresses


def segment_histories(X, ip_addresses):
  new_X = []
  new_ip_addresses = []

  total_segment_counts = 0
  for i in range(X.shape[0]):
    segments = shaping_utils.segment_vector(X[i], config.SEQ_LEN)
    new_X += segments
    new_ip_addresses += segments.shape[0] * [ip_addresses[i]]

    total_segment_counts += segments.shape[0]

  set_logger.debug("Average seg count: " +
                   str(total_segment_counts / len(new_X)))

  set_logger.debug("History segmentation complete.")
  return np.array(new_X, dtype=np.float32), new_ip_addresses


def label_data(X, ip_addresses):
  '''
  Label data.
  Args:
    - X (np.array).
    - ip_addresses (list of str).
  Returns:
    - X (np.array): the original X from args.
    - Y (np.array): the new labels for dataset.
  '''

  Y = np.full(X.shape[0], fill_value=-1, dtype=np.int32)
  for i, ip in enumerate(ip_addresses):
    Y[i] = shaping_utils.build_one_hot(
        1 if ip in config.malicious_ips else 0,
        [0, 1]
    )
  set_logger.debug("Data labelled!")

  class_counts = analysis_utils.count_classes(Y)
  set_logger.debug("Class distribution is malignant: '" +
                   str(class_counts['1']) +
                   "', benign: '" + str(class_counts['0']) + "'.")
  return X, Y


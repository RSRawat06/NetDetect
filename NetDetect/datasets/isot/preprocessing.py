from ..utils import csv_utils, shaping_utils, analysis_utils
from . import config
import numpy as np
import csv
from .logger import set_logger
from sklearn import preprocessing


def preprocess_file(file_path, n_steps):
  '''
  Return preprocessed dataset from raw data file.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
    - n_steps (int)
  '''

  set_logger.info("Starting preprocessing...")
  return label_data(*segment_histories(*separate_ips(
      *preprocess_features(*load_data(file_path))), n_steps))


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

  flat_X = []
  participating_ips = []

  with open(file_path, 'r') as f:
    set_logger.debug("Opened dataset csv...")
    for i, row in enumerate(csv.reader(f)):
      # We assume the first row is a header.
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        set_logger.debug("Headers key generated: " + str(headers_key))
        continue

      # We collect relevant features and corresponding IPs.
      flat_X.append(csv_utils.featurize_row(
          row, headers_key, config.numerical_fields))
      participating_ips.append(identify_participants(row, headers_key))

  set_logger.debug("Basic data loading complete.")

  return np.array(flat_X, dtype=np.float32), \
      participating_ips


def preprocess_features(X, ips):
  '''
  Scale the feature vectors using scikit preprocessing.
  '''

  assert(len(X.shape) == 2)  # Double check that X is 2d.
  X = preprocessing.maxabs_scale(X, copy=False)
  return X, ips


def separate_ips(flat_X, ips):
  '''
  Segment feature_vectors into their IPs.
  Essentially creates array of feature vectors that
  a given IP was involved in for each IP.
  Args:
    - flat_X (np.array) - 2d feature vecs.
    - ips (list(list(str)))
  Returns:
    - X (list of list of np.array): list of list of 1d feature vecs.
    - new_ips (list of str): list of ips
  '''

  X = []
  new_ips = []

  encountered_features = 0

  # Maps a given IP address to its history's index in X.
  ip_history_map = {}

  set_logger.debug("Mapping history for each IP...")
  for i in range(flat_X.shape[0]):
    for ip in ips[i]:
      if ip not in ip_history_map:
        ip_history_map[ip] = len(X)
        X.append([])
        new_ips.append(ip)
      X[ip_history_map[ip]].append(flat_X[i])
      encountered_features += 1

  set_logger.debug("Separation by IP is complete.")
  set_logger.debug(str(len(X)) + " IP addresses found.")
  set_logger.debug("Average history length for each ip: " +
                   str(encountered_features / len(X)))

  return X, new_ips


def segment_histories(X, ips, n_steps):
  """
  Segment histories into segments of uniform length.
  """

  new_X = []
  new_ips = []

  for i in range(len(X)):
    segments = shaping_utils.segment_vector(np.array(X[i]), n_steps)
    new_X += segments
    new_ips += len(segments) * [ips[i]]

  set_logger.debug("History segmentation complete.")
  set_logger.debug("Average seg count: " +
                   str(len(new_X) / len(X)))

  return np.array(new_X, dtype=np.float32), new_ips


def label_data(X, ips):
  '''
  Label data.
  Args:
    - X (3d np.array).
    - ip_addresses (list of str).
  Returns:
    - X (3d np.array): the original X from args.
    - Y (2d np.array): the new labels for dataset.
  '''

  set_logger.debug("Labelling data!")
  Y = np.full((len(X), 2), fill_value=-1, dtype=np.int32)
  for i, ip in enumerate(ips):
    Y[i] = shaping_utils.build_one_hot(
        1 if ip.lower() in config.malicious_ips else 0,
        [0, 1]
    )
  set_logger.debug("Data labelled!")

  class_counts = analysis_utils.count_classes(Y)
  set_logger.debug("Class distribution is malignant: '" +
                   str(class_counts['1']) +
                   "', benign: '" + str(class_counts['0']) + "'.")
  return np.array(X, dtype=np.float32), np.array(Y, dtype=np.uint8)


def identify_participants(row, headers_key):
  '''
  Return participants for a given row.
  '''

  participants = []
  for i, value in enumerate(row):
    if headers_key[i] in config.participant_fields:
      participants.append(str(value))
  return participants


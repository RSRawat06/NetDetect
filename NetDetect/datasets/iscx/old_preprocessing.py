from ..utils import csv_utils, shaping_utils, analysis_utils
import random
from . import config
import numpy as np
from .utils import identify_participants, parse_score
import csv
from .logger import set_logger
from sklearn import preprocessing


def preprocess_file(file_path):
  '''
  Return preprocessed dataset from raw data file.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
  '''

  set_logger.info("Starting preprocessing...")
  return segment_histories(*separate_ips(
      *preprocess_features(*load_data(file_path))))


def load_data(file_path):
  '''
  Load in a CSV and parse for data.
  Args:
    - file_path (str): path to raw dataset file.
  Returns:
    - X: np.array([[0.3, 0.3...]...], dtype=float32)
    - Y: np.array([0, 1.....], dtype=int16)
    - metadata: [{info:a, label:x}... (n_classes)]
  '''

  flat_X = []
  flat_Y = []
  participating_ips = []

  with open(file_path, 'r') as f:
    set_logger.debug("Opened dataset csv...")
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        set_logger.debug("Headers key generated: " + str(headers_key))
        continue

      flat_X.append(csv_utils.featurize_row(
          row, headers_key, config.numerical_fields))
      flat_Y.append(parse_score(csv_utils.featurize_row_singular(
          row, headers_key, config.score_field)))
      participating_ips.append(identify_participants(row, headers_key))

  set_logger.debug("Basic data loading complete.")

  return (np.array(flat_X, dtype=np.float32),
          np.array(flat_Y, dtype=np.uint16),
          participating_ips)


def preprocess_features(flat_X, flat_Y, participating_ips):
  normalized_X = preprocessing.maxabs_scale(flat_X, copy=False)
  return normalized_X, flat_Y, participating_ips


def separate_ips(flat_X, flat_Y, participating_ips):
  '''
  Segment feature_vectors into their IPs.
  Essentially creates array of feature vectors that
  a given IP was involved in for each IP.
  Args:
    - flat_X (np.array)
    - flat_Y (np.array)
    - participating_ips (list(str))
  Returns:
    - X (np.array): array of feature vectors
    - Y (np.array): array of feature vectors
  '''

  X = []
  Y = []
  encountered_ip_count = 0
  encountered_features = 0

  # Maps a given IP address to its history's index in X.
  ip_history_map = {}

  set_logger.debug("Mapping history for each IP...")
  for i in range(flat_X.shape[0]):
    for ip in participating_ips[i]:
      if ip not in ip_history_map:
        ip_history_map[ip] = encountered_ip_count
        X.append([])
        Y.append([])
        encountered_ip_count += 1
      X[ip_history_map[ip]].append(flat_X[i])
      Y[ip_history_map[ip]].append(flat_Y[i])
      encountered_features += 1

  set_logger.debug("Separation by IP is complete.")
  set_logger.debug(str(encountered_ip_count) + " IP addresses found.")
  set_logger.debug("Average history length for each ip: " +
                   str(encountered_features / encountered_ip_count))

  return np.array(X), np.array(Y)


def segment_histories(X, Y):
  """
  Segment histories into segments of uniform length.
  If segment contains no malign, seuqnece is considered normal.
  Else, malignant.
  """

  new_X = []
  new_Y = []

  total_segment_counts = 0
  for i in range(X.shape[0]):
    segments = shaping_utils.segment_vector(np.array(X[i]), config.SEQ_LEN)
    scores = shaping_utils.segment_vector(np.array(Y[i]), config.SEQ_LEN)

    for i in range(len(segments)):
      malignant = 0
      for score in scores[i]:
        if score == 1:
          malignant += 1
        elif score == 0:
          pass
        else:
          raise ValueError

      # if (malignant == 1):
      #   continue

      if malignant == 0:
        if random.randint(0, 33) > 1:
          # continue
          pass

      new_X.append(segments[i])
      new_Y.append(
          shaping_utils.build_one_hot(
              0 if (malignant == 0) else 1,
              [0, 1]
          )
      )
    total_segment_counts += 1

  set_logger.debug("History segmentation complete.")
  set_logger.debug("Average seg count: " +
                   str(len(new_X) / total_segment_counts))

  del(X)
  del(Y)

  class_counts = analysis_utils.count_classes(new_Y)
  set_logger.debug("Class distribution is malignant: '" +
                   str(class_counts['1']) +
                   "', benign: '" + str(class_counts['0']) + "'.")

  return new_X, new_Y


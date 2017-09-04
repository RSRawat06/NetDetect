'''
Preprocesses datasets:
  - Parsing data points
  - Shuffling data points
'''


from sklearn.utils import shuffle
import numpy as np


def parse_row(row, fields_key, parse_feature):
  '''
  Parse a data point (row) based off of provided
  custom parse_feature function
  '''
  feature_vector = []
  participant_ips = []
  for j, field in enumerate(row):
    value, is_flow_id, is_score, is_participant_ip, is_feature = parse_feature(fields_key[j], field)
    if is_score:
      score = value
    elif is_flow_id:
      flow_id = value
    elif is_participant_ip:
      participant_ips.append(value)
    elif is_feature:
      feature_vector += value
    else:
      raise ValueError
  return feature_vector, score, flow_id, participant_ips


def shuffle_points(X, Y, partition_len):
  '''
  Shuffle two large arrays in sync
  '''
  X_mid = []
  Y_mid = []
  # Break points into segments, and shuffle individual segments
  for i in range(0, len(Y), partition_len):
    sub_X, sub_Y = shuffle(X[i:i + partition_len], Y[i:i + partition_len], random_state=0)
    X_mid.append(sub_X)
    Y_mid.append(sub_Y)
  del(X)
  del(Y)
  # Shuffle the high level segments
  X_mid, Y_mid = shuffle(X_mid, Y_mid, random_state=0)
  X_shuffled = []
  Y_shuffled = []
  # Flatten the segments back into an array
  for i in range(len(Y_mid)):
    X_shuffled += list(X_mid[i])
    Y_shuffled += list(Y_mid[i])
  del(X_mid)
  del(Y_mid)
  return np.array(X_shuffled), np.array(Y_shuffled)


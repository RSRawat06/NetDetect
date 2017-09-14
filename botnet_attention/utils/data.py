'''
Module to handle the import and preprocessing of datasets
'''

import csv
import numpy as np
from sklearn.utils import shuffle
from .preprocess import parse_row, create_parse_feature, create_store_categoricals
from .segmenter import segment_packets, segment_flows


def load(config):
  parse_feature = create_parse_feature(*config.COLUMNS)
  store_categoricals = create_store_categoricals(*config.CATEGORICAL_COLUMNS)

  '''
  Fetch and preprocess dataset
  '''
  X = []
  metadata = []

  fields_key = []
  possible_categoricals = []

  with open(config.DATA_DIR + config.DATA_NAME, 'r') as f:
    # First pass through data
    for i, row in enumerate(csv.reader(f)):
      # If first row, use it to generate header rows dict
      if i == 0:
        for j, field in enumerate(row):
          fields_key[j] = field
        i += 1
        continue

      # Enumerate through possible categoricals
      for j, value in enumerate(row):
        field = fields_key[j]
        possible_categoricals = store_categoricals(field, value, possible_categoricals)

      # If data cap is hit
      if config.N_CAP:
        if (config.N_CAP > i):
          print("Points cap reached. Data loading safely terminated early.")
          break

    # Second pass through data
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        continue

      # If data cap is hit
      if config.N_CAP:
        if (config.N_CAP > i):
          print("Points cap reached. Data loading safely terminated early.")
          break

      # Parse row
      feature_vector, flow_id, participant_ips = parse_row(row, fields_key, parse_feature, possible_categoricals)
      X.append(feature_vector)

      participants = []
      for ip in participant_ips:
        if ip in config.MALICIOUS_IPS:
          participants.append({"score": 1, "ip": ip})
        else:
          participants.append({"score": 0, "ip": ip})
      metadata.append({"participants": participants, "flow_id": flow_id})

  X, flow_metadata = segment_packets(X, metadata, config.N_PACKETS)
  X, Y = segment_flows(X, flow_metadata, config.N_FLOWS)
  X, Y = shuffle_points(X, Y, config.SHUFFLE_PARTITION_LEN)

  # Calculate total true/false
  total_true = 0
  total_false = 0
  for i in Y:
    if i[1] is 1:
      total_true += 1
    else:
      total_false += 1

  print("Total true: ", total_true)
  print("Total false: ", total_false)
  print("Percentage true: ", 100 * total_true / (total_true + total_false))

  # Return end result
  return X, Y


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


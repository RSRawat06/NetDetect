'''
Module to handle the import and preprocessing of datasets
'''


import csv
import numpy as np
from .preprocess import parse_row, shuffle_points
from .segmenter import segment_packets, segment_flows


def load(config, parse_feature):
  '''
  Load dataset fetching functions
  '''
  def fetch(path=config.DATA_DIR + config.DATA_NAME):
    '''
    Fetch and preprocess dataset
    '''
    X = []
    Y = []
    metadata = []

    fields_key = []

    with open(path, 'r') as f:
      for i, row in enumerate(csv.reader(f)):
        # If first row, use it to generate header rows dict
        if i == 0:
          for j, field in enumerate(row):
            fields_key[j] = field
          i += 1
          continue

        # If data cap is hit
        if config.N_CAP:
          if (config.N_CAP > i):
            print("Points cap reached. Data loading safely terminated early.")
            break

        # Parse row
        feature_vector, score, flow_id, participant_ips = parse_row(row, fields_key, parse_feature)
        X.append(feature_vector)
        Y.append(score)
        metadata.append({"participant_ips": participant_ips, "flow_id": flow_id})

    X, Y = segment_packets(X, Y, metadata, config.MAX_FLOW_LENGTH)

    if config.USE_FLOW_SEQUENCES:
      X, Y = segment_flows(X, Y, metadata, config.MAX_FLOW_SEQUENCE_LENGTH)

    # Shuffle data points
    X, Y = shuffle_points(X, Y, config.SHUFFLE_PARTITION_LEN)

    # Calculate total true/false
    total_true = 0
    total_false = 0
    for i in X:
      if i[1]:
        total_true += 1
      else:
        total_false += 1
    print("Total true: ", total_true)
    print("Total false: ", total_false)
    print("Percentage true: ", 100 * total_true / (total_true + total_false))

    # Partition training/test data
    train_X = np.array(X[:-config.N_TEST])
    train_Y = np.array(Y[:-config.N_TEST])
    test_X = np.array(X[-config.N_TEST:])
    test_Y = np.array(Y[-config.N_TEST:])

    # Return end result
    return {"X": train_X, "Y": train_Y}, {"X": test_X, "Y": test_Y}
  return fetch


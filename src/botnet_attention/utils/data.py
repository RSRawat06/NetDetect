'''
Module to handle the import and preprocessing of datasets
'''


import csv
import numpy as np
from .preprocess import parse_row, shuffle_points
from .segmenter import segment_packets, segment_flows


def load(config, parse_feature, store_categoricals):
  '''
  Load dataset fetching functions
  '''
  def fetch(path=config.DATA_DIR + config.DATA_NAME):
    '''
    Fetch and preprocess dataset
    '''
    X = []
    metadata = []

    fields_key = []
    possible_categoricals = []

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

        for j, value in enumerate(row):
          field = fields_key[j]
          possible_categoricals = store_categoricals(field, value, possible_categoricals)

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

    X, flow_metadata = segment_packets(X, metadata, config.MAX_FLOW_LENGTH)
    X, Y = segment_flows(X, flow_metadata, config.MAX_FLOW_SEQUENCE_LENGTH)
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

    # Partition training/test data
    train_X = np.array(X[:-config.N_TEST])
    train_Y = np.array(Y[:-config.N_TEST])
    test_X = np.array(X[-config.N_TEST:])
    test_Y = np.array(Y[-config.N_TEST:])

    # Return end result
    return {"X": train_X, "Y": train_Y}, {"X": test_X, "Y": test_Y}
  return fetch


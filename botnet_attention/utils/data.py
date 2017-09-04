import csv
import numpy as np
from preprocess import parse_feature, shuffle_points
from segmenter import segment_packets, segment_flows


def load(config):
  def fetch(path=config.DATA_DIR + config.DATA_NAME):
    X = []
    Y = []
    flow_ids = []
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

        # Generate feature vectors
        feature_vector = []
        participant_ips = []
        for j, field in enumerate(row):
          if fields_key[j] == "Score":
            score = parse_feature(fields_key[j], field, config)
          elif fields_key[j] == "FlowNo.":
            flow_id = field
          elif fields_key[j] == "Source":
            participant_ips.append(field)
          elif fields_key[j] == "Destination":
            participant_ips.append(field)
          else:
            try:
              feature_vector += parse_feature(fields_key[j], field, config)
            except ValueError:
              continue

        # Add new point to the record
        X.append(feature_vector)
        Y.append(score)
        flow_ids.append(flow_id)
        metadata.append(participant_ips)

    print("All rows processed")
    X, Y, metadata = segment_packets(X, Y, flow_ids, metadata, config)

    if config.FLOW_SEQUENCES:
      X, Y = segment_flows(X, Y, metadata, config)

    # Shuffle data points
    X, Y = shuffle_points(X, Y, config)

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
    seq_train = X[:-config.N_TEST]
    seq_train_targets = Y[:-config.N_TEST]
    seq_test = X[-config.N_TEST:]
    seq_test_targets = Y[-config.N_TEST:]
    # seq_validation = X[-config.N_TEST:]
    # seq_validation_targets = Y[-config.N_TEST:]
    print("Data loading complete")

    # Return end result
    len_training = len(Y) - config.N_TEST
    len_testing = config.N_TEST
    return {"x": np.array(seq_train), "y": np.array(seq_train), "targets": np.array(seq_train_targets)}, {"x": np.array(seq_test), "y": np.array(seq_test), "targets": np.array(seq_test_targets)}, len_training, len_testing



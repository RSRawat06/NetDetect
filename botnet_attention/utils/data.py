'''
Module to handle the import and preprocessing of datasets
'''

import csv
from .preprocess import parse_row, create_parse_feature, create_store_categoricals
from .segmenter import segment_packets, segment_flows


def load(data_path, config):
  '''
  Fetch and preprocess dataset
  '''

  parse_feature = create_parse_feature(*config.COLUMNS)
  store_categoricals = create_store_categoricals(*config.CATEGORICAL_COLUMNS)

  X = []
  fields_key = []
  metadata = []
  possible_categoricals = []

  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      # For first row, build up headers
      if i == 0:
        for j, field in enumerate(row):
          fields_key[j] = field
        i += 1
        continue

      # Enumerate through possible categoricals
      for j, value in enumerate(row):
        field = fields_key[j]
        possible_categoricals = store_categoricals(field, value, possible_categoricals)

    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        continue

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

  return X, Y


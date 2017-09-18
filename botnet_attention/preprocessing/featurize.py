from . import utils
import csv


def featurize_csv(data_path, numerical_fields, protocol_fields, categorical_fields, port_fields):
  '''
  '''
  all_records = {'protocol': {}, 'categorical': {}, 'port': {}}
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      all_records = store_categoricals(row, headers_key, all_records, protocol_fields, categorical_fields, port_fields)

  X = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      X.append(featurize_row(row, headers_key, all_records, numerical_fields, protocol_fields, categorical_fields, port_fields))

  return X


def featurize_row(row, headers_key, all_records, numerical_fields, protocol_fields, categorical_fields, port_fields):
  '''
  Parse a data point (row) based off of provided
  custom parse_feature function
  '''
  def __create_one_hot(record, values):
    onehot = [0] * len(record)
    for value in values:
      ind = record.index(value)
      assert(ind >= 0)
      onehot[ind] = 1
    return onehot

  feature_vector = []
  for i, value in enumerate(row):
    field = headers_key[i]
    if field in numerical_fields:
      feature_vector.append(float(value))
    elif field in protocol_fields:
      feature_vector += __create_one_hot(all_records['protocol'][field], str(value).split(":"))
    elif field in categorical_fields:
      feature_vector += __create_one_hot(all_records['categorical'][field], value)
    elif field in port_fields:
      if value > 2000:
        value = 2000
      feature_vector += __create_one_hot(all_records['port'][field], value)

  return feature_vector


def store_categoricals(row, headers_key, all_records, protocol_fields, categorical_fields, port_fields, threshold=30):
  '''
  '''
  def __append_to_records(records, field, value):
    if field not in records:
      records[field] = []
    if value not in records[field]:
      records[field].append(value)
    if len(records[field]) > threshold:
      raise ValueError
    return records

  for i, value in enumerate(row):
    field = headers_key[i]
    if field in protocol_fields:
      for protocol in str(value).split(":"):
        __append_to_records(all_records['protocol'], field, protocol)

    elif headers_key[i] in categorical_fields:
      __append_to_records(all_records['categorical'], field, value)

    elif headers_key[i] in port_fields:
      if int(value) > 2000:
        value = 2000
      __append_to_records(all_records['port'], field, int(value))

  return all_records


def metadatize_csv(data_path, malicious_ips, flow_field, participant_fields):
  '''
  '''
  metadata = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      metadata.append(metadatize_row(row, headers_key, malicious_ips, flow_field, participant_fields))
  return metadata


def metadatize_row(row, headers_key, malicious_ips, flow_field, participant_fields):
  '''
  Parse a data point (row) based off of provided
  custom parse_feature function
  '''
  participants = []
  flow_id = None
  for i, value in enumerate(row):
    if headers_key[i] == flow_field:
      flow_id = int(value)
    elif headers_key[i] in participant_fields:
      score = 1 if int(value) in malicious_ips else 0
      participants.append({"score": score, "ip": int(value)})
  return flow_id, participants


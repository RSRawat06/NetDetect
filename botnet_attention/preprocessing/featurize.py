from . import utils
# Importing build_headers, 


def featurize_csv(csv, parse_feature, protocol_fields, categorical_fields, port_fields):
  '''
  '''
  all_records = {'protocol': {}, 'categorical': {}, 'port': {}}
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      all_records = store_categoricals(row, all_records, protocol_fields, categorical_fields, port_fields)
 
  X = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      X.append(featurize_row(row, headers_key, parse_feature, all_records))
  return metadata


def featurize_row(row, headers_key, parse_feature, all_records):
  '''
  Parse a data point (row) based off of provided
  custom parse_feature function
  '''
  feature_vector = []
  for i, value in enumerate(row):
    try:
      value = parse_feature(headers_key[i], value, all_records)
      feature_vector += value
    except ValueError:
      continue
  return feature_vector


def store_categoricals(row, headers_key, categorical_records, protocol_fields, categorical_fields, port_fields, threshold = 30):
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


def metadatize_csv(data_path, is_flow_id, is_participant_ip):
  '''
  '''
  metadata = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      metadata.append(metadatize_row(row, headers_key, malicious_ips, is_flow_id, is_participant_ip))
  return metadata


def metadatize_row(row, headers_key, malicious_ips, is_flow_id, is_participant_ip):
  '''
  Parse a data point (row) based off of provided
  custom parse_feature function
  '''
  participant = []
  flow_id = None
  for i, value in enumerate(row):
    if is_flow_id(headers_key[i]):
      flow_id = int(value)
    elif is_participant_ip(headers_key[i]):
      score = 1 if int(value) in malicious_ips else 0
      participants.append({"score": score, "ip": int(value)})
  return flow_id, participant


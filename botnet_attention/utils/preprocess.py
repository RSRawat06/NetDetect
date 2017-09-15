'''
Preprocesses datasets:
  - Parsing data points
  - Shuffling data points
'''


def parse_row(row, fields_key, parse_feature, records):
  '''
  Parse a data point (row) based off of provided
  custom parse_feature function
  '''
  feature_vector = []
  participant_ips = []
  for j, field in enumerate(row):
    value, is_flow_id, is_participant_ip, is_feature = parse_feature(fields_key[j], field, records)
    if is_flow_id:
      flow_id = value
    elif is_participant_ip:
      participant_ips.append(value)
    elif is_feature:
      feature_vector += value
    else:
      raise ValueError
  return feature_vector, flow_id, participant_ips


def create_store_categoricals(protocol_fields, categorical_fields, threshold=10):
  def store_categoricals(field, value, records):
    if field in protocol_fields:
      if field not in records:
        records['protocol'][field] = []
      for cat in value.split(":"):
        if cat not in records['protocol'][field]:
          records['protocol'][field].append(cat)
    elif field in categorical_fields:
      if field not in records:
        records['categorical'][field] = []
      if value not in records[field]:
        records['categorical'][field].append(value)
      if len(records['categorical'][field]) > threshold:
        print(records['categorical'][field])
        raise ValueError
    return records
  return store_categoricals


def create_parse_feature(numerical, categorical, protocol, participant, flow_field):
  def parse_feature(field, raw_value, records):
    '''
    Parse feature based on field type.
    Parsing mechanisms unique to ISCX.
    '''
    is_flow_id, is_participant_ip, is_feature = False, False, False

    if field in numerical:
      if raw_value == "":
        value = [0]
      else:
        value = [float(raw_value)]
      is_feature = True
    elif field in categorical:
      is_feature = True
      ind = records['categorical'][field].index(raw_value)
      assert(ind >= 0)
      value = [0] * len(records['categorical'][field])
      value[ind] = 1
    elif field in protocol:
      is_feature = True
      value = [0] * len(records['categorical'][field])
      for i, cat in enumerate(raw_value.split(":")):
        value[i] = 1
    elif field in participant:
      is_participant_ip = True
      value = str(raw_value)
    elif field == flow_field:
      is_flow_id = True
      value = int(raw_value)
    else:
      raise ValueError
    return value, is_flow_id, is_participant_ip, is_feature
  return parse_feature


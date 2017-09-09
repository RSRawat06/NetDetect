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


def create_store_categoricals(categorical_fields, threshold=10):
  def store_categoricals(field, value, records):
    if field in categorical_fields:
      if field not in records:
        records[field] = []
      if value not in records[field]:
        records[field].append(value)
      if len(records[field]) > threshold:
        raise ValueError
    return records
  return store_categoricals


def create_parse_feature(numerical_fields, categorical_fields, participant_fields, flow_field):
  def parse_feature(field, raw_value, records):
    '''
    Parse feature based on field type.
    Parsing mechanisms unique to ISCX.
    '''
    is_flow_id, is_participant_ip, is_feature = False, False, False
    if field in numerical_fields:
      if raw_value == "":
        value = [0]
      else:
        value = [float(raw_value)]
      is_feature = True
    # Categorical data
    elif field in categorical_fields:
      is_feature = True
      ind = records[field].index(raw_value)
      assert(ind >= 0)
      value = [0] * len(records[field])
      value[ind] = 1
    # Add participants
    elif field in participant_fields:
      is_participant_ip = True
      value = str(raw_value)
    # Flow Numbers
    elif field == flow_field:
      is_flow_id = True
      value = int(raw_value)
    else:
      raise ValueError
    return value, is_flow_id, is_participant_ip, is_feature
  return parse_feature


def test_load():
  def store_categoricals(field, value, records):
    if field in ["type"]:
      if field not in records:
        records[field] = []
      if value not in records[field]:
        records[field].append(value)
      if len(records[field]) > 10:
        raise ValueError
    return value


  def parse_feature(field, raw_value):
    is_flow_id, is_score, is_participant_ip, is_feature = False, False, False, False
    if field in ["size", "width"]:
      value = [int(raw_value)]
      is_feature = True
    elif field == "type":
      is_feature = True
      if raw_value.lower().strip() == "apple":
        value = [0, 1]
      elif raw_value.lower().strip() == "pear":
        value = [1, 0]
    elif field == "score":
      is_score = True
      value = int(raw_value / 3)
    elif field == "ip":
      is_participant_ip = True
      value = int(raw_value)
    elif field == "ip2":
      is_participant_ip = True
      value = int(raw_value)
    elif field == "flow_id":
      is_flow_id = True
      value = int(raw_value)
    else:
      raise ValueError

  fetch = load(config, parse_feature, store_categoricals)


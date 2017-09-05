from ...utils import preprocess


def test_parse_row():
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

    return value, is_flow_id, is_score, is_participant_ip, is_feature

  assert(([323, 23, 0, 1], 1, 5, [4]) == preprocess.parse_row([323, '23', 'apple', 5.0, 3.0, 4], {"size": 0, "width": 1, "type": 2, "flow_id": 3, "score": 4, "ip": 5}, parse_feature))
  assert(([22, 14, 1, 0], 1, 9, [4, 8]) == preprocess.parse_row([22.5, '14', 'pear', 9.0, 0, 4, 8], {"size": 0, "width": 1, "type": 2, "flow_id": 3, "score": 4, "ip": 5, "ip2": 6}, parse_feature))


def test_shuffle_points():
  x, y = preprocess.shuffle_points(list(range(100)), list(range(100)), 10)
  assert(type(x[0]) == 1)
  assert(len(x) == 100)
  assert(len(range(100)) == x.sort())
  assert(type(y[0]) == 1)
  assert(len(y) == 100)
  assert(len(range(100)) == y.sort())
  x, y = preprocess.shuffle_points(list(range(100)), list(range(100)), 3)
  assert(type(x[0]) == 1)
  assert(len(x) == 100)
  assert(len(range(100)) == x.sort())
  assert(type(y[0]) == 1)
  assert(len(y) == 100)
  assert(len(range(100)) == y.sort())
  x, y = preprocess.shuffle_points(list(range(100)), list(range(100)), 120)
  assert(type(x[0]) == 1)
  assert(len(x) == 100)
  assert(len(range(100)) == x.sort())
  assert(type(y[0]) == 1)
  assert(len(y) == 100)
  assert(len(range(100)) == y.sort())


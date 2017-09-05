from ...utils import preprocess
import numpy as np


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

  assert(([323, 23, 0, 1], 1, 5, [4]) == preprocess.parse_row([323, '23', 'apple', 5.0, 3.0, 4], {0: "size", 1: "width", 2: "type", 3: "flow_id", 4: "score", 5: "ip"}, parse_feature))
  assert(([22, 14, 1, 0], 0, 9, [4, 8]) == preprocess.parse_row([22.5, '14', 'pear', 9.0, 0, 4, 8], {0: "size", 1: "width", 2: "type", 3: "flow_id", 4: "score", 5: "ip", 6: "ip2"}, parse_feature))


def test_shuffle_points():
  x, y = preprocess.shuffle_points(list(range(100)), list(range(100)), 10)
  assert(len(x) == 100)
  assert(np.all(list(range(100)) == np.sort(x)))
  assert(len(y) == 100)
  assert(np.all(list(range(100)) == np.sort(y)))
  x, y = preprocess.shuffle_points(list(range(100)), list(range(100)), 3)
  assert(len(x) == 100)
  assert(np.all(list(range(100)) == np.sort(x)))
  assert(len(y) == 100)
  assert(np.all(list(range(100)) == np.sort(y)))
  x, y = preprocess.shuffle_points(list(range(100)), list(range(100)), 120)
  assert(len(x) == 100)
  assert(np.all(list(range(100)) == np.sort(x)))
  assert(len(y) == 100)
  assert(np.all(list(range(100)) == np.sort(y)))


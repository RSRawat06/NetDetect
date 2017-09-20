import numpy as np


def segment_by_ips(old_X, metadata, sequence_length):
  '''
  Segments flows into time-ordered sequences of flows.
  Flows are also scored.
  '''

  # Segment flows into fixed-vector sequences
  raw_X = []
  raw_Y = []
  member_ind_map = {}

  for i in range(len(old_X)):
    source = metadata[i]['participants'][0]
    if source['ip'] not in member_ind_map:
      member_ind_map[source['ip']] = len(raw_X)
      raw_X.append([])
      raw_Y.append(source['score'])
    raw_X[member_ind_map[source['ip']]].append(old_X[i])

    destination = metadata[i]['participants'][1]
    if destination['ip'] not in member_ind_map:
      member_ind_map[destination['ip']] = len(raw_X)
      raw_X.append([])
      raw_Y.append(destination['score'])
    raw_X[member_ind_map[destination['ip']]].append(old_X[i])

  X = []
  Y = []
  average_flow_sum = 0
  for i in range(len(raw_X)):
    average_flow_sum += len(raw_X[i])
    cutoff_x = raw_X[i][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    point = np.pad(cutoff_x, ((0, right_padding_width), (0, 0), (0, 0)), 'constant', constant_values=0)
    assert(point.shape[0] == sequence_length)
    X.append(point)

    score = np.zeros(2, dtype=np.int8)
    score[int(raw_Y[i])] = 1
    Y.append(score)

  print("Average number of flows in seq:", average_flow_sum / len(raw_X))
  return np.array(X), np.array(Y)


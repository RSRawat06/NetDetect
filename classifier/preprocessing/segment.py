import numpy as np


def segment_by_ips(flows, metadata, sequence_length):
  '''
  Segments flows into time-ordered sequences of flows.
  Flows are also scored.
  '''

  # Segment flows into fixed-vector sequences
  raw_X = []
  raw_Y = []
  member_ind_map = {}  # ip: index of raw_X entry

  for i in range(len(flows)):
    source = metadata[i]['participants'][0]
    if source['ip'] not in member_ind_map:
      member_ind_map[source['ip']] = len(raw_X)
      raw_X.append([])
      raw_Y.append(source['score'])
    raw_X[member_ind_map[source['ip']]].append(flows[i])

    destination = metadata[i]['participants'][1]
    if destination['ip'] not in member_ind_map:
      member_ind_map[destination['ip']] = len(raw_X)
      raw_X.append([])
      raw_Y.append(destination['score'])
    raw_X[member_ind_map[destination['ip']]].append(flows[i])

  print("Average number of flows in seq:", len(flows) / len(raw_X))
  del(member_ind_map)
  del(flows)
  del(metadata)


  X = []
  Y = []
  for i in range(len(raw_X)):
    cutoff_x = raw_X[i][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    point = np.pad(cutoff_x, ((0, right_padding_width), (0, 0)), 'constant', constant_values=0)
    assert(point.shape[0] == sequence_length)
    X.append(point)

    score = np.zeros(2, dtype=np.int8)
    score[int(raw_Y[i])] = 1
    Y.append(score)

  return np.array(X), np.array(Y)


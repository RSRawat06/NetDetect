'''
This module handles segmenting packets to flows and
segmenting flows into segmented sequences of flows
'''


import numpy as np


def segment_packets(X, Y, metadata, sequence_length):
  '''
  Input is packets, which are segmented into flows
  '''

  # Segment packets into fixed-vector sequences composing a flow
  # On the side, build up a {flow_id: metadata} dict
  flow_ids = [x['flow_id'] for x in metadata]

  features_size = len(X[0])
  # Assuming flow_ids start from 0 and neatly increment to x
  number_of_flows = max(flow_ids) + 1

  flow_X = [[] for _ in range(number_of_flows)]
  flow_Y = [[] for _ in range(number_of_flows)]

  # Append points in X and Y to write flow array
  for i in range(len(X)):
    flow_id = flow_ids[i]
    flow_X[flow_id].append(X[i])
    flow_Y[flow_id].append(Y[i])

  # Construct fixed vectors
  X = np.zeros((number_of_flows, sequence_length, features_size), dtype=np.int32)
  Y = np.zeros((number_of_flows, 2), dtype=np.int8)

  # Iterate through each flow
  for i in range(number_of_flows):
    # Verify each flow has the same score
    scores = []
    for score in flow_Y[i]:
      scores.append(score[1])
    Y[i] = flow_Y[i][0]
    assert(len(np.unique(scores)) == 1)

    # Pad + cut off X feature vector
    cutoff_x = flow_X[i][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    X[i] = np.pad(cutoff_x, ((0, right_padding_width), (0, 0)), 'constant', constant_values=0)
    assert(X[i].shape[0] == sequence_length)

  return X, Y


def segment_flows(flow_X, flow_Y, metadata, sequence_length):
  '''
  Segments flows into time-ordered sequences of flows.
  '''

  # Segment flows into fixed-vector sequences
  member_map = {}
  for meta_datum in metadata:
    source = meta_datum['participant_ips'][0]
    dest = meta_datum['participant_ips'][1]
    flow_id = meta_datum['flow_id']
    if source not in member_map:
      member_map[source] = []
    if dest not in member_map:
      member_map[dest] = []
    member_map[source].append(flow_id)
    member_map[dest].append(flow_id)

  X = []
  Y = []
  for user, flow_ids in member_map.items():
    # Ensure continuity of scores
    scores = []
    flow_ids = np.unique(flow_ids)
    for flow_id in flow_ids:
      scores.append(flow_Y[flow_id][1])
    if (len(np.unique(scores)) != 1):
      continue
    for i in range(0, len(flow_ids) - sequence_length):
      X.append([flow_X[flow_id] for flow_id in flow_ids[i:i + sequence_length]])
      Y.append(flow_Y[flow_ids[0]])

  return np.array(X), np.array(Y)


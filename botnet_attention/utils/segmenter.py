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

  flow_X = [[]] * (max(flow_ids) + 1)
  flow_Y = [[]] * (max(flow_ids) + 1)
  # Append points in X and Y to write flow array
  for i in range(len(X)):
    flow_id = flow_ids[i]
    flow_X[flow_id].append(X[i])
    flow_Y[flow_id].append(Y[i])

  # Construct fixed vectors
  X = np.zeros((max(flow_ids), sequence_length), dtype=np.dtype32)
  Y = np.zeros((max(flow_ids), 2), dtype=np.dtype32)

  # Iterate through each flow
  for i in range(max(flow_ids)):
    # Verify each flow has the same score
    scores = []
    for score in flow_Y[i]:
      scores.append(score[1])
    Y[i] = flow_Y[i][0]
    assert(len(np.unique(scores)) == 1)

    # Pad + cut off X feature vector
    cutoff_x = flow_X[i][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    X[i] = np.pad(cutoff_x, (0, right_padding_width), 'constant', constant_values=0)
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

    if flow_id not in member_map[source]:
      member_map[source].append(flow_id)
    if flow_id not in member_map[dest]:
      member_map[dest].append(flow_id)

  X = []
  Y = []
  for user, flow_ids in member_map.items():
    # Ensure continuity of scores
    scores = []
    for flow_id in flow_ids:
      scores.append(flow_Y[flow_id][1])
    if (len(np.unique(scores)) != 1):
      continue

    for i in range(0, len(flow_Y) - sequence_length):
      X.append([flow_X[flow_id] for flow_id in flow_ids[i:i + sequence_length]])
      Y.append(flow_Y[flow_ids[0]][0])

  return X, Y

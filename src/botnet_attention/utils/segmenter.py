'''
This module handles segmenting packets to flows and
segmenting flows into segmented sequences of flows
'''


import numpy as np


def segment_packets(X, metadata, sequence_length):
  '''
  Takes in packet features, packet metadata, and sequence
  length.
  Returns features organized into flow sequences and
  flow metadata that is the same as the packet metadata for
  that flow.
  '''

  number_of_flows = max([datum['flow_id'] for datum in metadata]) + 1

  flow_X = [[[], None] for _ in range(number_of_flows)]
  # flow_X = [[[packet_vector, ...], packet_metadata]...]

  # Append points in X and Y to write flow array
  for i in range(len(X)):
    flow_id = metadata[i]['flow_id']
    flow_X[flow_id][0].append(X[i])
    flow_X[flow_id][1] = metadata[i]

  # Construct fixed vectors
  X = np.zeros((number_of_flows, sequence_length, len(X[0])), dtype=np.int32)
  flow_metadata = []

  # Iterate through each flow
  for i in range(number_of_flows):
    # Pad + cut off X feature vector
    cutoff_x = flow_X[i][0][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    X[i] = np.pad(cutoff_x, ((0, right_padding_width), (0, 0)), 'constant', constant_values=0)
    assert(X[i].shape[0] == sequence_length)
    flow_metadata.append(X[i][1])

  return X, flow_metadata


def segment_flows(flow_X, flow_metadata, sequence_length):
  '''
  Segments flows into time-ordered sequences of flows.
  Flows are also scored.
  '''

  # Segment flows into fixed-vector sequences
  raw_X = []
  raw_Y = []
  member_ind_map = {}

  for i in range(len(flow_X)):
    source = flow_metadata[i]['participants'][0]
    if source['ip'] not in member_ind_map:
      member_ind_map[source['ip']] = len(raw_X)
      raw_X.append([])
      raw_Y.append(source['score'])
    raw_X[member_ind_map[source['ip']]].append(flow_X[i])

    destination = flow_metadata[i]['participants'][1]
    if destination['ip'] not in member_ind_map:
      member_ind_map[destination['ip']] = len(raw_X)
      raw_X.append([])
      raw_Y.append(destination['score'])
    raw_X[member_ind_map[destination['ip']]].append(flow_X[i])

  X = []
  Y = []
  for i in range(len(raw_X)):
    cutoff_x = raw_X[i][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    point = np.pad(cutoff_x, ((0, right_padding_width), (0, 0), (0, 0)), 'constant', constant_values=0)
    assert(point.shape[0] == sequence_length)
    X.append(point)

    score = np.zeros(2, dtype=np.int8)
    score[int(raw_Y[i])] = 1
    Y.append(score)

  return np.array(X), np.array(Y)


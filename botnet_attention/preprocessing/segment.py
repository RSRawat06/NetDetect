import numpy as np


def segment_by_flows(X, metadata, sequence_length):
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

  n_packets_sum = 0

  # Iterate through each flow
  for i in range(number_of_flows):
    # Pad + cut off X feature vector
    n_packets_sum += len(flow_X[i][0])
    cutoff_x = flow_X[i][0][:sequence_length]
    right_padding_width = sequence_length - len(cutoff_x)
    X[i] = np.pad(cutoff_x, ((0, right_padding_width), (0, 0)), 'constant', constant_values=0)
    assert(X[i].shape[0] == sequence_length)
    flow_metadata.append(flow_X[i][1])

  print("Average number of packets per flow:", n_packets_sum / number_of_flows)

  return X, flow_metadata


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


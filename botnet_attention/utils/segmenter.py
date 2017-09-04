import numpy as np


def segment_packets(X, Y, flow_ids, flow_members, config):
  # Segment packets into fixed-vector sequences composing a flow
  # On the side, build up a {flow_id: metadata} dict
  flow_X = [[]] * (max(flow_ids) + 1)
  flow_Y = [[]] * (max(flow_ids) + 1)
  metadata = {}
  for i in range(len(X)):
    flow_id = flow_ids[i]
    flow_X[flow_id].append(X[i])
    flow_Y[flow_id].append(Y[i])
    metadata[flow_id] = flow_members[i]

  # Construct fixed vectors
  X = np.zeros((max(flow_ids), config.MAX_PACKET_SEQUENCE_LENGTH), dtype=np.dtype32)
  Y = np.zeros((max(flow_ids), 2), dtype=np.dtype32)

  # Iterate through each flow
  for i in range(len(flow_ids)):
    # Verify each flow has the same score
    scores = []
    for score in flow_Y[i]:
      scores.append(score[1])
    Y[i] = flow_Y[i][0]
    assert(len(np.unique(scores)) == 1)

    # Pad + cut off X feature vector
    cutoff_x = flow_X[i][:config.MAX_PACKET_SEQUENCE_LENGTH]
    right_padding_width = config.MAX_PACKET_SEQUENCE_LENGTH - len(cutoff_x)
    X[i] = np.pad(cutoff_x, (0, right_padding_width), 'constant', constant_values=0)
    assert(X[i].shape[0] == config.MAX_PACKET_SEQUENCE_LENGTH)

  return flow_X, flow_Y, metadata


def segment_flows(flow_X, flow_Y, metadata, config):
  # Segment flows into fixed-vector sequences
  member_map = {}
  for flow_id, datum in metadata.items():
    source = datum[0]
    dest = datum[1]
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
    scores = []
    for flow_id in flow_ids:
      scores.append(flow_Y[flow_id][1])
    assert(len(np.unique(scores)) == 1)
    for i in range(0, len(flow_Y) - config.MAX_SEQUENCE_LENGTH):
      X.append([flow_X[flow_id] for flow_id in flow_ids[i:i + config.MAX_SEQUENCE_LENGTH]])
      Y.append(flow_Y[flow_ids[0]][0])

  return X, Y

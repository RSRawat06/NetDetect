from ...botnet_attention.utils import segmenter
import numpy as np


def test_segment_packets():
  X = [list(range(x, x + 10)) for x in range(100)]
  # X: [0, 1... 10], [1, 2... 11]... [99, 100... 109]
  # X.shape: (100, 10)
  Y = [[int(float(x) / 2), int(float(x) / 2) + 1] for x in range(100, 200)]
  # Y: [50, 51], [50, 51], [51, 52], [51, 52]... [99, 100], [99, 100]
  # Y.shape: (100, 2)

  metadata = [{"flow_id": int(float(x) / 2), "participant_ips": [int(float(x) / 4), int(float(x) / 4) + 1]} for x in range(100)]
  # flow_id: 0, 0, 1, 1, ... 99, 99
  # flow_id.shape: 100
  # participant_ids: [0, 1], [0, 1], [0, 1], [0, 1], [1, 2], [1, 2]... [24, 25]
  # participant_ids.shape: (100, 2)

  new_X, new_Y = segmenter.segment_packets(X, Y, metadata, 3)
  # new_X: [[0, 1... 10], [1, 2... 11], [0, 0... 0]], [[2, 3... 12], [3, 4... 13], [0, 0... 0]]...
  # new_X.shape: (50, 3, 10)
  # new_Y: [50, 51], [51, 52]...
  # new_Y.shape: (50, 2)
  assert(new_X.shape == (50, 3, 10))
  assert(np.all(sequence == [list(range((2 * x), (2 * x) + 10)), list(range((2 * x) + 1, (2 * x) + 11)), np.zeros(10)] for x, sequence in enumerate(new_X)))
  assert(new_Y.shape == (50, 2))
  assert(np.all(new_Y == [[x, x + 1] for x in range(50, 100)]))

  new_X, new_Y = segmenter.segment_packets(X, Y, metadata, 2)
  # new_X: [[0, 1... 10], [1, 2... 11]], [[2, 3... 12], [3, 4... 13]]...
  # new_X.shape: (50, 2, 10)
  # new_Y: [50, 51], [51, 52]...
  # new_Y.shape: (50, 2)
  assert(new_X.shape == (50, 2, 10))
  assert(np.all(sequence == [list(range((2 * x), (2 * x) + 10)), list(range((2 * x) + 1, (2 * x) + 11))] for x, sequence in enumerate(new_X)))
  assert(new_Y.shape == (50, 2))
  assert(np.all(new_Y == [[x, x + 1] for x in range(50, 100)]))

  new_X, new_Y = segmenter.segment_packets(X, Y, metadata, 1)
  # new_X: [[0, 1... 10]], [[2, 3... 12]]...
  # new_X.shape: (50, 1, 10)
  # new_Y: [50, 51], [51, 52]...
  # new_Y.shape: (50, 2)
  assert(new_X.shape == (50, 1, 10))
  assert(np.all(sequence == [list(range((2 * x), (2 * x) + 10))] for x, sequence in enumerate(new_X)))
  assert(new_Y.shape == (50, 2))
  assert(np.all(new_Y == [[x, x + 1] for x in range(50, 100)]))


def test_segment_flows():
  X = [list(range(x, x + 10)) for x in range(100)]
  # X: [0, 1... 10], [1, 2... 11]... [99, 100... 109]
  # X.shape: (100, 10)
  Y = [[int(float(x) / 2), int(float(x) / 2) + 1] for x in range(100, 200)]
  # Y: [50, 51], [50, 51], [51, 52], [51, 52]... [99, 100], [99, 100]
  # Y.shape: (100, 2)

  metadata = [{"flow_id": int(float(x) / 2), "participant_ips": [int(float(x) / 4), int(float(x) / 4) + 1]} for x in range(100)]
  # flow_id: 0, 0, 1, 1, ... 99, 99
  # flow_id.shape: 100
  # participant_ids: [0, 1], [0, 1], [0, 1], [0, 1], [1, 2], [1, 2]... [24, 25]
  # participant_ids.shape: (100, 2)

  new_X, new_Y = segmenter.segment_packets(X, Y, metadata, 3)
  # new_X: [[0, 1... 10], [1, 2... 11], [0, 0... 0]], [[2, 3... 12], [3, 4... 13], [0, 0... 0]]...
  # new_X.shape: (50, 3, 10)
  # new_Y: [50, 51], [51, 52]...
  # new_Y.shape: (50, 2)

  # first participant, last particiapnt = 2 flows, 4 packets
  # others particiapnts = 4 flows, 8 packets
  # users.shape = 25
  # member_map.shape = (25, 2, 3, 10) or (25, 4, 3, 10)

  X, Y = segmenter.segment_flows(new_X, new_Y, metadata, 2)
  assert(X.shape == (48, 3, 10))
  assert(Y.shape == (48, 2))


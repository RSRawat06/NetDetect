from ...utils import segmenter
import numpy as np

# 100 IPs
# 25 malignant
# 396 flows
# 396 * 3 packets
# 4 features/packet
all_ips = list(range(100))
all_scores = [(x % 4 == 0) for x in range(100)]

all_flow_participant_inds = [[{"ip": int(x / 4), "score": all_scores[int(x / 4)]}, {"ip": int(x / 4) + 1, "score": all_scores[int(x / 4) + 1]}] for x in range(99 * 4)]

X_struct = [[[3 * x] * 4, [(3 * x) + 1] * 4, [(3 * x) + 2] * 4] for x in range(396)]

X_flat = [[x] * 4 for x in range(0, 396 * 3)]

assert(np.all(np.reshape(X_struct, (396 * 3, 4)) == X_flat))

metadata = []
for i in range(396 * 3):
  metadata.append({"flow_id": int(i / 3), "participants": all_flow_participant_inds[int(i / 3)]})


def test_segment_packets():
  new_X, flow_metadata = segmenter.segment_packets(X_flat, metadata, 4)
  assert(new_X.shape == (396, 4, 4))
  assert(np.all(new_X == [x + [[0] * 4] for x in X_struct]))

  new_X, flow_metadata = segmenter.segment_packets(X_flat, metadata, 3)
  assert(new_X.shape == (396, 3, 4))
  assert(np.all(new_X == X_struct))

  new_X, flow_metadata = segmenter.segment_packets(X_flat, metadata, 2)
  assert(new_X.shape == (396, 2, 4))
  assert(np.all(new_X == [x[:2] for x in X_struct]))


def test_segment_flows():
  new_X, flow_metadata = segmenter.segment_packets(X_flat, metadata, 4)

  X, Y = segmenter.segment_flows(new_X, flow_metadata, 5)
  assert(X.shape == (100, 5, 4, 4))
  assert(Y.shape == (100, 2))
  assert(np.all(np.argmax(Y, axis=1) == [(x % 4 == 0) for x in range(100)]))


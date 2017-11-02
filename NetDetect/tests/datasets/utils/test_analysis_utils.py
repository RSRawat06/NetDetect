from ....datasets.utils import analysis_utils
import numpy as np


def test_count_classes():
  raw_Y = [[0, 1] * 3 + [1, 0] * 2 + [0, 1] * 4]
  Y = np.array(raw_Y)
  counts = analysis_utils.count_classes(Y)
  assert(counts == {"0": 7, "1": 2})


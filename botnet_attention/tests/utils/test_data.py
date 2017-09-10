from ...utils import data
import numpy as np


def test_shuffle_points():
  x, y = data.shuffle_points(list(range(100)), list(range(100)), 10)
  assert(len(x) == 100)
  assert(np.all(list(range(100)) == np.sort(x)))
  assert(len(y) == 100)
  assert(np.all(list(range(100)) == np.sort(y)))
  x, y = data.shuffle_points(list(range(100)), list(range(100)), 3)
  assert(len(x) == 100)
  assert(np.all(list(range(100)) == np.sort(x)))
  assert(len(y) == 100)
  assert(np.all(list(range(100)) == np.sort(y)))
  x, y = data.shuffle_points(list(range(100)), list(range(100)), 120)
  assert(len(x) == 100)
  assert(np.all(list(range(100)) == np.sort(x)))
  assert(len(y) == 100)
  assert(np.all(list(range(100)) == np.sort(y)))


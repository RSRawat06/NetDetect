import numpy as np


def count_classes(Y):
  """
  Return count of number of classes.
  """

  uniques, counts = np.unique(np.argmax(Y, 1), return_counts=True)
  counts = {}
  for i in range(len(uniques)):
    counts[str(uniques[i])] = counts[i]
  return counts


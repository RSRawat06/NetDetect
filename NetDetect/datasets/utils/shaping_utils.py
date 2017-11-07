'''
Module for shaping/manipulating data.
'''

import numpy as np


def build_one_hot(value, candidates):
  '''
  Returns proper one-hot representation of a value.
  Invalid value will raise Value Error.
  Args:
    - value (str): value belonging to candidates.
    - candidates (list of str): possible values.
  Returns:
    - vector (np.arr): one-hot vector represenation.
  '''

  for i, candidate in enumerate(candidates):
    if candidate == value:
      vector = np.zeros(len(candidates))
      vector[i] = 1
      return vector
  raise ValueError("Value not in candidates for one-hot.")


def fix_vector_length(vector, length):
  '''
  Fix a vector to a given length, padding and cutting
  as needed.
  Args:
    - vector (np.arr): input vector.
    - length (int): desired length.
  Returns:
    - vector (np.arr): vector with proper length in dim 1.
  '''

  right_padding = length - min(length, vector.shape[0])
  padding_shape = [[0, right_padding]] + [[0, 0] for _ in range
                                          (len(vector.shape) - 1)]
  vector = np.pad(vector[:length], padding_shape,
                  'constant', constant_values=0)

  # Sanity check.
  assert(vector.shape[0] == length)

  return vector


def segment_vector(vector, length):
  '''
  Get continuous samples of specified length from
  vector.
  Args:
    - vector (np.arr): input vector.
    - length (int): desired length.
  Returns:
    - vector (list of np.arr): list of vectors.
  '''

  if len(vector) <= length:
    return [fix_vector_length(vector, length)]

  cut_vectors = []
  for i in range(0, len(vector) + 1 - length):
    cut_vectors.append(vector[i:i + length])

  return cut_vectors


def partition_dataset(X, Y, n_test, n_val):
  '''
  Partition a dataset into train, test, validation
  splits.
  Args:
    - X (np.array)
    - Y (np.array)
    - n_test (int): number of test points
    - n_val (int): number of validation points
  Return:
    - result:
      {"train": {"X": np.arr, "Y": np.arr},
      "test": {"X": np.arr, "Y": np.arr},
      "val": {"X": np.arr, "Y": np.arr}}.
  '''

  # X, Y partitioned as: [train, test, val]
  n_train = X.shape[0] - n_test - n_val

  train_X = X[:n_train]
  test_X = X[n_train:(n_train + n_test)]
  val_X = X[(n_train + n_test):]
  del(X)

  train_Y = Y[:n_train]
  test_Y = Y[n_train:(n_train + n_test)]
  val_Y = Y[(n_train + n_test):]
  del(Y)

  return {"train": {"X": train_X, "Y": train_Y},
          "test": {"X": test_X, "Y": test_Y},
          "val": {"X": val_X, "Y": val_Y}}


def shuffle_twins(X, Y):
  '''
  Shuffle two np.arrays in parallel.
  Shuffles on axis=0.
  Args:
    - X (np.array)
    - Y (np.array)
  Return:
    - X_shuffled (np.array)
    - Y_shuffled (np.array)
  '''

  assert(X.shape[0] == Y.shape[0])

  rng_state = np.random.get_state()
  np.random.shuffle(X)
  np.random.set_state(rng_state)
  np.random.shuffle(Y)

  return X, Y


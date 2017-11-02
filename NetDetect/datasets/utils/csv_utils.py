'''
Module for csv functions.
'''

import numpy as np


def build_headers(row):
  '''
  Build headers for a row.
  Args:
    - row (list): from csv.
  Returns:
    - headers_key (dict): {0: "ip", 1: "name"}
  '''

  headers_key = {}
  for j, field in enumerate(row):
    headers_key[j] = field
  return headers_key


def featurize_row(row, headers_key, numerical_fields):
  '''
  Featurize a row into a real-valued vector.
  '''

  feature_vector = []
  for i, value in enumerate(row):
    if headers_key[i] in numerical_fields:
      feature_vector.append(float(value))
  assert(len(feature_vector) == len(numerical_fields))

  return np.array(feature_vector, dtype=np.float32)


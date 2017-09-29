from . import utils
import numpy as np
import csv


def featurize_csv(data_path, numerical_fields):
  '''
  Load in a CSV and convert to ANN-friendly numpy vectors.
  Args:
    - data_path (str): path to data file
  Returns:
    - X: list([0, 1, 3.0, 4, 5...])
  '''
  X = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      X.append(featurize_row(row, headers_key, numerical_fields))
  return X


def featurize_row(row, headers_key, numerical_fields):
  '''
  Featurize a row into a real-valued vector
  Args:
    - row (list of str): row of a csv from csv.reader
    - headers_key (dict): maps pos index in row to field name
    - numerical_fields (list of str): list of field names to be floated
  Returns:
    - feature_vector: list(0, 3, 4...)
  '''
  feature_vector = []
  for i, value in enumerate(row):
    if headers_key[i] in numerical_fields:
      feature_vector.append(float(value))
  assert(len(feature_vector) == len(numerical_fields))
  return np.array(feature_vector)


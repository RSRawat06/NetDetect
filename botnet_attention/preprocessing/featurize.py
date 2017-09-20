from . import utils
import numpy as np
import csv


def featurize_csv(data_path):
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
      X.append([float(x) for x in row])
 
  return X


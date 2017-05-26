import sys, os  

sys.path.append(os.getcwd())
from config import *

import csv
import numpy as np

def score_extraction(row, fields_key):
  point = []
  for field, index_ in fields_key.items():
    if field == "Score":
      if row[index_] == "0":
        target = [1,0]
      elif row[index_] == "1":
        target = [0,1]
      else:
        raise Exception
      continue
    point.append(row[index_])
  return point, target

def sequentialify(data, targets, key):
  seq_points = []
  seq_targets = []
  len_training = 0

  sequence_match = {}

  # Assign to streams
  for i, point in enumerate(data):
    assert(point)
    # Handle src
    if (key[i][0] in sequence_match):
      sequence_match[key[i][0]]['approved'].append(point)
    else:
      sequence_match[key[i][0]] = {'approved':[point]}
      sequence_match[key[i][0]]['score'] = targets[i]

    # Handle dest
    if (key[i][1] in sequence_match):
      sequence_match[key[i][1]]['approved'].append(point)
    else:
      sequence_match[key[i][1]] = {'approved':[point]}
      sequence_match[key[i][1]]['score'] = targets[i]

  # Segment into chunks of uniform length
  for usr, info in sequence_match.items():
    # print("Info: ", info)
    for i in range(0, len(info['approved'])-MAX_SEQUENCE_LENGTH):
      seq_points.append(info['approved'][i:i+MAX_SEQUENCE_LENGTH])
      seq_targets.append(info['score'])
      # print("Approved: ", info['approved'][i:i+MAX_SEQUENCE_LENGTH])
      # print("Approved select: ", info['approved'][i])
      # print("Score: ", info['score'])
      len_training += 1

  return np.array(seq_points), seq_targets, len_training


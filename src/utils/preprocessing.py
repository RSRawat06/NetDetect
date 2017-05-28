import sys, os  

sys.path.append(os.getcwd())
from config import *

import csv
import numpy as np

def score_extraction(row, fields_key):
  point = np.empty(len(fields_key) - 1, dtype=np.float32)
  i = 0
  for field, index_ in fields_key.items():
    if field == "Score":
      if row[index_] == "0":
        target = [1,0]
      elif row[index_] == "1":
        target = [0,1]
      else:
        raise Exception
      continue
    point[i] = row[index_]
    i += 1
  return point, target

def sequentialify(data, targets, supp_data):
  print("Initiating sequentificalication")
  
  sequence_match = []
  sequence_match_key = {}

  # Assign to streams
  for i, point in enumerate(data):

    # Handle src
    if (supp_data[i][0] in sequence_match_key):
      sequence_match[sequence_match_key[supp_data[i][0]]]['approved'].append(point)
    else:
      sequence_match_key[supp_data[i][0]] = len(sequence_match)
      sequence_match.append({'approved':[point], 'score':targets[i]})

    # Handle dest
    if (supp_data[i][1] in sequence_match_key):
      sequence_match[sequence_match_key[supp_data[i][1]]]['approved'].append(point)
    else:
      sequence_match_key[supp_data[i][1]] = len(sequence_match)
      sequence_match.append({'approved':[point], 'score':targets[i]})

  del(data)
  del(targets)
  del(supp_data)

  seq_points = []
  seq_targets = []
  len_training = 0

  # Segment into chunks of uniform length
  for usr, seq_id in sequence_match_key.items():
    info = sequence_match[seq_id]
    for i in range(0, len(info['approved'])-MAX_SEQUENCE_LENGTH):
      seq_points.append(info['approved'][i:i+MAX_SEQUENCE_LENGTH])
      seq_targets.append(info['score'])
      len_training += 1

  print("Sequentification complete")

  return np.array(seq_points), np.array(seq_targets), len_training


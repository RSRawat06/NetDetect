import sys, os  

sys.path.append(os.getcwd())
from config import *

import csv
import numpy as np

def point_extraction(row, fields_key):
  point = np.empty(len(fields_key), dtype=np.float32)
  i = 0
  for field, index_ in fields_key.items():
    point[i] = row[index_]
    i += 1
  return point

def is_malicious(ip):
  if ip in ["192.168.2.112","131.202.243.84","192.168.5.122","198.164.30.2","192.168.2.110","192.168.4.118","192.168.2.113","192.168.1.103","192.168.4.120","192.168.2.112","192.168.2.109","192.168.2.105","147.32.84.180","147.32.84.170","147.32.84.150","147.32.84.140","147.32.84.130","147.32.84.160","10.0.2.15","192.168.106.141","192.168.106.131","172.16.253.130","172.16.253.131","172.16.253.129","172.16.253.240","74.78.117.238","158.65.110.24","192.168.3.35","192.168.3.25","192.168.3.65","172.29.0.116","172.29.0.109","172.16.253.132","192.168.248.165","10.37.130.4"]:
    return [0, 1]
  else:
    return [1, 0]

def sequentialify(data, supp_data):
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
      sequence_match.append({'approved':[point], 'score':is_malicious(supp_data[i][0])})

    # Handle dest
    # if (supp_data[i][1] in sequence_match_key):
    #   sequence_match[sequence_match_key[supp_data[i][1]]]['approved'].append(point)
    # else:
    #   sequence_match_key[supp_data[i][1]] = len(sequence_match)
    #   sequence_match.append({'approved':[point], 'score':targets[i]})

  del(data)
  # del(targets)
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


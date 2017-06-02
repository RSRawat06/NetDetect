import sys, os  

sys.path.append(os.getcwd())
from config import *

sys.path.append(PROJ_ROOT + "src/utils/")
from preprocessing import *

import csv
from sklearn.utils import shuffle
import numpy as np

def load_data(path=DATASET_URL, n_points_cap=None):
  print("Initiating data loading")
  DESIRED_FIELDS = ["Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Header Length","Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate","Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"]
  # DESIRED_FIELDS = ['APL', 'AvgPktPerSec', 'IAT', 'NumForward', 'Protocol', 'BytesEx', 'BitsPerSec', 'NumPackets', 'StdDevLen', 'SameLenPktRatio', 'FPL', 'Duration', 'NPEx']
  UNDESIRED_FIELDS = ['Source IP', 'Destination IP']
  fields_key = {}
  unfields_key = {}

  points = []
  # targets = []
  key = []

  i = 0

  with open(path, 'r') as f:
    first_row = True
    training_data = []
    for row in csv.reader(f):
      # If first row, use it to generate header rows dict
      if first_row:
        j = 0
        for field in row:
          print(field)
          if field in DESIRED_FIELDS:
            fields_key[field] = j
          if field in UNDESIRED_FIELDS:
            unfields_key[field] = j
          j += 1
        for field in DESIRED_FIELDS:
          print(fields_key)
          assert(field in fields_key)
        first_row = False
        print("Header correlation complete")
        continue

      # Terminate if reached points cap
      if n_points_cap:
        if (n_points_cap > i):
          print("Points cap reached. Data loading safely terminated early.")
          break
      i += 1

      # Process row
      point = point_extraction(row, fields_key)

      points.append(point)
      # targets.append(target)
      key.append((row[unfields_key['Source IP']], row[unfields_key['Destination IP']]))


  print("All rows processed")
  seq_points, seq_targets, total_len = sequentialify(points, key)
  del(points)
  # del(targets)
  del(key)
  
  shuffle_points_mid = []
  shuffle_targets_mid = []

  for i in range(0, total_len, SHUFFLE_PARTITION_LEN):
    shuffled_points, shuffled_targets = shuffle(seq_points[i:i+SHUFFLE_PARTITION_LEN], seq_targets[i:i+SHUFFLE_PARTITION_LEN], random_state=0)
    shuffle_points_mid.append(shuffled_points)
    shuffle_targets_mid.append(shuffled_targets)

  del(seq_points)
  del(seq_targets)

  shuffle_points_mid, shuffle_targets_mid = shuffle(shuffle_points_mid, shuffle_targets_mid, random_state=0)

  seq_points_shuffled = []
  seq_targets_shuffled = []
  for i in range(len(shuffle_points_mid)):
    seq_points_shuffled += list(shuffle_points_mid[i])
    seq_targets_shuffled += list(shuffle_targets_mid[i])

  del(shuffle_points_mid)
  del(shuffle_targets_mid)
  
  total_true = 0
  total_false = 0

  for i in seq_targets_shuffled:
    if i[1]:
      total_true+=1
    else:
      total_false+=1

  print("Total true: ", total_true)
  print("Total false: ", total_false)
  print("Percentage true: ", 100*total_true/(total_true+total_false))

  seq_train = seq_points_shuffled[:-N_TEST]
  seq_train_targets = seq_targets_shuffled[:-N_TEST]
  seq_test = seq_points_shuffled[-N_TEST:]
  seq_test_targets = seq_targets_shuffled[-N_TEST:]

  del(seq_points_shuffled)
  del(seq_targets_shuffled)

  len_training = total_len - N_TEST
  len_testing = N_TEST
  print("Data loading complete")

  print("Data shuffled")

  return {"x":np.array(seq_train), "y":np.array(seq_train), "targets":np.array(seq_train_targets)}, {"x":np.array(seq_test), "y":np.array(seq_test), "targets":np.array(seq_test_targets)}, len_training, len_testing
  
if __name__ == "__main__":
  load_data()


from sklearn.utils import shuffle
import numpy as np


def parse_feature(feature_name, value, config):
  if feature_name == "Score":
    if value == "0":
      return np.array([1, 0])
    elif value == "1":
      return np.array([0, 1])
    else:
      raise Exception
  elif feature_name in ["FirstPacketLength", "NumberOfPackets", "NumberOfBytes", "StdDevOfPacketLength", "RatioOfSameLengthPackets", "Duration", "AveragePacketLength", "AverageBitsPerSecond", "AveragePacketsPerSecond", "NumberOfNullPackets", "IsNull"]:
    return [value]
  else:
    raise ValueError


def shuffle_points(X, Y, config):
  X_mid = []
  Y_mid = []
  for i in range(0, len(Y), config.SHUFFLE_PARTITION_LEN):
    sub_X, sub_Y = shuffle(X[i:i + config.SHUFFLE_PARTITION_LEN], Y[i:i + config.SHUFFLE_PARTITION_LEN], random_state=0)
    X_mid.append(sub_X)
    Y_mid.append(sub_Y)
  del(X)
  del(Y)
  X_mid, Y_mid = shuffle(X_mid, Y_mid, random_state=0)
  X_shuffled = []
  Y_shuffled = []
  for i in range(len(Y_mid)):
    X_shuffled += list(X_mid[i])
    Y_shuffled += list(Y_mid[i])
  del(X_mid)
  del(Y_mid)
  print("Data shuffled")
  return np.array(X_shuffled), np.array(Y_shuffled)


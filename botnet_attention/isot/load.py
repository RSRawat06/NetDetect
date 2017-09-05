"""
isot.load

Load in local ISOT dataset for use.
"""

from ..utils import data
from . import config
import numpy as np


def parse_feature(field, raw_value):
  '''
  Parse feature based on field type.
  Parsing mechanisms unique to ISOT.
  '''
  is_flow_id, is_score, is_participant_ip, is_feature = False, False, False, False
  # Load in score
  if field == "Score":
    is_score = True
    if raw_value == "0":
      value = np.array([1, 0])
    elif raw_value == "1":
      value = np.array([0, 1])
    else:
      raise Exception
  # Basic integer datas
  elif field in ["FirstPacketLength", "NumberOfPackets", "NumberOfBytes", "StdDevOfPacketLength", "RatioOfSameLengthPackets", "Duration", "AveragePacketLength", "AverageBitsPerSecond", "AveragePacketsPerSecond", "NumberOfNullPackets", "IsNull"]:
    value = [int(raw_value)]
    is_feature = True
  # Categorical data
  elif field == "type":
    is_feature = True
    if raw_value.lower().strip() == "apple":
      value = [0, 1]
    elif raw_value.lower().strip() == "pear":
      value = [1, 0]
  # Add participants
  elif field in ["Destination", "Source"]:
    is_participant_ip = True
    value = str(raw_value)
  # Flow Numbers
  elif field == "FlowNo":
    is_flow_id = True
    value = int(raw_value)
  else:
    raise ValueError
  return value, is_flow_id, is_score, is_participant_ip, is_feature


fetch_data = data.load(config, parse_feature)


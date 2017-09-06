"""
iscx.load

Load in local ISCX dataset for use.
"""

from ..utils import data
from . import config
import numpy as np


def store_categoricals(field, value, records):
  if field in ["eth.dst", "tcp.dstport", "tcp.srcport", "udp.dstport", "udp.srcport", "ip.proto", "ip.flags", "frame.protocols", "tcp.flags"]:
    if field not in records:
      records[field] = []
    if value not in records[field]:
      records[field].append(value)
    if len(records[field]) > 10:
      raise ValueError
  return value


def parse_feature(field, raw_value, records):
  '''
  Parse feature based on field type.
  Parsing mechanisms unique to ISCX.
  '''
  is_flow_id, is_participant_ip, is_feature = False, False, False
  if field in ["ip.len", "frame.time_epoch", "tcp.len", "udp.len"]:
    if raw_value == "":
      value = [0]
    else:
      value = [int(raw_value)]
    is_feature = True
  # Categorical data
  elif field in ["eth.dst", "tcp.dstport", "tcp.srcport", "udp.dstport", "udp.srcport", "ip.proto", "ip.flags", "frame.protocols", "tcp.flags"]:
    is_feature = True
    ind = records[field].index(raw_value)
    assert(ind >= 0)
    value = np.zeros(len(records[field]))
    value[ind] = 1
  # Add participants
  elif field in ["ip.src", "ip.dst"]:
    is_participant_ip = True
    value = [str(raw_value)]
  # Flow Numbers
  elif field == "FlowNo":
    is_flow_id = True
    value = [int(raw_value)]
  else:
    raise ValueError
  return value, is_flow_id, is_participant_ip, is_feature


fetch_data = data.load(config, parse_feature, store_categoricals)


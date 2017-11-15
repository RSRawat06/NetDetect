from . import config


def identify_participants(row, headers_key):
  '''
  Return participants for a given row.
  '''

  participants = []
  for i, value in enumerate(row):
    if headers_key[i] in config.participant_fields:
      participants.append(str(value))
  return participants


def parse_score(value):
  '''
  Parse score according to ISCX label
  standards.
  '''

  if value == "BENIGN":
    return 0
  elif value == "BOTNET":
    return 1
  else:
    raise ValueError


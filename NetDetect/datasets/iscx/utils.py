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


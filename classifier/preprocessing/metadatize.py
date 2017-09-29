import csv
from . import utils


def metadatize_csv(data_path, malicious_ips, flow_field, participant_fields):
  '''
  Load in a CSV and develop corresponding metadata listing flow id and participants.
  Args:
    - data_path (str): path to data file
    - malicious_ips (list): list of malicious ips
    - flow_field (str): name of header corresponding to flow number
    - participant_fields (list): list of headers corresponding to participants
  Returns:
    - metadata: list({"flow_id": 34, "participants": [{"score": 1, "ip":"323.343"]})
  '''
  metadata = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      metadata.append(metadatize_row(row, headers_key, malicious_ips, flow_field, participant_fields))

  return metadata


def metadatize_row(row, headers_key, malicious_ips, flow_field, participant_fields):
  '''
  Parse a row to extract the metadatum for the row.
  Args:
    - row (list of str): row of a csv from csv.reader
    - headers_key (dict): maps pos index in row to field name
    - malicious_ips (list): list of malicious ips
    - flow_field (str): name of header corresponding to flow number
    - participant_fields (list): list of headers corresponding to participants
  Returns:
    - metadatum: {"flow_id": 34, "participants": [{"score": 1, "ip":"323.343"]}
  '''
  participants = []
  flow_id = None
  for i, value in enumerate(row):
    if headers_key[i] == flow_field:
      flow_id = str(value)
    elif headers_key[i] in participant_fields:
      score = 1 if str(value) in malicious_ips else 0
      participants.append({"score": score, "ip": str(value)})
  return {'flow_id': flow_id, 'participants': participants}


from ...preprocessing import featurize
import numpy as np
from . import config

headers_key = {0: 'size', 1: 'width', 2: 'type', 3: 'flow_id', 4: 'ip', 5: 'ip2', 6: 'proto', 7: 'port'}

raw_first = [323, '23', 'apple', 5.0, 4, 6, 'chocolate:dank:shit', 10]
raw_second = [22.5, '14', 'pear', 9.0, 4, 8, 'chocolate:weird:blah', 2900]
raw_third = [12.5, '14', 'chocolate', 9.0, 4, 8, 'chocolate:weird:shit', 1998]
raw_fourth = [22.5, '14', 'pear', 9.0, 4, 8, 'chocolate:weird', 29000]

meta_first = {'flow_id': 5, 'participants': [{"score": 0, "ip": 4}, {"score": 1, "ip": 6}]}
meta_second = {'flow_id': 9, 'participants': [{"score": 0, "ip": 4}, {"score": 1, "ip": 8}]}
meta_third = {'flow_id': 9, 'participants': [{"score": 0, "ip": 4}, {"score": 1, "ip": 8}]}
meta_fourth = {'flow_id': 9, 'participants': [{"score": 0, "ip": 4}, {"score": 1, "ip": 8}]}

clean_first = [323.0, 23.0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
clean_second = [22.5, 14.0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
clean_third = [12.5, 14.0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
clean_fourth = [22.5, 14.0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]

flow_field = 'flow_id'
numerical_fields = ['size', 'width']
protocol_fields = ['proto']
categorical_fields = ['type']
port_fields = ['port']
participant_fields = ['ip', 'ip2']
malicious_ips = [6, 8]


def test_store_categoricals():
  all_records = {'categorical': {}, 'protocol': {}, 'port': {}}
  for row in [raw_first, raw_second]:
    all_records = featurize.store_categoricals(row, headers_key, all_records, 'proto', 'type', 'port')
  assert(all_records == {'categorical': {'type': ['apple', 'pear']}, 'protocol': {'proto': ['chocolate', 'dank', 'shit', 'weird', 'blah']}, 'port': {'port': [10, 200]}})


def test_store_categorical_over():
  all_records = {'categorical': {}, 'protocol': {}, 'port': {}}
  for i, row in enumerate([raw_first, raw_second, raw_third]):
    if i == 2:
      failed = False
      try:
        all_records = featurize.store_categoricals(row, headers_key, all_records, 'proto', 'type', 'port', 5, 5)
      except ValueError:
        failed = True
      assert(failed)
      continue
  assert(all_records == {'categorical': {'type': ['apple', 'pear']}, 'protocol': {'proto': ['chocolate', 'dank', 'shit', 'weird', 'blah']}, 'port': {'port': [10, 200]}})


def test_featurize_row():
  all_records = {'categorical': {}, 'protocol': {}, 'port': {}}
  for row in [raw_first, raw_second, raw_third, raw_fourth]:
    all_records = featurize.store_categoricals(row, headers_key, all_records, 'proto', 'type', 'port')
  for clean, raw in ((clean_first, raw_first), (clean_second, raw_second), (clean_third, raw_third), (clean_fourth, raw_fourth)):
    assert(clean == featurize.featurize_row(raw, headers_key, all_records, numerical_fields, protocol_fields, categorical_fields, port_fields))


def test_featurize_csv():
  X = featurize.featurize_csv(config.DATA_DIR + "sample_csv.csv", numerical_fields, protocol_fields, categorical_fields, port_fields)
  assert(np.all(X == np.array([clean_first, clean_second, clean_third, clean_fourth])))


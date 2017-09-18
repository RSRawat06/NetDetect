from ...preprocessing import metadatize
from . import config

headers_key = {0: 'size', 1: 'width', 2: 'type', 3: 'flow_id', 4: 'ip', 5: 'ip2', 6: 'proto', 7: 'port'}

raw_first = [323, '23', 'apple', 5.0, '4', '6', 'chocolate:dank:shit', 10]
raw_second = [22.5, '14', 'pear', 9.0, '4', '8', 'chocolate:weird:blah', 2900]
raw_third = [12.5, '14', 'chocolate', 9.0, '4', '8', 'chocolate:weird:shit', 1998]
raw_fourth = [22.5, '14', 'pear', 9.0, '4', 8, 'chocolate:weird', 29000]

meta_first = {'flow_id': 5, 'participants': [{"score": 0, "ip": '4'}, {"score": 1, "ip": '6'}]}
meta_second = {'flow_id': 9, 'participants': [{"score": 0, "ip": '4'}, {"score": 1, "ip": '8'}]}
meta_third = {'flow_id': 9, 'participants': [{"score": 0, "ip": '4'}, {"score": 1, "ip": '8'}]}
meta_fourth = {'flow_id': 9, 'participants': [{"score": 0, "ip": '4'}, {"score": 1, "ip": '8'}]}

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
malicious_ips = ['6', '8']


def test_metadatize_row():
  for meta, raw in ((meta_first, raw_first), (meta_second, raw_second), (meta_third, raw_third), (meta_fourth, raw_fourth)):
    assert(meta == metadatize.metadatize_row(raw, headers_key, malicious_ips, flow_field, participant_fields))


def test_metadatize_csv():
  metadata = metadatize.metadatize_csv(config.DATA_DIR + "sample_csv.csv", malicious_ips, flow_field, participant_fields)
  assert(metadata == [meta_first, meta_second, meta_third, meta_fourth])


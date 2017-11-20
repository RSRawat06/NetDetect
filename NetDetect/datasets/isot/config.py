'''
Standard configuration for isot dataset.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

RAW_NAME = "isot_full.csv"

PROCESSED_NAME = "processed_dataset.p"

DATASET_URL = ""

participant_fields = ['Source', 'Destination']
numerical_fields = [
    'APL', 'AvgPktPerSec', 'IAT', 'NumForward', 'Protocol', 'BytesEx',
    'BitsPerSec', 'NumPackets', 'StdDevLen', 'SameLenPktRatio',
    'FPL', 'Duration', 'NPEx'
]
malicious_ips = [
    'bb:bb:bb:bb:bb:bb',
    'aa:aa:aa:aa:aa:aa',
    'cc:cc:cc:cc:cc:cc',
    'cc:cc:cc:dd:dd:dd'
]


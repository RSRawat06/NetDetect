'''
Standard configuration for isot dataset.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

RAW_DATASET_PATH = DUMPS_DIR + "isot_full.csv"

participant_fields = ['Eth_val']
numerical_fields = [
    'APL', 'AvgPktPerSec', 'IAT', 'NumForward', 'Protocol', 'BytesEx',
    'BitsPerSec', 'NumPackets', 'StdDevLen', 'SameLenPktRatio',
    'FPL', 'Duration', 'NPEx'
]
malicious_ips = [
    'bb:bb:bb:bb:bb:bb',
    'aa:aa:aa:aa:aa:aa',
    'cc:cc:cc:cc:cc:cc',
    'cc:cc:cc:dd:dd:dd',
]


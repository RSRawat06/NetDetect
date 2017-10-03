'''
Standard configuration for isot dataset.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

RAW_SAVE_NAME = "isot_full.csv"
PROCESSED_SAVE_NAME = "processed_dataset.p"

MAX_SEQUENCE_LENGTH = 5

DESIRED_FIELDS = [
    'APL', 'AvgPktPerSec', 'IAT', 'NumForward', 'Protocol', 'BytesEx',
    'BitsPerSec', 'NumPackets', 'StdDevLen', 'SameLenPktRatio',
    'FPL', 'Duration', 'NPEx', 'Score'
]
UNDESIRED_FIELDS = ['Source', 'Destination']

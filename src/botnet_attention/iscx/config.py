"""
iscx.config

Configuration file for ISCX module.
"""

# GENERIC
import os
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/"
TEST_DATA_NAME = "testing_set.csv"
TEST_DATA_URL = "https://nyc3.digitaloceanspaces.com/homearchive/testing.csv?AWSAccessKeyId=XF3ED5IAPT7JR6UA6NCB&Expires=1505337450&Signature=uXOTxOWQwnAdgljo0hrdj0XqXMs%3D"
TRAIN_DATA_NAME = "training_set.csv"
TRAIN_DATA_URL = "https://nyc3.digitaloceanspaces.com/homearchive/training.csv?AWSAccessKeyId=XF3ED5IAPT7JR6UA6NCB&Expires=1505337468&Signature=Jewu5EgHi98IfeTnWJC1rVFmsLA%3D"
DATA_NAME = TRAIN_DATA_NAME

# MODEL
MODEL = "self"
MODEL_CONFIG = {
    "basic": [
        100,
        [1, 5],
        50,
        MAX_FLOW_SEQUENCE_LENGTH,
        MAX_FLOW_LENGTH,
        13
    ],
    "packets": [
        20,
        20,
        4
    ],
    "flows": [
        20,
        20,
        4
    ]
}

# DATASET
N_CAP = 10000
N_TEST = 1000
SHUFFLE_PARTITION_LEN = 1000000
MALICIOUS_IPS = []
MAX_FLOW_LENGTH = 40
MAX_FLOW_SEQUENCE_LENGTH = 40

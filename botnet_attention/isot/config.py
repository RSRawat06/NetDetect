"""
isot.config

Configuration file for ISOT module.
"""

# GENERIC
import os
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/"
DATA_NAME = "default_set.data"
DATA_URL = "https://dropbox.com/dankmemes"

# MODEL
MODEL = "double"
MODEL_CONFIG = {
    "N_HIDDEN": 5,
    "N_FEATURES": 13,
    "MAX_SEQUENCE_LENGTH": 5,
    "MAX_PACKET_SEQUENCE_LENGTH": 5,
    "ITERATIONS": 50,
    "BATCH_S": 50,
    "RESULT_WEIGHTING": [1, 5]
}

# DATASET
N_CAP = 10000
N_TEST = 1000
SHUFFLE_PARTITION_LEN = 1000000
FLOW_SEQUENCES = False

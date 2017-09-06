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
MALICIOUS_IPS = ["192.168.2.112", "131.202.243.84", "192.168.5.122", "198.164.30.2", "192.168.2.110", "192.168.5.122", "192.168.4.118", "192.168.5.122", "192.168.2.113", "192.168.5.122", "192.168.1.103", "192.168.5.122", "192.168.4.120", "192.168.5.122", "192.168.2.112", "192.168.2.110", "192.168.2.112", "192.168.4.120", "192.168.2.112", "192.168.1.103", "192.168.2.112", "192.168.2.113", "192.168.2.112", "192.168.4.118", "192.168.2.112", "192.168.2.109", "192.168.2.112", "192.168.2.105", "192.168.1.105", "192.168.5.122", "147.32.84.180", "147.32.84.170", "147.32.84.150", "147.32.84.140", "147.32.84.130", "147.32.84.160", "10.0.2.15", "192.168.106.141", "192.168.106.131", "172.16.253.130", "172.16.253.131", "172.16.253.129", "172.16.253.240", "74.78.117.238", "158.65.110.24", "192.168.3.35", "192.168.3.25", "192.168.3.65", "172.29.0.116", "172.29.0.109", "172.16.253.132", "192.168.248.165", "10.37.130.4"]
MAX_FLOW_LENGTH = 40
MAX_FLOW_SEQUENCE_LENGTH = 40

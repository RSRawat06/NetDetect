"""
models.config

Configuration file for models module.
"""

import os
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

VANILLA_MODEL_NAME = "vanilla.model"
VANILLA_META_NAME = "vanilla_tf.meta"

SELF_MODEL_NAME = "self.model"
SELF_META_NAME = "self_tf.meta"

FRAMEWORK = "tensorflow"
MAX_FLOW_LENGTH = 40
MAX_FLOW_SEQUENCE_LENGTH = 40

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

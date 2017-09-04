"""
models.config

Configuration file for models module.
"""

import os
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

ATTENTION_MODEL_NAME = "attention.model"
ATTENTION_META_NAME = "attention_tf.meta"

VANILLA_MODEL_NAME = "vanilla.model"
VANILLA_META_NAME = "vanilla_tf.meta"

DOUBLE_MODEL_NAME = "double.model"
DOUBLE_META_NAME = "double_tf.meta"

FRAMEWORK = "tensorflow"

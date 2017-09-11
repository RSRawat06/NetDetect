"""
models.config

Configuration file for models module.
"""

import os
MODEL_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

VANILLA_MODEL_NAME = "vanilla.model"
VANILLA_META_NAME = "vanilla_tf.meta"

SELF_MODEL_NAME = "self.model"
SELF_META_NAME = "self_tf.meta"

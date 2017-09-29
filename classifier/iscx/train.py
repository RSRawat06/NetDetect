"""
iscx.train

Train on the ISCX dataset
Run: `python3 -m botnet_attention.iscx.train`
"""

import pickle
import tensorflow as tf
from .. import models
from . import config


if __name__ == "__main__":
  with tf.Session() as sess:
    preprocessed = pickle.load(open(config.DATA_DIR + config.TRAIN_SAVE, "rb"))
    model = models.vanilla_gru.Vanilla_GRU(sess, config)
    model.initialize()
    model.train(*preprocessed)
    model.save()


"""
isot.train

Train on the ISOT dataset
Run: `python3 -m botnet_attention.isot.train`
"""

import tensorflow as tf
from .. import models
from . import load, config


if __name__ == "__main__":
  data = load.fetch_data()
  with tf.Session() as sess:
    if config.MODEL == "vanilla":
      model = models.vanilla_gru(sess, config.MODEL_CONFIG)
    elif config.MODEL == "self":
      model = models.self_attention(sess, config.MODEL_CONFIG)
    else:
      raise ValueError("Invalid choice of RNN model")
    model.build_model()
    model.initialize()
    model.train(*data)
    model.save()

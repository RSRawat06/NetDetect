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
      model = models.vanilla_rnn(sess, config.MODEL_CONFIG)
    elif config.MODEL == "attention":
      model = models.attention_rnn(sess, config.MODEL_CONFIG)
    elif config.MODEL == "double":
      model = models.double_rnn(sess, config.MODEL_CONFIG)
    else:
      raise ValueError("Invalid choice of RNN model")
    model.build_model()
    model.train(*data, config.ITERATIONS)

"""
iscx.train

Train on the ISCX dataset
Run: `python3 -m botnet_attention.iscx.train`
"""

import tensorflow as tf
from .. import models
from . import load, config

if __name__ == "__main__":
  data = load.fetch_data()
  with tf.Session() as sess:
    if config.MODEL == "vanilla":
      model = models.vanilla_gru.Vanilla_GRU(sess, config)
    elif config.MODEL == "self":
      model = models.self_attention.Self_Attention(sess, config)
    else:
      raise ValueError("Invalid choice of model")
    model.build_model()
    model.initialize()
    model.train(*data)
    model.save()

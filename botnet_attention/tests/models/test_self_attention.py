import tensorflow as tf
from ... import models
from . import test_config as config
import numpy as np


def test_self_attention():
  tf.reset_default_graph()
  with tf.Session() as sess:
    model = models.self_attention.Self_Attention(sess, config)
    model.build_model()
    model.initialize()
    predictions = model.predict(np.full((2 * config.BATCH_SIZE, config.N_FLOWS, config.N_PACKETS, config.N_FEATURES), 10))
    model.save()

  tf.reset_default_graph()
  with tf.Session() as sess:
    model2 = models.self_attention.Self_Attention(sess, config)
    model2.build_model()
    model2.load()
    predictions = model2.predict(np.full((2 * config.BATCH_SIZE, config.N_FLOWS, config.N_PACKETS, config.N_FEATURES), 10))


if __name__ == "__main__":
  test_self_attention()

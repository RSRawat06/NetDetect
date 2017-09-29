import tensorflow as tf
from ... import models
from . import test_config as config
import numpy as np


def test_self_attention():
  tf.reset_default_graph()
  with tf.Session() as sess:
    model = models.self_attention.Self_Attention(sess, config)
    model.initialize()
    model.predict(np.full((2 * models.config.BATCH_SIZE, models.config.NUMBERS['flows'], models.config.NUMBERS['flow_features']), 10))
    model.save()

  tf.reset_default_graph()
  with tf.Session() as sess:
    model2 = models.self_attention.Self_Attention(sess, config)
    model2.initialize()
    model2.restore()
    model2.predict(np.full((2 * models.config.BATCH_SIZE, models.config.NUMBERS['flows'], models.config.NUMBERS['flow_features']), 10))


if __name__ == "__main__":
  test_self_attention()

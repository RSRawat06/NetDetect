import tensorflow as tf
from ... import models
from . import test_config as config
import numpy as np


def test_vanilla_gru():
  tf.reset_default_graph()
  with tf.Session() as sess:
    model = models.vanilla_gru.Vanilla_GRU(sess, config)
    model.initialize()
    model.predict(np.full((2 * models.config.BATCH_SIZE, models.config.NUMBERS['flows'], models.config.NUMBERS['packets'], models.config.NUMBERS['packet_features']), 10))
    model.save()

  tf.reset_default_graph()
  with tf.Session() as sess:
    model2 = models.vanilla_gru.Vanilla_GRU(sess, config)
    model2.pseudoload()
    model2.build_model()
    model2.predict(np.full((2 * models.config.BATCH_SIZE, models.config.NUMBERS['flows'], models.config.NUMBERS['packets'], models.config.NUMBERS['packet_features']), 10))


if __name__ == "__main__":
  test_vanilla_gru()

import tensorflow as tf
from ... import models
from . import test_config as config
import numpy as np


def test_vanilla_gru():
  tf.reset_default_graph()
  with tf.Session() as sess:
    model = models.vanilla_gru.Vanilla_GRU(sess, config)
    model.build_model()
    model.initialize()
    predictions = model.predict(np.full((2 * config.BATCH_SIZE, config.N_FLOWS, config.N_PACKETS, config.N_FEATURES), 10))
    model.save()

  tf.reset_default_graph()
  with tf.Session() as sess:
    model2 = models.vanilla_gru.Vanilla_GRU(sess, config)
    model2.build_model()
    model2.load()
    predictions = model2.predict(np.full((2 * config.BATCH_SIZE, config.N_FLOWS, config.N_PACKETS, config.N_FEATURES), 10))

  tf.reset_default_graph()

if __name__ == "__main__":
  test_vanilla_gru()

import tensorflow as tf
from ... import models
from . import test_config as config

def test_vanilla_gru():
  with tf.Session() as sess:
    model = models.vanilla_gru.Vanilla_GRU(sess, config)
    model.build_model()
    # model.predict(np.zeros((config.N_BATCHES, config.N_FLOWS, config.N_PACKETS, config.N_FEATURES)))

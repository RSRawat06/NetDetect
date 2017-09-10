import tensorflow as tf
from ... import models
from . import test_config as config

def test_self_attention():
  with tf.Session() as sess:
    model = models.self_attention.Self_Attention(sess, config)
    model.build_model()
    # model.predict(np.zeros((config.N_BATCHES, config.N_FLOWS, config.N_PACKETS, config.N_FEATURES)))

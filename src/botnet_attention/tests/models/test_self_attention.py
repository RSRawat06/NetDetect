import tensorflow as tf
from .. import models
from . import load
from . import test_config as config

if __name__ == "__main__":
  with tf.Session() as sess:
    model = models.self_attention(sess, config)
    model.build_model()

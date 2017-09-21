"""
iscx.download

Download the ISCX dataset locally.
Run: `python3 -m botnet_attention.iscx.download`
"""

from . import config
from ..utils import network
from ..preprocessing import main
import tensorflow as tf
import numpy as np
import pickle

if __name__ == "__main__":
  X, Y = main.preprocess(config.DATA_DIR + config.TRAIN_SAVE, config)
  print("File preprocessed")
  with open('chocolate.p', 'wb') as f:
    pickle.dump((X, Y), f)
  print("File dumped")

  writer = tf.python_io.TFRecordWriter(config.DATA_DIR + config.TF_SAVE)
  for i in len(X):
    features = np.reshape(np.array(X[i], dtype=np.float32), (-1))
    label = np.array(Y[i], dtype=np.float32)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'label': tf.train.Feature(float32_list=tf.train.Float32List(value=label.astype("float32"))),
                'features': tf.train.Feature(float32_list=tf.train.Float32List(value=features.astype("float32"))),
            }
        )
    )
    serialized = example.SerializeToString()
    writer.write(serialized)
  writer.close()


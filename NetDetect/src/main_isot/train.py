from ...src.models import SequentialModel
from ...datasets.isot import load
from . import config
from .logger import train_logger
import numpy as np
import tensorflow as tf


def train():
  with tf.Session() as sess:
    model = SequentialModel(sess, config, train_logger)
    X, Y = load()
    shuffled_dataset = model.shuffle_and_partition(X, Y, 20, 20)
    del(X)
    del(Y)
    model.initialize()
    model.train(
        shuffled_dataset['train']['X'],
        shuffled_dataset['train']['Y'],
        shuffled_dataset['test']['X'],
        shuffled_dataset['test']['Y']
    )
    model.save()

    predictions = model.predict(shuffled_dataset['train']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['train']['Y'][:len(predictions)], 1))
    acc = np.mean(correct.astype(np.float32))
    print("Train acc:", acc)

    predictions = model.predict(shuffled_dataset['test']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['test']['Y'][:len(predictions)], 1))
    acc = np.mean(correct.astype(np.float32))
    print("Test acc:", acc)


if __name__ == "__main__":
  train()


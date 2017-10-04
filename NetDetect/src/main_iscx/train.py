from ...src.models import FlowModel
from ...datasets.iscx import load
from . import config
from .logger import train_logger
import tensorflow as tf


def train():
  with tf.Session() as sess:
    model = FlowModel(sess, config, train_logger)
    X, Y = load()
    shuffled_dataset = model.shuffle_and_partition(
        X, Y, config.BATCH_SIZE, config.BATCH_SIZE)
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


if __name__ == "__main__":
  train()


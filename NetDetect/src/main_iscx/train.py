from ...src.models import FlowModel
from ...datasets.iscx import load
from . import config
from .logger import train_logger
import tensorflow as tf


def train():
  with tf.Session() as sess:
    model = FlowModel(sess, config, train_logger)
    dataset = load(10 * config.BATCH_SIZE, 2 * config.BATCH_SIZE)

    def __epoch_evaluation(sub_epoch):
      loss, acc, tpr, fpr, summary = model.evaluate(
          dataset['train']['X'], dataset['train']['Y'], prefix="train")
      train_logger.info(
          "Epoch: %f has train loss: %f, train accuracy: %f, \
           TPR: %s, FPR: %s" % (sub_epoch, loss, acc, str(tpr), str(fpr)))
      model.train_writer.add_summary(summary, global_step=sub_epoch)

      loss, acc, tpr, fpr, summary = model.evaluate(
          dataset['test']['X'], dataset['test']['Y'], prefix="test")
      train_logger.info(
          "Epoch: %f has test loss: %f, test accuracy: %f, \
           TPR: %s, FPR: %s" % (sub_epoch, loss, acc, str(tpr), str(fpr)))
      model.test_writer.add_summary(summary, global_step=sub_epoch)

    model.initialize()
    model.train(
        dataset['train']['X'],
        dataset['train']['Y'],
        __epoch_evaluation
    )
    model.save()


if __name__ == "__main__":
  train()


from ...src.models import FlowModel
from ...datasets.iscx import load
from . import config
from .logger import train_logger
import tensorflow as tf


def train():
  with tf.Session() as sess:
    model = FlowModel(sess, config, train_logger)
    train_logger.info("Model created.")

    train_logger.info(
        "Parameters: " +
        "\nNumber of iterations: " + str(config.ITERATIONS) +
        "\nBatch size: " + str(config.BATCH_SIZE) +
        "\nTest size: " + str(config.TEST_SIZE) +
        "\nVal size: " + str(config.VAL_SIZE) +
        "\nN_features: " + str(config.N_FEATURES) +
        "\nN_steps: " + str(config.N_STEPS) +
        "\nLayers: " + str(config.LAYERS) +
        "\nn_classes: " + str(config.N_CLASSES) +
        "\nregularization: " + str(config.REGULARIZATION) +
        "\nreport interval: " + str(config.REPORT_INTERVAL) +
        "\nsave interval: " + str(config.SAVE_INTERVAL) +
        "\nModel: flow_model"
    )

    dataset = load(config.TEST_SIZE, config.VAL_SIZE)
    train_logger.info("Dataset loaded.")

    def __epoch_evaluation(self, sub_epoch):
      if sub_epoch % config.SAVE_INTERVAL == 0:
        loss, acc, tpr, fpr, summary = self.evaluate(
            dataset['train']['X'], dataset['train']['Y'], prefix="train")
        self.logger.info(
            "Epoch: %f has train loss: %f, train accuracy: %f, \
             TPR: %s, FPR: %s" % (sub_epoch, loss, acc, str(tpr), str(fpr)))
        self.train_writer.add_summary(summary, global_step=sub_epoch)

      loss, acc, tpr, fpr, summary = self.evaluate(
          dataset['test']['X'], dataset['test']['Y'], prefix="test")
      self.logger.info(
          "Epoch: %f has test loss: %f, test accuracy: %f, \
           TPR: %s, FPR: %s" % (sub_epoch, loss, acc, str(tpr), str(fpr)))
      self.test_writer.add_summary(summary, global_step=sub_epoch)

    model.build_model()
    model.initialize()
    model.train(
        dataset['train']['X'],
        dataset['train']['Y'],
        __epoch_evaluation
    )
    train_logger.info("Model trained.")
    model.save()
    train_logger.info("Model saved.")


if __name__ == "__main__":
  train()


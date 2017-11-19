from ...src.models import FlowAttModel
from ...datasets.iscx import load_full_test
from . import config
from .logger import eval_logger
import tensorflow as tf


def evaluate():
  with tf.Session() as sess:
    model = FlowAttModel(sess, config, eval_logger)
    eval_logger.info("Model created.")

    eval_logger.info(
        "Parameters: " +
        "\nNumber of iterations: " + str(config.ITERATIONS) +
        "\nBatch size: " + str(config.BATCH_SIZE) +
        "\nTest size: " + str(config.TEST_SIZE) +
        "\nN_features: " + str(config.N_FEATURES) +
        "\nN_steps: " + str(config.N_STEPS) +
        "\nLayers: " + str(config.LAYERS) +
        "\nn_classes: " + str(config.N_CLASSES) +
        "\nRegularization: " + str(config.REGULARIZATION) +
        "\nReport interval: " + str(config.REPORT_INTERVAL) +
        "\nSave interval: " + str(config.SAVE_INTERVAL) +
        "\nModel: flow_att_model"
    )

    dataset = load_full_test()
    eval_logger.info("Dataset loaded.")

    model.build_model()
    model.initialize()
    model.restore()

    loss, acc, tpr, fpr, summary = self.evaluate(
        dataset[0], dataset[1], prefix="test")
    print("Loss:", loss)
    self.logger.info(
        "Test loss: %f, test accuracy: %f, \
         TPR: %s, FPR: %s" % (loss, acc, str(tpr), str(fpr)))


if __name__ == "__main__":
  evaluate()


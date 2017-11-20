from ...src.models import FlowAttModel, FlowModel
from ...datasets.iscx import load
from .logger import train_logger
import tensorflow as tf
from . import config


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("s_batch", 32, "Batch size")
tf.app.flags.DEFINE_integer("s_test", 4096, "Size of test set")
tf.app.flags.DEFINE_integer("n_iterations", 100, "Number of iterations")
tf.app.flags.DEFINE_integer("n_features", 77, "Number of features")
tf.app.flags.DEFINE_integer("n_steps", 22, "Number of steps in input sequence")

tf.app.flags.DEFINE_integer("h_gru", 64, "Hidden units in GRU layer")
tf.app.flags.DEFINE_integer("h_att", 16, "Hidden units in attention mechanism")
tf.app.flags.DEFINE_integer("o_gru", 64, "Output units in GRU layer")
tf.app.flags.DEFINE_integer("h_dense", 64, "Hidden units in first dense layer")
tf.app.flags.DEFINE_integer("o_dense", 32, "Output units in first dense layer")
tf.app.flags.DEFINE_integer("h_dense2", 32, "Hidden units in second dense layer")
tf.app.flags.DEFINE_integer("o_dense2", 16, "Output units in second dense layer")
tf.app.flags.DEFINE_integer("n_classes", 2, "Number of label classes")

tf.app.flags.DEFINE_float("v_regularization", 0.15, "Value of regularization term")
tf.app.flags.DEFINE_integer("s_report_interval", 2000, "Number of epochs per report cycle")
tf.app.flags.DEFINE_integer("s_save_interval", 1000, "Number of epochs per save cycle")
0
tf.app.flags.DEFINE_string("model_name", "FlowAttModel", "FlowAttModel/FlowModel")

tf.app.flags.DEFINE_string("SAVES_DIR", config.SAVES_DIR, "Save directory")
tf.app.flags.DEFINE_string("GRAPHS_TRAIN_DIR", config.GRAPHS_TRAIN_DIR, "Graph train directory")
tf.app.flags.DEFINE_string("GRAPHS_TEST_DIR", config.GRAPHS_TEST_DIR, "Graph test directory")
tf.app.flags.DEFINE_string("CHECKPOINTS_DIR", config.CHECKPOINTS_DIR, "Checkpoints directory")


def train():
  with tf.Session() as sess:
    # Create model depending on spec.
    if FLAGS.model_name.lower() == "flowattmodel":
      model = FlowAttModel(sess, FLAGS, train_logger)
    elif FLAGS.model_name.lower() == "flowmodel":
      model = FlowModel(sess, FLAGS, train_logger)
    else:
      raise ValueError("No valid model spec.")
    train_logger.info("Model created.")

    # Log provided hyperparameters.
    param_desc = ""
    for name, val in FLAGS.__dict__['__flags'].items():
      param_desc += "\n" + name + str(val)
    train_logger.info("Parameters: " + param_desc)

    # Load dataset
    train_dataset, test_dataset = load(FLAGS.s_test)
    train_logger.info("Dataset loaded.")

    # Define epoch_eval func.
    def __epoch_evaluation(self, sub_epoch):
      # Evaluate on subset of training dataset.
      loss, acc, tpr, fpr, summary = self.evaluate(
          train_dataset[0][:FLAGS.s_test],
          train_dataset[1][:FLAGS.s_test],
          prefix="train"
      )
      self.logger.info(
          "Epoch: %f has train loss: %f, train accuracy: %f, \
           TPR: %s, FPR: %s" % (sub_epoch, loss, acc, str(tpr), str(fpr)))
      self.train_writer.add_summary(summary, global_step=sub_epoch)

      # Evaluate on subset of testing dataset.
      loss, acc, tpr, fpr, summary = self.evaluate(
          test_dataset[0], test_dataset[1], prefix="test")
      print("Loss:", loss)
      self.logger.info(
          "Epoch: %f has test loss: %f, test accuracy: %f, \
           TPR: %s, FPR: %s" % (sub_epoch, loss, acc, str(tpr), str(fpr)))
      self.test_writer.add_summary(summary, global_step=sub_epoch)

    # Build model
    model.build_model()
    model.initialize()

    # Train model
    model.train(
        train_dataset[0],
        train_dataset[1],
        __epoch_evaluation
    )
    train_logger.info("Model trained.")

    # Save model
    model.save(model.global_step)
    train_logger.info("Model saved.")


if __name__ == "__main__":
  train()


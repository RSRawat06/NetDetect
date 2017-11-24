from ...src.models import FlowAttModel, FlowModel
from ...datasets.isot import load
from .logger import train_logger
from . import config
import tensorflow as tf


def train(FLAGS):
  with tf.Session() as sess:
    ##############################
    ### Create model depending on spec.
    if FLAGS.model_type.lower() == "flowattmodel":
      model = FlowAttModel(sess, FLAGS, train_logger,
                           model_name=FLAGS.model_name)
    elif FLAGS.model_type.lower() == "flowmodel":
      model = FlowModel(sess, FLAGS, train_logger,
                        model_name=FLAGS.model_name)
    else:
      raise ValueError("No valid model spec.")
    train_logger.info("Model created.")
    ##############################

    ##############################
    ### Log provided hyperparameters.
    param_desc = ""
    for name, val in FLAGS.__dict__['__flags'].items():
      param_desc += "\n" + name + ": " + str(val)
    train_logger.info("Parameters: " + param_desc)
    ##############################

    ##############################
    ### Load dataset
    train_dataset, test_dataset = load(FLAGS.s_test, FLAGS.n_steps)
    train_logger.info("Dataset loaded.")
    ##############################

    ##############################
    ### Build model
    model.initialize()
    ##############################

    ##############################
    ### Define epoch_eval func.
    def __epoch_evaluation(self, iteration):
      # Evaluate on subset of training dataset.
      loss, acc, tpr, fpr, summary = self.evaluate(
          train_dataset[0][:FLAGS.s_test],
          train_dataset[1][:FLAGS.s_test],
          prefix="train"
      )
      self.logger.info(
          "Epoch: %f has train loss: %f, train accuracy: %f, \
           TPR: %s, FPR: %s" % (iteration, loss, acc, str(tpr), str(fpr)))
      self.train_writer.add_summary(summary, global_step=self.global_step)

      # Evaluate on subset of testing dataset.
      loss, acc, tpr, fpr, summary = self.evaluate(
          test_dataset[0], test_dataset[1], prefix="test")
      print("Loss:", loss)
      self.logger.info(
          "Epoch: %f has test loss: %f, test accuracy: %f, \
           TPR: %s, FPR: %s" % (iteration, loss, acc, str(tpr), str(fpr)))
      self.test_writer.add_summary(summary, global_step=self.global_step)

      # Determine min acc or save
      if self.min_acc is None:
        self.min_acc = acc
      elif (acc > self.min_acc):
        self.min_acc = acc
        self.save(self.global_step)

    ##############################

    ##############################
    ### Train model
    model.train(
        train_dataset[0],
        train_dataset[1],
        __epoch_evaluation
    )
    train_logger.info("Model trained.")
    ##############################

    ##############################
    ### Evaluate
    loss, acc, tpr, fpr, summary = model.evaluate(
        test_dataset[0], test_dataset[1], prefix="test")
    print("Total test loss:", loss)
    train_logger.info(
        "Total test loss: %f, total test accuracy: %f, \
         TPR: %s, FPR: %s" % (loss, acc, str(tpr), str(fpr)))
    ##############################


if __name__ == "__main__":
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string("model_name", "default.model",
                             "Name of model to be used in logs.")
  tf.app.flags.DEFINE_string("model_type", "FlowAttModel",
                             "FlowAttModel/FlowModel")
  tf.app.flags.DEFINE_integer("s_batch", 128,
                              "Size of batches")
  tf.app.flags.DEFINE_float("v_regularization", 0.1,
                            "Value of regularization term")

  tf.app.flags.DEFINE_integer("n_features", 77,
                              "Number of features")
  tf.app.flags.DEFINE_integer("n_steps", 22,
                              "Number of steps in input sequence")

  tf.app.flags.DEFINE_integer("h_gru", 64,
                              "Hidden units in GRU layer")
  tf.app.flags.DEFINE_integer("h_att", 16,
                              "Hidden units in attention mechanism")
  tf.app.flags.DEFINE_integer("o_gru", 64,
                              "Output units in GRU layer")
  tf.app.flags.DEFINE_integer("h_dense", 64,
                              "Hidden units in first dense layer")
  tf.app.flags.DEFINE_integer("o_dense", 32,
                              "Output units in first dense layer")
  tf.app.flags.DEFINE_integer("h_dense2", 32,
                              "Hidden units in second dense layer")
  tf.app.flags.DEFINE_integer("o_dense2", 16,
                              "Output units in second dense layer")
  tf.app.flags.DEFINE_integer("n_classes", 2,
                              "Number of label classes")

  tf.app.flags.DEFINE_integer("n_epochs", 10,
                              "Number of iterations")
  tf.app.flags.DEFINE_integer("s_test", 4096,
                              "Size of test set")
  tf.app.flags.DEFINE_integer("s_report_interval", 2000,
                              "Number of epochs per report cycle")

  tf.app.flags.DEFINE_string("graphs_train_dir", config.GRAPHS_TRAIN_DIR,
                             "Graph train directory")
  tf.app.flags.DEFINE_string("graphs_test_dir", config.GRAPHS_TEST_DIR,
                             "Graph test directory")
  tf.app.flags.DEFINE_string("checkpoints_dir", config.CHECKPOINTS_DIR,
                             "Checkpoints directory")

  train(FLAGS)



from ...src.models import FlowAttModel, FlowModel
from ...credentials import azure_account_name, azure_account_key
from ...datasets import iscx, isot
from .logger import train_logger
from . import config
from azure.storage.blob import BlockBlobService
import tensorflow as tf


def train(FLAGS):
  with tf.Session() as sess:
    tf.set_random_seed(1)

    ##############################
    ### Log hyperparameters.
    param_desc = FLAGS.model_name + ":   "
    for flag, val in FLAGS.__dict__['__flags'].items():
      param_desc += flag + ": " + str(val) + "; "
    train_logger.debug("Parameters " + param_desc)
    ##############################

    ##############################
    ### Instantiate model.
    ### Valid specs: flowattmodel, flowmodel.
    if FLAGS.model_type.lower() == "flowattmodel":
      model = FlowAttModel(sess, FLAGS, train_logger,
                           model_name=FLAGS.model_name)
    elif FLAGS.model_type.lower() == "flowmodel":
      model = FlowModel(sess, FLAGS, train_logger,
                        model_name=FLAGS.model_name)
    else:
      raise ValueError("Invalid model type.")
    ##############################

    ##############################
    ### Load dataset.
    ### Valid specs: iscx, isot.
    if FLAGS.dataset.lower() == "iscx":
      train_dataset, test_dataset = iscx.load(FLAGS.s_test, FLAGS.n_steps)
    elif FLAGS.dataset.lower() == "isot":
      train_dataset, test_dataset = isot.load(FLAGS.s_test, FLAGS.n_steps)
    else:
      raise ValueError("Invalid dataset.")
    ##############################

    ##############################
    ### Build model
    model.initialize()
    ##############################

    ##############################
    ### Define report func.
    def __report_func(self, iteration):
      # Evaluate on subset of training dataset.
      loss, acc, tpr, fpr, summary = self.evaluate(
          train_dataset[0][:FLAGS.s_test],
          train_dataset[1][:FLAGS.s_test],
          prefix="train"
      )
      self.logger.info(
          FLAGS.model_name +
          "; iteration: %f, train loss: %f, train accuracy: %f, "
          "train TPR: %s, train FPR: %s"
          % (iteration, loss, acc, str(tpr), str(fpr)))
      self.train_writer.add_summary(summary, global_step=iteration)

      # Evaluate on subset of testing dataset.
      loss, acc, tpr, fpr, summary = self.evaluate(
          test_dataset[0], test_dataset[1], prefix="test")
      self.logger.info(
          FLAGS.model_name +
          "; iteration: %f, test loss: %f, test accuracy: %f, "
          "test TPR: %s, test FPR: %s"
          % (iteration, loss, acc, str(tpr), str(fpr)))
      self.test_writer.add_summary(summary, global_step=iteration)

      # Determine min acc or save
      if self.min_acc is None:
        self.min_acc = acc
        self.min_tpr = tpr
        self.min_fpr = fpr
        self.min_iter = iteration
      elif (acc > self.min_acc):
        self.min_acc = acc
        self.min_tpr = tpr
        self.min_fpr = fpr
        self.min_iter = iteration
        self.save(iteration)
    ##############################

    ##############################
    ### Train model
    model.train(
        train_dataset[0],
        train_dataset[1],
        __report_func
    )
    print(FLAGS.model_name + ": training complete.")
    print(
        "Best test accuracy: %f, "
        "test TPR: %s, test FPR: %s"
        % (model.min_acc, str(model.min_tpr), str(model.min_fpr)))
    ##############################

    ##############################
    ### Upload model
    print("Location of best save:", FLAGS.checkpoints_dir + FLAGS.model_name +
          "-" + str(model.min_iter))
    block_blob_service = BlockBlobService(
        account_name=azure_account_name,
        account_key=azure_account_key
    )
    for suffix in [".meta", ".index", ".data-00000-of-00001"]:
      block_blob_service.create_blob_from_path(
          "models",
          FLAGS.model_name,
          FLAGS.checkpoints_dir + FLAGS.model_name + "-" + str(model.min_iter) + suffix
      )
    ##############################


if __name__ == "__main__":
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string("dataset", "blank",
                             "Which dataset to use: iscx/isot")
  tf.app.flags.DEFINE_string("model_name", "default",
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


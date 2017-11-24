from ...src.models import FlowAttModel, FlowModel
from ...datasets import iscx, isot
from .logger import eval_logger
from . import config
import tensorflow as tf


def evaluate(FLAGS):
  with tf.Session() as sess:
    ##############################
    ### Log hyperparameters.
    param_desc = FLAGS.model_name + ":   "
    for flag, val in FLAGS.__dict__['__flags'].items():
      param_desc += flag + ": " + str(val) + "; "
    eval_logger.debug("Parameters " + param_desc)
    ##############################

    ##############################
    ### Instantiate model.
    ### Valid specs: flowattmodel, flowmodel.
    if FLAGS.model_type.lower() == "flowattmodel":
      model = FlowAttModel(sess, FLAGS, eval_logger,
                           model_name=FLAGS.model_name)
    elif FLAGS.model_type.lower() == "flowmodel":
      model = FlowModel(sess, FLAGS, eval_logger,
                        model_name=FLAGS.model_name)
    else:
      raise ValueError("Invalid model type.")
    ##############################

    ##############################
    ### Load dataset.
    ### Valid specs: iscx, isot.
    if FLAGS.dataset.lower() == "iscx":
      test_dataset = iscx.load_full_test(FLAGS.n_steps)
    elif FLAGS.dataset.lower() == "isot":
      test_dataset = isot.load_full_test(FLAGS.n_steps)
    else:
      raise ValueError("Invalid dataset.")
    ##############################

    ##############################
    ### Build model
    model.initialize()
    model.restore()
    ##############################

    ##############################
    ### Evaluate
    loss, acc, tpr, fpr, summary = model.evaluate(
        test_dataset[0], test_dataset[1], prefix="test")
    eval_logger.info(
        FLAGS.model_name +
        "; loss: %f, accuracy: %f, TPR: %s, FPR: %s"
        % (loss, acc, str(tpr), str(fpr)))
    print(FLAGS.model_name + ": testing complete.")
    ##############################


if __name__ == "__main__":
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string("dataset", "blank",
                             "Which dataset to use: iscx/isot")
  tf.app.flags.DEFINE_string("model_name", "default.model",
                             "Name of model to be used in logs.")
  tf.app.flags.DEFINE_string("model_type", "FlowAttModel",
                             "FlowAttModel/FlowModel")
  tf.app.flags.DEFINE_integer("s_batch", 32,
                              "Size of batches")
  tf.app.flags.DEFINE_float("v_regularization", 0.15,
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

  tf.app.flags.DEFINE_string("graphs_train_dir", config.GRAPHS_TRAIN_DIR,
                             "Graph train directory")
  tf.app.flags.DEFINE_string("graphs_test_dir", config.GRAPHS_TEST_DIR,
                             "Graph test directory")
  tf.app.flags.DEFINE_string("checkpoints_dir", config.CHECKPOINTS_DIR,
                             "Checkpoints directory")

  evaluate(FLAGS)


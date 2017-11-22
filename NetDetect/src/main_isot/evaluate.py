from ...src.models import FlowAttModel, FlowModel
from ...datasets.isot import load_full_test
from .logger import eval_logger
from . import config
import tensorflow as tf


def evaluate():
  with tf.Session() as sess:
    ##############################
    ### Create model depending on spec.
    if FLAGS.model_name.lower() == "flowattmodel":
      model = FlowAttModel(sess, FLAGS, eval_logger)
    elif FLAGS.model_name.lower() == "flowmodel":
      model = FlowModel(sess, FLAGS, eval_logger)
    else:
      raise ValueError("No valid model spec.")
    eval_logger.info("Model created.")
    ##############################

    ##############################
    ### Log provided hyperparameters.
    param_desc = ""
    for name, val in FLAGS.__dict__['__flags'].items():
      param_desc += "\n" + name + ": " + str(val)
    eval_logger.info("Parameters: " + param_desc)
    ##############################

    ##############################
    ### Load dataset
    test_dataset = load_full_test(FLAGS.n_steps)
    eval_logger.info("Dataset loaded.")
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
    print("Total test loss:", loss)
    eval_logger.info(
        "Total test loss: %f, total test accuracy: %f, \
         TPR: %s, FPR: %s" % (loss, acc, str(tpr), str(fpr)))
    ##############################


if __name__ == "__main__":
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string("model_name", "FlowAttModel",
                             "FlowAttModel/FlowModel")
  tf.app.flags.DEFINE_integer("s_batch", 32,
                              "Size of batches")
  tf.app.flags.DEFINE_float("v_regularization", 0.15,
                            "Value of regularization term")

  tf.app.flags.DEFINE_integer("n_features", 13,
                              "Number of features")
  tf.app.flags.DEFINE_integer("n_steps", 22,
                              "Number of steps in input sequence")

  tf.app.flags.DEFINE_integer("h_gru", 16,
                              "Hidden units in GRU layer")
  tf.app.flags.DEFINE_integer("h_att", 8,
                              "Hidden units in attention mechanism")
  tf.app.flags.DEFINE_integer("o_gru", 16,
                              "Output units in GRU layer")
  tf.app.flags.DEFINE_integer("h_dense", 8,
                              "Hidden units in first dense layer")
  tf.app.flags.DEFINE_integer("o_dense", 8,
                              "Output units in first dense layer")
  tf.app.flags.DEFINE_integer("h_dense2", 8,
                              "Hidden units in second dense layer")
  tf.app.flags.DEFINE_integer("o_dense2", 8,
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


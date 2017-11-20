from ...src.models import FlowAttModel, FlowModel
from ...datasets.iscx import load_full_test
from .logger import eval_logger
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("s_batch", 32)
tf.app.flags.DEFINE_integer("n_features", 77)
tf.app.flags.DEFINE_integer("n_steps", 22)

tf.app.flags.DEFINE_integer("h_gru", 64)
tf.app.flags.DEFINE_integer("h_att", 16)
tf.app.flags.DEFINE_integer("o_gru", 64)
tf.app.flags.DEFINE_integer("h_dense", 64)
tf.app.flags.DEFINE_integer("o_dense", 32)
tf.app.flags.DEFINE_integer("h_dense2", 32)
tf.app.flags.DEFINE_integer("o_dense2", 16)
tf.app.flags.DEFINE_integer("n_classes", 2)

tf.app.flags.DEFINE_integer("v_regularization", 0.15)
tf.app.flags.DEFINE_string("model_name", "FlowAttModel")


def evaluate():
  with tf.Session() as sess:
    # Create model depending on spec.
    if FLAGS.model_name.lower() == "flowattmodel":
      model = FlowAttModel(sess, FLAGS, eval_logger)
    elif FLAGS.model_name.lower() == "flowmodel":
      model = FlowModel(sess, FLAGS, eval_logger)
    else:
      raise ValueError("No valid model spec.")
    eval_logger.info("Model created.")

    # Log provided hyperparameters.
    param_desc = ""
    for name, val in FLAGS.__dict__['__flags'].items():
      param_desc += "\n" + name + str(val)
    eval_logger.info("Parameters: " + param_desc)

    # Load dataset
    dataset = load_full_test()
    eval_logger.info("Dataset loaded.")

    # Build model
    model.build_model()
    model.initialize()

    # Restore
    model.restore()

    # Evaluate
    loss, acc, tpr, fpr, summary = model.evaluate(
        dataset[0], dataset[1], prefix="test")
    print("Loss:", loss)
    eval_logger.info(
        "Test loss: %f, test accuracy: %f, \
         TPR: %s, FPR: %s" % (loss, acc, str(tpr), str(fpr)))


if __name__ == "__main__":
  evaluate()


import tensorflow as tf


class StandardLayers():
  '''
  Standard TF components.
  '''

  def _prediction_layer(self, X, var_scope, config):
    '''
    Predicts end result
    Args:
      - X: input data of shape (batch, features)
      - var_scope: string name of tf variable scope.
      - config {
          'n_batches': number of batches,
          'n_input': number of input features,
          'n_classes': number of potential output classes
        }
    '''

    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'], config['n_input']))

    with tf.variable_scope(var_scope):
      W = tf.get_variable("W", shape=(config['n_input'], config['n_classes']))
      b = tf.get_variable("bias", shape=(config['n_classes']))
      prediction = tf.nn.softmax(tf.matmul(X, W) + b, name="prediction")
      assert(prediction.shape == (config['n_batches'], config['n_classes']))

      return prediction

  def _define_optimization_vars(self, target, prediction,
                                result_weights, regularization):
    '''
    Defines loss, optim, and various metrics to tarck training progress.
    Args:
      - target - correct labels of shape (batch, classes).
      - prediction - predictions of shape (batch, classes).
      - result_weights - array indicating how much to weight loss for each
                         class, ex: [1, 5].
      - regularization (float) - l2 regularization constant.
    Return:
      - loss (tf.float32): regularized loss for pred/target.
      - acc (tf.float32): decimal accuracy.
    '''

    with tf.variable_scope('optimization'):
      regularization_loss = tf.add_n([
          tf.nn.l2_loss(v) for v in tf.trainable_variables()
          if 'bias' not in v.name.lower()
      ]) * tf.constant(regularization, dtype=tf.float32)

      loss = regularization_loss - tf.reduce_sum(
          target * tf.log(prediction + 1e-10) *
          tf.constant(result_weights, dtype=tf.float32),
          name="loss"
      )

      return loss

  def _define_binary_metrics(self, target, prediction):
    '''
    Defines binary recall/precision metrics.
    Args:
      - target - correct labels of shape (batch, classes).
      - prediction - predictions of shape (batch, classes).
    Return:
      - TPR (tf.float32): true positive rate.
      - FPR (tf.float32): false positive rate.
    '''

    with tf.variable_scope('binary'):
      ones_target = tf.ones_like(tf.argmax(target, 1))
      zeros_target = tf.zeros_like(tf.argmax(target, 1))
      ones_prediction = tf.ones_like(tf.argmax(prediction, 1))
      zeros_prediction = tf.zeros_like(tf.argmax(prediction, 1))

      TN = tf.reduce_sum(
          tf.cast(
              tf.logical_and(
                  tf.equal(tf.argmax(prediction, 1), zeros_prediction),
                  tf.equal(tf.argmax(target, 1), zeros_target)
              ),
              tf.float32
          )
      )
      FN = tf.reduce_sum(
          tf.cast(
              tf.logical_and(
                  tf.equal(tf.argmax(prediction, 1), zeros_prediction),
                  tf.equal(tf.argmax(target, 1), ones_target)
              ),
              tf.float32
          )
      )
      TP = tf.reduce_sum(
          tf.cast(
              tf.logical_and(
                  tf.equal(tf.argmax(prediction, 1), ones_prediction),
                  tf.equal(tf.argmax(target, 1), ones_target)
              ),
              tf.float32
          )
      )
      FP = tf.reduce_sum(
          tf.cast(
              tf.logical_and(
                  tf.equal(tf.argmax(prediction, 1), ones_prediction),
                  tf.equal(tf.argmax(target, 1), zeros_target)
              ),
              tf.float32
          )
      )

      tpr = tf.divide(
          tf.cast(TP, tf.float32),
          tf.cast(TP, tf.float32) + tf.cast(FN, tf.float32),
          name="true_positive_rate"
      )
      fpr = tf.divide(
          tf.cast(FP, tf.float32),
          tf.cast(FP, tf.float32) + tf.cast(TN, tf.float32),
          name="false_positive_rate"
      )

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
      acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

      return tpr, fpr, acc

  def _instant_summaries(self):
    '''
    Define summaries for tensorboard use.
    '''

    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)
      tf.summary.scalar("accuracy", self.acc)
      try:
        tf.summary.scalar("true_positive_rate", self.tpr)
        tf.summary.scalar("false_positive_rate", self.fpr)
      except AttributeError:
        pass
      summary_op = tf.summary.merge_all()

      return summary_op


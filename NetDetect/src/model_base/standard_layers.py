import tensorflow as tf


class StandardLayers():
  '''
  Standard TF components.
  '''

  def _prediction_layer(self, X, var_scope, config):
    '''
    Predicts end result.
    Args:
      X - input data of shape (batch, features).
      var_scope - string name of tf variable scope.
      config {
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

  def _dense_layer(self, X, var_scope, config):
    '''
    Predicts end result.
    Args:
      X - input data of shape (batch, features).
      var_scope - string name of tf variable scope.
      config {
          'n_batches': number of batches,
          'n_input': number of input features,
          'n_hidden': number of hidden units,
          'n_output': number of potential output classes
        }
    '''

    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'], config['n_input']))

    with tf.variable_scope(var_scope):
      W_1 = tf.get_variable("W_1", shape=(config['n_input'],
                                          config['n_hidden']))
      b_1 = tf.get_variable("bias_1", shape=(config['n_hidden']))
      A = tf.tanh(tf.matmul(X, W_1) + b_1, name="A")

      W_2 = tf.get_variable("W_2", shape=(config['n_hidden'],
                                          config['n_output']))
      b_2 = tf.get_variable("bias_2", shape=(config['n_output']))
      output = tf.tanh(tf.matmul(A, W_2) + b_2, name="output")

      assert(output.shape == (config['n_batches'], config['n_output']))
      return output

  def _define_optimization_vars(self, target, prediction, result_weights=None):
    '''
    Defines loss, optim, and various metrics to tarck training progress.
    Args:
      - target - correct labels of shape (batch, classes).
      - prediction - predictions of shape (batch, classes).
      - result_weights - array indicating how much to weight loss for each
                         class, ex: [1, 5].
    Return:
      - loss (tf.float32): regularized loss for pred/target.
      - acc (tf.float32): decimal accuracy.
    '''

    with tf.variable_scope('optimization'):
      regularization = tf.add_n([
          tf.nn.l2_loss(v) for v in tf.trainable_variables()
          if 'bias' not in v.name
      ]) * tf.constant(0.01, dtype=tf.float32)

      delta = tf.constant(0.0001, dtype=tf.float32)
      if result_weights is None:
        loss = regularization - tf.reduce_sum(
            target * tf.log(prediction + delta), name="loss"
        )
      else:
        loss = regularization - tf.reduce_sum(
            target * tf.log(prediction + delta) *
            tf.constant(result_weights, dtype=tf.float32),
            name="loss"
        )

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
      acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

      return loss, acc

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

      return tpr, fpr

  def _summaries(self, binary=False):
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


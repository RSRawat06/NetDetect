import tensorflow as tf
import numpy as np
import time
from . import config as model_config

tf.logging.set_verbosity(tf.logging.ERROR)


class Vanilla_GRU():
  def __init__(self, sess, config):
    self.sess = sess
    self.var_init = tf.global_variables_initializers()
    self.saver = tf.train.Saver(tf.global_variables())
    self.config = config

  def save(self):
    self.saver.save(self.sess, model_config.MODEL_DIR + model_config.VANILLA_MODEL_NAME)

  def load(self, model_url):
    self.saver.restore(self.sess, model_config.MODEL_DIR + model_config.VANILLA_MODEL_NAME)

  def build_model(self):
    # Set initial vars
    self.x = tf.placeholder(tf.float32, [self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES])
    self.target = tf.placeholder(tf.float32, [self.config.N_BATCHES, 1], name="target")

    ##############################

    self.fwd_gru_p = tf.nn.rnn_cell.BasicGRUCell(self.config.N_PACKET_GRU_HIDDEN)
    self.bwd_gru_p = tf.nn.rnn_cell.BasicGRUCell(self.config.N_PACKET_GRU_HIDDEN)

    _, self.O_fwd_p, self.O_bwd_p = tf.nn.rnn.static_bidirectional_rnn(self.fwd_gru_p, self.bwd_gru_p, self.x)
    assert(self.O_fwd_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_PACKET_GRU_HIDDEN))
    assert(self.O_bwd_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_PACKET_GRU_HIDDEN))

    self.W_p = tf.get_variable("W_p", shape=[2 * self.config.N_PACKET_GRU_HIDDEN, self.config.N_PACKET_DENSE_HIDDEN])
    self.A_p = tf.tanh(tf.matmul(tf.concatenate(self.O_fwd_p, self.O_bwd_p), self.W_p))
    assert(self.A_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN))

    ##############################

    self.fwd_gru_f = tf.nn.rnn_cell.BasicGRUCell(self.config.N_FLOW_GRU_HIDDEN)
    self.bwd_gru_f = tf.nn.rnn_cell.BasicGRUCell(self.config.N_FLOW_GRU_HIDDEN)

    _, self.O_fwd_f, self.O_bwd_f = tf.nn.rnn.static_bidirectional_rnn(self.fwd_gru_f, self.bwd_gru_f, self.A_p)
    assert(self.O_fwd_f.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_FLOW_GRU_HIDDEN))
    assert(self.O_bwd_f.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_FLOW_GRU_HIDDEN))

    self.W_f = tf.get_variable("W_p", shape=[2 * self.config.N_FLOW_GRU_HIDDEN, self.config.N_FLOW_DENSE_HIDDEN])
    self.A_f = tf.tanh(tf.matmul(tf.concatenate(self.O_fwd_f, self.O_bwd_f), self.W_f))
    assert(self.A_f.shape == (self.config.N_BATCHES, self.config.N_FLOW_DENSE_HIDDEN))

    ##############################

    self.W_final = tf.get_variable("W_final", shape=[self.config.N_FLOW_DENSE_HIDDEN, 2])
    self.prediction = tf.nn.softmax(tf.matmul(self.A_f, self.W_final))
    assert(self.prediction.shape == (self.config.N_BATCHES, 2))

    self.loss = -tf.reduce_sum(self.target * tf.log(self.prediction) * tf.constant(self.config.RESULT_WEIGHTING))

    ##############################

    # Number of correct, not normalized
    correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.target, 1))
    # Accuracy
    self.acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    # Setting optimizers
    self.optimizer = tf.train.AdamOptimizer()
    self.optim = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())

  def train(self, training_data, testing_data):
    self.var_init.run()
    for j in range(self.config.ITERATIONS):
      start_time = time.time()

      for i in range(0, self.config.N_BATCHES, len(training_data['targets'])):
        feed_dict = {
            self.x: training_data['X'][i:i + self.config.N_BATCHES],
            self.target: training_data['Y'][i:i + self.config.N_BATCHES]
        }
        __, train_loss, train_acc = self.sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)
        print("Train loss: ", train_loss, "\nTrain acc on train: ", train_acc)

      total_testing_acc = []
      for i in range(0, self.config.N_BATCHES, len(testing_data['targets'])):
        feed_dict = {
            self.x: testing_data['X'][i:i + self.config.N_BATCHES],
            self.target: testing_data['Y'][i:i + self.config.N_BATCHES]
        }
        testing_acc = self.sess.run([self.acc], feed_dict=feed_dict)
        total_testing_acc.append(testing_acc)

      print("Testing acc on test: ", np.mean(total_testing_acc))
      elapsed_time = time.time() - start_time
      print("Iteration", j, "took: ", elapsed_time)

  def predict(self, input_data):
    self.var_init.run()
    all_predictions = []
    for i in range(0, self.config.N_BATCHES, len(input_data)):
      feed_dict = {
          self.x: input_data[i:i + self.config.N_BATCHES],
      }
      predictions = self.sess.run([self.prediction], feed_dict=feed_dict)
      all_predictions += list(predictions)
    return all_predictions


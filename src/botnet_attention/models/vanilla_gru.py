import tensorflow as tf
import numpy as np
import time
from . import config

tf.logging.set_verbosity(tf.logging.ERROR)
ITERATIONS, RESULT_WEIGHTING, N_BATCHES, N_FLOWS, N_PACKETS, N_FEATURES = config['basic']
N_HIDDEN_P, N_HIDDEN_M_P = config['packets']
N_HIDDEN_F, N_HIDDEN_A_F, R_F = config['flows']


class Self_Attention():
  def __init__(self, sess):
    self.sess = sess

  def save(self):
    self.saver.save(self.sess, config.MODEL_DIR + config.VANILLA_MODEL_NAME)

  def load(self, model_url):
    self.saver = tf.train.import_meta_graph(config.MODEL_DIR + config.VANILLA_META_NAME)
    self.saver.restore(self.sess, tf.train.latest_checkpoint(config.MODEL_DIR))
    self.sess.run(tf.global_variables_initializer())

  def build_model(self):
    # Set initial vars
    self.x = tf.placeholder(tf.float32, [N_BATCHES, N_FLOWS, N_PACKETS, N_FEATURES])
    self.target = tf.placeholder(tf.float32, [N_BATCHES, 1], name="target")

    ##############################

    self.fwd_lstm_p = tf.nn.rnn_cell.BasicGRUCell(N_HIDDEN_P)
    self.bwd_lstm_p = tf.nn.rnn_cell.BasicGRUCell(N_HIDDEN_P)

    _, self.O_fwd_p, self.O_bwd_p = tf.nn.rnn.static_bidirectional_rnn(self.fwd_lstm_p, self.bwd_lstm_p, self.x)
    # Shape: (n_batches, n_flows, u), (n_batches, n_flows, u)

    self.W_p = tf.get_variable("W_p", shape=[2 * N_HIDDEN_P, N_HIDDEN_M_P])
    # Shape: (2u, h)

    self.M_p = tf.tanh(tf.matmul(tf.concatenate(self.O_fwd_p, self.O_bwd_p), self.W_p))
    # Shape: (n_batches, n_flows, h)

    ##############################

    self.fwd_lstm_f = tf.nn.rnn_cell.BasicGRUCell(N_HIDDEN_F)
    self.bwd_lstm_f = tf.nn.rnn_cell.BasicGRUCell(N_HIDDEN_F)

    _, self.O_fwd_f, self.O_bwd_f = tf.nn.rnn.static_bidirectional_rnn(self.fwd_lstm_f, self.bwd_lstm_f, self.M_p)
    # Shape: (n_batches, u), (n_batches, u)

    self.W_f = tf.get_variable("W_p", shape=[2 * N_HIDDEN_F, N_HIDDEN_M_F])
    # Shape: (2u, h_f)

    self.M_f = tf.tanh(tf.matmul(tf.concatenate(self.O_fwd_f, self.O_bwd_f), self.W_f))
    # Shape: (n_batches, h_f)

    ##############################

    self.W_final = tf.get_variable("W_final", shape=[h_f])
    self.prediction = tf.nn.softmax(tf.dot(self.M_f, self.W_final))

    self.standard_loss = self.target * tf.log(self.prediction) * tf.constant(RESULT_WEIGHTING)
    self.packet_reg_weight = tf.constant(0.1)
    self.packet_attention_regularization = self.packet_reg_weight * tf.pow(tf.norm(tf.matmul(self.A_p, tf.transpose(self.A_p)) - tf.identity(R_P)), 0)
    self.flow_reg_weight = tf.constant(0.1)
    self.flow_attention_regularization = self.flow_reg_weight * tf.pow(tf.norm(tf.matmul(self.A_f, tf.transpose(self.A_f)) - tf.identity(R_F)), 2)

    self.loss = -tf.reduce_sum(self.standard_loss + self.packet_attention_regularization + self.flow_attention_regularization)

    ##############################

    # Number of correct, not normalized
    correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.target, 1))
    # Accuracy
    self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

    # Setting optimizers
    self.optimizer = tf.train.AdamOptimizer()
    self.optim = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())

  def train(self, training_data, testing_data):

    # Initializing tf + timing
    tf.global_variables_initializers().run()
    self.saver = tf.train.Saver(tf.global_variables())

    for j in range(ITERATIONS):
      print("\n################\nIteration: ", j)
      start_time = time.time()

      # Run through training data
      for i in range(0, N_BATCHES, len(training_data['targets'])):
        feed_dict = {
            self.x: training_data['X'][i:i + N_BATCHES],
            self.target: training_data['Y'][i:i + N_BATCHES]
        }
        __, train_loss, train_acc = self.sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)
        print("Train loss: ", train_loss, "\nTrain acc on train: ", train_acc)

      # Run through testing data
      total_testing_acc = []
      for i in range(0, N_BATCHES, len(testing_data['targets'])):
        feed_dict = {
            self.x: testing_data['X'][i:i + N_BATCHES],
            self.target: testing_data['Y'][i:i + N_BATCHES]
        }
        __, testing_loss, testing_acc = self.sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)
        total_testing_acc.append(testing_acc)

      print("Testing acc on test: ", np.mean(total_testing_acc))
      elapsed_time = time.time() - start_time
      print("Iteration took: ", elapsed_time)

import tensorflow as tf
import numpy as np
import time
from . import config


class Attention_Discriminator():
  def __init__(self, sess, model_config):
    self.sess = sess
    self.config = model_config

  def save(self):
    self.saver.save(self.sess, config.MODEL_DIR + config.VANILLA_MODEL_NAME)

  def load(self, model_url):
    self.saver = tf.train.import_meta_graph(config.MODEL_DIR + config.VANILLA_META_NAME)
    self.saver.restore(self.sess, tf.train.latest_checkpoint(config.MODEL_DIR))
    self.sess.run(tf.global_variables_initializer())

  def build_model(self):
    self.x = tf.placeholder(tf.float32, [self.config['BATCH_S'], self.config['MAX_PACKET_SEQUENCE_LENGTH'], self.config['N_FEATURES']], name="horizontal")
    self.target = tf.placeholder(tf.float32, [self.config['BATCH_S'], 2], name="target")
    # x.shape = [batch, time, input_feat]
    # target.shape = [batch, 2]

    with tf.variable_scope("encode_x"):
      self.fwd_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config['N_HIDDEN'], state_is_tuple=True)
      self.x_output, self.x_state = tf.nn.dynamic_rnn(cell=self.fwd_lstm, inputs=self.x, dtype=tf.float32)
      # x_output.shape = [batch, time, output_feat]
      # x_state.shape = [batch, output_feat]

    self.W = tf.get_variable("W", shape=[self.config['N_HIDDEN', 2]])
    # w.shape = [output_feat=N_HIDDEN, 2]
    self.pred = tf.nn.softmax(tf.matmul(self.x_state, self.W))
    # pred.shape = [batch, output_feat] * [output_feat, 2] = [batch, 2]

    self.loss = -tf.reduce_sum(self.target * tf.log(self.pred) * tf.constant(self.config['RESULT_WEIGHTING'], dtype=tf.float32), name="loss")
    # pred.shape = [batch, 2] * [batch, 2] * [batch, 2]

    self.optimizer = tf.train.AdamOptimizer()
    self.optim = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())
    __ = tf.scalar_summary("loss", self.loss)

    correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.target, 1))
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.pred, 1), tf.constant(1, dtype=tf.int64)), tf.equal(tf.argmax(self.target, 1), tf.constant(1, dtype=tf.int64))), "float"))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.pred, 1), tf.constant(0, dtype=tf.int64)), tf.equal(tf.argmax(self.target, 1), tf.constant(0, dtype=tf.int64))), "float"))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.pred, 1), tf.constant(1, dtype=tf.int64)), tf.equal(tf.argmax(self.target, 1), tf.constant(0, dtype=tf.int64))), "float"))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.pred, 1), tf.constant(0, dtype=tf.int64)), tf.equal(tf.argmax(self.target, 1), tf.constant(1, dtype=tf.int64))), "float"))

    self.false_neg_perc = fn / (tp + fn)
    self.false_pos_perc = fp / (tn + fp)
    self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

  def train(self, training_data, testing_data, len_training, len_testing, iterations):
    merged_sum = tf.merge_all_summaries()
    tf.initialize_all_variables().run()
    self.saver = tf.train.Saver()
    print("Starting to train on: ", len_training, "entries")

    for j in range(iterations):
      print("\n################\nIteration: ", j)
      start_time = time.time()
      total_training_acc = []

      # Run through training data
      for i in range(0, len_training, self.config['BATCH_S']):
        if len(training_data['x'][i:i + self.config['BATCH_S']]) != self.config['BATCH_S']:
          # Terminate if we've run out of data points to fulfill a batch
          break
        feed_dict = {
            self.x: np.array(training_data['x'][i:i + self.config['BATCH_S']]),
            self.y: np.array(training_data['y'][i:i + self.config['BATCH_S']]),
            self.target: np.array(training_data['targets'][i:i + self.config['BATCH_S']])
        }

        __, train_loss, train_acc, false_pos, false_neg, summ = self.sess.run([self.optim, self.loss, self.acc, self.false_pos_perc, self.false_neg_perc, merged_sum], feed_dict=feed_dict)
        total_training_acc.append(train_acc)

        if (i % 100000 == 0):
          print(int(100 * i / len_training), "%: train acc: ", train_acc, ", train loss: ", train_loss)
          print("Training fp on train: ", false_pos)
          print("Training fn on train: ", false_neg)

      print("Total training acc: ", np.mean(total_training_acc))

      # Run through testing data
      total_testing_acc = []
      total_fp = []
      total_fn = []
      for i in range(0, len_testing, self.config['BATCH_S']):
        if len(testing_data['x'][i:i + self.config['BATCH_S']]) != self.config['BATCH_S']:
          break
        feed_dict = {
            self.x: testing_data['x'][i:i + self.config['BATCH_S']],
            self.y: testing_data['y'][i:i + self.config['BATCH_S']],
            self.target: testing_data['targets'][i:i + self.config['BATCH_S']]
        }
        testing_loss, testing_acc, false_pos, false_neg, summ = self.sess.run([self.loss, self.acc, self.false_pos_perc, self.false_neg_perc, merged_sum], feed_dict=feed_dict)
        total_testing_acc.append(testing_acc)
        total_fp.append(false_pos)
        total_fn.append(false_neg)
      print("Testing acc on test: ", np.mean(total_testing_acc))
      print("Testing fp on test: ", np.mean(total_fp))
      print("Testing fn on test: ", np.mean(total_fn))

      elapsed_time = time.time() - start_time
      print("Iteration took: ", elapsed_time)


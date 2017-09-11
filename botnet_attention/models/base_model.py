import tensorflow as tf
import numpy as np
import time
from . import config as model_config


class Base_Model():
  def __init__(self, sess, config):
    self.sess = sess
    self.var_init = tf.global_variables_initializer()
    self.config = config
    self.saver = None
    self.model_file = model_config.DEFAULT_MODEL_NAME

  def save(self):
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())
    self.saver.save(self.sess, model_config.MODEL_DIR + self.model_file)

  def load(self):
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())
    self.saver.restore(self.sess, model_config.MODEL_DIR + self.model_file)

  def initialize(self):
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()

  def train(self, training_data, testing_data):
    # self.var_init.run()
    for j in range(self.config.ITERATIONS):
      start_time = time.time()

      for i in range(0, training_data['targets'].shape[0], self.config.BATCH_SIZE):
        feed_dict = {
            self.x: training_data['X'][i:i + self.config.BATCH_SIZE],
            self.target: training_data['Y'][i:i + self.config.BATCH_SIZE]
        }
        __, train_loss, train_acc = self.sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)
        print("Train loss: ", train_loss, "\nTrain acc on train: ", train_acc)

      total_testing_acc = []
      for i in range(0, testing_data['targets'].shape[0], self.config.BATCH_SIZE):
        feed_dict = {
            self.x: testing_data['X'][i:i + self.config.BATCH_SIZE],
            self.target: testing_data['Y'][i:i + self.config.BATCH_SIZE]
        }
        testing_acc = self.sess.run([self.acc], feed_dict=feed_dict)
        total_testing_acc.append(testing_acc)

      print("Testing acc on test: ", np.mean(total_testing_acc))
      elapsed_time = time.time() - start_time
      print("Iteration", j, "took: ", elapsed_time)

  def predict(self, input_data):
    # self.var_init.run()
    all_predictions = []
    for i in range(0, input_data.shape[0], self.config.BATCH_SIZE):
      feed_dict = {
          self.x: input_data[i:i + self.config.BATCH_SIZE],
      }
      predictions = self.sess.run([self.prediction], feed_dict=feed_dict)
      all_predictions += list(predictions)
    return all_predictions


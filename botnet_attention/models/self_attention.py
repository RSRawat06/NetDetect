import tensorflow as tf
import numpy as np
import time
from . import config as model_config
from .base_model import Base_Model

tf.logging.set_verbosity(tf.logging.ERROR)


class Self_Attention(Base_Model):
  def build_model(self):
    # Set initial vars
    self.x = tf.placeholder(tf.float32, [self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES])
    self.target = tf.placeholder(tf.float32, [self.config.N_BATCHES, 1], name="target")

    ##############################

    self.fwd_gru_p = tf.nn.rnn_cell.GRUCell(self.config.N_PACKET_GRU_HIDDEN)
    self.bwd_gru_p = tf.nn.rnn_cell.GRUCell(self.config.N_PACKET_GRU_HIDDEN)

    self.x_flatten = tf.reshape(self.x, (self.config.N_BATCHES * self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES))
    self.H_p_flat, _, _ = tf.nn.static_bidirectional_rnn(self.fwd_gru_p, self.bwd_gru_p, tf.unstack(self.x_flatten), dtype=tf.float32)
    assert(self.H_p_flat.shape == (self.config.N_BATCHES * self.config.N_FLOWS, self.config.N_PACKETS, 2 * self.config.N_PACKET_GRU_HIDDEN))
    self.H_p = tf.reshape(self.H_p_flat, (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS, -1))
    assert(self.H_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS, 2 * self.config.N_PACKET_GRU_HIDDEN))

    self.W_s_1_p = tf.get_variable("W_s_1_p", shape=[self.config.N_PACKET_ATTENTION_HIDDEN, 2 * self.config.N_PACKET_GRU_HIDDEN])
    self.W_s_2_p = tf.get_variable("W_s_2_p", shape=[self.config.N_PACKET_ATTENTION_HIDDEN])

    self.r_p = tf.nn.softmax(tf.matmul(self.W_s_2_p, tf.tanh(tf.matmul(self.W_s_1_p, tf.tranpose(self.H_p, (0, 1, 3, 2))))))
    assert(self.r_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKETS))

    self.M_p = tf.squeeze(tf.matmul(tf.transpose(self.H_p, (0, 1, 3, 2)), tf.expand_dims(self.r_p, 3)))
    assert(self.M_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, 2 * self.config.N_PACKET_GRU_HIDDEN))

    self.W_p = tf.get_variable("W_p", shape=[2 * self.config.N_PACKET_GRU_HIDDEN, self.config.N_PACKET_DENSE_HIDDEN])
    self.A_p = tf.tanh(tf.matmul(self.M_p, self.W_p))
    assert(self.A_p.shape == (self.config.N_BATCHES, self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN))

    ##############################

    self.fwd_gru_f = tf.nn.rnn_cell.GRUCell(self.config.N_FLOW_GRU_HIDDEN)
    self.bwd_gru_f = tf.nn.rnn_cell.GRUCell(self.config.N_FLOW_GRU_HIDDEN)

    self.H_f, _, _ = tf.nn.static_bidirectional_rnn(self.fwd_gru_f, self.bwd_gru_f, self.A_p)
    assert(self.H_f.shape == (self.config.N_BATCHES, self.config.N_FLOWS, 2 * self.config.N_FLOW_GRU_HIDDEN))

    self.W_s_1_f = tf.get_variable("W_s_1_f", shape=[self.config.N_FLOW_ATTENTION_HIDDEN, 2 * self.config.N_FLOW_GRU_HIDDEN])
    self.W_s_2_f = tf.get_variable("W_s_2_f", shape=[self.config.N_FLOW_ATTENTION_HIDDEN])

    self.r_f = tf.nn.softmax(tf.matmul(self.W_s_2_f, tf.tanh(tf.matmul(self.W_s_1_f, tf.tranpose(self.H_f, (0, 2, 1))))))
    assert(self.r_f.shape == (self.config.N_BATCHES, self.config.N_FLOWS))

    self.M_f = tf.squeeze(tf.matmul(tf.transpose(self.H_f, (0, 2, 1)), tf.expand_dims(self.r, 2)))
    assert(self.M_f.shape == (self.config.N_BATCHES, 2 * self.config.N_FLOW_GRU_HIDDEN))

    self.W_f = tf.get_variable("W_f", shape=[2 * self.config.N_FLOW_GRU_HIDDEN, self.config.N_FLOW_DENSE_HIDDEN])
    self.A_f = tf.tanh(tf.matmul(self.M_f, self.W_f))
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

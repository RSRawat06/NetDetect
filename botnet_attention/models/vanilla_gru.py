import tensorflow as tf
from .base_model import Base_Model

tf.logging.set_verbosity(tf.logging.ERROR)


class Vanilla_GRU(Base_Model):
  def build_model(self):
    # Set initial vars
    self.x = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES], name="x")
    self.target = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, 1], name="target")

    ##############################

    self.fwd_gru_p = tf.nn.rnn_cell.GRUCell(self.config.N_PACKET_GRU_HIDDEN)
    self.bwd_gru_p = tf.nn.rnn_cell.GRUCell(self.config.N_PACKET_GRU_HIDDEN)

    self.x_unstacked = tf.unstack(tf.transpose(tf.reshape(self.x, (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES)), (1, 0, 2)), name="x_unstacked")
    _, self.O_fwd_p, self.O_bwd_p = tf.nn.static_bidirectional_rnn(self.fwd_gru_p, self.bwd_gru_p, self.x_unstacked, dtype=tf.float32, scope="packet_rnn")
    assert(self.O_fwd_p.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKET_GRU_HIDDEN))
    assert(self.O_bwd_p.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKET_GRU_HIDDEN))

    self.W_p = tf.get_variable("W_p", shape=[2 * self.config.N_PACKET_GRU_HIDDEN, self.config.N_PACKET_DENSE_HIDDEN], dtype=tf.float32)
    self.A_p_unshaped = tf.tanh(tf.matmul(tf.concat((self.O_fwd_p, self.O_bwd_p), axis=1), self.W_p), name="A_p_unshaped")
    assert(self.A_p_unshaped.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN))
    self.A_p = tf.reshape(self.A_p_unshaped, (self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN), name="A_p")
    assert(self.A_p.shape == (self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN))

    ##############################

    self.fwd_gru_f = tf.nn.rnn_cell.GRUCell(self.config.N_FLOW_GRU_HIDDEN)
    self.bwd_gru_f = tf.nn.rnn_cell.GRUCell(self.config.N_FLOW_GRU_HIDDEN)
    self.A_p_unstacked = tf.unstack(tf.transpose(self.A_p, (1, 0, 2)), name="A_p_unstacked")
    _, self.O_fwd_f, self.O_bwd_f = tf.nn.static_bidirectional_rnn(self.fwd_gru_f, self.bwd_gru_f, self.A_p_unstacked, dtype=tf.float32, scope="flow_rnn")
    assert(self.O_fwd_f.shape == (self.config.BATCH_SIZE, self.config.N_FLOW_GRU_HIDDEN))
    assert(self.O_bwd_f.shape == (self.config.BATCH_SIZE, self.config.N_FLOW_GRU_HIDDEN))

    self.W_f = tf.get_variable("W_f", shape=[2 * self.config.N_FLOW_GRU_HIDDEN, self.config.N_FLOW_DENSE_HIDDEN])
    self.A_f = tf.tanh(tf.matmul(tf.concat((self.O_fwd_f, self.O_bwd_f), axis=1), self.W_f), name="A_f")
    assert(self.A_f.shape == (self.config.BATCH_SIZE, self.config.N_FLOW_DENSE_HIDDEN))

    ##############################

    self.W_final = tf.get_variable("W_final", shape=[self.config.N_FLOW_DENSE_HIDDEN, 2])
    self.prediction = tf.nn.softmax(tf.matmul(self.A_f, self.W_final), name="prediction")
    assert(self.prediction.shape == (self.config.BATCH_SIZE, 2))

    self.loss = -tf.reduce_sum(self.target * tf.log(self.prediction) * tf.constant(self.config.RESULT_WEIGHTING, dtype=tf.float32), name="loss")

    ##############################

    # Number of correct, not normalized
    correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.target, 1))
    # Accuracy
    self.acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    # Setting optimizers
    self.optimizer = tf.train.AdamOptimizer()
    self.optim = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())

    return self


import tensorflow as tf
from .base_model import Base_Model

tf.logging.set_verbosity(tf.logging.ERROR)


class Self_Attention(Base_Model):
  def __init__(self, sess, config, model_name="self_attention.model"):
    Base_Model.__init__(self, sess, config)
    self.model_name = model_name

  def build_model(self):
    # Set initial vars
    self.x = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES], name="x")
    self.target = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, 2], name="target")

    ##############################

    self.fwd_gru_p = tf.nn.rnn_cell.GRUCell(self.config.N_PACKET_GRU_HIDDEN)
    self.bwd_gru_p = tf.nn.rnn_cell.GRUCell(self.config.N_PACKET_GRU_HIDDEN)

    self.x_unstacked = tf.unstack(tf.transpose(tf.reshape(self.x, (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES)), (1, 0, 2)), name="x_unstacked")
    self.H_p_inverse, _, _ = tf.nn.static_bidirectional_rnn(self.fwd_gru_p, self.bwd_gru_p, self.x_unstacked, dtype=tf.float32, scope="packet_rnn")
    assert(tf.stack(self.H_p_inverse).shape == (self.config.N_PACKETS, self.config.BATCH_SIZE * self.config.N_FLOWS, 2 * self.config.N_PACKET_GRU_HIDDEN))
    self.H_p = tf.transpose(self.H_p_inverse, (1, 0, 2), name="H_p")
    assert(self.H_p.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKETS, 2 * self.config.N_PACKET_GRU_HIDDEN))

    self.W_s_1_p = tf.get_variable("W_s_1_p", shape=[2 * self.config.N_PACKET_GRU_HIDDEN, self.config.N_PACKET_ATTENTION_HIDDEN])
    self.W_s_2_p = tf.get_variable("W_s_2_p", shape=[self.config.N_PACKET_ATTENTION_HIDDEN, 1])

    self.r_p_mid = tf.tanh(tf.matmul(tf.reshape(self.H_p, (self.config.BATCH_SIZE * self.config.N_FLOWS * self.config.N_PACKETS, 2 * self.config.N_PACKET_GRU_HIDDEN)), self.W_s_1_p), name="r_p_mid")
    assert(self.r_p_mid.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS * self.config.N_PACKETS, self.config.N_PACKET_ATTENTION_HIDDEN))
    self.r_p = tf.nn.softmax(tf.reshape(tf.squeeze(tf.matmul(self.r_p_mid, self.W_s_2_p)), (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKETS)), name="r_p")
    assert(self.r_p.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKETS))

    self.M_p = tf.squeeze(tf.matmul(tf.transpose(self.H_p, (0, 2, 1)), tf.expand_dims(self.r_p, 2)), name="M_p")
    assert(self.M_p.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, 2 * self.config.N_PACKET_GRU_HIDDEN))

    self.W_p = tf.get_variable("W_p", shape=[2 * self.config.N_PACKET_GRU_HIDDEN, self.config.N_PACKET_DENSE_HIDDEN])
    self.A_p_unshaped = tf.tanh(tf.matmul(self.M_p, self.W_p), name="A_p_unshaped")
    assert(self.A_p_unshaped.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN))
    self.A_p = tf.reshape(self.A_p_unshaped, (self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN), name="A_p")
    assert(self.A_p.shape == (self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKET_DENSE_HIDDEN))

    ##############################

    self.fwd_gru_f = tf.nn.rnn_cell.GRUCell(self.config.N_FLOW_GRU_HIDDEN)
    self.bwd_gru_f = tf.nn.rnn_cell.GRUCell(self.config.N_FLOW_GRU_HIDDEN)

    self.A_p_unstacked = tf.unstack(tf.transpose(self.A_p, (1, 0, 2)), name="A_p_unstacked")
    self.H_f_inverted, _, _ = tf.nn.static_bidirectional_rnn(self.fwd_gru_f, self.bwd_gru_f, self.A_p_unstacked, dtype=tf.float32, scope="flow_rnn")
    assert(tf.stack(self.H_f_inverted).shape == (self.config.N_FLOWS, self.config.BATCH_SIZE, 2 * self.config.N_FLOW_GRU_HIDDEN))
    self.H_f = tf.transpose(self.H_f_inverted, (1, 0, 2), name="H_f")
    assert(self.H_f.shape == (self.config.BATCH_SIZE, self.config.N_FLOWS, 2 * self.config.N_FLOW_GRU_HIDDEN))

    self.W_s_1_f = tf.get_variable("W_s_1_f", shape=[2 * self.config.N_FLOW_GRU_HIDDEN, self.config.N_FLOW_ATTENTION_HIDDEN])
    self.W_s_2_f = tf.get_variable("W_s_2_f", shape=[self.config.N_FLOW_ATTENTION_HIDDEN, 1])
    
    self.r_f_mid = tf.tanh(tf.matmul(tf.reshape(self.H_f, (self.config.BATCH_SIZE * self.config.N_FLOWS, 2 * self.config.N_FLOW_GRU_HIDDEN)), self.W_s_1_f), name="r_f_mid")
    assert(self.r_f_mid.shape == (self.config.BATCH_SIZE * self.config.N_FLOWS, self.config.N_FLOW_ATTENTION_HIDDEN))
    self.r_f = tf.nn.softmax(tf.reshape(tf.squeeze(tf.matmul(self.r_f_mid, self.W_s_2_f)), (self.config.BATCH_SIZE, self.config.N_FLOWS)), name="r_f")
    assert(self.r_f.shape == (self.config.BATCH_SIZE, self.config.N_FLOWS))

    self.M_f = tf.squeeze(tf.matmul(tf.transpose(self.H_f, (0, 2, 1)), tf.expand_dims(self.r_f, 2)), name="M_f")
    assert(self.M_f.shape == (self.config.BATCH_SIZE, 2 * self.config.N_FLOW_GRU_HIDDEN))

    self.W_f = tf.get_variable("W_f", shape=[2 * self.config.N_FLOW_GRU_HIDDEN, self.config.N_FLOW_DENSE_HIDDEN])
    self.A_f = tf.tanh(tf.matmul(self.M_f, self.W_f), name="A_f")
    assert(self.A_f.shape == (self.config.BATCH_SIZE, self.config.N_FLOW_DENSE_HIDDEN))

    ##############################

    self.W_final = tf.get_variable("W_final", shape=[self.config.N_FLOW_DENSE_HIDDEN, 2])
    self.prediction = tf.nn.softmax(tf.matmul(self.A_f, self.W_final), name="prediction")
    assert(self.prediction.shape == (self.config.BATCH_SIZE, 2))

    self.loss = -tf.reduce_sum(self.target * tf.log(self.prediction) * tf.constant(self.config.RESULT_WEIGHTING, dtype=tf.float32), name="loss")

    ##############################

    correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.target, 1))
    self.acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    self.optimizer = tf.train.AdamOptimizer()
    self.optim = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())

    return self


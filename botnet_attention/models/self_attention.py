import tensorflow as tf
from .base_model import Base_Model

tf.logging.set_verbosity(tf.logging.ERROR)


class Self_Attention(Base_Model):
  '''
  Model that uses self attention for encoding sequences
  '''

  def __init__(self, sess, config, model_name="self_attention.model"):
    Base_Model.__init__(self, sess, config)
    self.model_name = model_name

  def build_model(self):
    # Verify inputs
    assert(self.x.shape == (self.config.BATCH_SIZE, self.config.N_FLOWS, self.config.N_PACKETS, self.config.N_FEATURES))
    assert(self.target.shape == (self.config.BATCH_SIZE, 2))

    # Packets encoder
    packets_encoder_config = {
        'n_seqs': self.config.BATCH_SIZE * self.config.N_FLOWS,
        'seq_len': self.config.N_PACKETS,
        'n_features': self.config.N_PACKET_FEATURES,
        'n_gru_hidden': self.config.N_HIDDEN['packets_gru'],
        'n_attention_hidden': self.config.N_HIDDEN['packets_attention'],
        'n_dense_hidden': self.config.N_FLOW_FEATURES
    }
    packet_x = tf.reshape(self.x, (packets_encoder_config['n_seqs'], packets_encoder_config['seq_len'], packets_encoder_config['n_features']))
    encoded_flows, att_matrix_p = self.__attention_encoder_layer(packet_x, "packets_encoder", packets_encoder_config)

    # Flow encoders
    flows_encoder_config = {
        'n_seqs': self.config.BATCH_SIZE,
        'seq_len': self.config.N_FLOWS,
        'n_features': self.config.N_FLOW_FEATURES,
        'n_gru_hidden': self.config.N_HIDDEN['flows_gru'],
        'n_attention_hidden': self.config.N_HIDDEN['flows_attention'],
        'n_dense_hidden': self.config.N_IP_FEATURES
    }
    encoded_ips, att_matrix_f = self.__attention_encoder_layer(encoded_flows, "flows_encoder", flows_encoder_config)

    # Get predictions
    predictor_config = {
        'n_batches': self.config.BATCH_SIZE,
        'n_input': self.config.N_IP_FEATURES,
        'n_classes': 2
    }
    self.prediction = self.__prediction_layer(encoded_ips, 'predictor', predictor_config)

    # Get loss and optimizer
    self.loss, self.optim, self.acc = self.__define_optimization_vars(self.target, self.prediction, self.config.RESULT_WEIGHTING)

    return self


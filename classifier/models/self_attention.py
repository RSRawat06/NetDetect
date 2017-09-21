import tensorflow as tf
from .base_model import Base_Model
from . import config


class Self_Attention(Base_Model):
  '''
  Model that uses self attention for encoding sequences
  '''

  def __init__(self, sess, data_config, model_name="self_attention.model"):
    Base_Model.__init__(self, sess, data_config)
    self.model_name = model_name

  def build_model(self, local_batch_size=config.BATCH_SIZE):
    assert(self.x.shape == (local_batch_size, config.NUMBERS['flows'], config.NUMBERS['flow_features']))
    assert(self.target.shape == (local_batch_size, 2))

    # Flow encoders
    flows_encoder_config = {
        'n_seqs': local_batch_size,
        'seq_len': config.NUMBERS['flows'],
        'n_features': config.NUMBERS['flow_features'],
        'n_gru_hidden': config.HIDDEN['flows_gru'],
        'n_attention_hidden': config.HIDDEN['flows_attention'],
        'n_dense_hidden': config.NUMBERS['ip_features']
    }
    encoded_ips, att_matrix_f = self._attention_encoder_layer(self.x, "flows_encoder", flows_encoder_config)

    # Get predictions
    predictor_config = {
        'n_batches': local_batch_size,
        'n_input': config.NUMBERS['ip_features'],
        'n_classes': 2
    }
    self.prediction = self._prediction_layer(encoded_ips, 'predictor', predictor_config)

    # Get loss and optimizer
    self.loss, self.optim, self.acc = self._define_optimization_vars(self.target, self.prediction, config.LOSS_WEIGHTING)

    self.summary_op = self._summaries()

    return self


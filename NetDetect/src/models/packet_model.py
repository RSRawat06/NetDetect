from ..model_base import Base, SequenceLayers
import tensorflow as tf


class PacketModel(Base, SequenceLayers):
  '''
  Model for predicting on packets.
  '''

  def __init__(self, sess, config, logger):
    logger.info('Instantiated packet model')
    Base.__init__(self, sess, config, logger)

  def build_model(self):
    '''
    Build the packet model.
    '''

    self.logger.info('Building model...')
    config = self.config

    self.x = tf.placeholder(
        tf.float32, (config.BATCH_SIZE, config.N_FLOW_STEPS,
                     config.N_PACKET_STEPS, config.N_FEATURES))
    self.target = tf.placeholder(tf.float32,
                                 (config.BATCH_SIZE, config.N_CLASSES))

    packet_encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_PACKET_STEPS,
        'n_features': config.N_FEATURES,
        'h_gru': config.LAYERS['h_packet_gru'],
        'h_dense': config.LAYERS['o_packet_gru']
    }
    packet_encoded_state = self._encoder_layer(
        tf.reshape(self.x, (-1, config.N_PACKET_STEPS, config.N_FEATURES)), 
        "packet_encoder", 
        packet_encoder_config
    )

    flow_encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_FLOW_STEPS,
        'n_features': config.LAYERS['o_packet_gru'],
        'h_gru': config.LAYERS['h_flow_gru'],
        'h_dense': config.LAYERS['o_flow_gru']
    }
    flow_encoded_state = self._encoder_layer(
        tf.reshape(packet_encoded_state, (config.BATCH_SIZE, config.N_FLOW_STEPS, 
                                          config.LAYERS['o_packet_gru'])), 
        "flow_encoder", 
        flow_encoder_config
    )

    predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.LAYERS['o_flow_gru'],
        'n_classes': config.N_CLASSES
    }
    self.prediction = self._prediction_layer(
        flow_encoded_state,
        'predictor',
        predictor_config)

    self.loss = self._define_optimization_vars(
        self.target,
        self.prediction,
        [1, 1.1],
        self.config.REGULARIZATION
    )
    self.tpr, self.fpr, self.acc = self._define_binary_metrics(
        self.target,
        self.prediction,
    )

    optimizer = tf.train.AdamOptimizer()
    self.optim = optimizer.minimize(
        self.loss,
        var_list=tf.trainable_variables(),
        global_step=self.global_step)

    self.logger.info('Model built.')

    return self



from ..model_base import Base, SequenceLayers
import tensorflow as tf


class FlowModel(Base, SequenceLayers):
  '''
  Model for predicting on flows.
  '''

  def __init__(self, sess, config, logger):
    logger.info('Instantiated flow model')
    Base.__init__(self, sess, config, logger)

  def build_model(self):
    '''
    Build the flow model.
    '''

    self.logger.info('Building model...')
    config = self.config

    self.x = tf.placeholder(
        tf.float32, (config.BATCH_SIZE, config.N_STEPS, config.N_FEATURES))
    self.target = tf.placeholder(tf.float32,
                                 (config.BATCH_SIZE, config.N_CLASSES))

    encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_STEPS,
        'n_features': config.N_FEATURES,
        'h_gru': config.LAYERS['h_gru'],
        'h_dense': config.LAYERS['o_gru']
    }
    encoded_state = self._encoder_layer(
        self.x, "encoder", encoder_config)

    dense_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.LAYERS['o_gru'],
        'n_hidden': config.LAYERS['h_dense'],
        'n_output': config.LAYERS['o_dense']
    }
    dense_state = self._dense_layer(
        encoded_state, "dense", dense_config)

    predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.LAYERS['o_dense'],
        'n_classes': config.N_CLASSES
    }
    self.prediction = self._prediction_layer(
        dense_state,
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


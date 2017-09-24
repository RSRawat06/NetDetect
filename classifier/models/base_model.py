import tensorflow as tf
from .layered_model import Layered_Model
from . import config

tf.logging.set_verbosity(tf.logging.ERROR)


class Base_Model(Layered_Model):
  '''
  Base model that handles save, restore, load, train functionalities
  '''

  def __init__(self, sess, data_config):
    assert(config.NUMBERS['flow_features'] == data_config.N_FEATURES)
    assert(config.NUMBERS['flows'] == data_config.N_FLOWS)

    tf.set_random_seed(4)
    self.sess = sess
    self.data_config = data_config
    self.saver = None
    self.model_name = "default.model"
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

  def initialize(self):
    '''
    Initialize models: builds model, loads data, initializes variables
    '''
    self.load_pipeline()
    self.build_model()
    self.writer = tf.summary.FileWriter(self.data_config.DATA_DIR + 'graphs', self.sess.graph)
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()

  def pseudoload(self):
    '''
    Pseudoload x as placeholders to support prediction
    '''
    self.x = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, config.NUMBERS['flows'], config.NUMBERS['flow_features']))
    self.target = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, 2))
    self.build_model()
    self.writer = tf.summary.FileWriter(self.data_config.DATA_DIR + 'graphs', self.sess.graph)
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()

  def save(self, global_step=None):
    '''
    Save the current variables in graph.
    Optional option to save for global_step (used in Train)
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())
    if global_step is None:
      self.saver.save(self.sess, self.data_config.DATA_DIR + 'checkpoints/' + self.model_name)
    else:
      self.saver.save(self.sess, self.data_config.DATA_DIR + 'checkpoints/' + self.model_name, global_step=self.global_step)

  def restore(self, resume=False):
    '''
    Load saved variable values into graph
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())

    if resume:
      ckpt = tf.train.get_checkpoint_state(self.data_config.DATA_DIR + 'checkpoints/checkpoint')
      if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      return

    self.saver.restore(self.sess, self.data_config.DATA_DIR + 'checkpoints/' + self.model_name)

  def load_pipeline(self):
    '''
    Load stream of data and batches input
    '''
    NUMBERS = config.NUMBERS

    filename_queue = tf.train.string_input_producer([self.data_config.DATA_DIR + self.data_config.TF_SAVE])
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized,
        features={
            'features': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        }
    )
    features = tf.reshape(tf.cast(features['features'], tf.float32), (NUMBERS['flows'], NUMBERS['flow_features']))
    label = tf.reshape(tf.cast(features['label'], tf.float32), 2)

    self.x, self.target = tf.train.shuffle_batch([features, label], batch_size=config.BATCH_SIZE, capacity=500, min_after_dequeue=100)
    assert(self.x.shape == (config.BATCH_SIZE, NUMBERS['flows'], NUMBERS['flow_features']))
    assert(self.target.shape == (config.BATCH_SIZE, 2))

  def train(self):
    '''
    Run model training. Model must have been initialized.
    '''

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    try:
      i = 0
      while not coord.should_stop():
        raw_x, raw_y, _, acc, loss, summary = self.sess.run([self.x, self.target, self.optim, self.acc, self.loss, self.summary_op])
        i += 1
        print("Epoch:", i, "has loss:", loss, "and accuracy:", acc)
        self.writer.add_summary(summary, global_step=self.global_step)
    finally:
      coord.request_stop()
      coord.join(threads)

  def predict(self, input_x):
    '''
    Predict classifications for new inputs
    '''
    predictions = []
    for i in range(0, len(input_x), config.BATCH_SIZE):
      feed_dict = {
          self.x: input_x[i:i + config.BATCH_SIZE]
      }
      prediction = list(self.sess.run([self.prediction], feed_dict=feed_dict))
      predictions += prediction
    return predictions


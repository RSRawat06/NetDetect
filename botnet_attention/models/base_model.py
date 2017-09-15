import tensorflow as tf
from .layered_model import Layered_Model
from . import config


class Base_Model(Layered_Model):
  '''
  Base model that handles save, restore, load, train functionalities
  '''

  def __init__(self, sess, data_config):
    self.sess = sess
    self.data_config = data_config
    self.saver = None
    self.model_name = "default.model"

  def save(self):
    '''
    Save the current variables in graph.
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())
    self.saver.save(self.sess, self.data_config.DATA_DIR + self.model_name)

  def restore(self):
    '''
    Load saved variable values into graph
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())
    self.saver.restore(self.sess, self.data_config.DATA_DIR + self.model_name)

  def load(self):
    '''
    Load stream of data and batches input
    '''
    NUMBERS = config.NUMBERS

    filename_queue = tf.train.string_input_producer([self.data_config.DATA_DIR + self.data_config.TF_SAVE], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature((2), tf.float32),
            'features': tf.FixedLenFeature((NUMBERS['flows'], NUMBERS['packets'], NUMBERS['packet_features']), tf.float32)
        }
    )
    label = features['label']
    features = features['features']

    self.x, self.target = tf.train.shuffle_batch([features, label], batch_size=config.BATCH_SIZE, capacity=500, min_after_dequeue=100)
    assert(self.x.shape == (config.BATCH_SIZE, NUMBERS['flows'], NUMBERS['packets'], NUMBERS['packet_features']))
    assert(self.target.shape == (config.BATCH_SIZE, 2))

  def initialize(self):
    '''
    Initialize models: builds model, loads data, initializes variables
    '''
    self.load()
    self.build_model()
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()

  def train(self):
    '''
    Run model training. Model must have been initialized.
    '''
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=self.sess, coord=coord)
    try:
      i = 0
      while not coord.should_stop():
        raw_x, raw_y, _, acc, loss = self.sess.run([self.x, self.target, self.optim, self.acc, self.loss])
        i += 1
        print("Epoch:", i, "has loss:", loss, "and accuracy:", acc)
    finally:
      coord.request_stop()

  def pseudoload(self):
    '''
    Pseudoload x as placeholders to support prediction
    '''
    self.raw_x = tf.placeholder(tf.int32)
    queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.int32], shapes=[()])
    self.enqueue_op = queue1.enqueue_many(self.raw_x)
    self.dequeue_op = queue1.dequeue()

    self.x = tf.train.shuffle_batch([self.dequeue_op], batch_size=config.BATCH_SIZE, capacity=50, min_after_dequeue=10)

  def predict(self, input_x):
    '''
    Predict classifications for new inputs
    '''
    predictions = []
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=self.sess, coord=coord)
    try:
      while not coord.should_stop():
        _, prediction = list(self.sess.run([self.enqueue_op, self.prediction], feed_dict={self.raw_x: input_x}))
        predictions += prediction
    finally:
      coord.request_stop()

    return predictions


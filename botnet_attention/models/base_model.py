import tensorflow as tf
from .layered_model import Layered_Model
from . import config


class Base_Model(Layered_Model):
  '''
  Base model that handles save, restore, load, train functionalities
  '''

  def __init__(self, sess, data_config):
    tf.set_random_seed(4)
    self.sess = sess
    self.data_config = data_config
    self.saver = None
    self.model_name = "default.model"
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

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
    self.writer = tf.summary.FileWriter(self.data_config.DATA_DIR + 'graphs', sess.graph) 
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
        raw_x, raw_y, _, acc, loss, summary = self.sess.run([self.x, self.target, self.optim, self.acc, self.loss, self.summary_op])
        i += 1
        print("Epoch:", i, "has loss:", loss, "and accuracy:", acc)
        self.writer.add_summary(summary, global_step=self.global_step)
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


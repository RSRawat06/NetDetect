import sys, os  

sys.path.append(os.getcwd())
from config import *

sys.path.append(PROJ_ROOT + "src/discrimination/")
from tf_discriminator_model import Attention_Discriminator

sys.path.append(PROJ_ROOT + "src/utils")
import data_utils

import tensorflow as tf

data = data_utils.load_data()
with tf.Session() as sess:
  model = Attention_Discriminator(sess)
  model.build_model()
  model.train(*data, ITERATIONS)

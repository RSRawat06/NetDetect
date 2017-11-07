import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

BATCH_SIZE = 16
TEST_SIZE = 3200
VAL_SIZE = 16
ITERATIONS = 10
N_FEATURES = 77
N_STEPS = 8
LAYERS = {
    'h_gru': 64,
    'o_gru': 32,
    'h_dense': 32,
    'o_dense': 16
}
N_CLASSES = 2
REGULARIZATION = 0.1

REPORT_INTERVAL = 100
SAVE_INTERVAL = 300


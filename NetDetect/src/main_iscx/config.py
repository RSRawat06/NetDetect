import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

BATCH_SIZE = 32
TEST_SIZE = 4096
ITERATIONS = 100
N_FEATURES = 77
N_STEPS = 22
LAYERS = {
    'h_gru': 64,
    'h_att': 16,
    'o_gru': 64,
    'h_dense': 64,
    'o_dense': 32,
    'h_dense2': 32,
    'o_dense2': 16
}
N_CLASSES = 2
REGULARIZATION = 0.1

REPORT_INTERVAL = 200
SAVE_INTERVAL = 600


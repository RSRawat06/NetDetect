BATCH_SIZE = 10
ITERATIONS = 100

LOSS_WEIGHTING = [1, 5]
NUMBERS = {
    'ip_features': 30,
    'flows': 11,
    'flow_features': 10,
    'packets': 20,
    'packet_features': 13
}
HIDDEN = {
    'flows_gru': 30,
    'flows_attention': 30,
    'packets_gru': 30,
    'packets_attention': 30
}
PARTITIONING = {
    'validation': 40,
    'testing': 40
}

from .featurize import featurize_csv
from .metadatize import metadatize_csv
from .segment import segment_by_ips, segment_by_flows


def preprocess(path, config):
  flat_X = featurize_csv(path, config.numerical_fields, config.protocol_fields, config.categorical_fields, config.port_fields)
  metadata = metadatize_csv(path, config.malicious_ips, config.flow_field, config.participant_fields)
  assert(flat_X.shape[1] == config.N_FEATURES)

  X_by_flows, metadata = segment_by_flows(flat_X, metadata, config.N_PACKETS)
  X, Y = segment_by_ips(X_by_flows, metadata, config.N_FLOWS)

  return X, Y


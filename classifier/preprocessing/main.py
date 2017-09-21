from .featurize import featurize_csv
from .metadatize import metadatize_csv
from .segment import segment_by_ips, segment_by_flows


def preprocess(path, config):
  flat_X = featurize_csv(path)
  print("Featurized")
  metadata = metadatize_csv(path, config.malicious_ips, config.flow_field, config.participant_fields)
  print("Metadatized")
  X, Y = segment_by_ips(X_by_flows, new_metadata, config.N_FLOWS)
  print("IP segmented")

  return X, Y


from .featurize import featurize_csv
from .metadatize import metadatize_csv
from .segment import segment_by_ips


def preprocess(path, config):
  flows = featurize_csv(path, config.numerical_fields)
  print("Featurized")
  metadata = metadatize_csv(path, config.malicious_ips, config.flow_field, config.participant_fields)
  print("Metadatized")
  X, Y = segment_by_ips(flows, metadata, config.N_FLOWS)
  print("IP segmented")

  return X, Y


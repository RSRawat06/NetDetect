def preprocess(path):
  flat_X = featurize_csv(path)
  metadata = featurize_csv(path)
  
  X_by_flows, metadata = segment_by_flows(flat_X, metadata)
  X, Y = segment_by_ips(X_by_flows, metadata) 

  return X, Y


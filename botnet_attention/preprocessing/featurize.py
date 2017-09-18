from . import utils
import csv


def featurize_csv(data_path, numerical_fields, protocol_fields, categorical_fields, port_fields):
  '''
  Load in a CSV and convert to ANN-friendly numpy vectors.
  Args:
    - data_path (str): path to data file
    - numerical_fields (list of str): list of field names to be floated
    - protocol_fields (list of str): list of field names with value of cls1:cls2:cls3
    - categorical_fields (list of str): list of field names with categorical features
    - port_fields (list of str): list of field names with categorical features to be capped at 2000
  Returns:
    - X: list([0, 1, 3.0, 4, 5...])
  '''
  all_records = {'protocol': {}, 'categorical': {}, 'port': {}}
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      all_records = store_categoricals(row, headers_key, all_records, protocol_fields, categorical_fields, port_fields)

  X = []
  with open(data_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = utils.build_headers(row)
        continue
      X.append(featurize_row(row, headers_key, all_records, numerical_fields, protocol_fields, categorical_fields, port_fields))

  return X


def featurize_row(row, headers_key, all_records, numerical_fields, protocol_fields, categorical_fields, port_fields):
  '''
  Featurize a row into a real-valued vector
  Args:
    - row (list of str): row of a csv from csv.reader
    - headers_key (dict): maps pos index in row to field name
    - all_records (dict): holds information on which fields have which possible categorical values
    - numerical_fields (list of str): list of field names to be floated
    - protocol_fields (list of str): list of field names with value of cls1:cls2:cls3
    - categorical_fields (list of str): list of field names with categorical features
    - port_fields (list of str): list of field names with categorical features to be capped at 2000
  Returns:
    - feature_vector: list(0, 3, 4...)
  '''
  def __create_one_hot(record, values):
    onehot = [0] * len(record)
    for value in values:
      ind = record.index(value)
      assert(ind >= 0)
      onehot[ind] = 1
    return onehot

  feature_vector = []
  for i, value in enumerate(row):
    field = headers_key[i]
    if field in numerical_fields:
      value = float(value)
      feature_vector.append(value)
    elif field in protocol_fields:
      value = str(value)
      feature_vector += __create_one_hot(all_records['protocol'][field], value.split(":"))
    elif field in categorical_fields:
      value = str(value)
      feature_vector += __create_one_hot(all_records['categorical'][field], [value])
    elif field in port_fields:
      value = int(float(value))
      if value > 2000:
        value = 2000
      feature_vector += __create_one_hot(all_records['port'][field], [value])

  return feature_vector


def store_categoricals(row, headers_key, all_records, protocol_fields, categorical_fields, port_fields, threshold=30):
  '''
  Store possible categorical values for each field in to a records dict
  Args:
    - row (list of str): row of a csv from csv.reader
    - headers_key (dict): maps pos index in row to field name
    - all_records (dict): holds information on which fields have which possible categorical values
    - protocol_fields (list of str): list of field names with value of cls1:cls2:cls3
    - categorical_fields (list of str): list of field names with categorical features
    - port_fields (list of str): list of field names with categorical features to be capped at 2000
    - threshold (int): limit on the length of a categorical one-hot vec
  Returns:
    - all_records (dict), similar to arg all_records but with new insights reflected
  '''
  def __append_to_records(records, field, value):
    if field not in records:
      records[field] = []
    if value not in records[field]:
      records[field].append(value)
    if len(records[field]) > threshold:
      raise ValueError
    return records

  for i, value in enumerate(row):
    field = headers_key[i]
    if field in protocol_fields:
      for protocol in str(value).split(":"):
        __append_to_records(all_records['protocol'], field, protocol)

    elif headers_key[i] in categorical_fields:
      __append_to_records(all_records['categorical'], field, value)

    elif headers_key[i] in port_fields:
      value = int(float(value))
      if value > 2000:
        value = 2000
      __append_to_records(all_records['port'], field, value)

  return all_records


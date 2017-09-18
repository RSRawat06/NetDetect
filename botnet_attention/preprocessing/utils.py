def build_headers(row):
  '''
  '''
  headers_key = {}
  for j, field in enumerate(row):
    headers_key[j] = field
  return headers_key

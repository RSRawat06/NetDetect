from ...preprocessing import utils

def test_build_headers():
  row = ["chocolate", "yummy", "weird"]
  headers_key = utils.build_headers(row)
  assert(headers_key = {1: "chocolate", 2: "yummy", 3: "weird"})

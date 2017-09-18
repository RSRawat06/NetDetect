from ...preprocessing import utils


def test_build_headers():
  row = ["chocolate", "yummy", "weird"]
  headers_key = utils.build_headers(row)
  assert(headers_key == {0: "chocolate", 1: "yummy", 2: "weird"})

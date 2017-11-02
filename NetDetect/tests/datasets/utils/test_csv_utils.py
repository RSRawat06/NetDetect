from ....datasets.utils import csv_utils
import unittest


class TestCSVUtils(unittest.TestCase):
  def test_build_headers(self):
    row = ["chocolate", "yummy", "weird"]
    headers_key = csv_utils.build_headers(row)
    self.assertEqual(headers_key, {0: "chocolate", 1: "yummy", 2: "weird"})

  def test_featurize_row(self):
    headers_key = ["a", "b", "c", "d", "none", "random", "e"]
    row = [5.0, 3.2, 4.5, 3.7, None, "Aasdf", 3.3]

    def __invalid_numerical_fields():
      numerical_fields = ["a", "b", "z"]
      failed = False
      try:
        csv_utils.featurize_row(row, headers_key, numerical_fields)
      except AssertionError:
        failed = True
      assert(failed)

    def __valid_numerical_fields():
      numerical_fields = ["a", "b", "e"]
      self.assertEqual(np.all(csv_utils.featurize_row(row, headers_key, numerical_fields),
                       np.array([5.0, 3.2, 3.3], np.float32)))
    
    __invalid_numerical_fields()
    __valid_numerical_fields()


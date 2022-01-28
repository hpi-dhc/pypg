"""
====================================
Tests the Filters Module
====================================
"""

import unittest

import numpy as np
import pandas as pd
from pypg.cycles import find_onset, find_with_template


class TestFilters(unittest.TestCase):
    """
    Unit tests for the cycles module.
    """
    def setUp(self):
        self.ppg = [2438.0, 2438.0, 2438.0, 2455.0, 2455.0, 2384.0, 2384.0, 2384.0, 2418.0,
                    2418.0, 2415.0, 2415.0, 2415.0, 2398.0, 2398.0, 2388.0, 2388.0, 2388.0,
                    2340.0, 2340.0, 2340.0, 2340.0, 2340.0, 2399.0, 2399.0, 2353.0, 2353.0,
                    2353.0, 2318.0, 2318.0, 2324.0, 2324.0, 2324.0, 2283.0, 2283.0, 2333.0,
                    2333.0, 2333.0, 2326.0, 2326.0, 2274.0, 2274.0, 2274.0, 2309.0, 2309.0,
                    2224.0, 2224.0, 2224.0, 2288.0, 2288.0, 2268.0, 2268.0, 2268.0, 2250.0]
        self.int_data = int(2438)
        self.pandas_data = pd.Series(self.ppg)
        self.numpy_data = np.array(self.ppg)

    def test_find_onset(self):
        """
        Tests if the returned data type is correct.
        """
        result_pandas = find_onset(self.pandas_data, 1000)
        result_numpy = find_onset(self.numpy_data, 1000)

        self.assertIsInstance(result_pandas, np.ndarray)
        self.assertIsInstance(result_numpy, np.ndarray)

        with self.assertRaises(Exception):
            find_onset(self.int_data, 1000)

    def test_find_with_template(self):
        """
        Tests if the returned data type is correct.
        """
        result_pandas = find_with_template(self.pandas_data, 1000)
        result_numpy = find_with_template(self.numpy_data, 1000)

        self.assertIsInstance(result_pandas, np.ndarray)
        self.assertIsInstance(result_numpy, np.ndarray)

        with self.assertRaises(Exception):
            find_with_template(self.int_data, 1000)


if __name__ == '__main__':
    unittest.main()

"""
====================================
Tests the Filters Module
====================================
"""

import unittest

import numpy as np
import pandas as pd
from pypg.filters import butterfy, chebyfy, movefy


class TestFilters(unittest.TestCase):
    """
    Unit tests for the filters module.
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

    def test_chebyfy(self):
        """
        Tests if the returned data type is correct.
        """
        result_pandas = chebyfy(self.pandas_data, [0.5, 10], 1000)
        result_numpy = chebyfy(self.numpy_data, [0.5, 10], 1000)

        self.assertIsInstance(result_pandas, pd.core.series.Series)
        self.assertIsInstance(result_numpy, np.ndarray)

        with self.assertRaises(Exception):
            chebyfy(self.int_data, [0.5, 10], 1000)

    def test_buttery(self):
        """
        Tests if the returned data type is correct.
        """
        result_pandas = butterfy(self.pandas_data, [0.5, 10], 1000)
        result_numpy = butterfy(self.numpy_data, [0.5, 10], 1000)

        self.assertIsInstance(result_pandas, pd.core.series.Series)
        self.assertIsInstance(result_numpy, np.ndarray)

        with self.assertRaises(Exception):
            butterfy(self.int_data, [0.5, 10], 1000)

    def test_movefy(self):
        """
        Tests if the returned data type is correct.
        """
        result_pandas = movefy(self.pandas_data, 100)
        result_numpy = movefy(self.numpy_data, 100)

        self.assertIsInstance(result_pandas, pd.core.series.Series)
        self.assertIsInstance(result_numpy, np.ndarray)

        with self.assertRaises(Exception):
            movefy(self.int_data, 100)


if __name__ == '__main__':
    unittest.main()

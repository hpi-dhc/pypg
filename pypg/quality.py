"""
=============================================
Extracts quality metrics from PPG segments (:mod:`pypg.quality`)
=============================================

This module is intended to extract quality metrics from PPG segments.

Quality
----------
skewness - Extracts the average or maximum skewness of a PPG segment.

References
----------
Krishnan, R., Natarajan, B., & Warren, S. (2010). Two-Stage Approach for
Detection and Reduction of Motion Artifacts in Photoplethysmographic Data. IEEE
Transactions on Biomedical Engineering, 57(8), 1867–1876.
https://doi.org/10.1109/TBME.2009.2039568

Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals.
Bioengineering, 3(4), 21. https://doi.org/10.3390/bioengineering3040021
"""
from math import sqrt

import numpy as np
import pandas as pd


def skewness(ppg, sampling_frequency, window=1, mode='avg'):
    """
    Calculates the skewness of a PPG segment using a moving window.

    The skewness is calculated for every window.
    The average or the maximum value is selected to represent the segment skewness.

    Parameters
    ----------
    ppg : pandas.Series, ndarray
        The PPG signal.
    sampling_frequency : int
        The sampling frequency of the signal in Hz.
    window : int
        The size of the sliding window, definied in seconds, defaults to 1 second.
    mode: str
        Either 'avg' which is the average skewness of the segmend or max which is
        the maximum skewness level, defaults to 'avg'.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.
        When mode is neither 'avg' or 'max'.

    Returns
    ----------
    skewness_value: float
        The skewness value for the PPG segment.

    References
    ----------
    Krishnan, R., Natarajan, B., & Warren, S. (2010). Two-Stage Approach for
    Detection and Reduction of Motion Artifacts in Photoplethysmographic Data. IEEE
    Transactions on Biomedical Engineering, 57(8), 1867–1876.
    https://doi.org/10.1109/TBME.2009.2039568

    Elgendi, M. (2016). Optimal signal quality index for photoplethysmogram signals.
    Bioengineering, 3(4), 21. https://doi.org/10.3390/bioengineering3040021

    https://en.wikipedia.org/wiki/Skewness
    """
    # verify ppg data sent
    if isinstance(ppg, pd.core.series.Series):
        ppg_values = np.trim_zeros(ppg.values) # remove 0s from beginning and ending
    elif isinstance(ppg, np.ndarray):
        ppg_values = np.trim_zeros(ppg) # remove 0s from beginning and ending
    else:
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if mode not in ['avg','max']:
        raise Exception('Invalid mode.')

    skewness_vector = []
    start = 0
    samples_in_window = sampling_frequency*window
    end = samples_in_window

    size = len(ppg_values)
    # check if the signal is longer than the window size
    if samples_in_window > size:
        raise Exception('PPG signal should be longer than '
                            + 'the window size of: {} seconds'.format( window))

    # windowing
    while end <= size:
        ppg_snippet = ppg_values[start:end]
        mean = np.mean(ppg_snippet)
        size_ppg = len(ppg_snippet)

        # implementation according to the Fisher-Pearson coefficient of skewness
        sum_n = 0
        sum_d = 0

        for i in ppg_snippet:
            sum_n = sum_n + (i - mean)**3
            sum_d = sum_d + (i - mean)**2
        numerator = sum_n/size_ppg
        denominator = sqrt((sum_d/size_ppg)**3)
        skewness_vector.append(numerator/denominator)

        start = start + 1
        end = end + 1

    if mode == 'avg':
        skewness_value = sum(skewness_vector)/len(skewness_vector)
    if mode == 'max':
        skewness_value = max(skewness_vector)

    return skewness_value

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
        The skewness value for the PPG segment. The higher the better.

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


# TODO: @Ari: Should we take over as well? CYLCE QUALITY CHECK
'''for cycle_end in ppg_valleys:

			### PPG cycle extraction
			#ppg_cycle_signal = filtered_ppg_signal_segment[cycle_start:cycle_end]
			#ppg_cycle_duration = len(ppg_cycle_signal)

			# Check whether ppg_cycle is full cycle, otherwise skip
			#min_cycle_len = ((60 / config['max_bpm']) * sampling_frequency)
			#max_cycle_len = ((60 / config['min_bpm']) * sampling_frequency)

			#is_normal_len = (min_cycle_len <= ppg_cycle_duration <= max_cycle_len)

			#if not is_normal_len:
			#	cycle_start = cycle_end
			#	continue'''

	
# TODO: @Ari: Should we take over as well? Feature Dataframe Quality
'''### Add NA_COUNT_THRESHOLD?!
		# EXAMPLES
		#timefeatures['BPM'] = np.mean(BPM) if BPM else 0
		#timefeatures['CT_mean'] = np.mean(CT) if CT else 0
		#timefeatures['CT_var'] = np.var(CT) if CT else 0

		# EXAMPLES
		#magfeatures['PPG_Mean'] = np.mean(cycle_means) if cycle_means else np.nan
		#magfeatures['PPG_Var'] = np.mean(cycle_var) if cycle_var else np.nan

		# EXAMPLES
		#NA_Count_Threshold = 0.8
		#for key in apgFeaturesWindow:
		#	apgFeatures[key] = np.mean(apgFeaturesWindow[key]) if len(
		#		apgFeaturesWindow[key]) / cycleCount >= NA_Count_Threshold else np.nan'''
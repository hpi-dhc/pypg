"""
=============================================
Extracts Features from PPG segments and cycles (:mod:`pypg.features`)
=============================================

This module is intended to extract features from PPG segments and cycles.

Features
----------
time       - Extract features from the time domain for a PPG segment.
time_cycle - Extract features from the time domain for a PPG cycle.

References
----------
Kurylyak, Y., Lamonaca, F., & Grimaldi, D. (2013). A Neural Network-based
method for continuous blood pressure estimation from a PPG signal.
Conference Record - IEEE Instrumentation and Measurement Technology
Conference, 280–283. https://doi.org/10.1109/I2MTC.2013.6555424

Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D.
(2013). Systolic Peak Detection in Acceleration Photoplethysmograms
Measured from Emergency Responders in Tropical Conditions. PLoS ONE,
8(10), 1–11. https://doi.org/10.1371/journal.pone.0076585

Li, Q., & Clifford, G. D. (2012). Dynamic time warping and machine
learning for signal quality assessment of pulsatile signals.
Physiological Measurement, 33(9), 1491–1501.
https://doi.org/10.1088/0967-3334/33/9/1491
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from .cycles import find_with_template

def time(ppg, sampling_frequency, factor=0.6667, unit='ms', verbose=False):
    """
    Extracts time domain features from PPG cycles in a give PPG segment as
    described by Kurylyak et al. (2013). Returns a pandas.DataFrame in
    which each line contains the features for a given valid cycle (as described
    by Li et al.) from the PPG segment given as input.

    Parameters
    ----------
    ppg : pandas.Series, ndarray
        The PPG signal.
    sampling_frequency : int
        The sampling frequency of the signal in Hz.
    factor: float, optional
        Number that is used to calculate the distance in relation to the
        sampling_frequency, by default 0.667 (or 66.7%). The factor is based
        on the paper by Elgendi et al. (2013).
    unit : str, optional
        The unit of the index, by default 'ms'.
    verbose : bool, optional
        If True, plots the features with and without outliers, by default False.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    Returns
    -------
    segment_features : pd.DataFrame
        A dataframe with the time-domain features in seconds for each valid
        cycle in the PPG segment.
    """
    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    cycles = find_with_template(ppg, sampling_frequency, factor=0.6667, verbose=verbose)
    # all signal peaks
    all_peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if verbose:
        print('All Peaks: ', all_peaks)
    if len(cycles) == 0:
        return pd.DataFrame()

    cur_index = 0
    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = segment_features.append(time_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose),
                                         ignore_index=True)
        if i > 0:
            segment_features.loc[cur_index-1, 'CP'] = (
                segment_features.loc[cur_index, 'sys_peak_ts'] - segment_features.loc[cur_index-1,
                'sys_peak_ts']).total_seconds()
        cur_index = cur_index + 1
    # last cycle or only cycle need to relies on the difference between the general peaks
    all_peaks_index = len(all_peaks)-1
    segment_features.loc[cur_index-1, 'CP'] = (
        all_peaks[all_peaks_index] - all_peaks[all_peaks_index-1])/sampling_frequency

    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    # remove outliers
    segment_features = _clean_segment_features_of_outliers(segment_features)

    if verbose:
        print('Cycle Features within Segment and no Outliers:')
    return segment_features

def time_cycle(ppg, sampling_frequency, factor=0.667, unit='ms', verbose=False):
    """
    Extracts time domain features for a PPG cycle. Returns a pandas.Series with
    the features described by Kurylyak et al. (2013).

    Parameters
    ----------
    ppg : pandas.Series, ndarray
        The PPG signal.
    sampling_frequency : int
        The sampling frequency of the signal in Hz.
    factor: float, optional
        Number that is used to calculate the distance in relation to the
        sampling_frequency, by default 0.667 (or 66.7%). The factor is based
        on the paper by Elgendi et al. (2013).
    unit : str, optional
        The unit of the index, by default 'ms'.
    verbose : bool, optional
        If True, plots the features with and without outliers, by default False.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    Returns
    -------
    cycle_features : pd.DataFrame
        A dataframe with the time-domain features in seconds for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit)

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min()
    max_amplitude = ppg.max()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    sys_peak_ts = ppg.index[peaks[0]]

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # features
    cycle_features = pd.DataFrame({
                        'start_ts': ppg.index.min(),
                        'sys_peak_ts': sys_peak_ts,
                        'SUT': (sys_peak_ts - ppg.index.min()).total_seconds(),
                        'DT': (ppg.index.max() - sys_peak_ts).total_seconds()
                        }, index=[0])
    for p_value in [10, 25, 33, 50, 66, 75]:
        p_ampl = p_value / 100 * max_amplitude
        x_1, x_2 = _find_xs_for_y(ppg, p_ampl, peaks[0])
        if verbose:
            plt.scatter([x_1, x_2], ppg[[x_1, x_2]])
        cycle_features.loc[0, 'DW_'+str(p_value)] = (x_2 - sys_peak_ts).total_seconds()
        cycle_features.loc[0, 'DW_SW_sum_'+str(p_value)] = (x_2 - x_1).total_seconds()
        cycle_features.loc[0, 'DW_SW_ratio_'+str(p_value)] = (
                                        x_2 - sys_peak_ts) / (sys_peak_ts - x_1)
    if verbose:
        plt.show()
    return cycle_features

# takes a dataframe of calculated features and removes the outliers occurring due
# to inaccuracies in the signal
def _clean_segment_features_of_outliers(segment_df, treshold=0.8):
    quant = segment_df.quantile(treshold)
    for col in segment_df.columns:
        if col.find('ts') == -1:
            segment_df = segment_df[segment_df[col] < quant[col]*2]
    return segment_df

# returns the x values for those samples in the signal, that are closest to some given y value
def _find_xs_for_y(y_s, y_val, sys_peak):
    diffs = abs(y_s - y_val)
    x_1 = diffs[:sys_peak].idxmin()
    x_2 = diffs[sys_peak:].idxmin()
    return x_1, x_2

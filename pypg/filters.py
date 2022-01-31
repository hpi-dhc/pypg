"""
=============================================
Filters for PPG Signals (:mod:`pypg.filters`)
=============================================

This module implements the most interesting filters found in the literature for
PPG signals.

Filters
----------
butterfy - Butterworth filter optimized for PPG signals.
chebyfy  - Chebyshev II filter optimized for PPG signals.
movefy   - Moving average filter.

Returns
----------
filtered : pandas.Series or ndarray
    Filtered PPG signal.

References
----------
Liang, Y., Elgendi, M., Chen, Z., & Ward, R. (2018). An optimal filter for
short photoplethysmogram signals. Scientific Data, 5(1), 180076.
https://doi.org/10.1038/sdata.2018.76

"""

import numpy as np
import pandas as pd
from scipy import signal

from .plots import simple_plot


def butterfy(ppg, cutoff_frequencies, sampling_frequency, filter_type='bandpass',
             filter_order=4, verbose=False, figure_path=None):
    """
    Implements the Butterworth filter with 'bandpass' as the filter_type and filter_order of
    4. Described by Liang et al. (2018) as the second best option for filtering PPG
    signals.

    Parameters
    ----------
    ppg : pandas.Series, ndarray
        The PPG signal.
    cutoff_frequencies : int or list
        The cutoff frequency(ies if bandpass) for the filter in Hz.
    sampling_frequency : int
        The sampling frequency of the signal in Hz.
    filter_type : str, optional
        Filter type (lowpass, highpass, bandpass). Defaults to 'bandpass'.
    filter_order : int, optional
        Filter Order. Defaults to 4.
    verbose : bool, optional
        Plots the signal. Defaults to False.
    figure_path : str, optional
        The path for the PPG plot to be saved.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    Returns
    ----------
    filtered: pandas.Series or ndarray
        Filtered PPG signal.

    References
    ----------
    Liang, Y., Elgendi, M., Chen, Z., & Ward, R. (2018). An optimal filter
    for short photoplethysmogram signals. Scientific Data, 5(1), 180076.
    https://doi.org/10.1038/sdata.2018.76
    """
    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
        signal_index = ppg.index
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or ndarray.')

    sos = signal.butter(filter_order, cutoff_frequencies, btype=filter_type,
                        fs=sampling_frequency, output='sos')
    filtered = signal.sosfiltfilt(sos, signal_values)

    if isinstance(ppg, pd.core.series.Series):
        filtered = pd.Series(filtered)
        filtered.index = signal_index

    if verbose:
        if isinstance(cutoff_frequencies, list):
            cut = ', '.join(str(s) for s in cutoff_frequencies)
            cut = ' ['+cut+']'
        elif isinstance(cutoff_frequencies, int):
            cut = str(cutoff_frequencies)
        label = 'Buterworth'+' '+filter_type+' '+'filter '+cut+' Hz'
        simple_plot(ppg=filtered, title='Filtered Signal', label=label, figure_path=figure_path)
    return filtered

def chebyfy(ppg, cutoff_frequencies, sampling_frequency, filter_type='bandpass',
            filter_order=4, band_attenuation=20, verbose=False, figure_path=None):
    """
    Imlements the Chebyshev II filter with 'bandpass' as the filter_type, a filter_order of 4,
    and a band_attenuation of 20 dB. Described by Liang et al. (2018) as the best filter for
    PPG signals.

    Parameters
    ----------
    ppg : pandas.Series or ndarray
            The PPG signal.
    cutoff_frequencies : int or list
            The cutoff frequency(ies if bandpass) for the filter in Hz.
    sampling_frequency : int
            The sampling frequency of the signal in Hz.
    filter_type : str, optional
            Filter type (lowpass, highpass, bandpass). Defaults to 'bandpass'.
    filter_order : int, optional
            Filter Order. Defaults to 4.
    band_attenuation : int, optional
            The attenuation required in the stop band.
            In decibels (dB), as a positive number. Defaults to 20.
    verbose : bool, optional
            Plots the signal. Defaults to False.
    figure_path : str, optional
            The path for the PPG plot to be saved.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    Returns
    ----------
    filtered: pandas.Series or ndarray
    Filtered PPG signal.

    References
    ----------
    Liang, Y., Elgendi, M., Chen, Z., & Ward, R. (2018). An optimal filter for
    short photoplethysmogram signals. Scientific Data, 5(1), 180076.
    https://doi.org/10.1038/sdata.2018.76
    """
    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
        signal_index = ppg.index
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or ndarray.')

    sos = signal.cheby2(filter_order, band_attenuation, cutoff_frequencies,
                        btype=filter_type, fs=sampling_frequency, output='sos')
    filtered = signal.sosfiltfilt(sos, signal_values)

    if isinstance(ppg, pd.core.series.Series):
        filtered = pd.Series(filtered)
        filtered.index = signal_index

    if verbose:
        if isinstance(cutoff_frequencies, list):
            cut = ', '.join(str(s) for s in cutoff_frequencies)
            cut = ' ['+cut+']'
        elif isinstance(cutoff_frequencies, int):
            cut = str(cutoff_frequencies)
        label = 'Chebyshev II'+' '+filter_type+' '+'filter '+cut+' Hz'
        simple_plot(ppg=filtered, title='Filtered Signal', label=label, figure_path=figure_path)
    return filtered

def movefy(ppg, size, verbose=False, figure_path=None):
    """
    Implements the moving average filter.

    Parameters
    ----------
    ppg : pandas.Series or ndarray
        The PPG signal.
    size : int
        Size of the np.ones mask to be convoluted.
    verbose : bool, optional
        Plot the signal. Defaults to False, by default False.
    figure_path : [type], optional
        The path for the PPG plot to be saved, by default None.

    Returns
    -------
    filtered: pandas.Series or ndarray
        Filtered PPG signal.

    Raises
    ------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    References
    ----------
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    """
    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
        signal_index = ppg.index
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or ndarray.')

    filtered = signal.convolve(signal_values, np.ones((size,))/size, mode='valid')

    if isinstance(ppg, pd.core.series.Series):
        filtered = pd.Series(filtered)
        padding = pd.Series(np.nan, index=range(len(signal_index) - len(filtered.index)))
        filtered = filtered.append(padding,ignore_index=True)
        filtered.index = signal_index

    if verbose:
        simple_plot(ppg=filtered, title='Filtered Signal',
                    label='Moving average filter', figure_path=figure_path)
    return filtered

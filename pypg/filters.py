"""
====================================
Filters for PPG Signals (:mod:`pypg.filters`)
====================================

Most interesting filters found in the literature for
PPG signals.

.. seealso::

    Liang Y, Elgendi M, Chen Z, Ward R. Analysis: An optimal filter for
    short photoplethysmogram signals. Sci Data 2018;5:1–12.
    https://doi.org/10.1038/sdata.2018.76.

Filters
----------

    chebyfy  - Chebyshev II filter optimized for PPG signals.
    butterfy - Butterworth filter optimized for PPG signals.
    movefy   - Moving average filter.


Returns
----------
    filtered : pandas.Series or ndarray
               Filtered PPG signal.
"""

import numpy as np
import pandas as pd
from scipy import signal

from .plots import simple_plot


def chebyfy(ppg, cutoff_frequencies, sampling_frequency, filter_type='bandpass',
            filter_order=4, band_attenuation=20, verbose=False, figure_path=None):
    """
    Chebyshev II filter with 'bandpass' as the filter_type, a filter_order of 4,
    and a band_attenuation of 20 dB. Described in the literature as the best filter for
    PPG signals.

    Parameters
    ----------
        ppg : pandas.Series or ndarray
              The raw PPG signal.
        cutoff_frequencies : int or list
              The cutoff frequency(ies if bandpass) for the filter in Hz.
        sampling_frequency : int
              The sampling frequency of the signal in Hz.
        filter_type : str, optional
              Filter type (low, high, band - pass). Defaults to 'bandpass'.
        filter_order : int, optional
              Filter Order. Defaults to 4.
        band_attenuation : int, optional
              The attenuation required in the stop band.
              In decibels (dB), as a positive number. Defaults to 20.
        verbose : bool, optional
              Plots the signal. Defaults to False.

    Returns
    ----------
        filtered : pandas.Series or ndarray
              Filtered PPG signal.

    References
    ----------
        Liang Y, Elgendi M, Chen Z, Ward R. Analysis: An optimal filter for
        short photoplethysmogram signals. Sci Data 2018;5:1–12.
        https://doi.org/10.1038/sdata.2018.76.
    """
    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or np.ndarray.')

    sos = signal.cheby2(filter_order, band_attenuation, cutoff_frequencies,
                        btype=filter_type, fs=sampling_frequency, output='sos')
    filtered = signal.sosfiltfilt(sos, signal_values)

    if isinstance(ppg, pd.core.series.Series):
        filtered = pd.Series(filtered)

    if verbose:
        simple_plot(filtered, 'Chebyshev II', filter_type, cutoff_frequencies, figure_path)
    return filtered

def butterfy(ppg, cutoff_frequencies, sampling_frequency, filter_type='bandpass',
             filter_order=4, verbose=False, figure_path=None):
    """
    Butterworth filter with 'bandpass' as the filter_type and filter_order of
    4. Described in the literature as the second best option for filtering PPG
    signals.

    Parameters
    ----------
        ppg : pandas.Series, ndarray
              The raw PPG signal.
        cutoff_frequencies : int or list
              The cutoff frequency(ies if bandpass) for the filter in Hz.
        sampling_frequency : int
              The sampling frequency of the signal in Hz.
        filter_type : str, optional
              Filter type (low, high, band - pass). Defaults to 'bandpass'.
        filter_order : int, optional
              Filter Order. Defaults to 4.
        verbose : bool, optional
              Plots the signal. Defaults to False.

    Returns
    ----------
        filtered : pandas.Series or ndarray
              Filtered PPG signal.

    References
    ----------
        Liang Y, Elgendi M, Chen Z, Ward R. Analysis: An optimal filter for
        short photoplethysmogram signals. Sci Data 2018;5:1–12.
        https://doi.org/10.1038/sdata.2018.76.
    """
    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or np.ndarray.')

    sos = signal.butter(filter_order, cutoff_frequencies, btype=filter_type,
                        fs=sampling_frequency, output='sos')
    filtered = signal.sosfiltfilt(sos, signal_values)

    if isinstance(ppg, pd.core.series.Series):
        filtered = pd.Series(filtered)

    if verbose:
        simple_plot(filtered, 'Buterworth', filter_type, cutoff_frequencies, figure_path)
    return filtered

def movefy(ppg, size, verbose=False, figure_path=None):
    """
    Moving average filter.

    Parameters
    ----------
    ppg : pandas.Series, ndarray
            The raw PPG signal.
    size : int
            Size of the np.ones mask to be convoluted.
    verbose : bool, optional
            Plot the signal. Defaults to False.

    Returns
    ----------
        filtered : pandas.Series or ndarray
            Filtered PPG signal.

    References
    ----------
        https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    """
    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or np.ndarray.')

    filtered = signal.convolve(signal_values, np.ones((size,))/size, mode='valid')

    if isinstance(ppg, pd.core.series.Series):
        filtered = pd.Series(filtered)

    if verbose:
        simple_plot(filtered, 'Moving Average', figure_path=figure_path)
    return filtered

"""
=============================================
Processess PPG Cycles (:mod:`pypg.cycles`)
=============================================

Identify characteristics and extract features
from PPG Cycles.

"""

import numpy as np
import pandas as pd
from scipy import signal

from .plots import marks_plot


def find_onset(ppg, sampling_frequency, factor=0.667, distance=None, height=None,
                      threshold=None, prominence=None, width=None, wlen=None,
                      rel_height=0.5, plateau_size=None, verbose=False):
    """
    Finds the local minima that correspond to the onsets/start of the cardiac cycle(s).

    Parameters
    ----------
        ppg : pandas.Series, ndarray
            The PPG signal.
        sampling_frequency : int
            The sampling frequency of the signal in Hz.
        factor: float, optional
            Number that is used to calculate the distance in relation to
            the sampling_frequency, by default 0.667 (or 66.7%).
        distance : number, optional
            Minimum horizontal distance (>=1) between the cycles start points, by default None.
            However, the function assumes (factor * sampling_frequency) when None is given.
            For more check the SciPy documentation.
        height : number or ndarray or sequence, optional
            Required height of peaks. Either a number, None, an array matching x or
            a 2-element sequence of the former. For more check the SciPy documentation.
        threshold : number or ndarray or sequence, optional
            Required threshold of peaks, the vertical distance to its neighboring samples.
            Either a number, None, an array matching x or a 2-element sequence of the former,
            by default None. For more check the SciPy documentation.
        prominence : number or ndarray or sequence, optional
            Required prominence of peaks. Either a number, None, an array matching x or
            a 2-element sequence of the former, by default None.
            For more check the SciPy documentation.
        width : number or ndarray or sequence, optional
            Required width of peaks in samples. Either a number, None, an array matching x
            or a 2-element sequence of the former, by default None.
            For more check the SciPy documentation.
        wlen : int, optional
            Used for calculation of the peaks prominences, thus it is only used if one of
            the arguments prominence or width is given, by default None.
            For more check the SciPy documentation.
        rel_height : float, optional
            Used for calculation of the peaks width, thus it is only used if width
            is given, by default 0.5 as defined by SciPy.
            For more check the SciPy documentation.
        plateau_size : number or ndarray or sequence, optional
            Required size of the flat top of peaks in samples. Either a number, None,
            an array matching x or a 2-element sequence of the former. , by default None
            For more check the SciPy documentation.

    Returns
    -------
        minima: np.ndarray
            Indices values in x with the start/onset of the PPG cycles.

    Raises
    ----------
        Exception
            When PPG values are neither pandas.Series nor ndarray.

    References
    ----------
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """

    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or ndarray.')

    if distance is None:
        distance=factor*sampling_frequency

    # inverts the signal to find the minimas
    peaks_data = signal.find_peaks(-signal_values, distance=distance, height=height,
                                threshold=threshold, prominence=prominence, width=width,
                                wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
    minima = peaks_data[0]
    if verbose:
        marks_plot(ppg, minima, figure_path=None)
    return minima

"""
=============================================
Extracts PPG Cycles (:mod:`pypg.cycles`)
=============================================

This module is intended to extract PPG cycles and their characteristics.

Cycles
----------
find_onset          - Find the cycle(s) onset.
find_with_template  - Return valid cycle(s) based on a template.

References
----------
Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D.
(2013). Systolic Peak Detection in Acceleration Photoplethysmograms
Measured from Emergency Responders in Tropical Conditions. PLoS ONE,
8(10), 1–11. https://doi.org/10.1371/journal.pone.0076585

Li, Q., & Clifford, G. D. (2012). Dynamic time warping and machine
learning for signal quality assessment of pulsatile signals.
Physiological Measurement, 33(9), 1491–1501.
https://doi.org/10.1088/0967-3334/33/9/1491
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from .plots import simple_plot, marks_plot


def find_onset(ppg, sampling_frequency, factor=0.667, distance=None, height=None,
                threshold=None, prominence=None, width=None, wlen=None,
                rel_height=0.5, plateau_size=None, verbose=False):
    """
    Finds the local minimun/minima that correspond to the onsets/start of the
    cardiac cycle(s).

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
    distance : number, optional
        Minimum horizontal distance (>=1) between the cycles start points,
        by default None. However, the function assumes (factor *
        sampling_frequency) when None is given. For more information check
        the SciPy documentation.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, None, an array matching x
        or a 2-element sequence of the former, by default None. For more information check
        the SciPy documentation.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its
        neighboring samples. Either a number, None, an array matching x or a
        2-element sequence of the former, by default None. For more
        information check the SciPy documentation.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, None, an array
        matching x or a 2-element sequence of the former, by default None.
        For more information check the SciPy documentation.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, None, an array
        matching x or a 2-element sequence of the former, by default None.
        For more information check the SciPy documentation.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used
        if one of the arguments prominence or width is given, by default
        None. For more information check the SciPy documentation.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if
        width is given, by default 0.5 as defined by SciPy, by
        default None. For more information check the SciPy documentation.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        None, an array matching x or a 2-element sequence of the former, by
        default None. For more information check the SciPy documentation.
    verbose : boolean, optinal
        Verbose is used to print a graph with the PPG signal and the values
        of each cycle onset, by default False.

    Returns
    -------
    minimum/minima: np.ndarray
        Indice(s) value(s) in x with the start/onset of the PPG cycle(s).

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D.
    (2013). Systolic Peak Detection in Acceleration Photoplethysmograms
    Measured from Emergency Responders in Tropical Conditions. PLoS ONE,
    8(10), 1–11. https://doi.org/10.1371/journal.pone.0076585
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

    # invert the signal to find the minimas
    peaks_data = signal.find_peaks(-signal_values, distance=distance, height=height,
                                threshold=threshold, prominence=prominence, width=width,
                                wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
    minima = peaks_data[0]
    if verbose:
        marks_plot(signal_values, minima)
    return minima

def find_with_template(ppg, sampling_frequency, return_type='original', factor=0.667,
                      distance=None, height=None, threshold=None, prominence=None, width=None,
                      wlen=None, rel_height=0.5, plateau_size=None, correlation_threshold=0.8,
                      verbose=False):
    """
    Finds PPG cycles in a PPG segment based on the method suggested by Li and
    Clifford (2012). All detected cycles in the same window are combined into a
    custom PPG signal template. Individual cycles are then compared with the
    template using two signal quality indices (SQI): (1) direct linear
    correlation and (2) direct linear correlation between the cycle, re-sampled
    to match the template length, and the template itself. Only if both
    correlations lie above a threshold, the cycle is considered valid.

    Parameters
    ----------
    ppg : pandas.Series, ndarray
        The PPG signal.
    sampling_frequency : int
        The sampling frequency of the signal in Hz.
    return_type: strm, optional
        The type of values to be returned (original or index), by default
        "original". Original returns a list of pd.Series or np.ndarray with the
        original data partioned into cycles. Index returns a list of tuples with
        the indexes of the begining and ending of each cycle in the original data.
    factor: float, optional
        Number that is used to calculate the distance in relation to the
        sampling_frequency, by default 0.667 (or 66.7%). The factor is based
        on the paper by Elgendi et al. (2013).
    distance : number, optional
        Minimum horizontal distance (>=1) between the cycles start points,
        by default None. However, the function assumes (factor *
        sampling_frequency) when None is given. For more information check
        the SciPy documentation.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, None, an array matching x
        or a 2-element sequence of the former, by default None. For more information check
        the SciPy documentation.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its
        neighboring samples. Either a number, None, an array matching x or a
        2-element sequence of the former, by default None. For more
        information check the SciPy documentation.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, None, an array
        matching x or a 2-element sequence of the former, by default None.
        For more information check the SciPy documentation.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, None, an array
        matching x or a 2-element sequence of the former, by default None.
        For more information check the SciPy documentation.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used
        if one of the arguments prominence or width is given, by default
        None. For more information check the SciPy documentation.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if
        width is given, by default 0.5 as defined by SciPy, by
        default None. For more information check the SciPy documentation.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        None, an array matching x or a 2-element sequence of the former, by
        default None. For more information check the SciPy documentation.
    correlation_threshold : float, optional
        Number that is used to calculate the correlation threshold from the
        template to the individual cycles, by default 0.8.
    verbose : boolean, optinal
        Verbose is used to print different graphs such as the onset of each
        cycle, the cycle template and all valid cycles, by default False.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.
        When return_type is neither 'original' or 'index'.

    Returns
    ----------
    cycles: list
        If "original" returns a list of pd.Series or np.ndarray with the valid PPG cycles.
        If "index" returns a list of tuples with the index of begining and
        ending of each valid PPG cycle.

    References
    ----------
    Li, Q., & Clifford, G. D. (2012). Dynamic time warping and machine
    learning for signal quality assessment of pulsatile signals.
    Physiological Measurement, 33(9), 1491–1501.
    https://doi.org/10.1088/0967-3334/33/9/1491
    """

    if isinstance(ppg, pd.core.series.Series):
        signal_values = ppg.values
    elif isinstance(ppg, np.ndarray):
        signal_values = ppg
    else:
        raise Exception('PPG values not accepted, enter either'
                        +' pandas.Series or ndarray.')

    if return_type not in ['index', 'original']:
        raise Exception('Wrong value for return_type.')

    initial_cycle_starts = find_onset(signal_values, sampling_frequency, factor, distance,
                                      height, threshold, prominence, width, wlen, rel_height,
                                      plateau_size, verbose)

    if len(initial_cycle_starts) <= 1:
        return []
    template_length = math.floor(np.median(np.diff(initial_cycle_starts)))
    cycle_starts = initial_cycle_starts[:-1]
    while cycle_starts[-1] + template_length > len(signal_values):
        cycle_starts = cycle_starts[:-1]
    template = []
    for i in range(template_length):
        template.append(np.mean(signal_values[cycle_starts + i]))

    corr_coef = []
    for cycle_start in cycle_starts:
        corr_coef.append(np.corrcoef(template,
                            signal_values[cycle_start:cycle_start+template_length])[0,1])

    valid_indices = np.argwhere(np.array(corr_coef) >= correlation_threshold)
    if (len(valid_indices) > len(cycle_starts) / 2) and len(valid_indices) > 1:
        cycle_starts = cycle_starts[np.squeeze(valid_indices)]
        template2 = []
        for i in range(template_length):
            template2.append(np.mean(signal_values[cycle_starts + i]))
        template = template2

    if verbose:
        simple_plot(template, title='Cycle Template')

    # check correlation of cycles with template SQI1: Pearson Correlation
    sqi1_corr = []
    for cycle_start in cycle_starts:
        corr, _ = stats.pearsonr(template, signal_values[cycle_start:cycle_start+template_length])
        sqi1_corr.append(corr)

    # SQI2: Pearson Correlation between the cycle, re-sampled to match the
    # template length, and the template itself
    sqi2_corr = []
    for cycle_start in cycle_starts:
        cycle_end = initial_cycle_starts[np.squeeze(
                                            np.argwhere(initial_cycle_starts==cycle_start)) + 1]
        corr, _ = stats.pearsonr(template, signal.resample(
                                    signal_values[cycle_start:cycle_end], template_length))
        sqi2_corr.append(corr)

    # filter for correlation >= correlation_threshold
    corrs = np.array([sqi1_corr, sqi2_corr]).transpose()
    cycle_starts = cycle_starts[np.all(corrs >= correlation_threshold, axis=1)]

    if verbose:
        print('Valid Cycles Detected')
        for cycle_start in cycle_starts:
            plt.plot(signal_values[cycle_start:cycle_start+template_length])
        plt.show()
        plt.close()

    cycles_indices = []
    for cycle_start in cycle_starts:
        cycle_end = initial_cycle_starts[np.squeeze(
                                            np.argwhere(initial_cycle_starts==cycle_start)) + 1]
        if (cycle_end - cycle_start) > template_length*1.2:
            cycle_end = cycle_start + template_length
        cycles_indices.append((cycle_start, cycle_end))

    if return_type == 'original':
        cycles = []
        for i, cycle_index in enumerate(cycles_indices):
            if isinstance(ppg, pd.core.series.Series):
                cycles.append(ppg.iloc[cycle_index[0]:cycle_index[1]])
            elif isinstance(ppg, np.ndarray):
                cycles.append(ppg[cycle_index[0]:cycle_index[1]])
        return cycles
    if return_type == 'index':
        return cycles_indices

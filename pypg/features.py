"""
=============================================
Extracts Features from PPG segments and cycles (:mod:`pypg.features`)
=============================================

This module is intended to extract features from PPG segments and cycles.

Features
----------
time             - Extract features from the time domain for a PPG segment.
time_cycle       - Extract features from the time domain for a PPG cycle.
nonlinear        - Extract features from the time domain computing non-linear relations for a PPG segment.
nonlinear_cycle  - Extract features from the time domain computing non-linear relations for a PPG cycle.
statistical      - Extract features from the time domain computing statistical values for a PPG segment.
statistical_cyle - Extract features from the time domain computing statistical values for a PPG cycle.
sdppg            - Extract features from the time domain's second derivative (SDPPG or APG) for a PPG segment.
sdppg_cycle      - Extract features from the time domain's second derivative (SDPPG or APG) for a PPG cycle.
frequency        - Extract features from the frequency domain for a PPG segment.
hrv              - Extract features from the computed Heart-Rate-Variability signal for a PPG segment.

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

from operator import index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nolds
from scipy import signal, stats, interpolate

from .cycles import find_with_template, find_with_SNR
from .plots import marks_plot


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

    cycles = find_with_template(ppg, sampling_frequency, factor=factor, verbose=verbose)
    # all signal peaks
    all_peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if verbose:
        print('All Peaks: ', all_peaks)
    if len(cycles) == 0:
        return pd.DataFrame()

    cur_index = 0
    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = pd.concat([segment_features, time_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose)],
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
        print(segment_features)
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
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # TODO: @Ari: Sampling Frequncy considered???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() # TODO: @Ari: If we do it for each cycle seperately, features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)
    max_amplitude = ppg.max()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if len(peaks) == 0:
        return pd.DataFrame()
    sys_peak_ts = ppg.index[peaks[0]] # TODO: @Ari: ASSUMING SYS PEAK IS ALWAYS FIRST MAXIMA > clean signal assumption (maybe add checks e.g. check peak height dictionary?)

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
                        'DT': (ppg.index.max() - sys_peak_ts).total_seconds(),
                        'CT': (ppg.index.max() - ppg.index.min()).total_seconds(),
                        'SUT_VAL': (ppg.values[peaks[0]])
                        }, index=[0])

    for p_value in [10, 25, 33, 50, 66, 75]:
        p_ampl = p_value / 100 * max_amplitude
        x_1, x_2 = _find_xs_for_y(ppg, p_ampl, peaks[0])
        if verbose:
            plt.scatter([x_1, x_2], ppg[[x_1, x_2]])
        cycle_features.loc[0, 'DW_'+str(p_value)] = (x_2 - sys_peak_ts).total_seconds()
        cycle_features.loc[0, 'SW_'+str(p_value)] = (sys_peak_ts - x_1).total_seconds()
        cycle_features.loc[0, 'DW_SW_sum_'+str(p_value)] = (x_2 - x_1).total_seconds()
        cycle_features.loc[0, 'DW_SW_ratio_'+str(p_value)] = (x_2 - sys_peak_ts) / (sys_peak_ts - x_1)
    
    if verbose:
        plt.show()
    
    return cycle_features


def nonlinear(ppg, sampling_frequency, factor=0.6667, unit='ms', verbose=False):
    """
    Extracts non-linear features from PPG cycles in a give PPG segment. Returns a pandas.DataFrame in
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
        A dataframe with the non-linear features for each valid
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

    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = pd.concat([segment_features, nonlinear_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose)],
                                         ignore_index=True)

    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    # remove outliers
    segment_features = _clean_segment_features_of_outliers(segment_features)

    if verbose:
        print('Cycle Features within Segment and no Outliers:')
        print(segment_features)

    return segment_features

def nonlinear_cycle(ppg, sampling_frequency, factor=0.667, unit='ms', verbose=False):
    """
    Extracts nonlinear features for a PPG cycle. Returns a pandas.Series.

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
        A dataframe with the nonlinear features for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # TODO: @Ari: Sampling Frequncy considered???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() # TODO: @Ari: If we do it for each cycle seperately, features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)
    ppg_cycle_duration = (ppg.index.max() - ppg.index.min()).total_seconds()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if len(peaks) == 0:
        return pd.DataFrame()
    sys_peak_ts = ppg.index[peaks[0]] # TODO: @Ari: ASSUMING SYS PEAK IS ALWAYS FIRST MAXIMA > clean signal assumption (maybe add checks e.g. check peak height dictionary?)
    sys_peak = (sys_peak_ts - ppg.index.min()).total_seconds()

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # features
    cycle_features = pd.DataFrame({
                        'ratio_WL_cycle_mean': (ppg_cycle_duration / np.mean(ppg.values)), # TODO: Add definition to description
                        'ratio_SUT_WL_DT': ((sys_peak / ppg_cycle_duration) / (ppg_cycle_duration - sys_peak)),
                        'ratio_SUT_DT': (sys_peak / (ppg_cycle_duration - sys_peak)),
                        'ratio_cycle_mean_cycle_var': (np.mean(ppg.values) / np.var(ppg.values))
                        }, index=[0])

    if verbose:
        plt.show()

    return cycle_features


def statistical(ppg, sampling_frequency, factor=0.6667, unit='ms', verbose=False):
    """
    Extracts statistical features from PPG cycles in a give PPG segment. Returns a pandas.DataFrame in
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
        A dataframe with the statistical features for each valid
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

    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = pd.concat([segment_features, statistical_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose)],
                                         ignore_index=True)

    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    # remove outliers
    segment_features = _clean_segment_features_of_outliers(segment_features)

    if verbose:
        print('Cycle Features within Segment and no Outliers:')
        print(segment_features)

    return segment_features

def statistical_cycle(ppg, sampling_frequency, factor=0.667, unit='ms', verbose=False):
    """
    Extracts statistical features for a PPG cycle. Returns a pandas.Series.

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
        A dataframe with the statistical features for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # TODO: @Ari: Sampling Frequncy considered???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() # TODO: @Ari: If we do it for each cycle seperately, features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if len(peaks) == 0:
        return pd.DataFrame()

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # features
    cycle_features = pd.DataFrame({
                        'ppg_mean': (np.mean(ppg.values)),
                        'ppg_var': (np.var(ppg.values)),
                        'ppg_skewness': (stats.skew(ppg.values)),
                        'ppg_kurtosis': (stats.kurtosis(ppg.values))
                        }, index=[0])

    if verbose:
        plt.show()

    return cycle_features


def sdppg(ppg, sampling_frequency, factor=0.6667, unit='ms', verbose=False):
    """
    Extracts features from the second derivative (SDPPG or APG) of the PPG cycles in a give PPG segment. Returns a pandas.DataFrame in
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
        A dataframe with the features from the second derivative (SDPPG or APG) of the PPG cycles in seconds for each valid
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

    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = pd.concat([segment_features, sdppg_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose)],
                                         ignore_index=True)
    
    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    # remove outliers
    segment_features = _clean_segment_features_of_outliers(segment_features)

    if verbose:
        print('Cycle Features within Segment and no Outliers:')
        print(segment_features)

    return segment_features

def sdppg_cycle(ppg, sampling_frequency, factor=0.667, unit='ms', verbose=False):
    """
    Extracts features from the second derivative (SDPPG or APG) for the PPG cycles. Returns a pandas.Series.

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
        A dataframe with the features from the second derivative (SDPPG or APG) in seconds for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # TODO: @Ari: Sampling Frequncy considered???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() # TODO: @Ari: If we do it for each cycle seperately, features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if len(peaks) == 0:
        return pd.DataFrame()
    sys_peak_ts = ppg.index[peaks[0]] # TODO: @Ari: ASSUMING SYS PEAK IS ALWAYS FIRST MAXIMA > clean signal assumption (maybe add checks e.g. check peak height dictionary?)
    sys_peak = (sys_peak_ts - ppg.index.min()).total_seconds()

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # second derviative of the PPG signal
    sdppg_signal = np.gradient(np.gradient(ppg.values))
    if sdppg_signal.size == 0:
        return pd.DataFrame()

    # TODO: features !!! WRITE DOWN REFERENCE NICELY IN TOP DESCRIPTION !!!
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0076585
    # https://www.researchgate.net/figure/Important-points-on-PPG-VPG-and-AVG-The-first-top-signal-is-an-epoch-of-PPG-The_fig4_339604879
    # a is global maximum;
    # b is the global minimum; (maybe get systolic_peak_index and get local minimum between a_index and systolic_peak_index)
    # e and c are the local maximum;
    # d is the local minimum in APG between e and c;
    # D is local minimum in APG after e
    # 
    # and Gaurav Paper

    # TODO: maybe add systolic peak index as reference for indices detection
    
    sdppg_signal_peaks, peak_dict = signal.find_peaks(sdppg_signal, height=0)
    sdppg_signal_valleys, valley_dict = signal.find_peaks(-sdppg_signal, height=0)
    if len(sdppg_signal_peaks) == 0 or len(sdppg_signal_valleys) == 0:
        return pd.DataFrame()

    # initialize SDPPG indices and variables
    a_index = b_index = c_index = d_index = e_index = None
    a_val = b_val = c_val = d_val = e_val = None
    AD = AE = CD = DF = D = E = None
    ratio_b_a = ratio_c_a = ratio_d_a = ratio_e_a = None
    ratio_AD_AF = ratio_CD_AF = ratio_AE_AF = ratio_DF_AF = None

    
    # a wave
    if len(sdppg_signal_peaks) != 0:
        
        a_index = (sdppg_signal_peaks[np.argmax(peak_dict['peak_heights'])])
        
        ## subset signal + dictionary to exclude a index + magnitude
        old_len = len(sdppg_signal_peaks)
        sdppg_signal_peaks = sdppg_signal_peaks[sdppg_signal_peaks > a_index]
        peak_dict['peak_heights'] = peak_dict['peak_heights'][old_len - len(sdppg_signal_peaks):]
        
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > a_index]
        valley_dict['peak_heights'] = valley_dict['peak_heights'][old_len - len(sdppg_signal_valleys):]
    
    # b wave
    if len(sdppg_signal_valleys) != 0:
        
        b_index = (sdppg_signal_valleys[np.argmax(valley_dict['peak_heights'])])
        
        ## subset signal + dictionary to exclude b index + magnitude
        old_len = len(sdppg_signal_peaks)
        sdppg_signal_peaks = sdppg_signal_peaks[sdppg_signal_peaks > b_index]
        peak_dict['peak_heights'] = peak_dict['peak_heights'][old_len - len(sdppg_signal_peaks):]
        
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > b_index]
        valley_dict['peak_heights'] = valley_dict['peak_heights'][old_len - len(sdppg_signal_valleys):]
    
    # c and e wave
    if len(sdppg_signal_peaks) >= 2:
        
        peaks = peak_dict['peak_heights'].argsort()[-2:]
        
        if len(peaks) == 2:
            if peaks[0] < peaks[1]:
                c_index = (sdppg_signal_peaks[peaks[0]])
                e_index = (sdppg_signal_peaks[peaks[1]])
            else:
                e_index = (sdppg_signal_peaks[peaks[0]])
                c_index = (sdppg_signal_peaks[peaks[1]])
            
            ## subset signal + dictionary to exlcude c and e indeces + magnitudes
            old_len = len(sdppg_signal_valleys)
            sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > c_index]
            valley_dict['peak_heights'] = valley_dict['peak_heights'][old_len - len(sdppg_signal_valleys):]
    
    # d wave
    if len(sdppg_signal_valleys) >= 1 and (c_index or e_index):
        
        ## get only valleys between c and e
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys < e_index]
        
        if old_len > len(sdppg_signal_valleys):
            valley_dict['peak_heights'] = valley_dict['peak_heights'][:-(old_len - len(sdppg_signal_valleys))]
        
        if len(valley_dict['peak_heights']) != 0:
            d_index = (sdppg_signal_valleys[valley_dict['peak_heights'].argmax()])
    

    ## SDPPG Signal values
    ### a
    if a_index:
        a_val = (sdppg_signal[a_index])
    
    ### b
    if b_index:
        b_val = (sdppg_signal[b_index])
    
    ### c
    if c_index:
        c_val = (sdppg_signal[c_index])
    
    ### d
    if d_index:
        d_val = (sdppg_signal[d_index])
    
    ### e
    if e_index:
        e_val = (sdppg_signal[e_index])
   
    
    if verbose:
        plt.figure(figsize=(17,6))
        plt.plot(sdppg_signal, label = 'SDPPG Signal')
        plt.scatter(np.array([a_index, b_index, c_index, d_index, e_index]), np.array([a_val, b_val, c_val, d_val, e_val]), c='g', label='Representative Points')
        plt.title('SDPPG Signal Wave Detection')
        plt.xlabel('Time')
        plt.ylabel('SDPPG')
        plt.legend()
    
    
    ## Time values of characteristics
    ### Location of c with respect to time (AD)
    if c_index:
        AD = (c_index / sampling_frequency)
    
    ### Location of d with respect to time (AE)
    if d_index:
        AE = (d_index / sampling_frequency)
    
    ### Difference of peak and dicrotic notch with respect to time (CD)
    if c_index and sys_peak:
        CD = ((c_index - sys_peak) / sampling_frequency)
    
    ### Difference of dicrotic notch and end of signal with respect to time (DF)
    if c_index:
        DF = ((len(sdppg_signal) - c_index)  / sampling_frequency)
    
    
    ## PPG Signal values
    ### Dicrotic notch (D')
    if c_index:
        D = (ppg.values[c_index])
    
    ### Diastolic point (E')
    if d_index:
        E = (ppg.values[d_index])
    
    
    ## Ratios
    ### First valley to the peak value of APG signal (b/a)
    if b_index and a_index:
        ratio_b_a = (b_val / a_val)
    
    ### Dicrotic notch to the peak value of APG signal (c/a)
    if c_index and a_index:
        ratio_c_a = (c_val / a_val)
    
    ### Diastolic point to the peak value of APG signal (d/a)
    if d_index and a_index:
        ratio_d_a = (d_val / a_val)
    
    ### e to the peak value of APG signal (e/a)
    if e_index and a_index:
        ratio_e_a = (e_val / a_val)
    
    ### Location of dicrotic notch with respect to the length of window (AD/AF)
    if c_index:
        ratio_AD_AF = (c_index / len(sdppg_signal))
    
    ### Difference of location of peak and dicrotic notch with respect to the length of window (CD/AF)
    if c_index and sys_peak:
        ratio_CD_AF = ((c_index - sys_peak) / len(sdppg_signal))
    
    ### Location of diastolic point on PPG signal with respect to the length of window (AE/AF)
    if d_index:
        ratio_AE_AF = (d_index / len(sdppg_signal))
    
    ### Difference of dichrotic notch and end with  respect ot length of window (DF/AF)
    if c_index:
        ratio_DF_AF = ((len(sdppg_signal) - c_index) / len(sdppg_signal))

    cycle_features = pd.DataFrame({
                        'a_val': a_val if a_val else np.nan,
                        'b_val': b_val if b_val else np.nan,
                        'c_val': c_val if c_val else np.nan,
                        'd_val': d_val if d_val else np.nan,
                        'e_val': e_val if e_val else np.nan,
                        'AD': AD if AD else np.nan,
                        'AE': AE if AE else np.nan,
                        'CD': CD if CD else np.nan,
                        'DF': DF if DF else np.nan,
                        'D': D if D else np.nan,
                        'E': E if E else np.nan,
                        'ratio_b_a': ratio_b_a if ratio_b_a else np.nan,
                        'ratio_c_a': ratio_c_a if ratio_c_a else np.nan,
                        'ratio_d_a': ratio_d_a if ratio_d_a else np.nan,
                        'ratio_e_a': ratio_e_a if ratio_e_a else np.nan,
                        'ratio_AD_AF': ratio_AD_AF if ratio_AD_AF else np.nan,
                        'ratio_CD_AF': ratio_CD_AF if ratio_CD_AF else np.nan,
                        'ratio_AE_AF': ratio_AE_AF if ratio_AE_AF else np.nan,
                        'ratio_DF_AF': ratio_DF_AF if ratio_DF_AF else np.nan,
                        }, index=[0])

    if verbose:
        plt.show()

    return cycle_features


def frequency(ppg, sampling_frequency, transformMethod='welch', cutoff_freq=12.5, interval_size=5, verbose=False):
    """
    Extracts frequency features from PPG cycles in a give PPG segment. Returns a pandas.DataFrame in
    which each line contains the features for a given valid cycle (as described
    by Li et al.) from the PPG segment given as input.

    Parameters
    ----------
    ppg : pandas.Series, ndarray
        The PPG signal.
    sampling_frequency : int
        The sampling frequency of the signal in Hz.
    transformMethod : str
        The method used for transforming the time signal to the frequency domain.
    cutoff_freq : int
        The frequency threshold used to exclude frequencies above threshold from analysis.
    interval_size : int
        The size of the interval used to split the frequency spectrum.
    verbose : bool, optional
        If True, plots the features, by default False.

    Raises
    ----------
    Exception
        When PPG values are neither pandas.Series nor ndarray.

    Returns
    -------
    segment_features : pd.DataFrame
        A dataframe with the frequency features in seconds for each valid
        cycle in the PPG segment.
    """
    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

     # Transform signal from time to frequency domain
    freq, mag = _transformSigfromTimetoFreq(ppg.values, sampling_frequency, transformMethod=transformMethod, verbose=verbose)
    if freq.size == 0 or mag.size == 0:
        return pd.DataFrame()

    # Initialize features dataframe
    segment_features = pd.DataFrame()

    # Cut off frequencies
    freq_cut = freq[freq <= cutoff_freq]
    mag_cut = mag[freq <= cutoff_freq]

    if verbose:
        plt.semilogy(freq_cut, mag_cut)
        plt.title('CutOff')
        plt.xlabel('Frequency')
        plt.xlim((freq_cut.min(), freq_cut.max()))
        plt.ylim((mag_cut.min(), mag_cut.max()))
        plt.ylabel('Magnitude')
        plt.show()

    ## -> Paper Wang # TODO: Add description to the top
    for start_index in np.arange(0, cutoff_freq, interval_size):
        end_index = start_index + interval_size
        if mag_cut[(freq_cut >= start_index) & (freq_cut < end_index)].size != 0:
            segment_features.loc[0, 'freqInt' + str(start_index) + '_' + str(end_index)] = np.nanmean(mag_cut[(freq_cut >= start_index) & (freq_cut < end_index)])
        else:
            segment_features.loc[0, 'freqInt' + str(start_index) + '_' + str(end_index)] = 0

    ## -> Paper Slapnicar # TODO: Add description to the top
    sorted_mag = mag_cut[np.argsort(mag_cut)[::-1]]
    if len(sorted_mag) >= 3:
        top3_psd = sorted_mag[:3]
    else:
        top3_psd = sorted_mag[:len(sorted_mag)]

    for i in np.arange(0, len(top3_psd)):
        segment_features.loc[0, 'mag_top' + str(i+1)] = top3_psd[i]

    sorted_freq = freq_cut[np.argsort(mag_cut)[::-1]]
    if len(sorted_freq) >= 3:
        top3_freq = sorted_freq[:3]
    else:
        top3_freq = sorted_freq[:len(sorted_freq)]

    for i in np.arange(0, len(top3_freq)):
        segment_features.loc[0, 'freq_top' + str(i+1)] = top3_freq[i]

    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    return segment_features


def hrv(ppg, sampling_frequency, factor=0.6667, unit='ms', verbose=False):
    """
    Extracts hrv features from a given PPG segment. 
    Returns a pandas.DataFrame in which each line contains the features 
    from the PPG segment given as input.

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
        A dataframe with the hrv-domain features in seconds for the PPG segment.
    """
    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')
    
    cycles = find_with_SNR(ppg, sampling_frequency, factor=0.6667, verbose=verbose)
    if len(cycles) == 0:
        return pd.DataFrame()
    
    # all signal peaks
    all_peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]

    if verbose:
        print('All Peaks: ', all_peaks)

    sys_peak_ts_curr = 0
    ibi_series = pd.Series()
    for i, peak in enumerate(all_peaks):
        sys_peak_ts_prior = sys_peak_ts_curr
        sys_peak_ts_curr = ppg.index[peak]

        if i > 0:
            ibi_series = ibi_series.append(pd.Series([sys_peak_ts_curr - sys_peak_ts_prior]), ignore_index=True)
    
    if len(ibi_series) < 5: # TODO: Check for minimum size (Ultra short HRV)
        if verbose:
            print('PPG Segment too small! Please provide bigger PPG segment')
        return pd.DataFrame()

    temporalHRVFeatures = _temporal_hrv(ibi_series)
    frequencyHRVFeatures = _frequency_hrv(ibi_series, sampling_frequency)

    segment_features = pd.concat([temporalHRVFeatures.reset_index(drop=True), frequencyHRVFeatures], axis=1)
 
    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    return segment_features


# compute temporal and frequency features of HRV
def _temporal_hrv(ibi_series):
    
    if isinstance(ibi_series, np.ndarray):
        ibi_series = pd.Series(ibi_series)
    elif not isinstance(ibi_series, pd.core.series.Series):
        raise Exception('Signal values not accepted, enter a pandas.Series or ndarray.')
    
    window = 5
    nn_threshold = 50 # TODO: @Ari: HOW TO SET THIS VALUE? > IBI_SERIES VALUES around 0.88 ish. Affect computation of nn_xx /// was 50 before
    
    # Prepare data
    instantaneous_hr = 60 / (ibi_series / 1000) # TODO: @Ari: why divided by 1000? from ms to s?
    rolling_mean_hr = instantaneous_hr.rolling(window).mean()
    rolling_24h = ibi_series.rolling(window)
    
    # Precalculate data for standard indexes
    nn_diff = np.diff(ibi_series)
    nn_xx = np.sum(np.abs(nn_diff) > nn_threshold)
    
    # Precalculate data for geometrical indeces
    bin_size = 7.8125
    hist_middle = (ibi_series.values.min() + ibi_series.values.max()) / 2
    hist_bound_lower = hist_middle - bin_size * np.ceil((hist_middle - ibi_series.values.min()) / bin_size)
    hist_length = int(np.ceil((ibi_series.values.max() - hist_bound_lower) / bin_size) + 1)
    hist_bins = hist_bound_lower + np.arange(hist_length) * bin_size
    hist, _ = np.histogram(ibi_series.values, hist_bins)
    hist_height = np.max(hist)
    
    # Calculate TINN measure
    hist_max_index = np.argmax(hist)
    min_n = 0
    min_m = len(hist) - 1
    min_error = np.finfo(np.float64).max
    # Ignore bins that do not contain any intervals
    nonzero_indices = np.nonzero(hist)
    for n in range(hist_max_index):
        for m in reversed(range(hist_max_index + 1, len(hist))):
            # Create triangular interpolation function for n and m
            tri_interp = interpolate.interp1d(
                [n, hist_max_index, m],
                [0, hist[hist_max_index], 0],
                bounds_error=False,
                fill_value=0
            )
            # Square difference of histogram and triangle
            error = np.trapz(
                [(hist[t] - tri_interp(t)) ** 2 for t in nonzero_indices],
                [hist_bins[t] for t in nonzero_indices]
            )
            if min_error > error:
                min_n = n
                min_m = m
                min_error = error
    n = hist_bins[min_n]
    m = hist_bins[min_m]
	
	# Non-Linear Parameters
    tolerance = ibi_series.std() * 0.2
	
    # features
    temporalHRVFeatures = pd.DataFrame({
                            'SampEn': float(nolds.sampen(ibi_series.to_numpy(), 2, tolerance)),
                            'MeanNN': ibi_series.mean(),
                            'MeanHR': instantaneous_hr.mean(),
                            'MaxHR': rolling_mean_hr.max(),
                            'MinHR': rolling_mean_hr.min(),
                            'STDHR': instantaneous_hr.std(),
                            'SDNN': np.std(ibi_series),
                            'SDNNindex': rolling_24h.std().mean(),
                            'SDANN': rolling_24h.mean().std(),
                            'RMSSD': np.sqrt(np.mean(nn_diff ** 2)),
                            f'NN{nn_threshold}': nn_xx,
                            f'pNN{nn_threshold}': nn_xx / len(ibi_series.values) * 100,
                            'HRVTriangularIndex': len(ibi_series.values) / hist_height,
                            'TINN': m - n}, 
                            index=[0])
        
    return temporalHRVFeatures

def _frequency_hrv(ibi_series, sampling_frequency):
    
    if isinstance(ibi_series, np.ndarray):
        ibi_series = pd.Series(ibi_series)
    elif not isinstance(ibi_series, pd.core.series.Series):
        raise Exception('Signal values not accepted, enter a pandas.Series or ndarray.')
	
	## TODO: @Ari: HYPERPARAMETERS to config dict
    fft_interpolation = 4.0
    
    vlf_limit = 0.04
    lf_limit = 0.15
    hf_limit = 0.4
    ulf_limit = 0
	
    freq, mag = _transformSigfromTimetoFreq(ibi_series.values, sampling_frequency, fft_interpolation=fft_interpolation)
    if freq.size == 0 or mag.size == 0:
        return pd.DataFrame()

    abs_index = freq <= hf_limit
    ulf_index = freq <= ulf_limit
    vlf_index = (freq >= ulf_limit) & (freq <= vlf_limit)
    lf_index = (freq >= vlf_limit) & (freq <= lf_limit)
    hf_index = (freq >= lf_limit) & (freq <= hf_limit)
	
	# Get power for each band by integrating over spectral density
    abs_power = np.trapz(mag[abs_index], freq[abs_index])
    ulf = np.trapz(mag[ulf_index], freq[ulf_index])
    vlf = np.trapz(mag[vlf_index], freq[vlf_index])
    lf = np.trapz(mag[lf_index], freq[lf_index])
    hf = np.trapz(mag[hf_index], freq[hf_index])
	
	# Normalized power for LF and HF band
    lf_nu = lf / (abs_power - vlf - ulf) * 100
    hf_nu = hf / (abs_power - vlf - ulf) * 100
	
	# Relative power of each band
    ulf_perc = (ulf / abs_power) * 100
    vlf_perc = (vlf / abs_power) * 100
    lf_perc = (lf / abs_power) * 100
    hf_perc = (hf / abs_power) * 100
	
	# Frequency with highest power
    vlf_peak = freq[vlf_index][np.argmax(mag[vlf_index])]
    lf_peak = freq[lf_index][np.argmax(mag[lf_index])]
    hf_peak = freq[hf_index][np.argmax(mag[hf_index])]
    
    freqencyHRVFeatures = pd.DataFrame({
                            'VLF_peak': vlf_peak if vlf_peak else np.nan,
                            'VLF_power': vlf if vlf else np.nan,
                            'VLF_power_log': np.log(vlf) if np.log(vlf) else np.nan,
                            'VLF_power_rel': vlf_perc if vlf_perc else np.nan,
                            'LF_peak': lf_peak if lf_peak else np.nan,
                            'LF_power': lf if lf else np.nan,
                            'LF_power_log': np.log(lf) if np.log(lf) else np.nan,
                            'LF_power_rel': lf_perc if lf_perc else np.nan,
                            'LF_power_norm': lf_nu if lf_nu else np.nan,
                            'HF_peak': hf_peak if hf_peak else np.nan,
                            'HF_power': hf if hf else np.nan,
                            'HF_power_log': np.log(hf) if np.log(hf) else np.nan,
                            'HF_power_rel': hf_perc if hf_perc else np.nan,
                            'HF_power_norm': hf_nu if hf_nu else np.nan,
                            'ratio_LF_HF': lf/hf if lf/hf else np.nan
                            }, index=[0])
    
    return freqencyHRVFeatures


# takes a dataframe of calculated features and removes the outliers occurring due
# to inaccuracies in the signal
def _clean_segment_features_of_outliers(segment_df, treshold=0.8):
    quant = segment_df.quantile(treshold)
    for col in segment_df.columns:
        if col.find('ts') == -1 and len(segment_df[col]) > 1:
            segment_df = segment_df[np.abs(segment_df[col]) < np.abs(quant[col]*2)]
    return segment_df

# returns the x values for those samples in the signal, that are closest to some given y value
def _find_xs_for_y(y_s, y_val, sys_peak):
    diffs = abs(y_s - y_val)
    x_1 = diffs[:sys_peak].idxmin()
    x_2 = diffs[sys_peak:].idxmin()
    return x_1, x_2

# transforms a given temporal signal into the spectral domain
# using different methods and allowing interpolation if required.
# returns the frequencies and their magnitudes.
def _transformSigfromTimetoFreq(timeSignal, sampling_frequency, transformMethod='welch', fft_interpolation=None, verbose=False):

    if fft_interpolation:
        x = np.cumsum(timeSignal)
        f_interpol = interpolate.interp1d(x, timeSignal, kind = "cubic", fill_value="extrapolate")
        t_interpol = np.arange(x[0], x[-1], fft_interpolation/sampling_frequency)
        
        timeSignal = f_interpol(t_interpol)
        sampling_frequency = fft_interpolation
    
    # TODO: Maybe add different methods
    if transformMethod == 'welch':
        freq, mag = signal.welch(timeSignal, sampling_frequency, window='hamming', scaling='density', detrend='linear')
    else:
        print('Transform method not available. Please select one of the following: [welch]')
    
    if verbose:
        plt.semilogy(freq, mag)
        plt.title('Spectral Analysis using Welch Method')
        plt.xlabel('Frequency')
        plt.ylabel('PSD')
        plt.show()
    
    return freq, mag
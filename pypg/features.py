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
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # ??? Sampling Frequncy considered ???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() ## ??? SHOULDNT IT BE MEAN VALUE: ppg - ppg.mean() ???  if we do it for each cycle individually features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)
    max_amplitude = ppg.max()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    sys_peak_ts = ppg.index[peaks[0]] # ??? ASSUMING SYS PEAK IS ALWAYS FIRST MAXIMA > clean signal assumption (maybe add checks) ???

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
                        'BPM': (60 / (ppg.index.max() - ppg.index.min()).total_seconds()), # ??? CHECK KURYLAK DEF ???
                        'CT': (ppg.index.max() - ppg.index.min()).total_seconds(),
                        'SUT_VAL': (ppg.values[sys_peak_ts])
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

    cur_index = 0
    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = segment_features.append(nonlinear_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose),
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
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # ??? Sampling Frequncy considered ???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() ## ??? SHOULDNT IT BE MEAN VALUE: ppg - ppg.mean() ???  if we do it for each cycle individually features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)
    ppg_cycle_duration = (ppg.index.max() - ppg.index.min()).total_seconds()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    sys_peak_ts = ppg.index[peaks[0]] # ??? ASSUMING SYS PEAK IS ALWAYS FIRST MAXIMA > clean signal assumption (maybe add checks) ???

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # features
    cycle_features = pd.DataFrame({
                        'ratio_WL_cycle_mean': (ppg_cycle_duration / np.mean(ppg.values)), # ??  WHERE DO DEFINITIONS COME FROM ??
                        'ratio_SUT_WL_DT': ((sys_peak_ts / ppg_cycle_duration) / (ppg_cycle_duration - sys_peak_ts)),
                        'ratio_SUT_DT': (sys_peak_ts / (ppg_cycle_duration - sys_peak_ts)),
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

    cur_index = 0
    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = segment_features.append(statistical_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose),
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
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # ??? Sampling Frequncy considered ???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() ## ??? SHOULDNT IT BE MEAN VALUE: ppg - ppg.mean() ???  if we do it for each cycle individually features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]

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

    cur_index = 0
    segment_features = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        segment_features = segment_features.append(sdppg_cycle(cycle,
                                         sampling_frequency, factor, unit, verbose=verbose),
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
        ppg.index = pd.to_datetime(ppg.index, unit=unit) # ??? Sampling Frequncy considered ???

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min() ## ??? SHOULDNT IT BE MEAN VALUE: ppg - ppg.mean() ???  if we do it for each cycle individually features based on ppg values might be wrong because offset will be different!!! (e.g. SUT_VAL or statistical values)

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    sys_peak_ts = ppg.index[peaks[0]] # ??? ASSUMING SYS PEAK IS ALWAYS FIRST MAXIMA > clean signal assumption (maybe add checks) ???

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # second derviative of the PPG signal
    sdppg_signal = np.gradient(np.gradient(ppg.values))

    # features !!! WRITE DOWN REFERENCE NICELY IN TOP DESCRIPTION !!!
    # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0076585
    # https://www.researchgate.net/figure/Important-points-on-PPG-VPG-and-AVG-The-first-top-signal-is-an-epoch-of-PPG-The_fig4_339604879
    # a is global maximum;
    # b is the global minimum; (maybe get systolic_peak_index and get local minimum between a_index and systolic_peak_index)
    # e and c are the local maximum;
    # d is the local minimum in APG between e and c;
    # D is local minimum in APG after e
    # 
    # and Gaurav Paper

    # TODO: Can be improved!
    # TODO: maybe add systolic peak index as reference for indices detection
    
    sdppg_signal_peaks, peak_dict = signal.find_peaks(sdppg_signal, height=0)
    sdppg_signal_valleys, valley_dict = signal.find_peaks(-sdppg_signal, height=0)
    
    # a wave
    if len(sdppg_signal_peaks) != 0:
        
        a_index = (sdppg_signal_peaks[np.argmax(peak_dict['peak_heights'])])
        
        ## subset signal + dictionary to exclude a index + magnitude
        old_len = len(sdppg_signal_peaks)
        sdppg_signal_peaks = sdppg_signal_peaks[sdppg_signal_peaks > a_index[0]]
        peak_dict['peak_heights'] = peak_dict['peak_heights'][old_len - len(sdppg_signal_peaks):]
        
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > a_index[0]]
        valley_dict['peak_heights'] = valley_dict['peak_heights'][old_len - len(sdppg_signal_valleys):]
    
    # b wave
    if len(sdppg_signal_valleys) != 0:
        
        b_index = (sdppg_signal_valleys[np.argmax(valley_dict['peak_heights'])])
        
        ## subset signal + dictionary to exclude b index + magnitude
        old_len = len(sdppg_signal_peaks)
        sdppg_signal_peaks = sdppg_signal_peaks[sdppg_signal_peaks > b_index[0]]
        peak_dict['peak_heights'] = peak_dict['peak_heights'][old_len - len(sdppg_signal_peaks):]
        
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > b_index[0]]
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
            sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > c_index[0]]
            valley_dict['peak_heights'] = valley_dict['peak_heights'][old_len - len(sdppg_signal_valleys):]
    
    # d wave
    if len(sdppg_signal_valleys) >= 1 and (c_index or e_index):
        
        ## get only valleys between c and e
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys < e_index[0]]
        
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
    if c_index and sys_peak_ts is not None:
        CD = ((c_index - sys_peak_ts) / sampling_frequency)
    
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
    if b_val and a_val:
        ratio_b_a = (b_val / a_val)
    
    ### Dicrotic notch to the peak value of APG signal (c/a)
    if c_val and a_val:
        ratio_c_a = (c_val / a_val)
    
    ### Diastolic point to the peak value of APG signal (d/a)
    if d_val and a_val:
        ratio_d_a = (d_val / a_val)
    
    ### e to the peak value of APG signal (e/a)
    if e_val and a_val:
        ratio_e_a = (e_val / a_val)
    
    ### Location of dicrotic notch with respect to the length of window (AD/AF)
    if c_index:
        ratio_AD_AF = (c_index / len(sdppg_signal))
    
    ### Difference of location of peak and dicrotic notch with respect to the length of window (CD/AF)
    if c_index and sys_peak_ts is not None:
        ratio_CD_AF = ((c_index - sys_peak_ts) / len(sdppg_signal))
    
    ### Location of diastolic point on PPG signal with respect to the length of window (AE/AF)
    if d_index:
        ratio_AE_AF = (d_index / len(sdppg_signal))
    
    ### Difference of dichrotic notch and end with  respect ot length of window (DF/AF)
    if c_index:
        ratio_DF_AF = ((len(sdppg_signal) - c_index) / len(sdppg_signal))

    cycle_features = pd.DataFrame({
                        'a_val': a_val,
                        'b_val': b_val,
                        'c_val': c_val,
                        'd_val': d_val,
                        'e_val': e_val,
                        'AD': AD,
                        'AE': AE,
                        'CD': CD,
                        'DF': DF,
                        'D': D,
                        'E': E,
                        'ratio_b_a': ratio_b_a,
                        'ratio_c_a': ratio_c_a,
                        'ratio_d_a': ratio_d_a,
                        'ratio_e_a': ratio_e_a,
                        'ratio_AD_AF': ratio_AD_AF,
                        'ratio_CD_AF': ratio_CD_AF,
                        'ratio_AE_AF': ratio_AE_AF,
                        'ratio_DF_AF': ratio_DF_AF,
                        }, index=[0])

    if verbose:
        plt.show()

    return cycle_features


def frequency(ppg, sampling_frequency, transformMethod, cutoff_freq=12.5, interval_size=0.5, verbose=False):
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
    freq, mag = _transformSigfromTimetoFreq(ppg, sampling_frequency, transformMethod, verbose)

	# features
    segment_features = pd.DataFrame()

    # Cut off frequencies
    freq_cut = freq[freq <= cutoff_freq]
    mag_cut = mag[freq <= cutoff_freq]

    if verbose:
        plt.semilogy(freq_cut, mag_cut)
        plt.title('CutOff')
        plt.xlabel('Frequency')
        plt.ylabel('mag')
        plt.show()

    ## -> Paper Wang
    for start_index in np.arange(0, cutoff_freq, interval_size):
        end_index = start_index + interval_size
        segment_features.append({
            'freqInt' + str(start_index) + '_' + str(end_index): 
            np.nanmean(mag_cut[(freq_cut >= start_index) & (freq_cut < end_index)])}, 
            ignore_index=True)

    ## -> Paper Slapnicar
    sorted_mag = mag_cut[np.argsort(mag_cut)[::-1]]
    if len(sorted_mag) >= 3:
        top3_psd = sorted_mag[:3]
    else:
        top3_psd = sorted_mag[:len(sorted_mag)]

    for i in np.arange(0, len(top3_psd)):
        segment_features.append({
            'mag_top' + str(i+1):
            top3_psd[i]},
            ignore_index=True)

    sorted_freq = freq_cut[np.argsort(mag_cut)[::-1]]
    if len(sorted_freq) >= 3:
        top3_freq = sorted_freq[:3]
    else:
        top3_freq = sorted_freq[:len(sorted_freq)]

    for i in np.arange(0, len(top3_freq)):
        segment_features.append({
            'freq_top' + str(i+1):
            top3_freq[i]},
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

    cycles = find_with_template(ppg, sampling_frequency, factor=0.6667, verbose=verbose)
    # all signal peaks
    all_peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    if verbose:
        print('All Peaks: ', all_peaks)
    if len(cycles) == 0:
        return pd.DataFrame()

    cur_index = 0
    CP = pd.DataFrame()
    for i, cycle in enumerate(cycles):
        CP = CP.append(time_cycle(cycle, sampling_frequency,
                                    factor, unit, verbose=verbose),
                                    ignore_index=True)
        if i > 0:
            CP.loc[cur_index-1, 'CP'] = (
                CP.loc[cur_index, 'sys_peak_ts'] - CP.loc[cur_index-1, 'sys_peak_ts']).total_seconds()
        cur_index = cur_index + 1
    # last cycle or only cycle need to relies on the difference between the general peaks
    all_peaks_index = len(all_peaks)-1
    CP.loc[cur_index-1, 'CP'] = (
        all_peaks[all_peaks_index] - all_peaks[all_peaks_index-1])/sampling_frequency
    
    temporalHRVFeatures = _temporal_hrv(CP['CP'])
    frequencyHRVFeatures = _frequency_hrv(CP['CP'], sampling_frequency)

    segment_features = pd.concat([temporalHRVFeatures.reset_index(drop=True), frequencyHRVFeatures], axis=1)
 
    if verbose:
        print('Cycle Features within Segment:')
        print(segment_features)

    # remove outliers
    segment_features = _clean_segment_features_of_outliers(segment_features)

    if verbose:
        print('Cycle Features within Segment and no Outliers:')
        print(segment_features)

    return segment_features


# compute temporal and frequency features of HRV
def _temporal_hrv(ibi_series):
    
    if isinstance(ibi_series, np.ndarray):
        ibi_series = pd.Series(ibi_series)
    elif not isinstance(ibi_series, pd.core.series.Series):
        raise Exception('Signal values not accepted, enter a pandas.Series or ndarray.')
    
    window = 5
    nn_threshold = 50
    
    # Prepare data
    instantaneous_hr = 60 / (ibi_series)
    rolling_mean_hr = instantaneous_hr.rolling(window).mean()
    rolling_24h = ibi_series.rolling(5)
    
    # Precalculate data for standard indexes
    nn_diff = np.diff(ibi_series)
    nn_xx = np.sum(np.abs(nn_diff) > nn_threshold)
    
    # Precalculate data for geometrical indeces
    bin_size = 7.8125
    hist_middle = (ibi_series.min() + ibi_series.max()) / 2
    hist_bound_lower = hist_middle - bin_size * np.ceil((hist_middle - ibi_series.min()) / bin_size)
    hist_length = int(np.ceil((ibi_series.max() - hist_bound_lower) / bin_size) + 1)
    hist_bins = hist_bound_lower + np.arange(hist_length) * bin_size
    hist, _ = np.histogram(ibi_series, hist_bins)
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
                            f'pNN{nn_threshold}': nn_xx / len(ibi_series) * 100,
                            'HRVTriangularIndex': len(ibi_series) / hist_height,
                            'TINN': m - n}, 
                            index=[0])
    
    return temporalHRVFeatures

def _frequency_hrv(signal, sampling_frequency):
    
    if isinstance(signal, np.ndarray):
        ibi_series = pd.Series(signal)
    elif not isinstance(signal, pd.core.series.Series):
        raise Exception('Signal values not accepted, enter a pandas.Series or ndarray.')
	
	## TODO: HYPERPARAMETERS to config dict
    fft_interpolation = 4.0
    use_ulf = False
    lomb_smoothing = 0.02
    
    vlf_limit = 0.04
    lf_limit = 0.15
    hf_limit = 0.4
    
    if not use_ulf or ibi_series.sum() < 300000: #TODO: check
		# Do not use ULF band on sample shorter than 5 minutes
        ulf_limit = 0
    else:
        ulf_limit = ulf_limit #TODO: ??????
	
	# TODO: export transformMethod
    transformMethod='welch'
    freq, mag = _transformSigfromTimetoFreq(ibi_series, sampling_frequency, transformMethod, fft_interpolation)
    
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
                            'VLF_peak': vlf_peak,
                            'VLF_power': vlf,
                            'VLF_power_log': np.log(vlf),
                            'VLF_power_rel': vlf_perc,
                            'LF_peak': lf_peak,
                            'LF_power': lf,
                            'LF_power_log': np.log(lf),
                            'LF_power_rel': lf_perc,
                            'LF_power_norm': lf_nu,
                            'HF_peak': hf_peak,
                            'HF_power': hf,
                            'HF_power_log': np.log(hf),
                            'HF_power_rel': hf_perc,
                            'HF_power_norm': hf_nu,
                            'ratio_LF_HF': lf/hf}, 
                            index=[0])
    
    # Add ULF parameters, if band available
    if use_ulf and np.sum(ulf_index) > 0:
        ulf_peak = freq[np.argmax(mag[ulf_index])]
        freqencyHRVFeatures.append({
                                'ULF_peak': ulf_peak,
                                'ULF_power': ulf,
                                'ULF_power_log': np.log(ulf),
                                'ULF_power_rel': ulf_perc}, 
                                index=[0])
    
    return freqencyHRVFeatures


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

# transforms a given temporal signal into the spectral domain
# using different methods and allowing interpolation if required.
# returns the frequencies and their magnitudes.
def _transformSigfromTimetoFreq(signal, fs, transformMethod, fft_interpolation=None, verbose=False):

    ## TODO: Can maybe be improved?!
    if fft_interpolation:
        x = np.cumsum(signal)
        f_interpol = interpolate.interp1d(x, signal, kind = "cubic", fill_value="extrapolate")
        t_interpol = np.arange(x[0], x[-1], fft_interpolation/fs)
        
        signal = f_interpol(t_interpol)
        fs = fft_interpolation
    
    if transformMethod == 'welch':
        freq, mag = signal.welch(signal, fs, window='hamming', scaling='density', detrend='linear')
    
    if verbose:
        plt.semilogy(freq, mag)
        plt.title('Spectral Analysis using Welch Method')
        plt.xlabel('Frequency')
        plt.ylabel('PSD')
        plt.show()
    
    return freq, mag
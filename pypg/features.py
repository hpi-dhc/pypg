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
from scipy import signal, stats

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
        A dataframe with the non-linear features in seconds for each valid
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
        A dataframe with the nonlinear features in seconds for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit)

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min()
    ppg_cycle_duration = len(ppg)

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    systolic_peak_index = peaks[0]

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # features
    cycle_features = pd.DataFrame({
                        'ratio_WL_cycle_mean': (ppg_cycle_duration / np.mean(ppg)),
                        'ratio_SUT_WL_DT': ((systolic_peak_index / ppg_cycle_duration) / (ppg_cycle_duration - systolic_peak_index)),
                        'ratio_SUT_DT': (systolic_peak_index / (ppg_cycle_duration - systolic_peak_index)),
                        'ratio_cycle_mean_cycle_var': (np.mean(ppg) / np.var(ppg))
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
        A dataframe with the statistical features in seconds for each valid
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
        A dataframe with the statistical features in seconds for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit)

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # features
    cycle_features = pd.DataFrame({
                        'ppg_mean': (np.mean(ppg)),
                        'ppg_var': (np.var(ppg)),
                        'ppg_skewness': (stats.skew(ppg)),
                        'ppg_kurtosis': (stats.kurtosis(ppg))
                        }, index=[0])

    if verbose:
        plt.show()
    return cycle_features


def sdppg(ppg, sampling_frequency, factor=0.6667, unit='ms', verbose=False):
    """
    Extracts features from the second derivative (APG) of the PPG cycles in a give PPG segment. Returns a pandas.DataFrame in
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
        A dataframe with the features from the second derivative (APG) of the PPG cycles in seconds for each valid
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
    return segment_features


def sdppg_cycle(ppg, sampling_frequency, factor=0.667, unit='ms', verbose=False):
    """
    Extracts features from the second derivative (APG) for the PPG cycles. Returns a pandas.Series.

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
        A dataframe with the features from the second derivative (APG) in seconds for the PPG cycle.
    """

    if isinstance(ppg, np.ndarray):
        ppg = pd.Series(ppg)
    elif not isinstance(ppg, pd.core.series.Series):
        raise Exception('PPG values not accepted, enter a pandas.Series or ndarray.')

    if not isinstance(ppg.index, pd.DatetimeIndex):
        ppg.index = pd.to_datetime(ppg.index, unit=unit)

    ppg = ppg.interpolate(method='time')
    ppg = ppg - ppg.min()

    peaks = signal.find_peaks(ppg.values, distance=factor*sampling_frequency)[0]
    systolic_peak_index = peaks[0]

    if verbose:
        plt.figure()
        plt.xlim((ppg.index.min(), ppg.index.max()))
        plt.scatter(ppg.index[peaks], ppg[peaks])
        plt.plot(ppg.index, ppg.values)

    # second derviative of the PPG signal
    sdppg_signal = np.gradient(np.gradient(ppg))

    # features
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
        
        ## subset signal + dictionary to only have left over indeces + magnitudes
        old_len = len(sdppg_signal_peaks)
        sdppg_signal_peaks = sdppg_signal_peaks[sdppg_signal_peaks > a_index[0]]
        peak_dict['peak_heights'] = peak_dict['peak_heights'][old_len - len(sdppg_signal_peaks):]
        
        old_len = len(sdppg_signal_valleys)
        sdppg_signal_valleys = sdppg_signal_valleys[sdppg_signal_valleys > a_index[0]]
        valley_dict['peak_heights'] = valley_dict['peak_heights'][old_len - len(sdppg_signal_valleys):]
    
    # b wave
    if len(sdppg_signal_valleys) != 0:
        
        b_index = (sdppg_signal_valleys[np.argmax(valley_dict['peak_heights'])])
        
        ## subset signal + dictionary to only have left over indeces + magnitudes
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
            
            ## subset signal + dictionary to only have left over indeces + magnitudes
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
    if c_index and systolic_peak_index is not None:
        CD = ((c_index - systolic_peak_index) / sampling_frequency)
    
    ### Difference of dicrotic notch and end of signal with respect to time (DF)
    if c_index:
        DF = ((len(sdppg_signal) - c_index)  / sampling_frequency)
    
    
    ## PPG Signal values
    ### Dicrotic notch (D')
    if c_index:
        D = (ppg[c_index])
    
    ### Diastolic point (E')
        if d_index:
            E = (ppg[d_index])
    
    
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
    if c_index and systolic_peak_index is not None:
        ratio_CD_AF = ((c_index - systolic_peak_index) / len(sdppg_signal))
    
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

"""
====================================
Plots for PPG Signals (:mod:`pypg.plots`)
====================================

Plots
----------

    simple_plot  - Plots a raw or filtered PPG signal.

"""

import matplotlib.pyplot as plt


def simple_plot(ppg, filter_name=None, filter_type=None, cutoff_frequencies=None, figure_path=None):
    """
    Plots raw or filtered PPG signals.

    Parameters
    ----------
        ppg : pandas.Series or ndarray
              The raw PPG signal.
        filter_name : str, optional
              Name of the filter (e. g. butterworth).
        filter_type : str, optional
              Filter type (low, high, band - pass).
        cutoff_frequencies : int or list, optional
              The cutoff frequency(ies if bandpass) for the filter.
        figure_path : str, optional
              Path to save the graph as a pdf figure.
    """
    # define the parameters of the figure
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    label = 'PPG Signal'
    title = 'PPG Signal'
    if filter_name:
        title = 'Filtered Signal'
        label = filter_name
        if filter_type:
            label = label+' '+filter_type+' '+'filter'
        if cutoff_frequencies:
            if isinstance(cutoff_frequencies, list):
                cut = ', '.join(str(s) for s in cutoff_frequencies)
                cut = ' ['+cut+']'
            elif isinstance(cutoff_frequencies, int):
                cut = str(cutoff_frequencies)
            label = label+cut+' Hz'

    # change the style of the axis spines
    _, axis = plt.subplots(figsize=(10, 5))
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['left'].set_smart_bounds(True)
    axis.spines['bottom'].set_smart_bounds(True)
    axis.yaxis.grid(color='#333F4B', linestyle=':', linewidth=0.2, which='major')

    axis.set_title(title, fontsize=16)
    axis.set_xlabel('Time', fontsize=14)
    axis.set_ylabel('Amplitude', fontsize=14)
    plt.plot(ppg, label=label, color='#e8335e')
    plt.legend()
    if figure_path:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', format="pdf")
    plt.show()
    plt.close()

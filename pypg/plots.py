"""
=============================================
Plots for PPG Signals (:mod:`pypg.plots`)
=============================================

This module implements a few plots for PPG signals.

Plots
----------
simple_plot - Plots a PPG signal.
marks_plot  - Plots a PPG signal with marks (e. g. peaks or valleys).

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _configure_plot(title=None, y_axis=None, x_axis=None):
    """
    Configures the plot colors, axis, font sizes, etc.

    Parameters
    ----------
    title : str, optional
        Title of the plot, by default None.
    y_axis : str, optional
        Y axis name, by default None.
    x_axis : str, optional
        X axis name, by default None.
    """
    # defines the parameters of the figure
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    # changes the style of the axis spines
    _, axis = plt.subplots(figsize=(10, 5))
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.yaxis.grid(color='#333F4B', linestyle=':', linewidth=0.2, which='major')

    if title:
        axis.set_title(title, fontsize=16)
    if y_axis:
        axis.set_ylabel(y_axis, fontsize=14)
    if x_axis:
        axis.set_xlabel(x_axis, fontsize=14)

def simple_plot(ppg, title='PPG Signal', label='PPG Signal', y_axis='Amplitude',
                x_axis=None, figure_path=None):
    """
    Plots PPG signals.

    Parameters
    ----------
    ppg : indexable objects are supported.
        The PPG signal.
    title : str, optional
        Title of the plot, by default 'PPG Signal'.
    label : str, optional
        Label of the plotted data, by default 'PPG Signal'.
    y_axis : str, optional
        Y axis name, by default 'Amplitude'.
    x_axis : str, optional
        X axis name, by default None.
    figure_path : str, optional
        The path for the plot to be saved in pdf, by default None.
    """
    _configure_plot(title, y_axis, x_axis)
    plt.plot(ppg, label=label, color='#e8335e')
    plt.legend()
    if figure_path:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', format="pdf")
    plt.show()
    plt.close()

def marks_plot(ppg, marks, title='PPG Signal with Marks', label_ppg='PPG Signal',
               label_marks='Marks', y_axis='Amplitude', x_axis=None, figure_path=None):
    """
    Plots PPG signals with marks.

    Parameters
    ----------
    ppg : indexable objects are supported.
        The PPG signal.
    marks : ndarray
        Marks to be plotted against the PPG signal.
    title : str, optional
        Title of the plot, by default 'PPG Signal'.
    label_ppg : str, optional
        Label for the PPG signal, by default 'PPG Signal'.
    label_marks : str, optional
        Label for the marks, by default 'Marks'.
    y_axis : str, optional
        Y axis name, by default 'Amplitude'.
    x_axis : str, optional
        X axis name, by default None.
    figure_path : str, optional
        The path for the plot to be saved in pdf, by default None.
    """
    _configure_plot(title, y_axis, x_axis)

    plt.plot(ppg, color='#e8335e')
    for mark in marks:
        if isinstance(ppg, pd.core.series.Series):
            plt.plot(ppg.index[mark], ppg.iloc[mark], marker='X', markersize=8, color='#6233E8')
        elif isinstance(ppg, np.ndarray):
            plt.plot(mark, ppg[mark], marker='X', markersize=8, color='#6233E8')
    plt.legend([label_ppg, label_marks])
    if figure_path:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', format="pdf")
    plt.show()
    plt.close()

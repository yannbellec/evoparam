"""Plots for the evoparam object.

Notes
-----
This file contains plotting functions that take as input a evoparam object.
"""

import numpy as np

from evoparam.core.utils import nearest_ind
from evoparam.core.modutils import safe_import, check_dependency
from evoparam.sim.gen import sim_erp
from evoparam.utils.data import trim_signal
from evoparam.utils.params import compute_fwhm
from evoparam.plts.signals import plot_signals
from evoparam.plts.settings import PLT_FIGSIZES, PLT_COLORS
from evoparam.plts.utils import check_ax, check_plot_kwargs, savefig
from evoparam.plts.style import style_plot, style_ERP_plot

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@check_dependency(plt, 'matplotlib')
def plot_evoparam(model, ax=None, y_label=None):
    """Plot ERP and model fit results."""

    # create figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=PLT_FIGSIZES['signal'])

    # plot signal
    ax.plot(model.uncropped_time, model.uncropped_signal, alpha=0.75, label='ERP', color=PLT_COLORS['data'])

    # plot fit
    if model._peak_fit is not None:
        # plot full model fit
        ax.plot(model.time, model._peak_fit, linestyle='--', color=PLT_COLORS['model'], label='Gaussian fit')
    
        # plot peak and half-mag points
        ax.scatter(model.time[model.peak_indices_[:,1]], model.signal[model.peak_indices_[:,1]], color='r', label='Peak fit')
        half_mag_indices = np.concatenate((model.peak_indices_[:,0], model.peak_indices_[:,2]))
        ax.scatter(model.time[half_mag_indices], model.signal[half_mag_indices], color='b', label='Half-mag fit')
    
    # label
    if y_label is not None:
        ax.set(xlabel="Time (s)", ylabel=y_label)
    else:
        ax.set(xlabel="Time (s)", ylabel="Amplitude")
    ax.legend()

    # style
    style_ERP_plot(ax)

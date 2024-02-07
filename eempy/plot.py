"""
Functions for plotting EEM-related data
Author: Yongmin Hu (yongminhu@outlook.com)
Last update: 2024-01-15
"""

from utils import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm, Normalize


def plot_eem(intensity, em_range, ex_range, auto_intensity_range=True, scale_type='linear', vmin=0, vmax=10000,
             n_cbar_ticks=5, cbar=True, cmap='jet', figure_size=(7, 7), label_font_size=20, title=None,
             cbar_label="Intensity (a.u.)", cbar_font_size=16, aspect='equal', rotate=False):
    """
    plot EEM or EEM-like data.

    Parameters
    ----------------
    intensity: np.ndarray (2d)
        The EEM.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    auto_intensity_range: bool
        Whether to use the colorbar range generated automatically by matplotlib for plotting fluorescence intensity.
    scale_type: str, {'linear', 'log'}
        The type of colorbar scale used for plotting fluorescence intensity.
    vmin: int or float
        The minimum intensity displayed, if you wish to adjust the intensity scale yourself (autoscale=False).
    vmax: int or float
        The maximum intensity displayed, if you wish to adjust the intensity scale yourself (autoscale=False).
    n_cbar_ticks: int
        The number of ticks in colorbar.
    cmap: str
        The colormap, see https://matplotlib.org/stable/users/explain/colors/colormaps.html.
    figure_size: tuple or list with two elements
        The figure size.
    label_font_size: int
        The fontsize of the x and y axes labels.
    title: str
        The figure title.
    cbar: bool
        Whether to plot the colorscale bar.
    cbar_label: str
        The label of the colorbar scale.
    cbar_font_size: int
        The label size of the colorbar scale.
    aspect: 'equal' or float
        The aspect ratio.
    rotate: bool
        Whether to rotate the EEM, so that the x-axis is excitation and y-axis is emission.
    """
    plt.figure(figsize=figure_size)
    font = {'size': label_font_size}
    plt.rc('font', **font)
    # reset the axis direction
    if scale_type == 'log':
        c_norm = LogNorm(vmin=vmin, vmax=vmax)
        t_cbar = np.logspace(math.log(vmin), math.log(vmax), n_cbar_ticks)
    else:
        c_norm = None
        t_cbar = np.linspace(vmin, vmax, n_cbar_ticks)
    if not rotate:
        extent = (em_range.min(), em_range.max(), ex_range.min(), ex_range.max())
        plt.xlabel('Emission wavelength [nm]')
        plt.ylabel('Excitation wavelength [nm]')
        plt.ylim([ex_range[0], ex_range[-1]])
        if not auto_intensity_range:
            if scale_type == 'log':
                plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, origin='upper', aspect=aspect,
                           norm=c_norm)
            else:
                plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, vmin=vmin, vmax=vmax,
                           origin='upper', aspect=aspect, norm=c_norm)
        else:
            plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, origin='upper', aspect=aspect)
    if rotate:
        extent = (ex_range.min(), ex_range.max(), em_range.min(), em_range.max())
        plt.ylabel('Emission wavelength [nm]')
        plt.xlabel('Excitation wavelength [nm]')
        plt.xlim([ex_range[0], ex_range[-1]])
        if not auto_intensity_range:
            if scale_type == 'log':
                plt.imshow(np.flipud(np.fliplr(intensity.T)), cmap=cmap, interpolation='none', extent=extent,
                           origin='upper', aspect=aspect, norm=c_norm)
            else:
                plt.imshow(np.flipud(np.fliplr(intensity.T)), cmap=cmap, interpolation='none', extent=extent, vmin=vmin,
                           vmax=vmax, origin='upper', aspect=aspect, norm=c_norm)
        else:
            plt.imshow(np.flipud(np.fliplr(intensity.T)), cmap=cmap, interpolation='none', extent=extent,
                       origin='upper', aspect=aspect)
    if title:
        plt.title(title)
    if cbar:
        cbar = plt.colorbar(ticks=t_cbar, fraction=0.03, pad=0.04)
        cbar.set_label(cbar_label, labelpad=1.5)
        cbar.ax.tick_params(labelsize=cbar_font_size)
    return


def plot_abs(absorbance, ex_range, xmax=0.05, ex_range_display=(200, 800)):
    """
    Plot the UV absorbance data

    Parameters
    ----------------
    xmax: float (0~1)
        the maximum absorbance diplayed
    ex_range_display: tuple with two elements
        the range of excitation wavelengths displayed
    """
    plt.figure(figsize=(6.5, 2))
    font = {'size': 18}
    plt.rc('font', **font)
    plt.plot(ex_range, absorbance)
    plt.xlim(ex_range_display)
    plt.ylim([0, xmax])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance [a.u.]')


# def saveplot(datdir, datname, autoscale, cmax_fig, cmin_fig, savedir, savename):
#     """
#     save the EEM plot
#     """
#     datpath = datdir + '/' + datname
#     savepath = savedir + '/' + savename
#     intensity, em_range, ex_range = read_eem(datpath)
#     plt = plot_eem(intensity, em_range, ex_range, autoscale, cmax_fig, cmin_fig)
#     plt.savefig(savepath, dpi=600)


# def plot_abs_at_wavelength(abs_stack, ex_range, ex, plot=True, timestamp=False):
#     Ex_ref = dichotomy_search(ex_range, ex)
#     y = abs_stack[:, Ex_ref]
#     x = np.array(range(1, abs_stack.shape[0] + 1))
#     if plot:
#         if timestamp:
#             x = timestamp
#         plt.figure(figsize=(15, 5))
#         plt.scatter(x, y)
#         plt.xlim([x[0] - timedelta(hours=1), x[-1] + timedelta(hours=1)])


def plot_fi(fi: pd.DataFrame, q: float = 0.05):
    """
    Plot the fluorescence intensities of samples at a specific pair of ex/em.

    Parameters
    ----------
    fi: pandas DataFrame
        A DataFrame of shape (n,1), where n is the number of samples. The output of EEMdataset.peak_picking() can be
        passed to this parameter.
    q: float
        Where to plot the reference lines. By default, this is set to be 0.05 - two horizontal lines will be plotted at
        95% and 105% of the average fluorescence intensity.
    """
    fi = fi.to_numpy()
    index = fi.index()
    rel_std = stats.variation(fi)
    std = np.std(fi)
    fi_mean = fi.mean()
    ql = abs(q * fi_mean)
    print("Mean: {mean}".format(mean=fi_mean))
    print("Standard deviation: {std}".format(std=std))
    print("Relative Standard deviation: {rel_std}".format(rel_std=rel_std))
    plt.figure(figsize=(15, 5))
    plt.plot(index, fi, markersize=13)
    m0 = plt.axhline(fi_mean, linestyle='--', label='mean', c='black')
    mq_u = plt.axhline(fi_mean + ql, linestyle='--', label='+3%', c='red')
    mq_l = plt.axhline(fi_mean - ql, linestyle='--', label='-3%', c='blue')
    if max(fi) > fi_mean + ql or min(fi) < fi_mean - ql:
        plt.ylim(min(fi) - ql, max(fi) + ql)
    plt.legend([m0, mq_u, mq_l], ['mean', '+3%', '-3%'], prop={'size': 10})
    plt.xticks(rotation=90)
    plt.show()
    return


def plot_fi_correlation(fi: pd.DataFrame, ref):
    """
    Plot fluorescence intensity versus the reference value.

    Parameters
    ----------
    fi: pandas DataFrame
        A DataFrame of shape (n,1), where n is the number of samples. The output of EEMdataset.peak_picking() can be
        passed to this parameter.
    ref: np.ndarray (1d)
        The reference value. It should be an 1d numpy array of shape (n,), where n is the number of samples.
    """
    fi = fi.to_numpy()
    x = ref
    x_reshaped = x.reshape(ref.shape[0], 1)
    # x is the reference. y is the fluorescence.
    reg = LinearRegression().fit(x_reshaped, fi)
    w = reg.coef_
    b = reg.intercept_
    r2 = reg.score(x_reshaped, fi)
    pearson_coef, p_value_p = stats.pearsonr(x, fi)
    spearman_coef, p_value_s = stats.spearmanr(x, fi)
    print('Linear regression model: y={w}x+{b}'.format(w=w[0], b=b))
    print('Linear regression R2:', '{r2}'.format(r2=r2))
    print('Pearson coefficient: {coef_p} (p-value = {p_p})'.format(coef_p=pearson_coef, p_p=p_value_p))
    print('Spearman coefficient: {coef_s} (p-value = {p_s})'.format(coef_s=spearman_coef, p_s=p_value_s))
    plt.figure(figsize=(6, 3))
    plt.scatter(x, fi)
    p = np.array([x.min(), x.max()])
    q = w[0] * p + b
    plt.plot(p, q)
    plt.xlabel('Reference', fontdict={"size": 14})
    plt.ylabel('Fluorescence Intensity', fontdict={"size": 14})
    plt.show()
    return

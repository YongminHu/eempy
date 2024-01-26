"""
Functions for plotting EEM-related data
Author: Yongmin Hu (yongminhu@outlook.com)
Last update: 2024-01-15
"""

from utils import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


def plot_eem(intensity, em_range, ex_range, autoscale=False, cmin=-2000, cmax=6000, cbar=True, cmap='jet',
             figure_size=(7, 7), fontsize=20, title=None, cbar_label="Intensity (a.u.)", cbar_labelsize=16,
             aspect='equal', rotate=False):
    """plot EEM as a heatmap

    Parameters
    ----------------
    intensity: np.ndarray (2d)
        the EEM
    em_range: np.ndarray (1d)
        the emission wavelengths
    ex_range: np.ndarray (1d)
        the excitation wavelengths
    autoscale: bool
        whether to use the scale of intensity generated automatically by matplotlib
    cmin: int or float
        The minimum intensity displayed, if you wish to adjust the intensity scale yourself (autoscale=False)
    cmax: int or float
        The maximum intensity displayed, if you wish to adjust the intensity scale yourself (autoscale=False)
    cmap: str
        The colormap, see https://matplotlib.org/stable/users/explain/colors/colormaps.html
    figure_size: tuple or list with two elements
        The figure size
    fontsize: int
        The fontsize of the x and y axes labels
    title: str
        The figure title
    cbar: bool
        Whether to plot the colorscale bar
    cbar_label: str
        The label of the colorbar scale
    cbar_labelsize:
        The label size of the colorbar scale
    aspect: 'equal' or float
        The aspect ratio
    rotate: bool
        Whether to rotate the EEM, so that the x-axis is excitation and y-axis is emission
    """
    plt.figure(figsize=figure_size)
    font = {'size': fontsize}
    plt.rc('font', **font)
    # reset the axis direction
    if not rotate:
        extent = (em_range.min(), em_range.max(), ex_range.min(), ex_range.max())
        plt.xlabel('Emission wavelength [nm]')
        plt.ylabel('Excitation wavelength [nm]')
        plt.ylim([ex_range[0], ex_range[-1]])
        if not autoscale:
            plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, vmin=cmin, vmax=cmax,
                       origin='upper', aspect=aspect)
        else:
            plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, origin='upper', aspect=aspect)
    if rotate:
        extent = (ex_range.min(), ex_range.max(), em_range.min(), em_range.max())
        plt.ylabel('Emission wavelength [nm]')
        plt.xlabel('Excitation wavelength [nm]')
        plt.xlim([ex_range[0], ex_range[-1]])
        if not autoscale:
            plt.imshow(np.flipud(np.fliplr(intensity.T)), cmap=cmap, interpolation='none', extent=extent, vmin=cmin,
                       vmax=cmax, origin='upper', aspect=aspect)
        else:
            plt.imshow(np.flipud(np.fliplr(intensity.T)), cmap=cmap, interpolation='none', extent=extent,
                       origin='upper', aspect=aspect)
    if title:
        plt.title(title)
    if cbar:
        cbar = plt.colorbar(fraction=0.03, pad=0.04)
        cbar.set_label(cbar_label, labelpad=1.5)
        cbar.ax.tick_params(labelsize=cbar_labelsize)
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


def plot_abs_at_wavelength(abs_stack, ex_range, ex, plot=True, timestamp=False):
    Ex_ref = dichotomy_search(ex_range, ex)
    y = abs_stack[:, Ex_ref]
    x = np.array(range(1, abs_stack.shape[0] + 1))
    if plot:
        if timestamp:
            x = timestamp
        plt.figure(figsize=(15, 5))
        plt.scatter(x, y)
        plt.xlim([x[0] - timedelta(hours=1), x[-1] + timedelta(hours=1)])
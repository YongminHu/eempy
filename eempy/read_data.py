"""
Functions for importing raw EEM data
Author: Yongmin Hu (yongminhu@outlook.com)
Last update: 2024-01-15
"""

import os
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from datetime import timedelta


def read_eem_aqualog(filepath):
    """import EEM from aqualog EEM file. This kind of files are named using the format "xxPEM.dat" by the aqualog
    software by default. The blank file generate by aqualog ('xxBEM.dat') can also be read with this function.

    Parameters
    ----------------
    filepath: str
        the filepath to the aqualog EEM file

    Returns
    ----------------
    intensity: np.ndarray (2d)
        the EEM matrix
    em_range: np.ndarray (1d)
        the emission wavelengths
    ex_range: np.ndarray (1d)
        the excitation wavelengths
    """
    with open(filepath, 'r') as of:
        # get header (the first line in this case the Ex wavelength)
        firstline = of.readline()

        # remove unwanted characters
        firstline = re.findall(r"\d+", firstline)
        header = np.array([list(map(int, firstline))])

        # get index (the first column, in this case the Em wavelength) and the eem matrix
        index = []
        data = np.zeros(np.shape(header))
        line = of.readline()  # start reading from the second line
        while line:
            initial = (line.split())[0]
            # check if items only contains digits
            try:
                initial = float(initial)
                index.append(initial)
                # get fluorescence intensity data from each line
                dataline = np.array([list(map(float, (line.split())[1:]))])
                try:
                    data = np.concatenate([data, dataline])
                except ValueError:
                    print('please check the consistancy of header and data dimensions:\n')
                    print('header dimension: ', np.size(data), '\n')
                    print('data dimension for each line: ', np.size(dataline))
                    break
            except ValueError:
                pass
            line = of.readline()
        of.close()
        index = np.array(list(map(float, index)))
        data = data[1:, :]

    # Transpose the data matrix to set Xaxis-Em and Yaxis-Ex due to the fact
    # that the wavelength range of Em is larger, and it is visually better to
    # set the longer axis horizontally.
    intensity = data.T
    em_range = index
    ex_range = header[0]
    if em_range[0] > em_range[1]:
        em_range = np.flipud(em_range)
    if ex_range[0] > ex_range[1]:
        ex_range = np.flipud(ex_range)
    return intensity, em_range, ex_range


def read_abs_aqualog(filepath):
    """import UV absorbance data from aqualog UV absorbance file. This kind of files are named using the format
    "xxABS.dat" by the aqualog software by default.

    Parameters
    ----------------
    filepath: str
        the filepath to the aqualog UV absorbance file

    Returns
    ----------------
    absorbance:np.ndarray (1d)
        the UV absorbance spectra
    ex_range: np.ndarray (1d)
        the excitation wavelengths
    """
    with open(filepath, 'r') as of:
        line = of.readline()
        index = []
        data = []
        while line:
            initial = float((line.split())[0])
            index.append(initial)
            try:
                value = float((line.split())[1])
                data.append(value)
            except IndexError:
                data.append(np.nan)
                # if empty, set the value to nan
            line = of.readline()
        of.close()
        ex_range = np.flipud(index)
        absorbance = np.flipud(data)
    return absorbance, ex_range


def read_reference_from_text(filepath):
    """Read reference data from text file. The reference data can be any 1D data (e.g., dissolved organic carbon
    concentration).

    Parameters
    ----------------
    filepath: str
        the filepath to the aqualog UV absorbance file

    Returns
    ----------------
    absorbance:np.ndarray (1d)
        the reference data
    header: str
        the header
    """
    reference_data = []
    with open(filepath, "r") as f:
        line = f.readline()
        header = line.split()[0]
        while line:
            try:
                line = f.readline()
                reference_data.append(float(line.split()[0]))
            except IndexError:
                pass
        f.close()
    return reference_data, header


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
    cbar_label: str
        The label of the colorbar scale
    cbar_labelsize:
        The label size of the colorbar scale
    aspect: 'equal' or float
        The aspect ratio
    """
    plt.figure(figsize=figure_size)
    font = {'size': fontsize}
    plt.rc('font', **font)
    # reset the axis direction
    if not rotate:
        extent = [em_range.min(), em_range.max(), ex_range.min(), ex_range.max()]
        plt.xlabel('Emission wavelength [nm]')
        plt.ylabel('Excitation wavelength [nm]')
        plt.ylim([ex_range[0], ex_range[-1]])
        if autoscale == False:
            plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, vmin=cmin, vmax=cmax,
                       origin='upper', aspect=aspect)
        else:
            plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, origin='upper', aspect=aspect)
    if rotate:
        extent = [ex_range.min(), ex_range.max(), em_range.min(), em_range.max()]
        plt.ylabel('Emission wavelength [nm]')
        plt.xlabel('Excitation wavelength [nm]')
        plt.xlim([ex_range[0], ex_range[-1]])
        if autoscale == False:
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


def saveplot(datdir, datname, autoscale, cmax_fig, cmin_fig, savedir, savename):
    """
    save the EEM plot
    """
    datpath = datdir + '/' + datname
    savepath = savedir + '/' + savename
    intensity, em_range, ex_range = read_eem_aqualog(datpath)
    plt = plot_eem(intensity, em_range, ex_range, autoscale, cmax_fig, cmin_fig)
    plt.savefig(savepath, dpi=600)


def get_filelist(filedir, kw):
    """
    get a list containing all filenames with a given keyword in a folder
    For example, this can be used for searching EEM files (with the keyword "PEM.dat")
    """
    filelist = os.listdir(filedir)
    datlist = [file for file in filelist if kw in file]
    return datlist


# def ts_reshape(ts, timezone_correction=1):
#     ts_reshaped = ts.drop_duplicates(subset=['time'], keep='last')
#     ts_reshaped["time"] = pd.to_datetime(ts_reshaped.time) + timedelta(hours=timezone_correction)
#     ts_reshaped = ts_reshaped.set_index("time")
#     return ts_reshaped


def read_parafac_model(filepath):
    """
    Import PARAFAC model from a text file written in the format suggested by OpenFluor (
    https://openfluor.lablicate.com/). Note that the models downloaded from OpenFluor normally don't have scores.

    Parameters
    ----------------
    filepath: str
        the filepath to the aqualog UV absorbance file

    Returns
    ----------------
    ex_df: pd.DataFrame
        excitation loadings
    em_df: pd.DataFrame
        emission loadings
    score_df: pd.DataFrame or None
        scores (if there's any)
    info_dict: dict
        a dictionary containing the model information
    """
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        line_count = 0
        while '#' in line:
            if "Fluorescence" in line:
                print("Reading fluorescence measurement info...")
            line = f.readline().strip()
            line_count += 1
        info_dict = {}
        while '#' not in line:
            phrase = line.split(sep='\t')
            if len(phrase) > 1:
                info_dict[phrase[0]] = phrase[1]
            else:
                info_dict[phrase[0]] = ''
            line = f.readline().strip()
            line_count += 1
        while '#' in line:
            if "Excitation" in line:
                print("Reading Ex/Em loadings...")
            line = f.readline().strip()
            line_count_spectra_start = line_count
            line_count += 1
        while "Ex" in line:
            line = f.readline().strip()
            line_count += 1
        line_count_ex = line_count
        ex_df = pd.read_csv(filepath, sep="\t", header=None, index_col=[0, 1],
                           skiprows=line_count_spectra_start + 1, nrows=line_count_ex - line_count_spectra_start - 1)
        component_label = ['component {rank}'.format(rank=r + 1) for r in range(ex_df.shape[1])]
        ex_df.columns = component_label
        ex_df.index.names = ['type', 'wavelength']
        while "Em" in line:
            line = f.readline().strip()
            line_count += 1
        line_count_em = line_count
        em_df = pd.read_csv(filepath, sep='\t', header=None, index_col=[0, 1],
                           skiprows=line_count_ex, nrows=line_count_em - line_count_ex)
        em_df.columns = component_label
        em_df.index.names = ['type', 'wavelength']
        score_df = None
        while '#' in line:
            if "Score" in line:
                print("Reading component scores...")
            line = f.readline().strip()
            line_count += 1
        line_count_score = line_count
        while 'Score' in line:
            line = f.readline().strip()
            line_count += 1
        while '#' in line:
            if 'end' in line:
                line_count_end = line_count
                score_df = pd.read_csv(filepath, sep="\t", header=None, index_col=[0, 1],
                                            skiprows=line_count_score, nrows=line_count_end - line_count_score)
                score_df.index = score_df.index.set_levels([score_df.index.levels[0], pd.to_datetime(score_df.index.levels[1])])
                score_df.columns = component_label
                score_df.index.names = ['type', 'time']
                print('Reading complete')
                line = f.readline().strip()
        f.close()
    return ex_df, em_df, score_df, info_dict


def read_parafac_models(datdir, kw):
    """
    Search all PARAFAC models in a folder by keyword in filenames and import all of them into a dictionary using
    read_parafac_model()
    """
    datlist = get_filelist(datdir, kw)
    parafac_results = []
    for f in datlist:
        filepath = datdir + '/' + f
        ex_df, em_df, score_df, info_dict = read_parafac_model(filepath)
        info_dict['filename'] = f
        d = {'info': info_dict, 'ex': ex_df, 'em': em_df, 'score':score_df}
        parafac_results.append(d)
    return parafac_results
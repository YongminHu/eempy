"""
Functions for fluorescence python toolkit
Author: Yongmin Hu (yongmin.hu@eawag.ch, yongminhu@outlook.com)
Last update: 2021-03-07
"""

import os
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def readEEM(filepath):
    with open(filepath, 'r') as of:
        # get header (the first line in this case the Ex wavelength)
        firstline = of.readline()

        # remove unwanted characters
        firstline = re.findall(r"\d+", firstline)
        header = np.array([list(map(int, firstline))])

        # get index (the first column, in this case the Em wavelength) and the data matrix
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
                ind = 1
            line = of.readline()
        of.close()
        index = np.array(list(map(float, index)))
        data = data[1:, :]

    # Transpose the data matrix to set Xaxis-Em and Yaxis-Ex due to the fact
    # that the wavelength range of Em is larger, and it is visually better to
    # set the longer axis horizontally.
    intensity = data.T
    Em_range = index
    Ex_range = header[0]
    if Em_range[0] > Em_range[1]:
        Em_range = np.flipud(Em_range)
    if Ex_range[0] > Ex_range[1]:
        Ex_range = np.flipud(Ex_range)
    return intensity, Em_range, Ex_range


def readABS(filepath):
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
        Ex_range = np.flipud(index)
        absorbance = np.flipud(data)
    return absorbance, Ex_range


def string_to_float_list(string):
    flist = np.array([float(i) for i in list(string.split(","))])
    return flist


def read_reference_from_text(filepath):
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


def plotABS(absorbance, Ex_range, xmax=0.05):
    plt.figure(figsize=(8, 2))
    font = {
            'size': 16}
    plt.plot(Ex_range, absorbance)
    plt.xlim([200, 800])
    plt.ylim([0, xmax])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance [a.u.]')


def plotABS_interact(filedir, filename):
    filepath = filedir + '/' + filename
    absorbance, Ex_range = readABS(filepath)
    plt = plotABS(absorbance, Ex_range)
    
    
def plot3DEEM(intensity, Em_range, Ex_range, autoscale=False, cmax=6000, cmin=-2000, new_figure=True, cbar=True, cmap='jet', cbar_label="Intensity (a.u.)",
              figure_size=(7,7), title=False, cbar_labelsize=16, fontsize=20):
    # reset the axis direction
    extent = [Em_range.min(), Em_range.max(), Ex_range.min(), Ex_range.max()]
    if new_figure:
        plt.figure(figsize=figure_size)
    #plt.figure()
    font = {
            'size': fontsize}
    plt.rc('font', **font)
    if autoscale == False:
        plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, vmin=cmin, vmax=cmax,
                   origin='upper')
    else:
        plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, origin='upper')
    plt.xlabel('Emission wavelength [nm]')
    plt.ylabel('Excitation wavelength [nm]')
    plt.ylim([Ex_range[0], Ex_range[-1]])
    if title:
        plt.title(title)
    if cbar:
        cbar = plt.colorbar(fraction=0.03, pad=0.04)
        cbar.set_label(cbar_label, labelpad=1.5)
        cbar.ax.tick_params(labelsize=cbar_labelsize)
    return


def SavePlot(datdir, datname, autoscale, cmax_fig, cmin_fig, savedir, savename):
    datpath = datdir + '/' + datname
    savepath = savedir + '/' + savename
    intensity, Em_range, Ex_range = readEEM(datpath)
    plt = plot3DEEM(intensity, Em_range, Ex_range, autoscale, cmax_fig, cmin_fig)
    plt.savefig(savepath, dpi=600)


def get_filelist(filedir, extension):
    filelist = os.listdir(filedir)
    datlist = [file for file in filelist if extension in file]
    return datlist


def dir_update(filedir):
    filelist = os.listdir(filedir)
    global datlist
    datlist = [file for file in filelist if '.dat' in file]


def ts_reshape(ts, timezone_correction=1):
    ts_reshaped = ts.drop_duplicates(subset=['time'], keep='last')
    ts_reshaped["time"] = pd.to_datetime(ts_reshaped.time) + timedelta(hours=timezone_correction)
    ts_reshaped = ts_reshaped.set_index("time")
    return ts_reshaped


def find_filtration_phase(mbr2press, transition_time=9, timezone_correction=1):
    """
    status_code:
    2: filtration_phase
    1: transition time between filtration phase and stagnation phase
    0: stagnation phase
    """
    #    pres = ts_reshape(mbr2press, timezone_correction)
    pres = mbr2press
    pres_10min = pres.resample('10T').mean()
    pres_diff = np.array(pres_10min.bp[:-1]) - np.array(pres_10min.bp[1:])
    filtration_status = np.zeros(pres_diff.shape)
    for i in range(pres_diff.shape[0]):
        if pres_diff[i] > 0.3 and pres_diff[i] < 5:
            filtration_status[i] = 2
        elif i > 1 and i <= pres_diff.shape[0] - transition_time:
            if filtration_status[i - 1] == 2:
                filtration_status[i:i + transition_time] = 1
        elif i > 1 and i > pres_diff.shape[0] - transition_time:
            if filtration_status[i - 1] == 2:
                filtration_status[i:] = 1
    filtration_start = []
    filtration_end = []
    filtration_end_mbr = []
    for j in range(filtration_status.shape[0] - 1):
        if (j == 0 and filtration_status[j] == 2) or (filtration_status[j] == 2 and filtration_status[j - 1] < 2):
            filtration_start.append(pres_10min.index[j])
            # print("start: ", pres_10min.index[j])
        if filtration_status[j] == 2 and (not filtration_status[j + 1] == 2):
            filtration_end_mbr.append(pres_10min.index[j])
            # print("end_mbr", pres_10min.index[j])
        if filtration_status[j] == 1 and (not filtration_status[j + 1] == 1):
            filtration_end.append(pres_10min.index[j])
            # print("end: ", pres_10min.index[j])
    if not filtration_status[-1] == 0:
        filtration_end.append(pres_10min.index[-1])
    elif filtration_status[-1] == 2:
        filtration_end_mbr.append(pres_10min.index[-1])
    filtration_duration = [end - start for start, end in zip(filtration_start, filtration_end)]
    filtration_start = [filtration_start[i] for i in range(len(filtration_start)) if
                        filtration_duration[i] > timedelta(hours=1.75)]
    filtration_end = [filtration_end[i] for i in range(len(filtration_end)) if
                      filtration_duration[i] > timedelta(hours=1.75)]
    filtration_end_mbr = [filtration_end_mbr[i] for i in range(len(filtration_end_mbr)) if
                          filtration_duration[i] > timedelta(hours=1.75)]
    filtration_duration = [end - start for start, end in zip(filtration_start, filtration_end)]
    filtration_duration_mbr = [end - start for start, end in zip(filtration_start, filtration_end_mbr)]

    flow_rate = [60*(pres.loc[start-timedelta(minutes=20):start+timedelta(minutes=20)].max()
                     - pres.loc[end-timedelta(minutes=20):end+timedelta(minutes=20)].min())/duration.seconds for
                 start, end, duration in zip(filtration_start, filtration_end_mbr, filtration_duration_mbr)]
    # for start, end in zip(filtration_start, filtration_end_mbr):
    #     print("high:", pres_10min.loc[start][0], " low: ", pres_10min.loc[end][0])
    return filtration_start, filtration_end, filtration_duration, flow_rate



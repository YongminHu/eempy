"""
Functions for fluorescence python toolkit
Author: Yongmin Hu (yongmin.hu@eawag.ch, yongminhu@outlook.com)
Last update: 2021-03-07
"""

import os
import matplotlib.pyplot as plt
import re
import numpy as np


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
    
    
def plot3DEEM(intensity, Em_range, Ex_range, autoscale=False, cmax=6000, cmin=-2000, cmap='jet', cbar_label="Intensity (a.u.)",
              figure_size=(8,8), title=False):
    # reset the axis direction
    extent = [Em_range.min(), Em_range.max(), Ex_range.min(), Ex_range.max()]
    plt.figure(figsize=figure_size)
    font = {
            'size': 16}
    plt.rc('font', **font)
    if autoscale == False:
        plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, aspect='1.2', vmin=cmin, vmax=cmax,
                   origin='upper')
    else:
        plt.imshow(intensity, cmap=cmap, interpolation='none', extent=extent, aspect='1.2', origin='upper')
    plt.xlabel('Emission wavelength [nm]')
    plt.ylabel('Excitation wavelength [nm]')
    plt.ylim([Ex_range[0], Ex_range[-1]])
    if title:
        plt.title(title)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label(cbar_label, labelpad=1.5)
    cbar.ax.tick_params(labelsize=12)
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

"""
Functions for EEM analysis
Author: Yongmin Hu (yongmin.hu@eawag.ch, yongminhu@outlook.com)
Last update: 2024-01-10
"""

from read_data import *
from plot import *
from utils import *
import scipy.stats as stats
import random
import pandas as pd
import numpy as np
import statistics
import itertools
import string
import warnings
import math
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from datetime import datetime, timedelta
from tensorly.decomposition import parafac, non_negative_parafac
from tensorly.tenalg import mode_dot
from tensorly.cp_tensor import cp_to_tensor
from IPython.display import display
from pandas.plotting import register_matplotlib_converters
from scipy.sparse.linalg import ArpackError
from sklearn.ensemble import IsolationForest
from sklearn import svm
from matplotlib.cm import get_cmap
from typing import Literal, Union

register_matplotlib_converters()


def stack_eem(datdir, kw='PEM.dat', data_format='aqualog', timestamp_reference_list=None, custom_datlist=None,
              wavelength_synchronization=True, em_range_display=(250, 800), ex_range_display=(240, 500)):
    if not custom_datlist:
        datlist = get_filelist(datdir, kw)
    else:
        datlist = custom_datlist
    if timestamp_reference_list:
        datlist = [x for _, x in sorted(zip(timestamp_reference_list, datlist))]
    path = datdir + '/' + datlist[0]
    intensity, ex_range, em_range = read_eem(path, data_format=data_format)
    intensity, ex_range, em_range = eem_cutting(intensity, ex_range, em_range,
                                                em_min=em_range_display[0],
                                                em_max=em_range_display[1],
                                                ex_min=ex_range_display[0], ex_max=ex_range_display[1])
    num_datfile = len(datlist)
    eem_stack = np.zeros([num_datfile, intensity.shape[0], intensity.shape[1]])
    try:
        eem_stack[0, :, :] = intensity
    except ValueError:
        print('Check data dimension: ', datlist[0])
    em_range_old = np.copy(em_range)
    ex_range_old = np.copy(ex_range)
    for n in range(1, len(datlist)):
        path = datdir + '/' + datlist[n]
        intensity, ex_range, em_range = read_eem(path, data_format=data_format)
        intensity, ex_range, em_range = eem_cutting(intensity, ex_range, em_range,
                                                    em_min=em_range_display[0],
                                                    em_max=em_range_display[1],
                                                    ex_min=ex_range_display[0], ex_max=ex_range_display[1])
        if wavelength_synchronization:
            em_interval_new = (np.max(em_range) - np.min(em_range)) / (em_range.shape[0] - 1)
            em_interval_old = (np.max(em_range_old) - np.min(em_range_old)) / (em_range_old.shape[0] - 1)
            ex_interval_new = (np.max(ex_range) - np.min(ex_range)) / (ex_range.shape[0] - 1)
            ex_interval_old = (np.max(ex_range_old) - np.min(ex_range_old)) / (ex_range_old.shape[0] - 1)
            if em_interval_new > em_interval_old:
                em_range_target = em_range_old
            else:
                em_range_target = em_range
            if ex_interval_new > ex_interval_old:
                ex_range_target = ex_range_old
            else:
                ex_range_target = ex_range
            if em_interval_new > em_interval_old or ex_interval_new > ex_interval_old:
                intensity = eem_interpolation(intensity, em_range, np.flip(ex_range), em_range_target,
                                              np.flip(ex_range_target))
                em_range = np.copy(em_range_old)
                ex_range = np.copy(ex_range_old)
            if em_interval_new < em_interval_old or ex_interval_new < ex_interval_old:
                eem_stack = process_eem_stack(eem_stack, eem_interpolation, em_range_old, np.flip(ex_range_old), em_range_target,
                                              np.flip(ex_range_target))
        try:
            eem_stack[n, :, :] = intensity
        except ValueError:
            print('Check data dimension: ', datlist[n])
        em_range_old = np.copy(em_range)
        ex_range_old = np.copy(ex_range)
    return eem_stack, ex_range, em_range, datlist


def stack_abs(datdir, datlist, wavelength_synchronization=True, ex_range_display=(240, 500)):
    path = datdir + '/' + datlist[0]
    absorbance, ex_range = read_abs(path)
    absorbance = absorbance[np.logical_and(ex_range >= ex_range_display[0], ex_range <= ex_range_display[1])]
    ex_range = ex_range[np.logical_and(ex_range >= ex_range_display[0], ex_range <= ex_range_display[1])]
    num_datfile = len(datlist)
    abs_stack = np.zeros([num_datfile, absorbance.shape[0]])
    abs_stack[0, :] = absorbance
    ex_range_old = ex_range
    for n in range(1, len(datlist)):
        path = datdir + '/' + datlist[n]
        absorbance, ex_range = read_abs(path)
        absorbance = absorbance[np.logical_and(ex_range >= ex_range_display[0], ex_range <= ex_range_display[1])]
        ex_range = ex_range[np.logical_and(ex_range >= ex_range_display[0], ex_range <= ex_range_display[1])]
        if wavelength_synchronization and n > 0:
            ex_interval_new = (np.max(ex_range) - np.min(ex_range)) / (ex_range.shape[0] - 1)
            ex_interval_old = (np.max(ex_range_old) - np.min(ex_range_old)) / (ex_range_old.shape[0] - 1)
            if ex_interval_new > ex_interval_old:
                f = interpolate.interp1d(ex_range, absorbance)
                absorbance = f(ex_range_old)
            if ex_interval_new < ex_interval_old:
                abs_stack_new = np.zeros([num_datfile, absorbance.shape[0]])
                for i in range(n):
                    f = interpolate.interp1d(ex_range_old, abs_stack[i, :])
                    abs_stack_new[i, :] = f(ex_range)
                abs_stack = abs_stack_new
        abs_stack[n, :] = absorbance
        ex_range_old = ex_range
    return abs_stack, ex_range, datlist


def process_eem_stack(eem_stack, f, *args, **kwargs):
    processed_eem_stack = []
    other_outputs = []
    for i in range(eem_stack.shape[0]):
        f_output = f(eem_stack[i,:,:], *args, **kwargs)
        if isinstance(f_output, tuple):
            processed_eem_stack.append(f_output[0])
            other_outputs.append(f_output[1:])
        else:
            processed_eem_stack.append(f_output)
    if len(set([eem.shape for eem in processed_eem_stack])) > 1:
        warnings.warn("Processed EEMs have different shapes")
    return np.array(processed_eem_stack), other_outputs


def eem_threshold_masking(intensity, ex_range, em_range, threshold, fill, mask_type='greater', plot=False,
                          autoscale=True, cmin=0, cmax=4000):
    mask = np.ones(intensity.shape)
    intensity_masked = intensity.astype(float)
    extent = (em_range.min(), em_range.max(), ex_range.min(), ex_range.max())
    if mask_type == 'smaller':
        mask[np.where(intensity < threshold)] = np.nan
    if mask_type == 'greater':
        mask[np.where(intensity > threshold)] = np.nan
    intensity_masked[np.isnan(mask)] = fill
    if plot:
        plt.figure(figsize=(8, 8))
        plot_eem(intensity_masked, em_range=em_range, ex_range=ex_range, autoscale=autoscale, cmin=cmin, cmax=cmax)
        plt.imshow(mask, extent=extent, alpha=0.9, cmap="binary")
        # plt.title('Relative STD<{threshold}'.format(threshold=threshold))
    return intensity_masked, mask


def eem_region_masking(intensity, ex_range, em_range, em_min=250, em_max=810, ex_min=230, ex_max=500,
                       fill_value='nan'):
    masked = intensity.copy()
    em_min_idx = dichotomy_search(em_range, em_min)
    em_max_idx = dichotomy_search(em_range, em_max)
    ex_min_idx = dichotomy_search(ex_range, ex_min)
    ex_max_idx = dichotomy_search(ex_range, ex_max)
    mask = np.ones(intensity.shape)
    mask[ex_range.shape[0] - ex_max_idx - 1:ex_range.shape[0] - ex_min_idx,
    em_min_idx:em_max_idx + 1] = 0
    if fill_value == 'nan':
        masked[mask == 0] = np.nan
    elif fill_value == 'zero':
        masked[mask == 0] = 0
    return masked, mask


def eem_gaussian_filter(intensity, sigma=1, truncate=3):
    intensity_filtered = gaussian_filter(intensity, sigma=sigma, truncate=truncate)
    return intensity_filtered


def eem_cutting(intensity, ex_range, em_range, em_min, em_max, ex_min, ex_max):
    em_min_idx = dichotomy_search(em_range, em_min)
    em_max_idx = dichotomy_search(em_range, em_max)
    ex_min_idx = dichotomy_search(ex_range, ex_min)
    ex_max_idx = dichotomy_search(ex_range, ex_max)
    intensity_cut = intensity[ex_range.shape[0] - ex_max_idx - 1:ex_range.shape[0] - ex_min_idx,
                    em_min_idx:em_max_idx + 1]
    em_range_cut = em_range[em_min_idx:em_max_idx + 1]
    ex_range_cut = ex_range[ex_min_idx:ex_max_idx + 1]
    return intensity_cut, ex_range_cut, em_range_cut


def eem_grid_imputing(intensity, ex_range, em_range, method='linear', fill_value='linear_ex', prior_mask=None):
    x, y = np.meshgrid(em_range, ex_range[::-1])
    xx = x[~np.isnan(intensity)].flatten()
    yy = y[~np.isnan(intensity)].flatten()
    zz = intensity[~np.isnan(intensity)].flatten()
    interpolated = None
    if isinstance(fill_value, float):
        interpolated = interpolate.griddata((xx, yy), zz, (x, y), method=method, fill_value=fill_value)
    elif fill_value == 'linear_ex':
        interpolated = interpolate.griddata((xx, yy), zz, (x, y), method=method)
        for i in range(interpolated.shape[1]):
            col = interpolated[:, i]
            mask = np.isnan(col)
            if np.any(mask):
                interp_func = interpolate.interp1d(np.flatnonzero(~mask), col[~mask], kind='linear',
                                                   fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            interpolated[:, i] = col
    elif fill_value == 'linear_em':
        interpolated = interpolate.griddata((xx, yy), zz, (x, y), method=method)
        for j in range(interpolated.shape[0]):
            col = interpolated[j, :]
            mask = np.isnan(col)
            if np.any(mask):
                interp_func = interpolate.interp1d(np.flatnonzero(~mask), col[~mask], kind='linear',
                                                   fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            interpolated[j, :] = col
    if prior_mask is None:
        return interpolated
    else:
        intensity_imputed = intensity.copy()
        intensity_imputed[prior_mask == 0] = interpolated[prior_mask == 0]
        return intensity_imputed


def eem_raman_normalization(intensity, ex_range_blank=None, em_range_blank=None, blank=None, from_blank=False,
                            integration_time=1, ex_lb=349, ex_ub=351, bandwidth_type='wavenumber', bandwidth=1800,
                            rsu_standard=20000, manual_rsu=1):
    if not from_blank:
        return intensity / manual_rsu, manual_rsu
    else:
        ex_range_cut = ex_range_blank[(ex_range_blank >= ex_lb) & (ex_range_blank <= ex_ub)]
        rsu_tot = 0
        for ex in ex_range_cut.tolist():
            if bandwidth_type == 'wavenumber':
                em_target = -ex / (0.00036 * ex - 1)
                wn_target = 10000000 / em_target
                em_lb = 10000000 / (wn_target + bandwidth)
                em_rb = 10000000 / (wn_target - bandwidth)
                rsu, _, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
                                                     [em_lb, em_rb],
                                                     [ex, ex])
            else:
                rsu, _, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
                                                     [ex - bandwidth, ex + bandwidth],
                                                     [ex, ex])
            rsu_tot += rsu
        return intensity * rsu_standard / rsu_tot / integration_time, rsu_standard / rsu_tot / integration_time


def eem_raman_masking(intensity, ex_range, em_range, tolerance=5, method='linear', axis='grid'):
    intensity_masked = np.array(intensity)
    raman_mask = np.ones(intensity.shape)
    lambda_em = -ex_range / (0.00036 * ex_range - 1)
    tol_emidx = int(np.round(tolerance / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em[s] <= em_range[0] and lambda_em[s] + tolerance >= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em[s] + tolerance)
            raman_mask[exidx, 0: emidx + 1] = 0
        elif lambda_em[s] - tolerance <= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em[s])
            raman_mask[exidx, 0: emidx + tol_emidx + 1] = 0
        else:
            emidx = dichotomy_search(em_range, lambda_em[s] - tolerance)
            raman_mask[exidx, emidx: emidx + 2 * tol_emidx + 1] = 0

    if method == 'nan':
        intensity_masked[np.where(raman_mask == 0)] = np.nan
    else:
        if axis == 'ex':
            for j in range(0, intensity.shape[1]):
                try:
                    mask_start_idx = np.min(np.where(raman_mask[:, j] == 0)[0])
                    mask_end_idx = np.max(np.where(raman_mask[:, j] == 0)[0])
                    x = np.flipud(ex_range)[np.where(raman_mask[:, j] == 1)]
                    y = intensity_masked[:, j][np.where(raman_mask[:, j] == 1)]
                    f1 = interpolate.interp1d(x, y, kind=method, fill_value='extrapolate')
                    y_predict = f1(np.flipud(ex_range))
                    intensity_masked[:, j] = y_predict
                except ValueError:
                    continue

        if axis == 'em':
            for i in range(0, intensity.shape[0]):
                try:
                    mask_start_idx = np.min(np.where(raman_mask[i, :] == 0)[0])
                    mask_end_idx = np.max(np.where(raman_mask[i, :] == 0)[0])
                    x = em_range[np.where(raman_mask[i, :] == 1)]
                    y = intensity_masked[i, :][np.where(raman_mask[i, :] == 1)]
                    f1 = interpolate.interp1d(x, y, kind=method, fill_value='extrapolate')
                    y_predict = f1(em_range)
                    intensity_masked[i, :] = y_predict
                except ValueError:
                    continue

        if axis == 'grid':
            old_nan = np.isnan(intensity)
            intensity_masked[np.where(raman_mask == 0)] = np.nan
            intensity_masked = eem_grid_imputing(intensity_masked, method=method)
            # restore the nan values in non-raman-scattering region
            intensity_masked[old_nan] = np.nan
    return intensity_masked, raman_mask


def eem_rayleigh_masking(intensity, ex_range, em_range, tolerance_o1=15, tolerance_o2=15,
                         axis_o1='grid', axis_o2='grid', method_o1='zero', method_o2='linear'):
    intensity_masked = np.array(intensity)
    rayleigh_mask_o1 = np.ones(intensity.shape)
    rayleigh_mask_o2 = np.ones(intensity.shape)
    lambda_em_o1 = ex_range
    tol_emidx_o1 = int(np.round(tolerance_o1 / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em_o1[s] <= em_range[0] and lambda_em_o1[s] + tolerance_o1 >= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em_o1[s] + tolerance_o1)
            rayleigh_mask_o1[exidx, 0:emidx + 1] = 0
        elif lambda_em_o1[s] - tolerance_o1 <= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em_o1[s])
            rayleigh_mask_o1[exidx, 0: emidx + tol_emidx_o1 + 1] = 0
        else:
            emidx = dichotomy_search(em_range, lambda_em_o1[s])
            rayleigh_mask_o1[exidx, emidx: emidx + tol_emidx_o1 + 1] = 0
            intensity_masked[exidx, 0: emidx] = 0
    lambda_em_o2 = ex_range * 2
    tol_emidx_o2 = int(np.round(tolerance_o2 / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em_o2[s] <= em_range[0] and lambda_em_o2[s] + tolerance_o2 >= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em_o2[s] + tolerance_o2)
            rayleigh_mask_o2[exidx, 0:emidx + 1] = 0
        elif lambda_em_o2[s] - tolerance_o2 <= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em_o2[s])
            rayleigh_mask_o2[exidx, 0: emidx + tol_emidx_o2 + 1] = 0
        else:
            emidx = dichotomy_search(em_range, lambda_em_o2[s] - tolerance_o2)
            rayleigh_mask_o2[exidx, emidx: emidx + 2 * tol_emidx_o2 + 1] = 0

    for axis, itp, mask in zip([axis_o1, axis_o2], [method_o1, method_o2], [rayleigh_mask_o1, rayleigh_mask_o2]):
        if itp == 'zero':
            intensity_masked[np.where(mask == 0)] = 0
        elif itp == 'nan':
            intensity_masked[np.where(mask == 0)] = np.nan
        else:
            if axis == 'ex':
                for j in range(0, intensity.shape[1]):
                    try:
                        mask_start_idx = np.min(np.where(mask[:, j] == 0)[0])
                        mask_end_idx = np.max(np.where(mask[:, j] == 0)[0])
                        x = np.flipud(ex_range)[np.where(mask[:, j] == 1)]
                        y = intensity_masked[:, j][np.where(mask[:, j] == 1)]
                        f1 = interpolate.interp1d(x, y, kind=itp, fill_value='extrapolate')
                        y_predict = f1(np.flipud(ex_range))
                        intensity_masked[:, j] = y_predict
                    except ValueError:
                        continue
            if axis == 'em':
                for i in range(0, intensity.shape[0]):
                    try:
                        mask_start_idx = np.min(np.where(mask[i, :] == 0)[0])
                        mask_end_idx = np.max(np.where(mask[i, :] == 0)[0])
                        x = em_range[np.where(mask[i, :] == 1)]
                        y = intensity_masked[i, :][np.where(mask[i, :] == 1)]
                        f1 = interpolate.interp1d(x, y, kind=itp, fill_value='extrapolate')
                        y_predict = f1(em_range)
                        intensity_masked[i, :] = y_predict
                    except ValueError:
                        continue
            if axis == 'grid':
                old_nan = np.isnan(intensity)
                old_nan_o1 = np.isnan(intensity_masked)
                intensity_masked[np.where(mask == 0)] = np.nan
                intensity_masked = eem_grid_imputing(intensity_masked, method=itp)
                # restore the nan values in non-raman-scattering region
                intensity_masked[old_nan] = np.nan
                intensity_masked[old_nan_o1] = np.nan
    return intensity_masked, (rayleigh_mask_o1, rayleigh_mask_o2)


def eem_ife_correction(intensity, ex_range, em_range, absorbance, ex_range_abs, cuvette_length=1, ex_lower_limit=200,
                       ex_upper_limit=825):
    ex_range2 = np.concatenate([[ex_upper_limit], ex_range_abs, [ex_lower_limit]])
    absorbance = np.concatenate([[0], absorbance, [max(absorbance)]])
    f1 = interpolate.interp1d(ex_range2, absorbance, kind='linear', bounds_error=False, fill_value='extrapolate')
    absorbance_ex = np.fliplr(np.array([f1(ex_range)]))
    absorbance_em = np.array([f1(em_range)])
    ife_factors = 10 ** (cuvette_length * (absorbance_ex.T.dot(np.ones(absorbance_em.shape)) +
                                           np.ones(absorbance_ex.shape).T.dot(absorbance_em)))
    intensity_corrected = intensity * ife_factors
    return intensity_corrected


def eem_regional_integration(intensity, ex_range, em_range, em_boundary, ex_boundary):
    intensity_cut, em_range_cut, ex_range_cut = eem_cutting(intensity, ex_range, em_range,
                                                            em_min=em_boundary[0], em_max=em_boundary[1],
                                                            ex_min=ex_boundary[0], ex_max=ex_boundary[1])
    if intensity_cut.shape[0] == 1:
        integration = np.trapz(intensity_cut, em_range_cut, axis=1)
    elif intensity_cut.shape[1] == 1:
        integration = np.trapz(intensity_cut, ex_range_cut, axis=0)
    else:
        result_x = np.trapz(intensity_cut, np.flip(ex_range_cut), axis=0)
        integration = np.absolute(np.trapz(result_x, em_range_cut, axis=0))
    # number of effective pixels (i.e. pixels with positive intensity)
    num_pixels = intensity[intensity > 0].shape[0]
    avg_regional_intensity = integration / num_pixels
    return integration, avg_regional_intensity, num_pixels


def eem_interpolation(intensity, ex_range_old, em_range_old, ex_range_new, em_range_new):
    f = interpolate.interp2d(em_range_old, ex_range_old, intensity, kind='linear')
    intensity_interpolated = f(em_range_new, ex_range_new)
    return intensity_interpolated


def eems_tf_normalization(eem_stack):
    eem_stack_normalized = eem_stack.copy()
    tf_list = []
    for i in range(eem_stack.shape[0]):
        tf = eem_stack[i].sum()
        tf_list.append(tf)
    weights = tf_list / np.mean(tf_list)
    eem_stack_normalized = eem_stack / np.array(weights)[:, np.newaxis, np.newaxis]
    return eem_stack_normalized, np.array(weights)


def eems_outlier_detection_if(eem_stack, ex_range, em_range, tf_normalization=True, grid_size=(10, 10),
                              contamination=0.02):
    '''
    tells whether or not (+1 or -1) it should be considered as an inlier according to the fitted model.
    :param eem_stack:
    :param ex_range:
    :param em_range:
    :param tf_normalization:
    :param grid_size:
    :param contamination:
    :return:
    '''
    if tf_normalization:
        eem_stack, _ = eems_tf_normalization(eem_stack)
    em_range_new = np.arange(em_range[0], em_range[-1], grid_size[1])
    ex_range_new = np.arange(ex_range[0], ex_range[-1], grid_size[0])
    eem_stack_interpolated = process_eem_stack(eem_stack, eem_interpolation, ex_range, em_range, ex_range_new,
                                               em_range_new)
    eem_stack_unfold = eem_stack_interpolated.reshape(eem_stack_interpolated.shape[0],
                                                      eem_stack_interpolated.shape[1] * eem_stack_interpolated.shape[2])
    eem_stack_unfold = np.nan_to_num(eem_stack_unfold)
    clf = IsolationForest(random_state=0, n_estimators=200, contamination=contamination).fit(eem_stack_unfold)
    label = clf.predict(eem_stack_unfold)
    return label


def eems_outlier_detection_ocs(eem_stack, ex_range, em_range, tf_normalization=True, grid_size=(10, 10), nu=0.02,
                               kernel="rbf", gamma=10000):
    '''

    :param eem_stack:
    :param ex_range:
    :param em_range:
    :param tf_normalization:
    :param grid_size:
    :param nu:
    :param kernel:
    :param gamma:
    :return:
    '''
    if tf_normalization:
        eem_stack, _ = eems_tf_normalization(eem_stack)
    em_range_new = np.arange(em_range[0], em_range[-1], grid_size[1])
    ex_range_new = np.arange(ex_range[0], ex_range[-1], grid_size[0])
    eem_stack_interpolated = process_eem_stack(eem_stack, eem_interpolation, ex_range, em_range, ex_range_new,
                                               em_range_new)
    eem_stack_unfold = eem_stack_interpolated.reshape(eem_stack_interpolated.shape[0],
                                                      eem_stack_interpolated.shape[1] * eem_stack_interpolated.shape[2])
    eem_stack_unfold = np.nan_to_num(eem_stack_unfold)
    clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma).fit(eem_stack_unfold)
    label = clf.predict(eem_stack_unfold)
    return label


# def eems_random_sampling(eem_stack):
#     i = random.randrange(eem_stack.shape[0])
#     eems_selected = eem_stack[i]
#     return eems_selected, i


class EEM_dataset:
    def __init__(self, eem_stack, ex_range, em_range, ref=None, index=None):
        # The Em/Ex ranges should be sorted in ascending order
        self.eem_stack = eem_stack
        self.ex_range = ex_range
        self.em_range = em_range
        self.ref = ref
        self.index = index
        self.extent = (self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max())

    def zscore(self):
        transformed_data = stats.zscore(self.eem_stack, axis=0)
        return transformed_data

    def mean(self):
        mean = np.mean(self.eem_stack, axis=0)
        return mean

    def variance(self):
        variance = np.var(self.eem_stack, axis=0)
        return variance

    def rel_std(self, threshold=0.05):
        coef_variation = stats.variation(self.eem_stack, axis=0)
        rel_std = abs(coef_variation)
        if threshold:
            mean = np.mean(self.eem_stack, axis=0)
            qualified_pixel_proportion = np.count_nonzero(rel_std < threshold) / np.count_nonzero(~np.isnan(rel_std))
            print("The proportion of pixels with relative STD < {t}: ".format(t=threshold),
                  qualified_pixel_proportion)
        return rel_std

    def std(self):
        return np.std(self.eem_stack, axis=0)

    def total_fluorescence(self):
        return self.intensity.sum()

    def regional_integration(self, em_boundary, ex_boundary):
        integrations, _ = process_eem_stack(self.eem_stack, eem_regional_integration, ex_range=self.ex_range,
                                         em_range=self.em_range, em_boundary=em_boundary, ex_boundary=ex_boundary)
        return integrations

    def pixel_rel_std(self, em, ex, plot=True, timestamp=False, baseline=False, output=True,
                      labels=False):
        M = self.eem_stack
        em_ref = dichotomy_search(self.em_range, em)
        ex_ref = dichotomy_search(self.ex_range, ex)
        y = M[:, self.ex_range.shape[0] - ex_ref - 1, em_ref]
        cod = (self.em_range[em_ref], self.ex_range[ex_ref])
        rel_std = stats.variation(y)
        std = np.std(y)
        x = np.array(range(1, M.shape[0] + 1))
        y_mean = y.mean()
        q5 = abs(0.05 * y_mean)
        q3 = abs(0.03 * y_mean)
        print("Mean: {mean}".format(mean=y_mean))
        print("Standard deviation: {std}".format(std=std))
        print("Relative Standard deviation: {rel_std}".format(rel_std=rel_std))
        if timestamp:
            x = timestamp
        if not timestamp:
            if labels:
                x = labels
        if output:
            table = pd.DataFrame(y)
            table.index = x
            table.columns = ['Intensity (Ex/Em = {ex}/{em})'.format(ex=cod[1], em=cod[0])]
            display(table)
        if plot:
            plt.figure(figsize=(15, 5))
            marker = itertools.cycle(('o', 'v', '^', 's', 'D'))
            plt.plot(x, y, marker=next(marker), markersize=13)
            if timestamp:
                plt.xlim([x[0] - timedelta(hours=1), x[-1] + timedelta(hours=1)])
            if baseline:
                m0 = plt.axhline(y_mean, linestyle='--', label='mean', c='black')
                mp3 = plt.axhline(y_mean + q3, linestyle='--', label='+3%', c='red')
                mn3 = plt.axhline(y_mean - q3, linestyle='--', label='-3%', c='blue')
                if max(y) > y_mean + q5 or min(y) < y_mean - q5:
                    plt.ylim(min(y) - q3, max(y) + q3)
                    # plt.text(1, max(y)+abs(0.15*y_mean), "rel_STD={rel_std}".format(rel_std=round(rel_std,3)))
                else:
                    plt.ylim(y_mean - q5, y_mean + q5)
                    # plt.text(1, y_mean+abs(0.03*y_mean), "rel_STD={rel_std}".format(rel_std=round(rel_std,3)))
                plt.legend([m0, mp3, mn3], ['mean', '+3%', '-3%'], prop={'size': 10})
            plt.xticks(rotation=90)
            plt.title('(Ex, Em) = {cod}'.format(cod=(cod[1], cod[0])))
            plt.show()
        return rel_std

    def pixel_linreg(self, em, ex, plot=True, output=True):
        # Find the closest point to the given coordinate (Em, Ex)
        Em_ref = dichotomy_search(self.em_range, em)
        Ex_ref = dichotomy_search(self.ex_range, ex)
        M = self.eem_stack
        x = self.ref
        x_reshaped = x.reshape(M.shape[0], 1)
        cod = (self.em_range[Em_ref], self.ex_range[Ex_ref])
        print('The closest data point (Em, Ex) is: {cod}'.format(cod=cod))
        y = (M[:, self.ex_range.shape[0] - Ex_ref - 1, Em_ref])
        # x is the reference. y is the fluorescence.
        reg = LinearRegression().fit(x_reshaped, y)
        w = reg.coef_
        b = reg.intercept_
        r2 = reg.score(x_reshaped, y)
        pearson_coef, p_value_p = stats.pearsonr(x, y)
        spearman_coef, p_value_s = stats.spearmanr(x, y)
        print('Linear regression model: y={w}x+{b}'.format(w=w[0], b=b))
        print('Linear regression R2:', '{r2}'.format(r2=r2))
        print('Pearson coefficient: {coef_p} (p-value = {p_p})'.format(coef_p=pearson_coef, p_p=p_value_p))
        print('Spearman coefficient: {coef_s} (p-value = {p_s})'.format(coef_s=spearman_coef, p_s=p_value_s))
        if output:
            intensity_label = 'Intensity (Em, Ex)={cod}'.format(cod=cod)
            table = pd.DataFrame(np.concatenate([[x], [y]]).T)
            table.columns = ['Reference value', intensity_label]
            display(table)
        if plot:
            plt.figure(figsize=(6, 3))
            plt.scatter(x, y)
            p = np.array([x.min(), x.max()])
            q = w[0] * p + b
            plt.plot(p, q)
            plt.title('(Em, Ex) = {cod}'.format(cod=cod), fontdict={"size": 18})
            plt.xlabel('Time', fontdict={"size": 14})
            plt.ylabel('Intensity [a.u.]', fontdict={"size": 14})
            # plt.text(p.min() + 0.1 * (p.max() - p.min()), q.max() - 0.1 * (p.max() - p.min()),
            #          "$R^2$={r2}".format(r2=round(r2, 5)))
            plt.show()
        return table

    def eem_linreg(self, plot=True, scale='log', vmin=0.0001, vmax=0.1, mode='diff', num=4):
        M = self.eem_stack
        x = self.ref
        x = x.reshape(M.shape[0], 1)
        W = np.empty([M.shape[1], M.shape[2]])
        B = np.empty([M.shape[1], M.shape[2]])
        R2 = np.empty([M.shape[1], M.shape[2]])
        E = np.empty(M.shape)
        for i in range(M.shape[1]):
            for j in range(M.shape[2]):
                try:
                    y = (M[:, i, j])
                    reg = LinearRegression().fit(x, y)
                    W[i, j] = reg.coef_
                    B[i, j] = reg.intercept_
                    R2[i, j] = reg.score(x, y)
                    E[:, i, j] = reg.predict(x) - y
                except:
                    pass
        if plot:
            if mode == 'diff':
                X = 1 - R2
                title = '1-$R^2$'
            elif mode == 'abs':
                X = R2
                title = '$R^2$'
            plt.figure(figsize=(6, 6))
            extent = [self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max()]
            if scale == 'log':
                plt.imshow(X, cmap='jet', interpolation='none', extent=extent, aspect='1',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
                t = np.logspace(math.log(vmin), math.log(vmax), num)
                plt.colorbar(ticks=t, fraction=0.03, pad=0.04)
            elif scale == 'linear':
                plt.imshow(X, cmap='jet', interpolation='none', extent=extent, aspect='1', vmin=vmin, vmax=vmax)
                t = np.linspace(vmin, vmax, num)
                plt.colorbar(ticks=t, fraction=0.03, pad=0.04)
            plt.title(title)
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')
        return W, B, R2, E

    def eem_pearson_coef(self, plot=True, crange=(0, 1)):
        M = self.eem_stack
        x = self.ref
        C_p = np.empty([M.shape[1], M.shape[2]])
        P_p = np.empty([M.shape[1], M.shape[2]])
        for i in range(M.shape[1]):
            for j in range(M.shape[2]):
                try:
                    y = (M[:, i, j])
                    pc, p_value_p = stats.pearsonr(x, y)
                    C_p[i, j] = pc
                    P_p[i, j] = p_value_p
                except:
                    pass
        if plot:
            plt.figure(figsize=(8, 8))
            extent = [self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max()]
            plt.imshow(abs(C_p), cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(crange[0], crange[1]))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('abs(Pearson correlation coefficient)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')

            plt.figure(figsize=(8, 8))
            extent = [self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max()]
            plt.imshow(P_p, cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(vmin=0, vmax=0.1))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('p-value (Pearson)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')

    def eem_spearman_coef(self, plot=True, crange=(0, 1)):
        M = self.eem_stack
        x = self.ref
        C_s = np.empty([M.shape[1], M.shape[2]])
        P_s = np.empty([M.shape[1], M.shape[2]])
        for i in range(M.shape[1]):
            for j in range(M.shape[2]):
                try:
                    y = (M[:, i, j])
                    sc, p_value_s = stats.spearmanr(x, y)
                    C_s[i, j] = sc
                    P_s[i, j] = p_value_s
                except:
                    pass
        if plot:
            plt.figure(figsize=(8, 8))
            extent = [self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max()]
            plt.imshow(abs(C_s), cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(crange[0], crange[1]))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('abs(Spearman correlation coefficient)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')

            plt.figure(figsize=(8, 8))
            extent = (self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max())
            plt.imshow(P_s, cmap='jet', interpolation='none', extent=extent, aspect='1.2',
                       norm=Normalize(vmin=0, vmax=0.1))
            plt.colorbar(fraction=0.03, pad=0.04)
            plt.title('p-value (Spearman)')
            plt.xlabel('Emission wavelength [nm]')
            plt.ylabel('Excitation wavelength [nm]')

    # def plot_eem_linreg_error(self, error):
    #     x = self.ref
    #     x_index = self.index
    #     extent = (self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max())
    #     plt.imshow(error[x_index, :, :], cmap='jet', interpolation='none', extent=extent, aspect='1.2', vmin=-100,
    #                vmax=100)
    #     plt.title('x={x}'.format(x=x[x_index]))
    #     plt.xlabel('Emission wavelength [nm]')
    #     plt.ylabel('Excitation wavelength [nm]')

    def threshold_masking(self, threshold, mask_type='greater', plot=False, autoscale=True, cmin=0, cmax=4000,
                          copy=True):
        eem_stack_masked, masks = process_eem_stack(self.eem_stack, eem_threshold_masking, ex_range=self.ex_range,
                                                    em_range=self.em_range, threshold=threshold, mask_type=mask_type,
                                                    plot=plot, autoscale=autoscale, cmin=cmin, cmax=cmax)
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked, masks

    def gaussian_filter(self, sigma=1, truncate=3, copy=True):
        eem_stack_filtered = process_eem_stack(self.eem_stack, eem_gaussian_filter, sigma=sigma, truncate=truncate)
        if not copy:
            self.eem_stack = eem_stack_filtered
        return eem_stack_filtered

    def region_masking(self, ex_min, ex_max, em_min, em_max, fill_value='nan', copy=True):
        eem_stack_masked, _ = process_eem_stack(self.eem_stack, eem_region_masking, ex_range=self.ex_range,
                                                em_range=self.em_range, ex_min=ex_min, ex_max=ex_max, em_min=em_min,
                                                em_max=em_max, fill_value=fill_value)
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked

    def cutting(self, ex_min, ex_max, em_min, em_max, copy=True):
        eem_stack_cut, new_ranges = process_eem_stack(self.eem_stack, eem_cutting, ex_range=self.ex_range,
                                             em_range=self.em_range,
                                             ex_min=ex_min, ex_max=ex_max, em_min=em_min, em_max=em_max)
        if not copy:
            self.eem_stack = eem_stack_cut
            self.ex_range = new_ranges[0][0]
            self.em_range = new_ranges[0][1]
        return eem_stack_cut, new_ranges[0][0], new_ranges[0][1]

    def grid_imputing(self, method='linear', fill_value='linear_ex', prior_mask=None, copy=True):
        eem_stack_imputed = process_eem_stack(self.eem_stack, eem_grid_imputing, ex_range=self.ex_range,
                                              em_range=self.em_range, method=method, fill_value=fill_value,
                                              prior_mask=prior_mask)
        if not copy:
            self.eem_stack = eem_stack_imputed
        return eem_stack_imputed

    def raman_normalization(self, ex_range_blank=None, em_range_blank=None, blank=None, from_blank=False,
                            integration_time=1, ex_lb=349, ex_ub=351, bandwidth_type='wavenumber', bandwidth=1800,
                            rsu_standard=20000, manual_rsu=1, copy=True):
        eem_stack_normalized = process_eem_stack(self.eem_stack, eem_raman_normalization, ex_range_blank=ex_range_blank,
                                                 em_range_blank=em_range_blank, blank=blank, from_blank=from_blank,
                                                 integration_time=integration_time, ex_lb=ex_lb, ex_ub=ex_ub,
                                                 bandwidth_type=bandwidth_type, bandwidth=bandwidth,
                                                 rsu_standard=rsu_standard, manual_rsu=manual_rsu)
        if not copy:
            self.eem_stack = eem_stack_normalized
        return eem_stack_normalized

    def tf_normalization(self, copy=True):
        eem_stack_normalized, weights = eems_tf_normalization(self.eem_stack)
        if not copy:
            self.eem_stack = eem_stack_normalized
        return eem_stack_normalized, weights

    def raman_masking(self, tolerance=5, method='linear', axis='grid', copy=True):
        eem_stack_masked, _ = process_eem_stack(self.eem_stack, eem_raman_masking, ex_range=self.ex_range,
                                                em_range=self.em_range, tolerance=tolerance, method=method, axis=axis)
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked

    def rayleigh_masking(self, tolerance_o1=15, tolerance_o2=15, axis_o1='grid', axis_o2='grid', method_o1='zero',
                         method_o2='linear', copy=True):
        eem_stack_masked, _ = process_eem_stack(self.eem_stack, eem_rayleigh_masking, ex_range=self.ex_range,
                                                em_range=self.em_range, tolerance_o1=tolerance_o1,
                                                tolerance_o2=tolerance_o2, axis_o1=axis_o1, axis_o2=axis_o2,
                                                method_o1=method_o1, method_o2=method_o2)
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked

    def ife_correction(self, absorbance, ex_range_abs, cuvette_length=1, ex_lower_limit=200, ex_upper_limit=825,
                       copy=True):
        eem_stack_corrected = process_eem_stack(self.eem_stack, eem_ife_correction, ex_range=self.ex_range,
                                                em_range=self.em_range, absorbance=absorbance,
                                                ex_range_abs=ex_range_abs, cuvette_length=cuvette_length,
                                                ex_lower_limit=ex_lower_limit, ex_upper_limit=ex_upper_limit)
        if not copy:
            self.eem_stack = eem_stack_corrected
        return eem_stack_corrected

    def interpolation(self, ex_range_new, em_range_new, copy=True):
        eem_stack_interpolated = process_eem_stack(self.eem_stack, eem_interpolation, ex_range_old=self.ex_range,
                                                   em_range_old=self.em_range, ex_range_new=ex_range_new,
                                                   em_range_new=em_range_new)
        if not copy:
            self.eem_stack = eem_stack_interpolated
            self.ex_range = ex_range_new
            self.em_range = em_range_new
        return eem_stack_interpolated, ex_range_new, em_range_new

    def outlier_detection_if(self, tf_normalization=True, grid_size=(10,10), contamination=0.02, deletion=False):
        labels = eems_outlier_detection_if(eem_stack=self.eem_stack, ex_range=self.ex_range, em_range=self.em_range,
                                           tf_normalization=tf_normalization, grid_size=grid_size,
                                           contamination=contamination)
        if deletion:
            self.eem_stack = self.eem_stack[labels != -1]
            self.ref = self.ref[labels != -1]
            self.index = [idx for i, idx in enumerate(self.index) if labels[i] != -1]
        return labels

    def outlier_detection_ocs(self, tf_normalization=True, grid_size=(10,10), nu=0.02, kernel='rbf', gamma=10000,
                              deletion=False):
        labels = eems_outlier_detection_ocs(eem_stack=self.eem_stack, ex_range=self.ex_range, em_range=self.em_range,
                                            tf_normalization=tf_normalization, grid_size=grid_size, nu=nu,
                                            kernel=kernel, gamma=gamma)
        if deletion:
            self.eem_stack = self.eem_stack[labels != -1]
            self.ref = self.ref[labels != -1]
            self.index = [idx for i, idx in enumerate(self.index) if labels[i] != -1]
        return labels


class PARAFAC:
    def __init__(self, rank, non_negativity=True, init='svd', tf_normalization=True,
                 loadings_normalization: Union[Literal['loadings_sd', 'component_maximum'], None] = 'loadings_sd'):
        self.rank = rank
        self.non_negativity = non_negativity
        self.init = init
        self.tf_normalization = tf_normalization
        self.loadings_normalization = loadings_normalization

    def fit(self, eem_dataset: EEM_dataset):
        if self.tf_normalization:
            _, tf_weights = eem_dataset.tf_normalization(copy=False)
        try:
            if not self.non_negativity:
                if np.isnan(eem_dataset.eem_stack).any():
                    mask = np.where(np.isnan(eem_dataset.eem_stack), 0, 1)
                    _, factors = parafac(eem_dataset.eem_stack, rank=self.rank, mask=mask, init=self.init)
                else:
                    _, factors = parafac(eem_dataset.eem_stack, rank=self.rank, init=self.init)
            else:
                if np.isnan(eem_dataset.eem_stack).any():
                    mask = np.where(np.isnan(eem_dataset.eem_stack), 0, 1)
                    _, factors = non_negative_parafac(eem_dataset.eem_stack, rank=self.rank, mask=mask, init=self.init)
                else:
                    _, factors = non_negative_parafac(eem_dataset.eem_stack, rank=self.rank, init=self.init)
        except ArpackError:
            print(
                "PARAFAC failed possibly due to the presence of patches of nan values. Please consider cut or "
                "interpolate the nan values.")
        a, b, c = factors
        max_idx_r = []

        for r in range(self.rank):
            if a[:, r].sum() < 0:
                a[:, r] = -a[:, r]
                if abs(b[:, r].min()) > b[:, r].max():
                    b[:, r] = -b[:, r]
                elif abs(c[:, r].min()) > c[:, r].max():
                    c[:, r] = -c[:, r]
            elif abs(b[:, r].min()) > b[:, r].max() and abs(c[:, r].min()) > c[:, r].max():
                b[:, r] = -b[:, r]
                c[:, r] = -c[:, r]
            if self.loadings_normalization == 'loadings_sd':
                stdj = b[:, r].std()
                stdk = c[:, r].std()
                b[:, r] = b[:, r] / stdj
                c[:, r] = c[:, r] / stdk
                a[:, r] = a[:, r] * stdj * stdk
            component = np.array([b[:, r]]).T.dot(np.array([c[:, r]]))
            if self.loadings_normalization == 'components_maximum':
                w = 1 / component.max()
                component = component * w
                a[:, r] = a[:, r] / w
            if r == 0:
                component_stack = np.zeros([self.rank, component.shape[0], component.shape[1]])
            component_stack[r, :, :] = component
        if self.tf_normalization:
            a = np.multiply(a, tf_weights[:, np.newaxis])
        if self.loadings_normalization == 'components_maximum':
            score_df = pd.DataFrame(a / a.mean(axis=0))
        else:
            score_df = pd.DataFrame(a)


def decomposition_interact(eem_stack, em_range, ex_range, rank, index=[], decomposition_method='parafac', init='svd',
                           dataset_normalization=False, score_normalization=False, loadings_normalization=True,
                           component_normalization=False, plot_loadings=True,
                           plot_components=True, plot_fmax=True, display_score=True, component_cmin=0, component_cmax=1,
                           component_autoscale=False, title=True, cbar=True, cmap="jet", sort_em=True, rotate=False):
    plt.close()
    if dataset_normalization:
        eem_stack, tf = eems_tf_normalization(eem_stack)
    try:
        if decomposition_method == 'parafac':
            if np.isnan(eem_stack).any():
                mask = np.where(np.isnan(eem_stack), 0, 1)
                _, factors = parafac(eem_stack, rank=rank, mask=mask, init=init)
            else:
                _, factors = parafac(eem_stack, rank=rank, init=init)
        if decomposition_method == 'non_negative_parafac':
            if np.isnan(eem_stack).any():
                mask = np.where(np.isnan(eem_stack), 0, 1)
                _, factors = non_negative_parafac(eem_stack, rank=rank, mask=mask, init=init)
            else:
                _, factors = non_negative_parafac(eem_stack, rank=rank, init=init)
    except ArpackError:
        print("Please check if there's blank space in the fluorescence footprint in 'section 2. Fluorescence preview "
              "and parameter selection'. If so, please removed the sample, or adjust the excitation "
              "wavelength range to avoid excessive inner filter effect")
    I, J, K = factors
    max_idx_r = []

    for r in range(rank):
        if I[:, r].sum() < 0:
            I[:, r] = -I[:, r]
            if abs(J[:, r].min()) > J[:, r].max():
                J[:, r] = -J[:, r]
            elif abs(K[:, r].min()) > K[:, r].max():
                K[:, r] = -K[:, r]
        elif abs(J[:, r].min()) > J[:, r].max() and abs(K[:, r].min()) > K[:, r].max():
            J[:, r] = -J[:, r]
            K[:, r] = -K[:, r]
        if loadings_normalization:
            stdj = J[:, r].std()
            stdk = K[:, r].std()
            J[:, r] = J[:, r] / stdj
            K[:, r] = K[:, r] / stdk
            I[:, r] = I[:, r] * stdj * stdk
        component = np.array([J[:, r]]).T.dot(np.array([K[:, r]]))
        if component_normalization:
            w = 1 / component.max()
            component = component * w
            I[:, r] = I[:, r] / w
        if r == 0:
            component_stack = np.zeros([rank, component.shape[0], component.shape[1]])
        component_stack[r, :, :] = component

    if dataset_normalization:
        I = np.multiply(I, tf[:, np.newaxis])
    if score_normalization:
        score_df = pd.DataFrame(I / I.mean(axis=0))
    else:
        score_df = pd.DataFrame(I)

    exl_df = pd.DataFrame(np.flipud(J), index=ex_range)
    eml_df = pd.DataFrame(K, index=em_range)
    ex_column = ["Ex" for i in range(ex_range.shape[0])]
    em_column = ["Em" for i in range(em_range.shape[0])]
    score_column = ["Score" for i in range(score_df.shape[0])]
    exl_df.index = pd.MultiIndex.from_tuples(list(zip(*[ex_column, ex_range.tolist()])),
                                             names=('type', 'wavelength'))
    eml_df.index = pd.MultiIndex.from_tuples(list(zip(*[em_column, em_range.tolist()])),
                                             names=('type', 'wavelength'))

    if index:
        score_df.index = pd.MultiIndex.from_tuples(list(zip(*[score_column, index])),
                                                   names=('type', 'time'))
    else:
        score_df.index = score_column
    if sort_em:
        em_peaks = [c[1] for c in eml_df.idxmax()]
        peak_rank = list(enumerate(stats.rankdata(em_peaks)))
        order = [i[0] for i in sorted(peak_rank, key=lambda x: x[1])]
        exl_df = pd.DataFrame({'component {r}'.format(r=i + 1): exl_df.iloc[:, order[i]] for i in range(rank)})
        eml_df = pd.DataFrame({'component {r}'.format(r=i + 1): eml_df.iloc[:, order[i]] for i in range(rank)})
        score_df = pd.DataFrame({'component {r}'.format(r=i + 1): score_df.iloc[:, order[i]] for i in range(rank)})
        component_stack = component_stack[order]
    else:
        column_labels = ['component {r}'.format(r=i + 1) for i in range(rank)]
        exl_df.columns = column_labels
        eml_df.columns = column_labels
        score_df.columns = column_labels
    if plot_components:
        for r in range(rank):
            if title:
                title_r = score_df.columns[r]
            if not title:
                title_r = False
            plot_eem(component_stack[r, :, :], em_range, ex_range, cmin=component_cmin,
                     cmax=component_cmax, title=title_r, autoscale=component_autoscale, cbar=cbar,
                     cmap=cmap, aspect=1, rotate=rotate)

    if plot_loadings:
        fig_ex = exl_df.unstack(level=0).plot.line()
        handles_ex, labels_ex = fig_ex.get_legend_handles_labels()
        plt.legend(handles_ex, labels_ex, prop={'size': 10})
        plt.xticks(np.arange(0, ex_range[-1] + 1, 50))
        plt.xlim([ex_range[0], ex_range[-1]])
        plt.xlabel("Wavelength [nm]")
        fig_em = eml_df.unstack(level=0).plot.line()
        handles_em, labels_em = fig_em.get_legend_handles_labels()
        plt.legend(handles_em, labels_em, prop={'size': 10})
        plt.xticks(np.arange(0, em_range[-1], 50))
        plt.xlim([em_range[0], em_range[-1]])
        plt.xlabel("Wavelength [nm]")
    I_standardized = I / np.mean(I, axis=0)
    plt.figure(figsize=(15, 5))
    legend = []
    marker = itertools.cycle(('o', 'v', '^', 's', 'D'))
    if display_score:
        display(score_df)
        for r in range(rank):
            if index:
                plt.plot(index, I_standardized[:, r], marker=next(marker), markersize=13)
            else:
                plt.plot(I_standardized[:, r], marker=next(marker), markersize=13)
            legend.append('component {rank}'.format(rank=r + 1))
            plt.xlabel('Time')
            plt.xticks(rotation=90)
            plt.ylabel('Score')
        plt.legend(legend)

    fmax = I * component_stack.max(axis=(1, 2))
    if sort_em:
        fmax_df = pd.DataFrame({'component {r}'.format(r=i + 1): fmax[:, order[i]] for i in range(rank)})
    else:
        fmax_df = pd.DataFrame(fmax, columns=['component {r}'.format(r=i + 1) for i in range(rank)])
    if plot_fmax:
        if index:
            fmax_column = ["Fmax" for i in range(score_df.shape[0])]
            fmax_df.index = pd.MultiIndex.from_tuples(list(zip(*[fmax_column, index])),
                                                      names=('type', 'time'))
        display(fmax_df)
        plt.figure(figsize=(15, 5))
        n_sample = np.arange(fmax.shape[0])
        legend = []
        fmax_tot = 0
        for r in range(rank):
            fmax_r = fmax[:, r]
            plt.bar(n_sample, fmax_r, bottom=fmax_tot)
            fmax_tot += fmax_r
            legend.append('component {rank}'.format(rank=r + 1))
        plt.xticks(n_sample, index, rotation=90)
        plt.ylabel('Fmax')
        plt.legend(legend)
    return score_df, exl_df, eml_df, fmax_df, component_stack, max_idx_r


def match_parafac_components(model_ex, model_em, model_ref_ex, model_ref_em, criteria='TCC',
                             wavelength_synchronization=True, mode='mean', dtw=False):
    ex1_loadings, em1_loadings, ex2_loadings, em2_loadings = \
        model_ex.unstack(level=0), model_em.unstack(level=0), model_ref_ex.unstack(level=0), model_ref_em.unstack(
            level=0),
    ex_range1, em_range1, ex_range2, em_range2 = \
        ex1_loadings.index, em1_loadings.index, ex2_loadings.index, em2_loadings.index
    if wavelength_synchronization:
        em_interval1 = (em_range1.max() - em_range1.min()) / (em_range1.shape[0] - 1)
        em_interval2 = (em_range2.max() - em_range2.min()) / (em_range2.shape[0] - 1)
        ex_interval1 = (ex_range1.max() - ex_range1.min()) / (ex_range1.shape[0] - 1)
        ex_interval2 = (ex_range2.max() - ex_range2.min()) / (ex_range2.shape[0] - 1)
        if em_interval2 > em_interval1:
            em_range_target = em_range1
        else:
            em_range_target = em_range2
        if ex_interval2 > ex_interval1:
            ex_range_target = ex_range1
        else:
            ex_range_target = ex_range2
        f_ex1 = interpolate.interp1d(ex_range_target, ex1_loadings.to_numpy().T)
        f_em1 = interpolate.interp1d(em_range_target, em1_loadings.to_numpy().T)
        f_ex2 = interpolate.interp1d(ex_range_target, ex2_loadings.to_numpy().T)
        f_em2 = interpolate.interp1d(em_range_target, em2_loadings.to_numpy().T)
        ex1_loadings, em1_loadings, ex2_loadings, em2_loadings = \
            f_ex1(ex_range1), f_em1(em_range1), f_ex2(ex_range2), f_em2(em_range2)
    else:
        ex1_loadings, em1_loadings, ex2_loadings, em2_loadings = \
            ex1_loadings.to_numpy().T, em1_loadings.to_numpy().T, ex2_loadings.to_numpy().T, em2_loadings.to_numpy().T

    m_sim = np.zeros([model_ref_ex.shape[1], model_ex.shape[1]])
    matched_index = []
    max_sim = []
    for n2 in range(model_ref_ex.shape[1]):
        for n1 in range(model_ex.shape[1]):
            if dtw:
                ex1_aligned, ex2_aligned = dynamic_time_warping(ex1_loadings[n1], ex2_loadings[n2])
                em1_aligned, em2_aligned = dynamic_time_warping(em1_loadings[n1], em2_loadings[n2])
            else:
                ex1_aligned, ex2_aligned = [ex1_loadings[n1], ex2_loadings[n2]]
                em1_aligned, em2_aligned = [em1_loadings[n1], em2_loadings[n2]]
            if criteria == 'TCC':
                stat_ex = stats.pearsonr(ex1_aligned, ex2_aligned)[0]
                stat_em = stats.pearsonr(em1_aligned, em2_aligned)[0]
                if mode == 'mean':
                    m_sim[n1, n2] = statistics.mean([stat_ex, stat_em])
                if mode == 'min':
                    m_sim[n1, n2] = min([stat_ex, stat_em])
                if mode == 'max':
                    m_sim[n1, n2] = max([stat_ex, stat_em])
            # if criteria == 'SSC'
    memory = []
    m_sim_copy = m_sim.copy()
    for n2 in range(model_ref_ex.shape[1]):
        max_index = np.argmax(m_sim[:, n2])
        while max_index in memory:
            m_sim_copy[max_index, n2] = 0
            max_index = np.argmax(m_sim_copy[:, n2])
        memory.append(max_index)
        matched_index.append((max_index, n2))
        max_sim.append(m_sim[max_index, n2])
    order = [o[0] for o in matched_index]
    model_ex_sorted, model_em_sorted = model_ex.copy(), model_em.copy()
    for i, o in enumerate(order):
        model_ex_sorted['component {i}'.format(i=i + 1)] = model_ex['component {i}'.format(i=o + 1)]
        model_em_sorted['component {i}'.format(i=i + 1)] = model_em['component {i}'.format(i=o + 1)]
    return m_sim, matched_index, max_sim, [model_ex_sorted, model_em_sorted]


def fast_core_consistency(eem_stack, rank=[1, 2, 3, 4, 5], decomposition_method='non_negative_parafac', init='svd',
                          dataset_normalization=False, plot_cc=True):
    # Reference: [1]https://github.com/willshiao/pycorcondia [2]Papalexakis E E, Faloutsos C. Fast efficient and
    # scalable core consistency diagnostic for the parafac decomposition for big sparse tensors[C]//2015 IEEE
    # International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015: 5441-5445.

    def kronecker_mat_ten(matrices, X):
        for k in range(len(matrices)):
            M = matrices[k]
            X = mode_dot(X, M, k)
        return X

    # Shortcut to invert singular values.
    # Given a vector of singular values, returns the inverted matrix

    def invert_sing(s):
        return np.diag(1.0 / s)

    if isinstance(rank, int):
        rank = [rank]
    if dataset_normalization:
        eem_stack, tf = eems_tf_normalization(eem_stack)
    cc_list = []
    for r in rank:
        if decomposition_method == 'parafac':
            _, factors = parafac(eem_stack, r, init=init)
        elif decomposition_method == 'non_negative_parafac':
            _, factors = non_negative_parafac(eem_stack, r, init=init)
        I, J, K = factors
        Ui, Si, Vi = np.linalg.svd(I, full_matrices=False)
        Uj, Sj, Vj = np.linalg.svd(J, full_matrices=False)
        Uk, Sk, Vk = np.linalg.svd(K, full_matrices=False)
        inverted = [invert_sing(x) for x in (Si, Sj, Sk)]
        part1 = kronecker_mat_ten([Ui.T, Uj.T, Uk.T], eem_stack)
        part2 = kronecker_mat_ten(inverted, part1)
        G = kronecker_mat_ten([Vi.T, Vj.T, Vk.T], part2)
        for i in range(r):
            G[:, :, i] = G[:, :, i] / G[i, i, i]
        T = np.zeros((r, r, r))
        for i in range(r):
            T[i, i, i] = 1
        cc_list.append(round(100 * (1 - ((G - T) ** 2).sum() / float(r)), 2))
    if plot_cc:
        plt.close()
        plt.figure(figsize=(10, 5))
        for i in range(len(rank)):
            plt.plot(rank[i], cc_list[i], '-o')
            plt.annotate(cc_list[i], (rank[i] + 0.2, cc_list[i]))
        plt.xlabel("Rank")
        plt.xticks(rank)
        plt.ylabel("Core consistency")
    return cc_list


def explained_variance(eem_stack, rank=[1, 2, 3, 4, 5], decomposition_method='non_negative_parafac', init='svd',
                       dataset_normalization=False, plot_ve=True):
    if isinstance(rank, int):
        rank = [rank]
    if dataset_normalization:
        eem_stack, tf = eems_tf_normalization(eem_stack)
    ev_list = []
    for r in rank:
        if decomposition_method == 'parafac':
            weight, factors = parafac(eem_stack, r, init=init)
        elif decomposition_method == 'non_negative_parafac':
            weight, factors = non_negative_parafac(eem_stack, r, init=init)
        eem_stack_reconstruct = cp_to_tensor((weight, factors))
        y_train = eem_stack.reshape(-1)
        y_pred = eem_stack_reconstruct.reshape(-1)
        ev_list.append(round(100 * (1 - np.var(y_pred - y_train) / np.var(y_train)), 2))
    if plot_ve:
        plt.close()
        plt.figure(figsize=(10, 5))
        for i in range(len(rank)):
            plt.plot(rank[i], ev_list[i], '-o')
            plt.annotate(ev_list[i], (rank[i] + 0.2, ev_list[i]))
        plt.xlabel("Rank")
        plt.xticks(rank)
        plt.ylabel("Variance explained [%]")
    return ev_list


def parafac_pixel_error(eem_stack, em_range, ex_range, rank,
                        decomposition_method='non_negative_parafac', init='svd',
                        dataset_normalization=False):
    if dataset_normalization:
        eem_stack_nor, tf = eems_tf_normalization(eem_stack)
        if decomposition_method == 'parafac':
            weight, factors = parafac(eem_stack_nor, rank, init=init)
        elif decomposition_method == 'non_negative_parafac':
            weight, factors = non_negative_parafac(eem_stack_nor, rank, init=init)
        eem_stack_reconstruct = cp_to_tensor((weight, factors)) * tf[:, np.newaxis, np.newaxis]
    else:
        if decomposition_method == 'parafac':
            weight, factors = parafac(eem_stack, rank, init=init)
        elif decomposition_method == 'non_negative_parafac':
            weight, factors = non_negative_parafac(eem_stack, rank, init=init)
        eem_stack_reconstruct = cp_to_tensor((weight, factors))
    res_abs = eem_stack - eem_stack_reconstruct
    with np.errstate(divide='ignore', invalid='ignore'):
        res_ratio = 100 * (eem_stack - eem_stack_reconstruct) / eem_stack
    return res_abs, res_ratio


def parafac_sample_error(eem_stack, index, rank, error_type='MSE',
                         decomposition_method='non_negative_parafac', init='svd',
                         dataset_normalization=False, plot_error=True):
    def ssim(eem1, eem2, k1=0.01, k2=0.03, l=255):
        c1 = (k1 * l) ** 2
        c2 = (k2 * l) ** 2
        mu1 = np.mean(eem1)
        mu2 = np.mean(eem2)
        sigma1 = np.std(eem1)
        sigma2 = np.std(eem2)
        sigma12 = np.cov(eem1.flat, eem2.flat)[0, 1]
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2)
        ssim_index = numerator / denominator
        return ssim_index

    err_list = []
    if dataset_normalization:
        eem_stack_nor, tf = eems_tf_normalization(eem_stack)

        if decomposition_method == 'parafac':
            weight, factors = parafac(eem_stack_nor, rank, init=init)
        elif decomposition_method == 'non_negative_parafac':
            weight, factors = non_negative_parafac(eem_stack_nor, rank, init=init)
        eem_stack_reconstruct = cp_to_tensor((weight, factors))
        eem_stack_reconstruct = eem_stack_reconstruct * tf[:, np.newaxis, np.newaxis]
    else:
        if decomposition_method == 'parafac':
            weight, factors = parafac(eem_stack, rank, init=init)
        elif decomposition_method == 'non_negative_parafac':
            weight, factors = non_negative_parafac(eem_stack, rank, init=init)
        eem_stack_reconstruct = cp_to_tensor((weight, factors))

    for i in range(eem_stack.shape[0]):
        if error_type == 'MSE':
            err_list.append(np.mean(np.square(eem_stack[i] - eem_stack_reconstruct[i])))
        if error_type == 'PSNR':
            mse = np.mean(np.square(eem_stack[i] - eem_stack_reconstruct[i]))
            err_list.append(20 * np.log10(eem_stack[i].max() / np.sqrt(mse)))
        if error_type == 'SSIM':
            err_list.append(ssim(matrix_dtype_to_uint8(eem_stack[i]), matrix_dtype_to_uint8(eem_stack_reconstruct[i])))
    if plot_error:
        plt.figure(figsize=(10, 5))
        plt.plot(index, err_list)
        plt.xlabel("Sample")
        plt.xticks(rotation=90)
        plt.ylabel(error_type)
    return err_list


def eem_stack_spliting(eem_stack, datlist, n_split=4, rule='random'):
    idx_eems = [i for i in range(eem_stack.shape[0])]
    split_set = []
    datlist_set = []
    if rule == 'random':
        random.shuffle(idx_eems)
        idx_splits = np.array_split(idx_eems, n_split)
    if rule == 'chronological':
        idx_splits = np.array_split(idx_eems, n_split)
    for split in idx_splits:
        split_set.append(np.array([eem_stack[i] for i in split]))
        datlist_set.append([datlist[i] for i in split])
    return split_set, datlist_set


def split_validation(eem_stack, em_range, ex_range, rank, datlist, decomposition_method,
                     n_split=4, combination_size='half', n_test='max', rule='random', index=[],
                     criteria='TCC', plot_all_combos=True, dataset_normalization=False,
                     init='svd'):
    split_set, _ = eem_stack_spliting(eem_stack, datlist, n_split=n_split, rule=rule)
    if combination_size == 'half':
        cs = int(n_split) / 2
    else:
        cs = int(combination_size)
    combos = []
    combo_labels = []
    for i, j in zip(itertools.combinations([i for i in range(n_split)], int(cs * 2)),
                    itertools.combinations(list(string.ascii_uppercase)[0:n_split], int(cs * 2))):
        elements = list(itertools.combinations(i, int(cs)))
        codes = list(itertools.combinations(j, int(cs)))
        for k in range(int(len(elements) / 2)):
            combos.append([elements[k], elements[-1 - k]])
            combo_labels.append([''.join(codes[k]), ''.join(codes[-1 - k])])
    if n_test == 'max':
        n_t = len(combos)
    elif isinstance(n_test, int):
        if n_test > len(combos):
            n_t = len(combos)
        else:
            n_t = n_test
    idx = random.sample(range(len(combos)), n_t)
    test_count = 0
    sims = {}
    models = []
    while test_count < n_t:
        c1 = combos[idx[test_count]][0]
        c2 = combos[idx[test_count]][1]
        label = combo_labels[idx[test_count]]
        eem_stack_c1 = np.concatenate([split_set[i] for i in c1], axis=0)
        eem_stack_c2 = np.concatenate([split_set[i] for i in c2], axis=0)
        score1_df, exl1_df, eml1_df, _, _, _, _ = decomposition_interact(eem_stack_c1, em_range, ex_range, rank,
                                                                         index=index,
                                                                         decomposition_method=decomposition_method,
                                                                         dataset_normalization=dataset_normalization,
                                                                         score_normalization=False,
                                                                         loadings_normalization=True,
                                                                         component_normalization=False,
                                                                         plot_loadings=False,
                                                                         plot_components=False, display_score=False,
                                                                         component_autoscale=True, sort_em=True,
                                                                         init=init, plot_fmax=False
                                                                         )
        score2_df, exl2_df, eml2_df, _, _, _, _ = decomposition_interact(eem_stack_c2, em_range, ex_range, rank,
                                                                         index=index,
                                                                         decomposition_method=decomposition_method,
                                                                         dataset_normalization=dataset_normalization,
                                                                         score_normalization=False,
                                                                         loadings_normalization=True,
                                                                         component_normalization=False,
                                                                         plot_loadings=False,
                                                                         plot_components=False, display_score=False,
                                                                         component_autoscale=True, sort_em=True,
                                                                         init=init, plot_fmax=False
                                                                         )
        if test_count > 0:
            _, matched_index_prev, _, _ = match_parafac_components(models[test_count - 1][0][1],
                                                                   models[test_count - 1][0][2], exl1_df, eml1_df,
                                                                   criteria=criteria,
                                                                   wavelength_synchronization=False, mode='mean')
            order = [o[1] for o in matched_index_prev]
            exl1_df = pd.DataFrame({'component {r}'.format(r=i + 1): exl1_df.iloc[:, order[i]] for i in range(rank)})
            eml1_df = pd.DataFrame({'component {r}'.format(r=i + 1): eml1_df.iloc[:, order[i]] for i in range(rank)})
            score1_df = pd.DataFrame(
                {'component {r}'.format(r=i + 1): score1_df.iloc[:, order[i]] for i in range(rank)})

        m_sim, matched_index, max_sim, _ = match_parafac_components(exl1_df, eml1_df, exl2_df, eml2_df,
                                                                    criteria=criteria,
                                                                    wavelength_synchronization=False, mode='mean')
        for l in matched_index:
            if l[0] != l[1]:
                warnings.warn('Component {c1} of model {m1} does not match with '
                              'component {c1} of model {m2}, which is replaced by Component {c2} of model {m2}'
                              .format(c1=l[0] + 1, c2=l[1] + 1, m1=label[0], m2=label[1]))

        order = [o[1] for o in matched_index]
        exl2_df = pd.DataFrame({'component {r}'.format(r=i + 1): exl2_df.iloc[:, order[i]] for i in range(rank)})
        eml2_df = pd.DataFrame({'component {r}'.format(r=i + 1): eml2_df.iloc[:, order[i]] for i in range(rank)})
        score2_df = pd.DataFrame({'component {r}'.format(r=i + 1): score2_df.iloc[:, order[i]] for i in range(rank)})
        models.append([[score1_df, exl1_df, eml1_df, label[0]], [score2_df, exl2_df, eml2_df, label[1]]])
        sims['test {n}: {l1} vs. {l2}'.format(n=test_count + 1, l1=label[0], l2=label[1])] = max_sim
        test_count += 1
    sims_df = pd.DataFrame(sims, index=['component {c}'.format(c=c + 1) for c in range(rank)])
    if plot_all_combos:
        cmap = get_cmap('tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, 2 * len(models))]
        for r in range(rank):
            plt.figure()
            for i in range(len(models)):
                score1_df, exl1_df, eml1_df, label1 = models[i][0]
                score2_df, exl2_df, eml2_df, label2 = models[i][1]
                plt.plot(exl1_df.index.get_level_values(1), exl1_df.iloc[:, r],
                         color=colors[2 * i], linewidth=1, label=label1 + '-ex')
                plt.plot(exl2_df.index.get_level_values(1), exl2_df.iloc[:, r],
                         color=colors[2 * i + 1], linewidth=1, label=label2 + '-ex')
                plt.plot(eml1_df.index.get_level_values(1), eml1_df.iloc[:, r],
                         color=colors[2 * i], linewidth=1, linestyle='dashed', label=label1 + '-em')
                plt.plot(eml2_df.index.get_level_values(1), eml2_df.iloc[:, r],
                         color=colors[2 * i + 1], linewidth=1, linestyle='dashed', label=label2 + '-em')
                plt.xlabel('Wavelength [nm]', fontsize=15)
                plt.xticks(np.arange(min(ex_range), max(em_range), 50), fontsize=12)
                plt.ylabel('Loadings', fontsize=15)
                plt.yticks(fontsize=12)
                plt.title('component {rank}'.format(rank=r + 1))
            plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        print('Similarity of each test:')
        display(sims_df)
    return models, sims_df


def decomposition_reconstruction_interact(I, J, K, intensity, em_range, ex_range, datlist, data_to_view,
                                          crange=[0, 1000], manual_component=[], rmse=True, plot=True):
    idx = datlist.index(data_to_view)
    rank = I.shape[1]
    if not manual_component:
        num_component = np.arange(0, rank, 1)
    else:
        num_component = [i - 1 for i in manual_component]
    for r in num_component:
        component = np.array([J[:, r]]).T.dot(np.array([K[:, r]]))
        if r == 0:
            sample_r = I[idx, r] * component
        else:
            sample_r += I[idx, r] * component
        # reconstruction_error = np.linalg.norm(sample_r - eem_stack[idx])
        if plot:
            plot_eem(sample_r, em_range, ex_range, autoscale=False, cmin=crange[0], cmax=crange[1], figure_size=(8, 8),
                      title='Accumulate to component {rank}'.format(rank=r + 1))
    if rmse:
        error = np.sqrt(np.mean((sample_r - intensity) ** 2))
        # print("MSE of the final reconstructed EEM: ", error)
    if plot:
        plot_eem(intensity, em_range, ex_range, autoscale=False, cmin=crange[0], cmax=crange[1],
                  title='Original footprint')
        plot_eem(intensity - sample_r, em_range, ex_range, autoscale=False, cmin=crange[0], cmax=crange[1],
                  title='Residuals')
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(intensity - sample_r, intensity)
        ratio[ratio == np.inf] = np.nan
        plot_eem(ratio, em_range, ex_range, autoscale=False, cmin=crange[0], cmax=crange[1], title='Error [%]')
    return sample_r, error


def export_parafac(filepath, score_df, exl_df, eml_df, name, creator, date, email='', doi='', reference='', unit='',
                   toolbox='', fluorometer='', nSample='', decomposition_method='', validation='',
                   dataset_calibration='', preprocess='', sources='', description=''):
    info_dict = {'name': name, 'creator': creator, 'email': email, 'doi': doi, 'reference': reference,
                 'unit': unit, 'toolbox': toolbox, 'date': date, 'fluorometer': fluorometer, 'nSample': nSample,
                 'dateset_calibration': dataset_calibration, 'preprocess': preprocess,
                 'decomposition_method': decomposition_method,
                 'validation': validation, 'sources': sources, 'description': description}
    with open(filepath, 'w') as f:
        f.write('# \n# Fluorescence Model \n# \n')
        for key, value in info_dict.items():
            f.write(key + '\t' + value)
            f.write('\n')
        f.write('# \n# Excitation/Emission (Ex, Em), wavelength [nm], component_n [loading] \n# \n')
        f.close()
    with pd.option_context('display.multi_sparse', False):
        exl_df.to_csv(filepath, mode='a', sep="\t", header=None)
        eml_df.to_csv(filepath, mode='a', sep="\t", header=None)
    with open(filepath, 'a') as f:
        f.write('# \n# timestamp, component_n [Score] \n# \n')
        f.close()
    score_df.to_csv(filepath, mode='a', sep="\t", header=None)
    with open(filepath, 'a') as f:
        f.write('# end #')
    return info_dict

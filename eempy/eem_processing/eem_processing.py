"""
Functions for EEM preprocessing and post-processing
Author: Yongmin Hu (yongmin.hu@eawag.ch, yongminhu@outlook.com)
Last update: 2024-07-03
"""
from matplotlib import pyplot as plt

from eempy.utils import *
import scipy.stats as stats
import random
import pandas as pd
import numpy as np
import itertools
import string
import warnings
import copy
import json
import tensorly as tl
import torch
from math import sqrt
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, silhouette_score
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse.linalg import ArpackError
from scipy.linalg import khatri_rao
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment, nnls
from scipy.spatial.distance import cdist
from tensorly.solvers.nnls import hals_nnls
from tensorly.decomposition import parafac, non_negative_parafac, non_negative_parafac_hals
from tensorly.cp_tensor import cp_to_tensor, CPTensor, cp_normalize
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import unfolding_dot_khatri_rao
from tlviz.model_evaluation import core_consistency
from tlviz.outliers import compute_leverage
from tlviz.factor_tools import permute_cp_tensor
from pandas.plotting import register_matplotlib_converters
from typing import Optional

register_matplotlib_converters()


def process_eem_stack(eem_stack, f, **kwargs):
    """
    Apply an EEM processing function across all EEMs in an EEM stack.

    Parameters
    ----------
    eem_stack: np.ndarray (3d)
        The EEM stack.
    f: callable
        The EEM processing function to apply. Available functions include all functions named in the format of
        "eem_xxx()".
    **kwargs: f function parameters
        The parameters of the EEM processing function.

    Returns
    -------
    processed_eem_stack: np.ndarray
        The processed EEM stack.
    other_outputs: tuple, optional
        If the EEM processing function have more than 1 returns, the rest of the returns will be stored in a tuple,
        where each element is the return of EEM processing function applied to every EEM.
    """
    processed_eem_stack = []
    other_outputs = []

    # ------Absorbance and blank are two parameters that are potentially sample-specific--------
    if "absorbance" in kwargs:
        abs = kwargs.pop('absorbance')
        abs_passed = True
    else:
        abs_passed = False

    if "blank" in kwargs:
        b = kwargs.pop('blank')
        b_passed = True
    else:
        b_passed = False

    for i in range(eem_stack.shape[0]):
        if abs_passed:
            f_output = f(eem_stack[i, :, :], absorbance=abs[i], **kwargs)
        elif b_passed:
            f_output = f(eem_stack[i, :, :], blank=b[i], **kwargs)
        else:
            f_output = f(eem_stack[i, :, :], **kwargs)

        if isinstance(f_output, tuple):
            processed_eem_stack.append(f_output[0])
            other_outputs.append(f_output[1:])
        else:
            processed_eem_stack.append(f_output)
    if len(set([eem.shape for eem in processed_eem_stack])) > 1:
        warnings.warn("Processed EEMs have different shapes")
    if other_outputs:
        return np.array(processed_eem_stack), other_outputs
    else:
        return np.array(processed_eem_stack)


def eem_threshold_masking(intensity, threshold, fill=np.nan, mask_type='greater'):
    """
    Mask the fluorescence intensities above or below a certain threshold in an EEM.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    threshold:
        The intensity threshold.
    fill: float
        The value to fill the masked area
    mask_type: str, {"greater","smaller"}
        Whether to mask the intensities greater or smaller than the threshold.

    Returns
    -------
    intensity_masked: np.ndarray
        The masked EEM.
    mask: np.ndarray
        The mask matrix. +1: unmasked area; np.nan: masked area.
    """
    mask = np.ones(intensity.shape)
    intensity_masked = intensity.astype(float)
    if mask_type == 'smaller':
        mask[np.where(intensity < threshold)] = np.nan
    if mask_type == 'greater':
        mask[np.where(intensity > threshold)] = np.nan
    intensity_masked[np.isnan(mask)] = fill
    return intensity_masked, mask


def eem_region_masking(intensity, ex_range, em_range, ex_min=230, ex_max=500, em_min=250, em_max=810,
                       fill='nan'):
    """
    Mask the fluorescence intensities in a specified rectangular region.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    ex_min: float
        The lower boundary of excitation wavelength of the masked area.
    ex_max: float
        The upper boundary of excitation wavelength of the masked area.
    em_min: float
        The lower boundary of emission wavelength of the masked area.
    em_max: float
        The upper boundary of emission wavelength of the masked area.
    fill: float
        The value to fill the masked area

    Returns
    -------
    intensity_masked: np.ndarray
        The masked EEM.
    mask: np.ndarray
        The mask matrix. +1: unmasked area; np.nan: masked area.
    """
    intensity_masked = intensity.copy()
    em_min_idx = dichotomy_search(em_range, em_min)
    em_max_idx = dichotomy_search(em_range, em_max)
    ex_min_idx = dichotomy_search(ex_range, ex_min)
    ex_max_idx = dichotomy_search(ex_range, ex_max)
    mask = np.ones(intensity.shape)
    mask[ex_range.shape[0] - ex_max_idx - 1:ex_range.shape[0] - ex_min_idx, em_min_idx:em_max_idx + 1] = 0
    if fill == 'nan':
        intensity_masked[mask == 0] = np.nan
    elif fill == 'zero':
        intensity_masked[mask == 0] = 0
    return intensity_masked, mask


def eem_gaussian_filter(intensity, sigma=1, truncate=3):
    """
    Apply Gaussian filtering to an EEM. Reference: scipy.ndimage.gaussian_filter

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    sigma: float
        Standard deviation for Gaussian kernel.
    truncate: float
        Truncate the filter at this many standard deviations.

    Returns
    -------
    intensity_filtered: np.ndarray
        The filtered EEM.
    """
    intensity_filtered = gaussian_filter(intensity, sigma=sigma, truncate=truncate)
    return intensity_filtered


def eem_median_filter(intensity, footprint=(3, 3), mode='reflect'):
    """
    Apply Median filtering to an EEM. Reference: scipy.ndimage.median_filter

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    footprint: tuple of two integers
        Gives the shape that is taken from the input array, at every element position, to define the input to the filter
        function.
    mode: str, {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
        The mode parameter determines how the input array is extended beyond its boundaries.

    Returns
    -------
    eem_stack_filtered: np.ndarray
        The filtered EEM.
    """
    intensity_filtered = median_filter(intensity, footprint=np.ones(footprint), mode=mode)
    return intensity_filtered


def eem_cutting(intensity, ex_range_old, em_range_old, ex_min_new, ex_max_new, em_min_new, em_max_new):
    """
    To cut the EEM.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range_old: np.ndarray (1d)
        The excitation wavelengths.
    em_range_old: np.ndarray (1d)
        The emission wavelengths.
    ex_min_new: float
        The lower boundary of excitation wavelength of the EEM after cutting.
    ex_max_new: float
        The upper boundary of excitation wavelength of the EEM after cutting.
    em_min_new: float
        The lower boundary of emission wavelength of the EEM after cutting.
    em_max_new: float
        The upper boundary of emission wavelength of the EEM after cutting.

    Returns
    -------
    intensity_cut: np.ndarray
        The cut EEM.
    ex_range_cut: np.ndarray
        The cut ex wavelengths.
    em_range_cut:np.ndarray
        The cut em wavelengths.
    """
    em_min_idx = dichotomy_search(em_range_old, em_min_new)
    em_max_idx = dichotomy_search(em_range_old, em_max_new)
    ex_min_idx = dichotomy_search(ex_range_old, ex_min_new)
    ex_max_idx = dichotomy_search(ex_range_old, ex_max_new)
    intensity_cut = intensity[ex_range_old.shape[0] - ex_max_idx - 1:ex_range_old.shape[0] - ex_min_idx,
                    em_min_idx:em_max_idx + 1]
    em_range_cut = em_range_old[em_min_idx:em_max_idx + 1]
    ex_range_cut = ex_range_old[ex_min_idx:ex_max_idx + 1]
    return intensity_cut, ex_range_cut, em_range_cut


def eem_nan_imputing(intensity, ex_range, em_range, method: str = 'linear', fill_value: str = 'linear_ex'):
    """
    Impute the NaN values in an EEM.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    method: str, {"linear", "cubic"}
        The method for imputation.
    fill_value: str, {"linear_ex", "linear_em"}
        The method to extrapolate the NaN that are outside the non-NaN ranges.

    Returns
    -------
    intensity_imputed: np.ndarray
        The imputed EEM.
    """
    x, y = np.meshgrid(em_range, ex_range[::-1])
    xx = x[~np.isnan(intensity)].flatten()
    yy = y[~np.isnan(intensity)].flatten()
    zz = intensity[~np.isnan(intensity)].flatten()
    if isinstance(fill_value, float):
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method, fill_value=fill_value)
    elif fill_value == 'linear_ex':
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method)
        for i in range(intensity_imputed.shape[1]):
            col = intensity_imputed[:, i]
            mask = np.isnan(col)
            if np.any(mask) and np.any(~mask):
                interp_func = interp1d(np.flatnonzero(~mask), col[~mask], kind='linear',
                                       fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            intensity_imputed[:, i] = col
    elif fill_value == 'linear_em':
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method)
        for j in range(intensity_imputed.shape[0]):
            col = intensity_imputed[j, :]
            mask = np.isnan(col)
            if np.any(mask) and np.any(~mask):
                interp_func = interp1d(np.flatnonzero(~mask), col[~mask], kind='linear',
                                       fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            intensity_imputed[j, :] = col
    return intensity_imputed


def eem_raman_normalization(intensity, blank=None, ex_range_blank=None, em_range_blank=None, from_blank=True,
                            integration_time=1, ex_target=350, bandwidth=5,
                            rsu_standard=20000, manual_rsu: Optional[float] = 1):
    """
    Normalize the EEM using the Raman scattering unit (RSU) given directly or calculated from a blank EEM.
    RSU_final = RSU_raw / (RSU_standard * integration_time).

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    blank: np.ndarray (2d) or np.ndarray (3d)
        The blank. If this function is called by implementing process_eem_stack(), a 3d array with length n (the number
        of samples) at axis 0 may be passed. In this case, each EEM will be normalized with a sample-specific blank.
    ex_range_blank: np.ndarray (1d)
        The excitation wavelengths of blank.
    em_range_blank: np.ndarray (1d)
        The emission wavelengths of blank.
    from_blank: bool
        Whether to calculate the RSU from a blank. If False, manual_rsu will be used.
    integration_time: float
        The integration time of the blank measurement.
    ex_target: float
        The excitation wavelength at which the RSU is calculated.
    bandwidth: float
        The bandwidth of Raman scattering peak.
    rsu_standard: float
        A factor used to divide the raw RSU. This is used to control the magnitude of RSU so that the normalized
        intensity of EEM would not be numerically too high or too low.
    manual_rsu: float
        If from_blank = False, this will be used as the RSU_final directly.

    Returns
    -------
    intensity_normalized: np.ndarray
        The normalized EEM.
    rsu_final: float
        The Final RSU used in normalization.
    """
    if not from_blank:
        return intensity / manual_rsu, manual_rsu
    else:
        em_target = -ex_target / (0.00036 * ex_target - 1)
        rsu, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
                                          ex_min=ex_target, ex_max=ex_target, em_min=em_target - bandwidth / 2,
                                          em_max=em_target + bandwidth / 2)
        # elif bandwidth_type == 'wavenumber':
        #     em_target = -ex / (0.00036 * ex - 1)
        #     wn_target = 10000000 / em_target
        #     em_lb = 10000000 / (wn_target + bandwidth)
        #     em_rb = 10000000 / (wn_target - bandwidth)
        #     rsu, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
        #                                       ex_min=ex, ex_max=ex, em_min=em_lb, em_max=em_rb)
    rsu_final = rsu / (integration_time * rsu_standard)
    intensity_normalized = intensity / rsu_final
    return intensity_normalized, rsu_final


def eem_raman_scattering_removal(intensity, ex_range, em_range, width=5, interpolation_method='linear',
                                 interpolation_dimension='2d'):
    """
    Remove and interpolate the Raman scattering.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    width: float
        The width of Raman scattering.
    interpolation_method: str, {"linear", "cubic", "nan"}
        The method used to interpolate the Raman scattering.
    interpolation_dimension: str, {"1d-ex", "1d-em", "2d"}
        The axis along which the Raman scattering is interpolated. "1d-ex": interpolation is conducted along the excitation
        wavelength; "1d-em": interpolation is conducted along the emission wavelength; "2d": interpolation is conducted
        on the 2D grid of both excitation and emission wavelengths.

    Returns
    -------
    intensity_masked: np.ndarray
        The EEM with Raman scattering interpolated.
    raman_mask: np.ndarray
        Indicate the pixels that are interpolated. 0: pixel is interpolated; 1: pixel is not interpolated.
    """
    intensity_masked = np.array(intensity)
    width = width / 2
    raman_mask = np.ones(intensity.shape)
    lambda_em = -ex_range / (0.00036 * ex_range - 1)
    tol_emidx = int(np.round(width / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em[s] <= em_range[0] <= lambda_em[s] + width:
            emidx = dichotomy_search(em_range, lambda_em[s] + width)
            raman_mask[exidx, 0: emidx + 1] = 0
        elif lambda_em[s] - width <= em_range[0] < lambda_em[s]:
            emidx = dichotomy_search(em_range, lambda_em[s])
            raman_mask[exidx, 0: emidx + tol_emidx + 1] = 0
        elif em_range[0] < lambda_em[s] - width < em_range[-1]:
            emidx = dichotomy_search(em_range, lambda_em[s] - width)
            raman_mask[exidx, emidx: min(em_range.shape[0], emidx + 2 * tol_emidx + 1)] = 0

    if interpolation_method == 'nan':
        intensity_masked[np.where(raman_mask == 0)] = np.nan
    else:
        if interpolation_dimension == '1d-ex':
            for j in range(0, intensity.shape[1]):
                try:
                    x = np.flipud(ex_range)[np.where(raman_mask[:, j] == 1)]
                    y = intensity_masked[:, j][np.where(raman_mask[:, j] == 1)]
                    f1 = interp1d(x, y, kind=interpolation_method, fill_value='extrapolate')
                    y_predict = f1(np.flipud(ex_range))
                    intensity_masked[:, j] = y_predict
                except ValueError:
                    continue

        if interpolation_dimension == '1d-em':
            for i in range(0, intensity.shape[0]):
                try:
                    x = em_range[np.where(raman_mask[i, :] == 1)]
                    y = intensity_masked[i, :][np.where(raman_mask[i, :] == 1)]
                    f1 = interp1d(x, y, kind=interpolation_method, fill_value='extrapolate')
                    y_predict = f1(em_range)
                    intensity_masked[i, :] = y_predict
                except ValueError:
                    continue

        if interpolation_dimension == '2d':
            old_nan = np.isnan(intensity)
            intensity_masked[np.where(raman_mask == 0)] = np.nan
            intensity_masked = eem_nan_imputing(intensity_masked, ex_range, em_range, method=interpolation_method)
            # restore the nan values in non-raman-scattering region
            intensity_masked[old_nan] = np.nan
    return intensity_masked, raman_mask


def eem_rayleigh_scattering_removal(intensity, ex_range, em_range, width_o1=15, width_o2=15,
                                    interpolation_dimension_o1='2d', interpolation_dimension_o2='2d',
                                    interpolation_method_o1='zero',
                                    interpolation_method_o2='linear'):
    """
    Remove and interpolate the Rayleigh scattering.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    width_o1: float
        The width or 1st order Rayleigh scattering.
    width_o2: float
        The width or 2nd order Rayleigh scattering.
    interpolation_dimension_o1: str, {"1d-ex", "1d-em", "2d"}
        The axis along which the 1st order Rayleigh scattering is interpolated. "ex": interpolation is conducted along
        the excitation wavelength; "em": interpolation is conducted along the emission wavelength; "2d": interpolation
        is conducted on the 2D grid of both excitation and emission wavelengths.
    interpolation_dimension_o2: str, {"1d-ex", "1d-em", "2d"}
        The axis along which the 2nd order Rayleigh scattering is interpolated.
    interpolation_method_o1: str, {"linear", "cubic", "nan"}
        The method used to interpolate the 1st order Rayleigh scattering.
    interpolation_method_o2: str, {"linear", "cubic", "nan"}
        The method used to interpolate the 2nd order Rayleigh scattering.

    Returns
    -------
    intensity_masked: np.ndarray
        The EEM with Rayleigh scattering interpolated.
    rayleigh_mask_o1: np.ndarray
        Indicate the pixels that are interpolated due to 1st order Rayleigh scattering.
        0: pixel is interpolated; 1: pixel is not interpolated.
    rayleigh_mask_o2
        Indicate the pixels that are interpolated due to 1st order Rayleigh scattering.
        0: pixel is interpolated; 1: pixel is not interpolated.
    """
    intensity_masked = np.array(intensity)
    rayleigh_mask_o1 = np.ones(intensity.shape)
    rayleigh_mask_o2 = np.ones(intensity.shape)
    # convert the entire width to half-width
    width_o1 = width_o1 / 2
    width_o2 = width_o2 / 2
    lambda_em_o1 = ex_range
    tol_emidx_o1 = int(np.round(width_o1 / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em_o1[s] <= em_range[0] <= lambda_em_o1[s] + width_o1:
            emidx = dichotomy_search(em_range, lambda_em_o1[s] + width_o1)
            rayleigh_mask_o1[exidx, 0:emidx + 1] = 0
        elif lambda_em_o1[s] - width_o1 <= em_range[0] < lambda_em_o1[s]:
            emidx = dichotomy_search(em_range, lambda_em_o1[s])
            rayleigh_mask_o1[exidx, 0: emidx + tol_emidx_o1 + 1] = 0
        elif em_range[0] < lambda_em_o1[s] - width_o1 < em_range[-1]:
            emidx = dichotomy_search(em_range, lambda_em_o1[s] - width_o1)
            rayleigh_mask_o1[exidx, emidx: min(em_range.shape[0], emidx + 2 * tol_emidx_o1 + 1)] = 0
            intensity_masked[exidx, 0: emidx] = 0

    lambda_em_o2 = ex_range * 2
    tol_emidx_o2 = int(np.round(width_o2 / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em_o2[s] <= em_range[0] <= lambda_em_o2[s] + width_o2:
            emidx = dichotomy_search(em_range, lambda_em_o2[s] + width_o2)
            rayleigh_mask_o2[exidx, 0:emidx + 1] = 0
        elif lambda_em_o2[s] - width_o2 <= em_range[0] < lambda_em_o2[s]:
            emidx = dichotomy_search(em_range, lambda_em_o2[s])
            rayleigh_mask_o2[exidx, 0: emidx + tol_emidx_o2 + 1] = 0
        elif em_range[0] < lambda_em_o2[s] - width_o2 <= em_range[-1]:
            emidx = dichotomy_search(em_range, lambda_em_o2[s] - width_o2)
            rayleigh_mask_o2[exidx, emidx: min(em_range.shape[0], emidx + 2 * tol_emidx_o2 + 1)] = 0

    for axis, itp, mask in zip([interpolation_dimension_o1, interpolation_dimension_o2],
                               [interpolation_method_o1, interpolation_method_o2],
                               [rayleigh_mask_o1, rayleigh_mask_o2]):
        if itp == 'zero':
            intensity_masked[np.where(mask == 0)] = 0
        elif itp == 'nan':
            intensity_masked[np.where(mask == 0)] = np.nan
        elif itp == 'none':
            pass
        else:
            if axis == '1d-ex':
                for j in range(0, intensity.shape[1]):
                    try:
                        x = np.flipud(ex_range)[np.where(mask[:, j] == 1)]
                        y = intensity_masked[:, j][np.where(mask[:, j] == 1)]
                        f1 = interp1d(x, y, kind=itp, fill_value='extrapolate')
                        y_predict = f1(np.flipud(ex_range))
                        intensity_masked[:, j] = y_predict
                    except ValueError:
                        continue
            if axis == '1d-em':
                for i in range(0, intensity.shape[0]):
                    try:
                        x = em_range[np.where(mask[i, :] == 1)]
                        y = intensity_masked[i, :][np.where(mask[i, :] == 1)]
                        f1 = interp1d(x, y, kind=itp, fill_value='extrapolate')
                        y_predict = f1(em_range)
                        intensity_masked[i, :] = y_predict
                    except ValueError:
                        continue
            if axis == '2d':
                old_nan = np.isnan(intensity)
                old_nan_o1 = np.isnan(intensity_masked)
                intensity_masked[np.where(mask == 0)] = np.nan
                intensity_masked = eem_nan_imputing(intensity_masked, ex_range, em_range, method=itp,
                                                    fill_value='linear_ex')
                # restore the nan values in non-raman-scattering region
                intensity_masked[old_nan] = np.nan
                intensity_masked[old_nan_o1] = np.nan
    return intensity_masked, rayleigh_mask_o1, rayleigh_mask_o2


def eem_ife_correction(intensity, ex_range_eem, em_range_eem, absorbance, ex_range_abs):
    """
    Correct the inner filter effect (IFE).

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range_eem: np.ndarray (1d)
        The excitation wavelengths of EEM.
    em_range_eem: np.ndarray (1d)
        The emission wavelengths of EEM.
    absorbance: np.ndarray
        The absorbance. If this function is called by itself, an array of shape (i, ) should be passed, where i is the
        length of the absorbance spectrum. If this function is called by process_eem_stack(), an array of shape (n, i)
        should be passed, where n is the number samples, and i is the length of the absorbance spectrum.
    ex_range_abs: np.ndarray (1d)
        The excitation wavelengths of absorbance.

    Returns
    -------
    intensity_corrected: np.ndarray
        The corrected EEM.
    """
    # if absorbance.ndim == 1:
    #     absorbance_reshape = absorbance.reshape((1, absorbance.shape[0]))
    f1 = interp1d(ex_range_abs, absorbance, kind='linear', bounds_error=False, fill_value='extrapolate')
    absorbance_ex = np.fliplr(np.array([f1(ex_range_eem)]))
    absorbance_em = np.array([f1(em_range_eem)])
    ife_factors = 10 ** ((absorbance_ex.T.dot(np.ones(absorbance_em.shape)) +
                          np.ones(absorbance_ex.shape).T.dot(absorbance_em)) / 2)
    intensity_corrected = intensity * ife_factors
    return intensity_corrected


def eem_regional_integration(intensity, ex_range, em_range, ex_min, ex_max, em_min, em_max):
    """
    Calculate the regional fluorescence integration (RFI) over a rectangular region.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths
    ex_min: float
        The lower boundary of excitation wavelengths of the integrated region.
    ex_max: float
        The upper boundary of excitation wavelengths of the integrated region.
    em_min: float
        The lower boundary of emission wavelengths of the integrated region.
    em_max: float
        The upper boundary of emission wavelengths of the integrated region.

    Returns
    -------
    integration: float
        The RFI.
    avg_regional_intensity:
        The average fluorescence intensity in the region.
    """

    ex_range_interpolated = np.sort(np.unique(np.concatenate([ex_range, [ex_min, ex_max]])))
    em_range_interpolated = np.sort(np.unique(np.concatenate([em_range, [em_min, em_max]])))
    intensity_interpolated = eem_interpolation(intensity, ex_range, em_range, ex_range_interpolated,
                                               em_range_interpolated, method='linear')
    intensity_cut, ex_range_cut, em_range_cut = eem_cutting(intensity_interpolated, ex_range_interpolated,
                                                            em_range_interpolated,
                                                            ex_min_new=ex_min, ex_max_new=ex_max,
                                                            em_min_new=ex_min, em_max_new=em_max)
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
    return integration, avg_regional_intensity


def eem_interpolation(intensity, ex_range_old, em_range_old, ex_range_new, em_range_new, method: str = 'linear'):
    """
    Interpolate EEM on given ex/em ranges. This function is typically used for changing the ex/em ranges of an EEM
    (e.g., in order to synchronize EEMs to the same ex/em ranges). It may not be able to interpolate EEM containing nan
    values. For nan value imputation, please consider eem_nan_imputing().

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range_old: np.ndarray (1d)
        The original excitation wavelengths.
    em_range_old: np.ndarray (1d)
        The original emission wavelengths.
    ex_range_new: np.ndarray (1d)
        The new excitation wavelengths.
    em_range_new: np.ndarray (1d)
        The new emission wavelengths.
    method: str, {"linear", "nearest", "slinear", "cubic", "quintic", "pchip"}
        The interpolation method.

    Returns
    -------
    intensity_interpolated: np.ndarray
        The interpolated EEM.
    """
    interp = RegularGridInterpolator((ex_range_old[::-1], em_range_old), intensity, method=method, bounds_error=False)
    x, y = np.meshgrid(ex_range_new[::-1], em_range_new, indexing='ij')
    intensity_interpolated = interp((x, y)).reshape(ex_range_new.shape[0], em_range_new.shape[0])
    return intensity_interpolated


def eems_tf_normalization(intensity):
    """
    Normalize EEMs by the total fluorescence of each EEM.

    Parameters
    ----------
    intensity: np.ndarray (3d)
        The EEM stack.

    Returns
    -------
    intensity_normalized: np.ndarray
        The normalized EEM stack.
    weights: np.ndarray
        The total fluorescence of each EEM.
    """
    tf_list = []
    for i in range(intensity.shape[0]):
        tf = intensity[i].sum()
        tf_list.append(tf)
    weights = np.array(tf_list) / np.mean(tf_list)
    intensity_normalized = intensity / weights[:, np.newaxis, np.newaxis]
    return intensity_normalized, weights


def eems_fit_components(eem_stack, components, fit_intercept=False, positive=True):
    assert eem_stack.shape[1:] == components.shape[1:], "EEM and component have different shapes"
    eem_stack[np.isnan(eem_stack)] = 0
    components[np.isnan(components)] = 0
    score_sample = []
    fmax_sample = []
    max_values = np.amax(components, axis=(1, 2))
    eem_stack_pred = np.zeros(eem_stack.shape)
    for i in range(eem_stack.shape[0]):
        y_true = eem_stack[i].reshape([-1])
        x = components.reshape([components.shape[0], -1]).T
        reg = LinearRegression(fit_intercept=fit_intercept, positive=positive)
        reg.fit(x, y_true)
        y_pred = reg.predict(x)
        eem_stack_pred[i, :, :] = y_pred.reshape((eem_stack.shape[1], eem_stack.shape[2]))
        score_sample.append(reg.coef_)
        fmax_sample.append(reg.coef_ * max_values)
    score_sample = np.array(score_sample)
    fmax_sample = np.array(fmax_sample)
    return score_sample, fmax_sample, eem_stack_pred


def eems_error(eem_stack_true, eem_stack_pred, metric: str = 'mse'):
    assert eem_stack_true.shape == eem_stack_pred.shape, "eem_stack_true and eem_stack_pred have different shapes"
    error = []
    for i in range(eem_stack_true.shape[0]):
        y_true = eem_stack_true[i].reshape([-1])
        y_pred = eem_stack_pred[i].reshape([-1])
        if metric == 'mse':
            error.append(mean_squared_error(y_true, y_pred))
        elif metric == 'explained_variance':
            error.append(explained_variance_score(y_true, y_pred))
        elif metric == 'r2':
            error.append(r2_score(y_true, y_pred))
    return np.array(error)


def eems_decomposition_initialization(eem_stack, rank, method="nndsvd"):
    a, b, c = eem_stack.shape
    M_a = eem_stack.reshape(a, b * c)
    M_b = eem_stack.transpose(1, 0, 2).reshape(b, a * c)
    M_c = eem_stack.transpose(2, 0, 1).reshape(c, a * b)

    if method == "random":
        La = np.random.uniform(0, sqrt(np.mean(M_a) / rank), (a, rank))
        Lb = np.random.uniform(0, sqrt(np.mean(M_b) / rank), (b, rank))
        Lc = np.random.uniform(0, sqrt(np.mean(M_c) / rank), (c, rank))
        return La, Lb, Lc

    elif method == 'svd':
        weights_init, factors_init = initialize_cp(
            eem_stack, non_negative=True, init="svd", rank=rank
        )
        return factors_init

    else:
        # def NNDSVD(M):
        #     # Step 1: Compute SVD of V
        #     U, S, VT = np.linalg.svd(M, full_matrices=False)  # SVD decomposition
        #
        #     # Step 2: Keep the top-r components
        #     U_r = U[:, :rank]
        #     S_r = S[:rank]
        #     VT_r = VT[:rank, :]
        #
        #     # Step 3: Initialize W and H
        #     W = np.zeros((M.shape[0], rank))
        #     H = np.zeros((rank, M.shape[1]))
        #
        #     for k in range(rank):
        #         u_k = U_r[:, k]
        #         v_k = VT_r[k, :]
        #
        #         # Positive and negative parts
        #         u_k_pos = np.maximum(u_k, 0)
        #         u_k_neg = np.maximum(-u_k, 0)
        #         v_k_pos = np.maximum(v_k, 0)
        #         v_k_neg = np.maximum(-v_k, 0)
        #
        #         # Normalize
        #         u_norm_pos = np.linalg.norm(u_k_pos)
        #         v_norm_pos = np.linalg.norm(v_k_pos)
        #
        #         # Assign components
        #         if u_norm_pos * v_norm_pos > 0:
        #             W[:, k] = np.sqrt(S_r[k]) * (u_k_pos / u_norm_pos)
        #             H[k, :] = np.sqrt(S_r[k]) * (v_k_pos / v_norm_pos)
        #         else:
        #             W[:, k] = np.sqrt(S_r[k]) * (u_k_neg / np.linalg.norm(u_k_neg))
        #             H[k, :] = np.sqrt(S_r[k]) * (v_k_neg / np.linalg.norm(v_k_neg))
        #
        #     # Step 4: Handle zero entries
        #     if method == 'nndsvd':
        #         # W[W == 0] = np.random.uniform(0, 1e-4, W[W == 0].shape)
        #         # H[H == 0] = np.random.uniform(0, 1e-4, H[H == 0].shape)
        #         pass
        #     if method == 'nndsvda':
        #         W[W == 0] = np.mean(M)
        #         H[H == 0] = np.mean(M)
        #     if method == 'nndsvdar':
        #         W[W == 0] = np.random.uniform(0, np.mean(M) / 100, W[W == 0].shape)
        #         H[H == 0] = np.random.uniform(0, np.mean(M) / 100, H[H == 0].shape)
        #     return W, H

        La, _ = unfolded_eem_stack_initialization(M_a, rank, method)
        Lb, _ = unfolded_eem_stack_initialization(M_b, rank, method)
        Lc, _ = unfolded_eem_stack_initialization(M_c, rank, method)

        return La, Lb, Lc


class EEMDataset:
    """
    Build an EEM dataset.

    Parameters
    ----------
    eem_stack: np.ndarray (3d)
        A stack of EEM. It should have a shape of (N, I, J), where N is the number of samples, I is the number of
        excitation wavelengths, and J is the number of emission wavelengths.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    ref: pd.DataFrame or None
        Optional. The reference data, e.g., the DOC of each sample. It should have a length equal to the number of
        samples in the eem_stack.
    index: list or None
        Optional. The index used to label each sample. The number of elements in the list should equal the number
        of samples in the eem_stack.
    cluster: list or None
        Optional. The classification of samples, e.g., the output of EEM clustering algorithms.
    """

    def __init__(self, eem_stack: np.ndarray, ex_range: np.ndarray, em_range: np.ndarray,
                 index: Optional[list] = None, ref: Optional[pd.DataFrame] = None, cluster: Optional[list] = None):

        # ------------------parameters--------------------
        # The Em/Ex ranges should be sorted in ascending order
        self.eem_stack = eem_stack
        self.ex_range = ex_range
        self.em_range = em_range
        self.ref = ref
        self.index = index
        self.cluster = cluster
        self.extent = (self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max())

    # def to_json_serializable(self):
    #     self.eem_stack = self.eem_stack.tolist()
    #     self.ex_range = self.ex_range.tolist()
    #     self.em_range = self.em_range.tolist()
    #     self.ref = self.ref.tolist() if self.ref else None

    # --------------------EEM dataset features--------------------
    def zscore(self):
        """
        Calculate zscore of each pixel over all samples.

        Returns
        -------
        zscore: np.ndarray
        """
        zscore = stats.zscore(self.eem_stack, axis=0)
        return zscore

    def mean(self):
        """
        Calculate mean of each pixel over all samples.

        Returns
        -------
        mean: np.ndarray
        """
        mean = np.mean(self.eem_stack, axis=0)
        return mean

    def variance(self):
        """
        Calculate variance of each pixel over all samples.

        Returns
        -------
        variance: np.ndarray
        """
        variance = np.var(self.eem_stack, axis=0)
        return variance

    def std(self):
        """
        Calculate standard deviation of each pixel over all samples.

        Returns
        -------
        std: np.ndarray
        """
        return np.std(self.eem_stack, axis=0)

    def total_fluorescence(self):
        """
        Calculate total fluorescence of each sample.

        Returns
        -------
        tf: np.ndarray
        """
        return self.eem_stack.sum(axis=(1, 2))

    def regional_integration(self, ex_min, ex_max, em_min, em_max):
        """
        Calculate regional integration of samples.

        Parameters
        ----------
        See eempy.eem_processing.eem_regional_integration

        Returns
        -------
        integrations: pd.DataFrame
        """
        integrations, _ = process_eem_stack(
            self.eem_stack, eem_regional_integration, ex_range=self.ex_range,
            em_range=self.em_range, ex_min=ex_min, ex_max=ex_max, em_min=em_min, em_max=em_max
        )
        ri_name = f'RI (ex=[{ex_min}, {ex_max}] nm, em=[{em_min}, {em_max}] nm)'
        if self.index:
            ri = pd.DataFrame(integrations, index=self.index, columns=[ri_name])
        else:
            ri = pd.DataFrame(integrations, index=np.arange(integrations.shape[0]), columns=[ri_name])
        return ri

    def peak_picking(self, ex, em):
        """
        Return the fluorescence intensities at the location closest the given (ex, em)

        Parameters
        ----------
        ex: float or int
            excitation wavelength of the wanted location
        em: float or int
            emission wavelength of the wanted location

        Returns
        -------
        fi: pandas.DataFrame
            table of fluorescence intensities at the wanted location for all samples
        ex_actual:
            the actual ex of the extracted fluorescence intensities
        em_actual:
            the actual em of the extracted fluorescence intensities
        """
        em_idx = dichotomy_search(self.em_range, em)
        ex_idx = dichotomy_search(self.ex_range, ex)
        fi = self.eem_stack[:, - ex_idx - 1, em_idx]
        ex_actual = self.ex_range[ex_idx]
        em_actual = self.em_range[em_idx]
        fi_name = f'Intensity (ex={ex_actual} nm, em={em_actual} nm)'
        if self.index:
            fi = pd.DataFrame(fi, index=self.index, columns=[fi_name])
        else:
            fi = pd.DataFrame(fi, index=np.arange(fi.shape[0]), columns=[fi_name])

        return fi, ex_actual, em_actual

    def hix(self):
        """
        Calculate the humification index (HIX).

        Returns
        -------
        hix: pandas.DataFrame
            HIX
        """
        em1 = 300
        em2 = 345
        em3 = 435
        em4 = 480
        ex = 254
        if self.em_range.min() <= em1 and self.em_range.max() >= em4 and self.ex_range.min() <= ex <= self.ex_range.max():
            numerator = self.regional_integration(ex, ex, em3, em4).to_numpy()
            denominator = self.regional_integration(ex, ex, em1, em2).to_numpy() + numerator
            hix = numerator / denominator
            return pd.DataFrame(hix, index=self.index, columns=['HIX'])
        else:
            raise ValueError("The ranges of excitation or emission wavelengths do not support the calculation.")

    def bix(self):
        """
        Calculate the biological index (BIX).

        Returns
        -------
        bix: pandas.DataFrame
            BIX

        """
        ex = 310
        em1 = 380
        em2 = 430
        if self.em_range.min() <= em1 and self.em_range.max() >= em2 and self.ex_range.min() <= ex <= self.ex_range.max():
            numerator, _, _ = self.peak_picking(ex, em1)
            denominator, _, _ = self.peak_picking(ex, em2)
            bix = numerator.to_numpy() / denominator.to_numpy()
            return pd.DataFrame(bix, index=self.index, columns=['BIX'])
        else:
            raise ValueError("The ranges of excitation or emission wavelengths do not support the calculation.")

    def fi(self):
        """
        Calculate the fluorescence index (FI).

        Returns
        -------
        fi: pandas.DataFrame
            FI
        """
        ex = 370
        em1 = 470
        em2 = 520
        if self.em_range.min() <= em1 and self.em_range.max() >= em2 and self.ex_range.min() <= ex <= self.ex_range.max():
            numerator, _, _ = self.peak_picking(ex, em1)
            denominator, _, _ = self.peak_picking(ex, em2)
            fi = numerator.to_numpy() / denominator.to_numpy()
            return pd.DataFrame(fi, index=self.index, columns=['BIX'])
        else:
            raise ValueError("The ranges of excitation or emission wavelengths do not support the calculation.")

    def aqy(self, abs_stack, ex_range_abs, target_ex=None):
        """
        Calculate the apparent_quantum_yield (AQY).

        Parameters
        ----------
        abs_stack: np.ndarray
            absorbance spectra stack
        ex_range_abs: np.ndarray
            excitation wavelengths of absorbance spectra
        target_ex: float or None
            excitation wavelength for AQY. If None is passed, all excitation wavelengths will be returned.

        Returns
        -------
        aqy: pandas.DataFrame
            apparent quantum yield (AQY)
        """
        aqy = []
        for i in range(self.eem_stack.shape[0]):
            intensity = self.eem_stack[i]
            abs = abs_stack[i]
            f1 = interp1d(ex_range_abs, abs, kind='linear', bounds_error=False, fill_value='extrapolate')
            abs_interpolated = f1(self.ex_range)
            aqy.append(np.sum(intensity, axis=1)[::-1] / abs_interpolated)
        aqy = pd.DataFrame(aqy, index=self.index, columns=[f'AQY (ex = {l} nm)' for l in list(self.ex_range)])
        if target_ex is None:
            return aqy
        else:
            return aqy[f'AQY (ex = {target_ex} nm)']

    def correlation(self, variables, fit_intercept=True):
        """
        Analyze the correlation between reference and fluorescence intensity at each pair of ex/em.

        Params
        -------
        variables: list
            List of variables (i.e., the headers of the reference table) to be fitted
        fit_intercept: bool, optional
            Whether to fit the intercept for linear regression.

        Returns
        -------
        corr_dict: dict
            A dictionary containing multiple correlation evaluation metrics.
        """
        m = self.eem_stack
        corr_dict = {var: None for var in variables}
        for var in variables:
            x = np.array(self.ref.loc[:, [var]])
            w = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            b = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            r2 = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            pc = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            pc_p = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            sc = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            sc_p = np.full((m.shape[1], m.shape[2]), fill_value=np.nan)
            e = np.full(m.shape, fill_value=np.nan)
            for i in range(m.shape[1]):
                for j in range(m.shape[2]):
                    try:
                        y = (m[:, i, j])
                        reg = LinearRegression(fit_intercept=fit_intercept)
                        reg.fit(x, y)
                        w[i, j] = reg.coef_
                        b[i, j] = reg.intercept_
                        r2[i, j] = reg.score(x, y)
                        e[:, i, j] = reg.predict(x) - y
                        pc[i, j] = stats.pearsonr(x.reshape(-1), y).statistic
                        pc_p[i, j] = stats.pearsonr(x.reshape(-1), y).pvalue
                        sc[i, j] = stats.spearmanr(x.reshape(-1), y).statistic
                        sc_p[i, j] = stats.spearmanr(x.reshape(-1), y).pvalue
                    except ValueError:
                        pass
            corr_dict.var = {'slope': w, 'intercept': b, 'r_square': r2, 'linear regression residual': e,
                             'Pearson corr. coef.': pc, 'Pearson corr. coef. p-value': pc_p, 'Spearman corr. coef.': sc,
                             'Spearman corr. coef. p-value': sc_p}
        return corr_dict

    # -----------------EEM dataset processing methods-----------------

    def threshold_masking(self, threshold, fill, mask_type='greater', copy=True):
        """
        Mask the fluorescence intensities above or below a certain threshold in an EEM.

        Parameters
        ----------
        threshold, fill, mask_type:
            See eempy.eem_processing.eem_threshold_masking
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_masked: np.ndarray
            The masked EEM.
        mask: np.ndarray
            The mask matrix. +1: unmasked area; np.nan: masked area.
        """
        eem_stack_masked, masks = process_eem_stack(self.eem_stack, eem_threshold_masking, threshold=threshold,
                                                    fill=fill, mask_type=mask_type)
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked, masks

    def gaussian_filter(self, sigma=1, truncate=3, copy=True):
        """
        Apply Gaussian filtering to an EEM.

        Parameters
        ----------
        sigma, truncate:
            See eempy.eem_processing.eem_gaussian_filter
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_filtered: np.ndarray
            The filtered EEM.
        """
        eem_stack_filtered = process_eem_stack(self.eem_stack, eem_gaussian_filter, sigma=sigma, truncate=truncate)
        if not copy:
            self.eem_stack = eem_stack_filtered
        return eem_stack_filtered

    def median_filter(self, footprint=(3, 3), mode='reflect', copy=True):
        """
        Apply median filtering to an EEM.

        Parameters
        ----------
        footprint: tuple of two integers
            Gives the shape that is taken from the input array, at every element position, to define the input to the filter
            function.
        mode: str, {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
            The mode parameter determines how the input array is extended beyond its boundaries.
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_filtered: np.ndarray
            The filtered EEM.
        """
        eem_stack_filtered = process_eem_stack(self.eem_stack, eem_median_filter, footprint=(3, 3), mode='reflect')
        if not copy:
            self.eem_stack = eem_stack_filtered
        return eem_stack_filtered

    def region_masking(self, ex_min, ex_max, em_min, em_max, fill_value='nan', copy=True):
        """
        Mask the fluorescence intensities in a specified rectangular region.

        Parameters
        ----------
        ex_min, ex_max, em_min, em_max, fill_value:
            See eempy.eem_processing.eem_region_masking
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_masked: np.ndarray
            The masked EEM.
        """
        eem_stack_masked, _ = process_eem_stack(
            self.eem_stack, eem_region_masking, ex_range=self.ex_range,
            em_range=self.em_range, ex_min=ex_min, ex_max=ex_max, em_min=em_min,
            em_max=em_max, fill_value=fill_value
        )
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked

    def cutting(self, ex_min, ex_max, em_min, em_max, copy=True):
        """
        Calculate the regional fluorescence integration (RFI) over a rectangular region.

        Parameters
        ----------
        ex_min, ex_max, em_min, em_max:
            See eempy.eem_processing.eem_cutting
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        intensity_cut: np.ndarray
            The cut EEM.
        ex_range_cut: np.ndarray
            The cut ex wavelengths.
        em_range_cut:np.ndarray
            The cut em wavelengths.
        """
        eem_stack_cut, new_ranges = process_eem_stack(
            self.eem_stack, eem_cutting, ex_range_old=self.ex_range,
            em_range_old=self.em_range,
            ex_min_new=ex_min, ex_max_new=ex_max, em_min_new=em_min,
            em_max_new=em_max
        )
        if not copy:
            self.eem_stack = eem_stack_cut
            self.ex_range = new_ranges[0][0]
            self.em_range = new_ranges[0][1]
        return eem_stack_cut, new_ranges[0][0], new_ranges[0][1]

    def nan_imputing(self, method='linear', fill_value='linear_ex', copy=True):
        """
        Impute the NaN values in an EEM.

        Parameters
        ----------
        method, fill_value
            See eempy.eem_processing.eem_nan_imputing
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_imputed: np.ndarray
            The imputed EEM.
        """
        eem_stack_imputed = process_eem_stack(self.eem_stack, eem_nan_imputing, ex_range=self.ex_range,
                                              em_range=self.em_range, method=method, fill_value=fill_value)
        if not copy:
            self.eem_stack = eem_stack_imputed
        return eem_stack_imputed

    def raman_normalization(self, ex_range_blank=None, em_range_blank=None, blank=None, from_blank=False,
                            integration_time=1, ex_target=350, bandwidth=5,
                            rsu_standard=20000, manual_rsu=1, copy=True):
        """
        Normalize the EEM using the Raman scattering unit (RSU) given directly or calculated from a blank EEM.
        RSU_final = RSU_raw / (RSU_standard * integration_time).

        Parameters
        ----------
        blank, ex_range_blank, em_range_blank, from_blank, integration_time, ex_target, bandwidth
            See eempy.eem_processing.eem_raman_normalization
        rsu_standard, manual_rsu
            See eempy.eem_processing.eem_raman_normalization
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_normalized: np.ndarray
            The normalized EEM.
        """
        eem_stack_normalized, rsu = process_eem_stack(
            self.eem_stack, eem_raman_normalization, ex_range_blank=ex_range_blank,
            em_range_blank=em_range_blank, blank=blank, from_blank=from_blank,
            integration_time=integration_time, ex_target=ex_target,
            bandwidth=bandwidth, rsu_standard=rsu_standard, manual_rsu=manual_rsu
        )
        if not copy:
            self.eem_stack = eem_stack_normalized
        return eem_stack_normalized, rsu

    def tf_normalization(self, copy=True):
        """
        Normalize EEMs by the total fluorescence of each EEM.

        Parameters
        ----------
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_normalized: np.ndarray
            The normalized EEM stack.
        weights: np.ndarray
            The weighted total fluorescence of each EEM.
        """
        eem_stack_normalized, weights = eems_tf_normalization(self.eem_stack)
        if not copy:
            self.eem_stack = eem_stack_normalized
        return eem_stack_normalized, weights

    def raman_scattering_removal(self, width=5, interpolation_method='linear', interpolation_dimension='2d', copy=True):
        """
        Remove and interpolate the Raman scattering.

        Parameters
        ----------
        width, interpolation_method, interpolation_dimension:
            See eempy.eem_processing.eem_raman_scattering_removal
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_masked: np.ndarray
            The EEM with Raman scattering interpolated.
        """
        eem_stack_masked, _ = process_eem_stack(
            self.eem_stack, eem_raman_scattering_removal, ex_range=self.ex_range,
            em_range=self.em_range, width=width,
            interpolation_method=interpolation_method,
            interpolation_dimension=interpolation_dimension
        )
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked

    def rayleigh_scattering_removal(self, width_o1=15, width_o2=15, interpolation_dimension_o1='2d',
                                    interpolation_dimension_o2='2d', interpolation_method_o1='zero',
                                    interpolation_method_o2='linear', copy=True):
        """
        Remove and interpolate the Rayleigh scattering.

        Parameters
        ----------
        width_o1, width_o2, interpolation_dimension_o1, interpolation_dimension_o2, interpolation_method_o1, interpolation_method_o2:
            See eempy.eem_processing.eem_rayleigh_scattering_removal
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_masked: np.ndarray
            The EEM with Rayleigh scattering interpolated.
        """
        eem_stack_masked, _ = process_eem_stack(
            self.eem_stack, eem_rayleigh_scattering_removal, ex_range=self.ex_range,
            em_range=self.em_range, width_o1=width_o1,
            width_o2=width_o2,
            interpolation_dimension_o1=interpolation_dimension_o1,
            interpolation_dimension_o2=interpolation_dimension_o2,
            interpolation_method_o1=interpolation_method_o1,
            interpolation_method_o2=interpolation_method_o2
        )
        if not copy:
            self.eem_stack = eem_stack_masked
        return eem_stack_masked

    def ife_correction(self, absorbance, ex_range_abs, copy=True):
        """
        Correct the inner filter effect (IFE).

        Parameters
        ----------
        absorbance, ex_range_abs:
            See eempy.eem_processing.eem_ife_correction
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_corrected: np.ndarray
            The corrected EEM.
        """
        eem_stack_corrected = process_eem_stack(
            self.eem_stack, eem_ife_correction, ex_range_eem=self.ex_range,
            em_range_eem=self.em_range, absorbance=absorbance,
            ex_range_abs=ex_range_abs
        )
        if not copy:
            self.eem_stack = eem_stack_corrected
        return eem_stack_corrected

    def interpolation(self, ex_range_new, em_range_new, method, copy=True):
        """
        Interpolate EEM on given ex/em ranges. This function is typically used for changing the ex/em ranges of an EEM
        (e.g., in order to synchronize EEMs to the same ex/em ranges). It may not be able to interpolate EEM containing
        nan values. For nan value imputation, please consider eem_nan_imputing().

        Parameters
        ----------
        ex_range_new, em_range_new, method:
            See eempy.eem_processing.eem_interpolation
        copy: bool
            if False, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_stack_interpolated: np.ndarray
            The interpolated EEM.
        """
        eem_stack_interpolated = process_eem_stack(
            self.eem_stack, eem_interpolation, ex_range_old=self.ex_range,
            em_range_old=self.em_range, ex_range_new=ex_range_new,
            em_range_new=em_range_new, method=method
        )
        if not copy:
            self.eem_stack = eem_stack_interpolated
            self.ex_range = ex_range_new
            self.em_range = em_range_new
        return eem_stack_interpolated

    def splitting(self, n_split, rule: str = 'random', random_state=None):
        """
        To split the EEM dataset and form multiple sub-datasets.

        Parameters
        ----------
        n_split: int
            The number of splits.
        rule: str, {'random', 'sequential'}
            If 'random' is passed, the split will be generated randomly. If 'sequential' is passed, the dataset will be
            split according to index order.
        random_state: int, optional
            Random seed for splitting.

        Returns
        -------
        model_list: list.
            A list of sub-datasets. Each of them is an EEMDataset object.
        """
        idx_eems = [i for i in range(self.eem_stack.shape[0])]
        subset_list = []
        if rule == 'random':
            if random_state is not None:
                random.seed(random_state)
            random.shuffle(idx_eems)
            idx_splits = np.array_split(idx_eems, n_split)
        elif rule == 'sequential':
            idx_splits = np.array_split(idx_eems, n_split)
        else:
            raise ValueError("'rule' should be either 'random' or 'sequential'")
        for split in idx_splits:
            if self.ref is not None:
                ref = self.ref.iloc[split]
            else:
                ref = None
            if self.index is not None:
                index = [self.index[i] for i in split]
            else:
                index = None
            if self.cluster is not None:
                cluster = [self.cluster[i] for i in split]
            else:
                cluster = None
            m = EEMDataset(eem_stack=np.array([self.eem_stack[i] for i in split]), ex_range=self.ex_range,
                           em_range=self.em_range, ref=ref, index=index, cluster=cluster)
            subset_list.append(m)
        return subset_list

    def subsampling(self, portion=0.8, copy=True):
        """
        Randomly select a portion of the EEM.

        Parameters
        ----------
        portion: float
            The portion.
        copy: bool
            if False, overwrite the EEMDataset object.

        Returns
        -------
        eem_dataset_sub: np.ndarray
            New EEM dataset.
        selected_indices: np.ndarray
            Indices of selected EEMs.
        """
        n_samples = self.eem_stack.shape[0]
        selected_indices = np.random.choice(n_samples, size=int(n_samples * portion), replace=False)
        eem_stack_new = self.eem_stack[selected_indices, :, :]
        if self.index is not None:
            index_new = [self.index[i] for i in selected_indices]
        else:
            index_new = None
        if self.ref is not None:
            ref_new = self.ref.iloc[selected_indices]
        else:
            ref_new = None
        if self.cluster is not None:
            cluster_new = [self.cluster[i] for i in selected_indices]
        else:
            cluster_new = None
        if not copy:
            self.eem_stack = eem_stack_new
            self.index = index_new
            self.ref = ref_new
            self.cluster = cluster_new
        eem_dataset_sub = EEMDataset(eem_stack=eem_stack_new, ex_range=self.ex_range, em_range=self.em_range,
                                     index=index_new, ref=ref_new, cluster=cluster_new)
        eem_dataset_sub.sort_by_index()
        selected_indices = sorted(selected_indices)
        return eem_dataset_sub, selected_indices

    def sort_by_index(self):
        """
        Sort the sample order of eem_stack, index and reference (if exists) by the index.

        Returns
        -------
        sorted_indices: np.ndarray
            The sorted sample order
        """
        sorted_indices = sorted(range(len(self.index)), key=lambda i: self.index[i])
        self.index = sorted(self.index)
        self.eem_stack = self.eem_stack[sorted_indices]
        if self.ref is not None:
            self.ref = self.ref.iloc[sorted_indices]
        if self.cluster is not None:
            self.cluster = [self.cluster[i] for i in sorted_indices]
        return sorted_indices

    def filter_by_index(self, mandatory_keywords, optional_keywords, copy=True):
        """
        Select the samples whose indexes contain the given keyword.

        Parameters
        -------
        mandatory_keywords: str or list of str
            Keywords for selecting samples whose indexes contain all the mandatory keywords.
        optional_keywords: str or list of str
            Keywords for selecting samples whose indexes contain any of the optional keywords.
        copy: bool
            if False, overwrite the EEMDataset object.

        Returns
        -------
        eem_dataset_filtered: EEMDataset
            Filtered EEM dataset.
        sample_number_all_filtered: list
            Indexes (orders in the list) of samples that have been preserved after filtering.
        """
        if mandatory_keywords is None and optional_keywords is None:
            return self.eem_stack, self.index, self.ref, self.cluster, []

        if self.index is None:
            raise ValueError('index is not defined')
        if isinstance(mandatory_keywords, str):
            mandatory_keywords = [mandatory_keywords]
        if isinstance(optional_keywords, str):
            optional_keywords = [optional_keywords]
        sample_number_mandatory_filtered = []
        sample_number_all_filtered = []

        if mandatory_keywords is not None:
            for i, f in enumerate(self.index):
                if all(kw in f for kw in mandatory_keywords):
                    sample_number_mandatory_filtered.append(i)
        else:
            sample_number_mandatory_filtered = list(range(len(self.index)))

        if optional_keywords is not None:
            for j in sample_number_mandatory_filtered:
                if any(kw in self.index[j] for kw in optional_keywords):
                    sample_number_all_filtered.append(j)
        else:
            sample_number_all_filtered = sample_number_mandatory_filtered
        eem_stack_filtered = self.eem_stack[sample_number_all_filtered, :, :]
        index_filtered = [self.index[i] for i in sample_number_all_filtered]
        cluster_filtered = [self.cluster[i] for i in sample_number_all_filtered] if self.cluster is not None else None
        ref_filtered = self.ref.iloc[sample_number_all_filtered] if self.ref is not None else None
        eem_dataset_filtered = EEMDataset(eem_stack_filtered,
                                          ex_range=self.ex_range,
                                          em_range=self.em_range,
                                          index=index_filtered,
                                          ref=ref_filtered)
        if not copy:
            self.eem_stack = eem_stack_filtered
            self.index = index_filtered
            self.ref = ref_filtered
            self.cluster = cluster_filtered
        return eem_dataset_filtered, sample_number_all_filtered

    def filter_by_cluster(self, cluster_names, copy=True):
        """
        Select the samples belong to certain cluster(s).

        Parameters
        -------
        cluster_names: int/float/str or list of int/float/str
            cluster names.
        copy: bool
            if False, overwrite the EEMDataset object.

        Returns
        -------
        eem_dataset_filtered: EEMDataset
            Filtered EEM dataset.
        sample_number_all_filtered: list
            Indexes (orders in the list) of samples that have been preserved after filtering.
        """

        if self.cluster is None:
            raise ValueError('cluster of EEMDataset is not defined')
        if not isinstance(cluster_names, list):
            cluster_names = [cluster_names]
        sample_number_all_filtered = []
        for i, f in enumerate(self.cluster):
            if f in cluster_names:
                sample_number_all_filtered.append(i)
        eem_stack_filtered = self.eem_stack[sample_number_all_filtered, :, :]
        index_filtered = [self.index[i] for i in sample_number_all_filtered]
        cluster_filtered = [self.cluster[i] for i in sample_number_all_filtered] if self.cluster is not None else None
        ref_filtered = self.ref.iloc[sample_number_all_filtered] if self.ref is not None else None
        eem_dataset_filtered = EEMDataset(eem_stack_filtered,
                                          ex_range=self.ex_range,
                                          em_range=self.em_range,
                                          index=index_filtered,
                                          ref=ref_filtered)
        if not copy:
            self.eem_stack = eem_stack_filtered
            self.index = index_filtered
            self.ref = ref_filtered
            self.cluster = cluster_filtered
        return eem_dataset_filtered, sample_number_all_filtered


def combine_eem_datasets(list_eem_datasets):
    """
    Combine all EEMDataset objects in a list

    Parameters
    ----------
    list_eem_datasets: list.
        List of EEM datasets.

    Returns
    -------
    eem_dataset_combined: EEMDataset
        EEM dataset combined.
    """
    eem_stack_combined = []
    index_combined = []
    cluster_combined = []
    ref_combined = None
    ex_range_0 = list_eem_datasets[0].ex_range
    em_range_0 = list_eem_datasets[0].em_range
    for i, d in enumerate(list_eem_datasets):
        eem_stack_combined.append(d.eem_stack)
        if d.index is not None:
            index_combined = index_combined + d.index
        else:
            index_combined = index_combined + ['N/A' for i in range(d.eem_stack.shape[0])]
        if d.ref is not None:
            ref_combined = ref_combined.combine_first(d.ref) if ref_combined is not None else d.ref
        if d.cluster is not None:
            cluster_combined = cluster_combined + d.cluster
        else:
            cluster_combined = cluster_combined + ['N/A' for i in range(d.eem_stack.shape[0])]
        if not np.array_equal(d.ex_range, ex_range_0) or not np.array_equal(d.em_range, em_range_0):
            warnings.warn(
                'ex_range and em_range of the datasets must be identical. If you want to combine EEM datasets '
                'having different ex/em ranges, please consider unify the ex/em ranges using the interpolation() '
                'method of EEMDataset object')
    eem_stack_combined = np.concatenate(eem_stack_combined, axis=0)
    if ref_combined is not None:
        ref_combined = ref_combined.reindex(index_combined)
    eem_dataset_combined = EEMDataset(eem_stack=eem_stack_combined, ex_range=ex_range_0, em_range=em_range_0,
                                      ref=ref_combined, index=index_combined, cluster=cluster_combined)
    return eem_dataset_combined


class PARAFAC:
    """
    Parallel factor analysis (PARAFAC) model for EEM dataset.

    Parameters
    ----------
    n_components: int
        The number of components
    non_negativity: bool
        Whether to apply the non-negativity constraint
    solver: str, {'mu', 'hals'}
        Optimizer to for PARAFAC. 'mu' for multiplicative update optimizer, 'hals' for hierarchical alternating least
        square. 'hals' can only be applied together with non-negativity contraint.
    init: str or tensorly.CPTensor, {‘svd’, ‘random’, CPTensor}
        Type of factor matrix initialization
    tf_normalization: bool
        Whether to normalize the EEMs by the total fluorescence in PARAFAC model establishment
    loadings_normalization: str or None, {'sd', 'maximum', None}
        Type of normalization applied to loadings. if 'sd' is passed, the standard deviation will be normalized
        to 1. If 'maximum' is passed, the maximum will be normalized to 1. The scores will be adjusted accordingly.
    sort_em: bool
        Whether to sort components by emission peak position from lowest to highest. If False is passed, the
        components will be sorted by the contribution to the total variance.

    Attributes
    ----------
    score: pandas.DataFrame
        Score table.
    ex_loadings: pandas.DataFrame
        Excitation loadings table.
    em_loadings: pandas.DataFrame
        Emission loadings table.
    fmax: pandas.DataFrame
        Fmax table.
    components: np.ndarray
        PARAFAC components.
    cptensors: tensorly CPTensor
        The output of PARAFAC in the form of tensorly CPTensor.
    eem_stack_train: np.ndarray
        EEMs used for PARAFAC model establishment.
    eem_stack_reconstructed: np.ndarray
        EEMs reconstructed by the established PARAFAC model.
    ex_range: np.ndarray
        Excitation wavelengths.
    em_range: np.ndarray
        Emission wavelengths.
    """

    def __init__(self, n_components, non_negativity=True, solver='hals', init='svd',
                 tf_normalization=True, loadings_normalization: Optional[str] = 'maximum', sort_em=True,
                 alpha_sample=0, alpha_ex=0, alpha_em=0, l1_ratio=1,
                 prior_dict_sample=None, prior_dict_ex=None, prior_dict_em=None,
                 gamma_sample=0, gamma_ex=0, gamma_em=0, prior_ref_components=None,
                 idx_top=None, idx_bot=None, lam=0,
                 max_iter_als=100, tol=1e-06, max_iter_nnls=500, random_state=None
                 ):

        # ----------parameters--------------
        self.n_components = n_components
        self.non_negativity = non_negativity
        self.init = init
        self.tf_normalization = tf_normalization
        self.loadings_normalization = loadings_normalization
        self.sort_em = sort_em
        self.solver = solver
        self.alpha_sample = alpha_sample
        self.alpha_ex = alpha_ex
        self.alpha_em = alpha_em
        self.l1_ratio = l1_ratio
        self.prior_dict_sample = prior_dict_sample
        self.prior_dict_ex = prior_dict_ex
        self.prior_dict_em = prior_dict_em
        self.gamma_sample = gamma_sample
        self.gamma_ex = gamma_ex
        self.gamma_em = gamma_em
        self.prior_ref_components = prior_ref_components
        self.idx_top = idx_top
        self.idx_bot = idx_bot
        self.lam = lam
        self.max_iter_als = max_iter_als
        self.tol = tol
        self.max_iter_nnls = max_iter_nnls
        self.random_state = random_state

        # -----------attributes---------------
        self.score = None
        self.ex_loadings = None
        self.em_loadings = None
        self.fmax = None
        self.nnls_fmax = None
        self.components = None
        self.cptensors = None
        self.eem_stack_train = None
        self.eem_stack_reconstructed = None
        self.ex_range = None
        self.em_range = None
        self.beta = None

    # --------------methods------------------
    def fit(self, eem_dataset: EEMDataset):
        """
        Establish a PARAFAC model based on a given EEM dataset

        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset that the PARAFAC model establishes on.

        Returns
        -------
        self: object
            The established PARAFAC model
        """
        if self.tf_normalization:
            eem_stack_tf, tf_weights = eem_dataset.tf_normalization(copy=True)
        else:
            eem_stack_tf = eem_dataset.eem_stack.copy()
        try:
            if not self.non_negativity:
                if np.isnan(eem_stack_tf).any():
                    mask = np.where(np.isnan(eem_stack_tf), 0, 1)
                    cptensors = parafac(eem_stack_tf, rank=self.n_components, mask=mask, init=self.init,
                                        n_iter_max=self.max_iter_als, tol=self.tol)
                else:
                    cptensors = parafac(eem_stack_tf, rank=self.n_components, init=self.init,
                                        n_iter_max=self.max_iter_als, tol=self.tol)
                a, b, c = cptensors[1]
            else:
                if np.isnan(eem_stack_tf).any():
                    mask = np.where(np.isnan(eem_stack_tf), 0, 1)
                    if self.solver == 'hals':
                        cptensors = non_negative_parafac_hals(eem_stack_tf, rank=self.n_components, mask=mask,
                                                              init=self.init,
                                                              n_iter_max=self.max_iter_als, tol=self.tol)
                    elif self.solver == 'mu':
                        cptensors = non_negative_parafac(eem_stack_tf, rank=self.n_components, mask=mask,
                                                         init=self.init,
                                                         n_iter_max=self.max_iter_als, tol=self.tol)
                    a, b, c = cptensors[1]
                else:
                    if self.solver == 'hals':
                        if self.idx_top is not None and self.idx_bot is not None:
                            a, b, c, beta = cp_hals_prior_ratio(
                                eem_stack_tf,
                                rank=self.n_components,
                                init=self.init,
                                prior_dict_A=self.prior_dict_sample,
                                prior_dict_B=self.prior_dict_ex,
                                prior_dict_C=self.prior_dict_em,
                                alpha_A=self.alpha_sample,
                                alpha_B=self.alpha_ex,
                                alpha_C=self.alpha_em,
                                l1_ratio=self.l1_ratio,
                                gamma_A=self.gamma_sample,
                                gamma_B=self.gamma_ex,
                                gamma_C=self.gamma_em,
                                idx_top=self.idx_top,
                                idx_bot=self.idx_bot,
                                lam=self.lam,
                                max_iter_als=self.max_iter_als,
                                max_iter_nnls=self.max_iter_nnls,
                                prior_ref_components=self.prior_ref_components,
                                random_state=self.random_state
                            )
                            self.beta = beta
                        else:
                            a, b, c = cp_hals_prior(
                                eem_stack_tf,
                                rank=self.n_components,
                                init=self.init,
                                prior_dict_A=self.prior_dict_sample,
                                prior_dict_B=self.prior_dict_ex,
                                prior_dict_C=self.prior_dict_em,
                                alpha_A=self.alpha_sample,
                                alpha_B=self.alpha_ex,
                                alpha_C=self.alpha_em,
                                l1_ratio=self.l1_ratio,
                                gamma_A=self.gamma_sample,
                                gamma_B=self.gamma_ex,
                                gamma_C=self.gamma_em,
                                max_iter_als=self.max_iter_als,
                                max_iter_nnls=self.max_iter_nnls,
                                prior_ref_components=self.prior_ref_components,
                                random_state=self.random_state
                            )
                        cptensors = [[1] * self.n_components, [a, b, c]]
                    elif self.solver == 'mu':
                        cptensors = non_negative_parafac(eem_stack_tf, rank=self.n_components, init=self.init,
                                                         n_iter_max=self.max_iter_als, tol=self.tol)
                        a, b, c = cptensors[1]
        except ArpackError:
            print(
                "PARAFAC failed possibly due to the presence of patches of nan values. Please consider cut or "
                "interpolate the nan values.")
        components = np.zeros([self.n_components, b.shape[0], c.shape[0]])
        for r in range(self.n_components):
            # when non_negativity is not applied, ensure the scores are generally positive
            if not self.non_negativity:
                if a[:, r].sum() < 0:
                    a[:, r] = -a[:, r]
                    if abs(b[:, r].min()) > b[:, r].max():
                        b[:, r] = -b[:, r]
                    else:
                        c[:, r] = -c[:, r]
                elif abs(b[:, r].min()) > b[:, r].max() and abs(c[:, r].min()) > c[:, r].max():
                    b[:, r] = -b[:, r]
                    c[:, r] = -c[:, r]

            if self.loadings_normalization == 'sd':
                stdb = b[:, r].std()
                stdc = c[:, r].std()
                b[:, r] = b[:, r] / stdb
                c[:, r] = c[:, r] / stdc
                a[:, r] = a[:, r] * stdb * stdc
            elif self.loadings_normalization == 'maximum':
                maxb = b[:, r].max()
                maxc = c[:, r].max()
                b[:, r] = b[:, r] / maxb
                c[:, r] = c[:, r] / maxc
                a[:, r] = a[:, r] * maxb * maxc
            component = np.array([b[:, r]]).T.dot(np.array([c[:, r]]))
            components[r, :, :] = component
        if self.tf_normalization:
            fmax = pd.DataFrame(a * tf_weights[:, np.newaxis])
        else:
            fmax = pd.DataFrame(a)
        a, _, _ = eems_fit_components(eem_dataset.eem_stack, components, fit_intercept=False, positive=True)
        score = pd.DataFrame(a)
        nnls_fmax = a * components.max(axis=(1, 2))
        ex_loadings = pd.DataFrame(np.flipud(b), index=eem_dataset.ex_range)
        em_loadings = pd.DataFrame(c, index=eem_dataset.em_range)
        if self.sort_em:
            em_peaks = [c for c in em_loadings.idxmax()]
            peak_rank = list(enumerate(stats.rankdata(em_peaks)))
            order = [i[0] for i in sorted(peak_rank, key=lambda x: x[1])]
            components = components[order]
            ex_loadings = pd.DataFrame({'component {r} ex loadings'.format(r=i + 1): ex_loadings.iloc[:, order[i]]
                                        for i in range(self.n_components)})
            em_loadings = pd.DataFrame({'component {r} em loadings'.format(r=i + 1): em_loadings.iloc[:, order[i]]
                                        for i in range(self.n_components)})
            score = pd.DataFrame({'component {r} score'.format(r=i + 1): score.iloc[:, order[i]]
                                  for i in range(self.n_components)})
            fmax = pd.DataFrame({'component {r} score'.format(r=i + 1): fmax.iloc[:, order[i]]
                                  for i in range(self.n_components)})
            nnls_fmax = pd.DataFrame({'component {r} fmax'.format(r=i + 1): nnls_fmax[:, order[i]]
                                 for i in range(self.n_components)})
        else:
            column_labels = ['component {r}'.format(r=i + 1) for i in range(self.n_components)]
            ex_loadings.columns = column_labels
            em_loadings.columns = column_labels
            score.columns = ['component {r} PARAFAC-score'.format(r=i + 1) for i in range(self.n_components)]
            nnls_fmax = pd.DataFrame(nnls_fmax, columns=['component {r} PARAFAC-Fmax'.format(r=i + 1) for i in
                                               range(self.n_components)])

        ex_loadings.index = eem_dataset.ex_range.tolist()
        em_loadings.index = eem_dataset.em_range.tolist()

        if eem_dataset.index:
            score.index = eem_dataset.index
            fmax.index = eem_dataset.index
            nnls_fmax.index = eem_dataset.index
        else:
            score.index = [i + 1 for i in range(a.shape[0])]
            fmax.index = [i + 1 for i in range(a.shape[0])]
            nnls_fmax.index = [i + 1 for i in range(a.shape[0])]

        eem_stack_reconstructed = np.tensordot(score.to_numpy(), components, axes=(1, 0))

        self.score = score
        self.ex_loadings = ex_loadings
        self.em_loadings = em_loadings
        self.fmax = fmax
        self.nnls_fmax = nnls_fmax
        self.components = components
        self.cptensors = cptensors
        self.eem_stack_train = eem_dataset.eem_stack
        self.ex_range = eem_dataset.ex_range
        self.em_range = eem_dataset.em_range
        self.eem_stack_reconstructed = eem_stack_reconstructed
        return self

    def predict(self, eem_dataset: EEMDataset, fit_intercept=False, fit_beta=False, idx_top=None, idx_bot=None):
        """
        Predict the score and Fmax of a given EEM dataset using the component fitted. This method can be applied to a
        new EEM dataset independent of the one used in NMF model establishment.

        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset to be predicted.
        fit_intercept: bool
            Whether to calculate the intercept.
        fit_beta: bool
            Whether to fit the beta parameter (the proportions between "top" and "bot" samples).
        idx_top: list, optional
            List of indices of samples serving as numerators in ratio calculation.
        idx_bot: list, optional
            List of indices of samples serving as denominators in ratio calculation.

        Returns
        -------
        score_sample: pd.DataFrame
            The fitted score.
        fmax_sample: pd.DataFrame
            The fitted Fmax.
        eem_stack_pred: np.ndarray (3d)
            The EEM dataset reconstructed.
        """
        if not fit_beta:
            score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset.eem_stack, self.components,
                                                                            fit_intercept=fit_intercept)
        else:
            assert self.beta is not None, "Parameter beta must be provided through fitting."
            assert idx_top is not None and idx_bot is not None, "idx_top and idx_bot must be provided."
            max_values = np.amax(self.components, axis=(1, 2))
            score_sample = np.zeros([eem_dataset.eem_stack.shape[0], self.n_components])
            score_sample_bot = solve_W(
                X1=eem_stack_to_2d(eem_dataset.eem_stack)[idx_bot],
                X2=eem_stack_to_2d(eem_dataset.eem_stack)[idx_top],
                H=self.components.reshape([self.n_components, -1]),
                beta=self.beta
            )
            score_sample[idx_bot] = score_sample_bot
            score_sample[idx_top] = score_sample_bot * self.beta
            fmax_sample = score_sample * max_values
            eem_stack_pred = score_sample @ self.components.reshape([self.n_components, -1])
            eem_stack_pred = eem_stack_pred.reshape(eem_dataset.eem_stack.shape)
        score_sample = pd.DataFrame(score_sample, index=eem_dataset.index, columns=self.nnls_fmax.columns)
        fmax_sample = pd.DataFrame(fmax_sample, index=eem_dataset.index, columns=self.nnls_fmax.columns)
        return score_sample, fmax_sample, eem_stack_pred

    def component_peak_locations(self):
        """
        Get the ex/em of component peaks

        Returns
        -------
        max_exem: list
            A List of (ex, em) of component peaks.
        """
        max_exem = []
        for r in range(self.n_components):
            max_index = np.unravel_index(np.argmax(self.components[r, :, :]), self.components[r, :, :].shape)
            max_exem.append((self.ex_range[-(max_index[0] + 1)], self.em_range[max_index[1]]))
        return max_exem

    def residual(self):
        """
        Get the residual of the established PARAFAC model, i.e., the difference between the original EEM dataset and
        the reconstructed EEM dataset.

        Returns
        -------
        res: np.ndarray (3d)
            the residual
        """
        res = self.eem_stack_train - self.eem_stack_reconstructed
        return res

    def variance_explained(self):
        """
        Calculate the explained variance of the established PARAFAC model

        Returns
        -------
        ev: float
            the explained variance
        """
        ss_total = tl.norm(self.eem_stack_train) ** 2
        ss_residual = tl.norm(self.eem_stack_train - self.eem_stack_reconstructed) ** 2
        variance_explained = (ss_total - ss_residual) / ss_total * 100
        return variance_explained

    def core_consistency(self):
        """
        Calculate the core consistency of the established PARAFAC model

        Returns
        -------
        cc: float
            core consistency
        """
        cc = core_consistency(self.cptensors, self.eem_stack_train)
        return cc

    def leverage(self, mode: str = 'sample'):
        """
        Calculate the leverage of a selected mode.

        Parameters
        ----------
        mode: str, {'ex', 'em', 'sample'}
            The mode of which the leverage is calculated.

        Returns
        -------
        lvr: pandas.DataFrame
            The table of leverage

        """
        if mode == 'ex':
            lvr = compute_leverage(self.ex_loadings)
            lvr.columns = ['leverage-ex']
        elif mode == 'em':
            lvr = compute_leverage(self.em_loadings)
            lvr.columns = ['leverage-em']
        elif mode == 'sample':
            lvr = compute_leverage(self.score)
            lvr.columns = ['leverage-sample']
        else:
            raise ValueError("'mode' should be 'ex' or 'em' or 'sample'.")
        # lvr.index = lvr.index.set_levels(['leverage of {m}'.format(m=mode)] * len(lvr.index.levels[0]), level=0)
        return lvr

    def sample_rmse(self):
        """
        Calculate the root mean squared error (RMSE) of EEM of each sample.

        Returns
        -------
        rmse: pandas.DataFrame
            Table of RMSE
        """
        res = self.residual()
        # res = process_eem_stack(res, eem_rayleigh_scattering_removal, ex_range=self.ex_range, em_range=self.em_range)
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        rmse = pd.DataFrame(np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels),
                            index=self.nnls_fmax.index, columns=['RMSE'])
        return rmse

    def sample_relative_rmse(self):
        """
        Calculate the normalized root mean squared error (normalized RMSE) of EEM of each sample. It is defined as the
        RMSE divided by the mean of original signal.

        Returns
        -------
        relative_rmse: pandas.DataFrame
            Table of normalized RMSE
        """
        res = self.residual()
        # res = process_eem_stack(res, eem_rayleigh_scattering_removal, ex_range=self.ex_range, em_range=self.em_range)
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        relative_rmse = pd.DataFrame(
            np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels) / np.average(self.eem_stack_train, axis=(1, 2)),
            index=self.score.index, columns=['Relative RMSE'])
        return relative_rmse

    def sample_summary(self):
        """
        Get a table showing the score, Fmax, leverage, RMSE and normalized RMSE for each sample.

        Returns
        -------
        summary: pandas.DataFrame
            Table of samples' score, Fmax, leverage, RMSE and normalized RMSE.
        """
        lvr = self.leverage()
        rmse = self.sample_rmse()
        normalized_rmse = self.sample_normalized_rmse()
        summary = pd.concat([self.score, self.nnls_fmax, lvr, rmse, normalized_rmse], axis=1)
        return summary

    def export(self, filepath, info_dict):
        """
        Export the PARAFAC model to a text file that can be uploaded to the online PARAFAC model database Openfluor
        (https://openfluor.lablicate.com/#).

        Parameters
        ----------
        filepath: str
            Location of the saved text file. Please specify the ".csv" extension.
        info_dict: dict
            A dictionary containing the model information. Possible keys include: name, creator
            date, email, doi, reference, unit, toolbox, fluorometer, nSample, decomposition_method, validation,
            dataset_calibration, preprocess, sources, description

        Returns
        -------
        info_dict: dict
            A dictionary containing the information of the PARAFAC model.

        """

        ex_column = ["Ex"] * self.ex_range.shape[0]
        em_column = ["Em"] * self.em_range.shape[0]
        score_column = ["Score"] * self.score.shape[0]
        exl, eml, score = (self.ex_loadings.copy(), self.em_loadings.copy(), self.score.copy())
        exl.index = pd.MultiIndex.from_tuples(list(zip(*[ex_column, self.ex_range.tolist()])),
                                              names=('type', 'wavelength'))
        eml.index = pd.MultiIndex.from_tuples(list(zip(*[em_column, self.em_range.tolist()])),
                                              names=('type', 'wavelength'))
        score.index = pd.MultiIndex.from_tuples(list(zip(*[score_column, self.score.index])),
                                                names=('type', 'time'))

        with open(filepath, 'w') as f:
            f.write('# \n# Fluorescence Model \n# \n')
            for key, value in info_dict.items():
                f.write(key + '\t' + value)
                f.write('\n')
            f.write('# \n# Excitation/Emission (Ex, Em), wavelength [nm], component_n [loading] \n# \n')
            f.close()
        with pd.option_context('display.multi_sparse', False):
            exl.to_csv(filepath, mode='a', sep="\t", header=None)
            eml.to_csv(filepath, mode='a', sep="\t", header=None)
        with open(filepath, 'a') as f:
            f.write('# \n# timestamp, component_n [Score] \n# \n')
            f.close()
        score.to_csv(filepath, mode='a', sep="\t", header=None)
        with open(filepath, 'a') as f:
            f.write('# end #')
        return info_dict


def loadings_similarity(loadings1: pd.DataFrame, loadings2: pd.DataFrame, wavelength_alignment=False, dtw=False):
    """
    Calculate the Tucker's congruence between each pair of components of two loadings (of excitation or emission).

    Parameters
    ----------
    loadings1: pandas.DataFrame
        The first loadings. Each column of the table corresponds to one component.
    loadings2: pandas.DataFrame
        The second loadings. Each column of the table corresponds to one component.
    wavelength_alignment: bool
        Align the ex/em ranges of the components. This is useful if the PARAFAC models have different ex/em wavelengths.
        Note that ex/em will be aligned according to the ex/em ranges with the lower intervals between the two PARAFAC
        models.
    dtw: bool
        Apply dynamic time warping (DTW) to align the component loadings before calculating the similarity. This is
        useful for matching loadings with similar but shifted shapes.

    Returns
    -------
    m_sim: pandas.DataFrame
        The table of loadings similarities between each pair of components.
    """
    wl_range1, wl_range2 = (loadings1.index, loadings2.index)
    if wavelength_alignment:
        wl_interval1 = (wl_range1.max() - wl_range1.min()) / (wl_range1.shape[0] - 1)
        wl_interval2 = (wl_range2.max() - wl_range2.min()) / (wl_range2.shape[0] - 1)
        if wl_interval2 > wl_interval1:
            f2 = interp1d(wl_range2, loadings2.to_numpy(), axis=0)
            loadings2 = f2(wl_range1)
        elif wl_interval1 > wl_interval2:
            f1 = interp1d(wl_range1, loadings1.to_numpy(), axis=0)
            loadings1 = f1(wl_range2)
    else:
        loadings1, loadings2 = (loadings1.to_numpy(), loadings2.to_numpy())
    m_sim = np.zeros([loadings1.shape[1], loadings2.shape[1]])
    for n2 in range(loadings2.shape[1]):
        for n1 in range(loadings1.shape[1]):
            if dtw:
                l1, l2 = dynamic_time_warping(loadings1[:, n1], loadings2[:, n2])
            else:
                l1, l2 = [loadings1[:, n1], loadings2[:, n2]]
            m_sim[n1, n2] = stats.pearsonr(l1, l2)[0]
    m_sim = pd.DataFrame(m_sim, index=['model1 C{i}'.format(i=i + 1) for i in range(loadings1.shape[1])],
                         columns=['model2 C{i}'.format(i=i + 1) for i in range(loadings2.shape[1])])
    return m_sim


def component_similarity(components1: np.ndarray, components2: np.ndarray):
    m_sim = np.zeros([components1.shape[0], components2.shape[0]])
    for n2 in range(components2.shape[0]):
        for n1 in range(components1.shape[0]):
            c1_unfolded, c2_unfolded = [components1[n1].reshape(-1), components2[n2].reshape(-1)]
            m_sim[n1, n2] = stats.pearsonr(c1_unfolded, c2_unfolded)[0]
    m_sim = pd.DataFrame(m_sim, index=['model1 C{i}'.format(i=i + 1) for i in range(components1.shape[0])],
                         columns=['model2 C{i}'.format(i=i + 1) for i in range(components2.shape[0])])
    return m_sim


def align_components_by_loadings(models_dict: dict, ex_ref: pd.DataFrame, em_ref: pd.DataFrame,
                                 wavelength_alignment=False):
    """
    Align the components of PARAFAC models according to given reference ex/em loadings so that similar components
    are labelled by the same name.

    Parameters
    ----------
    models_dict: dict
        Dictionary of PARAFAC objects, the models to be aligned.
    ex_ref: pandas.DataFrame
        Ex loadings of the reference
    em_ref: pandas.DataFrame
        Em loadings of the reference
    wavelength_alignment: bool
        Align the ex/em ranges of the components. This is useful if the PARAFAC models have different ex/em wavelengths.
        Note that ex/em will be aligned according to the ex/em ranges with the lower intervals between the two PARAFAC
        models.

    Returns
    -------
    models_dict_new: dict
        Dictionary of the aligned PARAFAC object.
    """
    component_labels_ref = ex_ref.columns
    models_dict_new = {}
    for model_label, model in models_dict.items():
        m_sim_ex = loadings_similarity(ex_ref, model.ex_loadings, wavelength_alignment=wavelength_alignment)
        m_sim_em = loadings_similarity(em_ref, model.em_loadings, wavelength_alignment=wavelength_alignment)
        m_sim = (m_sim_ex + m_sim_em) / 2
        padded_matrix = np.zeros((max(m_sim.shape), max(m_sim.shape)))
        padded_matrix[:m_sim.shape[0], :m_sim.shape[1]] = m_sim
        row_ind, col_ind = linear_sum_assignment(-padded_matrix)
        pairs = [(i, j) for i, j in zip(row_ind, col_ind) if i < m_sim.shape[0]]
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        row_ind, col_ind = zip(*sorted_pairs)
        col_ind = list(col_ind)
        non_ordered_index = list(set([i for i in range(m_sim.shape[1])]) - set(col_ind))
        permutation = col_ind + non_ordered_index
        if non_ordered_index:
            component_labels_ref_extended = component_labels_ref + ['O{i}'.format(i=i + 1) for i in
                                                                    range(len(non_ordered_index))]
        else:
            component_labels_ref_extended = component_labels_ref
        component_labels_var = [0] * len(permutation)
        for i, nc in enumerate(permutation):
            component_labels_var[nc] = component_labels_ref_extended[i]
        model.score.columns, model.ex_loadings.columns, model.em_loadings.columns, model.nnls_fmax.columns = (
                [component_labels_var] * 4)
        model.score = model.score.iloc[:, permutation]
        model.ex_loadings = model.ex_loadings.iloc[:, permutation]
        model.em_loadings = model.em_loadings.iloc[:, permutation]
        model.nnls_fmax = model.nnls_fmax.iloc[:, permutation]
        model.components = model.components[permutation, :, :]
        model.cptensor = permute_cp_tensor(model.cptensors, permutation)
        models_dict_new[model_label] = model
    return models_dict_new


def align_components_by_components(models_dict: dict, components_ref: dict, model_type='parafac'):
    """
    Align the components of PARAFAC or NMF models according to given reference ex/em loadings so that similar components
    are labelled by the same name.

    Parameters
    ----------
    models_dict: dict
        Dictionary of PARAFAC objects, the models to be aligned.
    components_ref: dict
        Dictionary where each item is a reference component. The keys are the component labels, the values are the
        components (np.ndarray).
    model_type: str, {'parafac', 'nmf'}
        The type of model.

    Returns
    -------
    models_dict_new: dict
        Dictionary of the aligned PARAFAC object.
    """
    component_labels_ref = list(components_ref.keys())
    components_stack_ref = np.array([c.flatten() for c in components_ref.values()])
    models_dict_new = {}

    for model_label, model in models_dict.items():
        comp = model.components.reshape(model.components.shape[0], -1)
        cost_mat = cdist(comp, components_stack_ref, metric='correlation')

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        permutation = list(row_ind)
        matched_index = list(col_ind)

        # Generate new labels
        component_labels_var = [component_labels_ref[j] for j in matched_index]
        if len(permutation) < comp.shape[0]:
            unmatched = list(set(range(comp.shape[0])) - set(permutation))
            component_labels_var += [f"O{i+1}" for i in range(len(unmatched))]
            permutation += unmatched
        if model_type == 'parafac':
            model.score.columns, model.ex_loadings.columns, model.em_loadings.columns, model.nnls_fmax.columns = (
                    [component_labels_var] * 4)
            model.score = model.score.iloc[:, permutation]
            model.ex_loadings = model.ex_loadings.iloc[:, permutation]
            model.em_loadings = model.em_loadings.iloc[:, permutation]
            model.nnls_fmax = model.nnls_fmax.iloc[:, permutation]
            model.components = model.components[permutation, :, :]
            model.cptensor = permute_cp_tensor(model.cptensors, permutation)
        elif model_type == 'nmf':
            model.fmax.columns, model.nnls_fmax.columns = (
                    [component_labels_var] * 2)
            model.fmax = model.fmax.iloc[:, permutation]
            model.nnls_fmax = model.nnls_fmax.iloc[:, permutation]
            model.components = model.components[permutation, :, :]
        models_dict_new[model_label] = model
    return models_dict_new


class SplitValidation:
    """
    Conduct PARAFAC model validation by evaluating the consistency of PARAFAC models established on EEM sub-datasets.

    Parameters
    ----------
    rank: int
        Number of components in PARAFAC.
    n_split: int
        Number of splits.
    combination_size: int or str, {int, 'half'}
        The number of splits assembled into one combination. If 'half' is passed, each combination will include
        half of the splits (i.e., the split-half validation).
    rule: str, {'random', 'sequential'}
        Whether to split the EEM dataset randomly. If 'sequential' is passed, the dataset will be split according
        to index order.
    non_negativity: bool
        Whether to apply non-negativity constraint in PARAFAC.
    tf_normalization: bool
        Whether to normalize the EEM by total fluorescence in PARAFAC.

    Attributes
    -----------
    eem_subsets: dict
        Dictionary of EEM sub-datasets.
    subset_specific_models: dict
        Dictionary of PARAFAC models established on sub-datasets.
    """

    def __init__(self, rank, n_split=4, combination_size='half', rule='random', similarity_metric='TCC',
                 non_negativity=True, tf_normalization=True):
        # ---------------Parameters-------------------
        self.rank = rank
        self.n_split = n_split
        self.combination_size = combination_size
        self.rule = rule
        self.similarity_metric = similarity_metric
        self.non_negativity = non_negativity
        self.tf_normalization = tf_normalization

        # ----------------Attributes------------------
        self.eem_subsets = None
        self.subset_specific_models = None

    def fit(self, eem_dataset: EEMDataset):
        split_set = eem_dataset.splitting(n_split=self.n_split, rule=self.rule)
        if self.combination_size == 'half':
            cs = int(self.n_split) / 2
        else:
            cs = int(self.combination_size)
        elements = list(itertools.combinations([i for i in range(self.n_split)], int(cs)))
        codes = list(itertools.combinations(list(string.ascii_uppercase)[0:self.n_split], int(cs)))
        model_complete = PARAFAC(n_components=self.rank, non_negativity=self.non_negativity,
                                 tf_normalization=self.tf_normalization)
        model_complete.fit(eem_dataset=eem_dataset)
        sims_ex, sims_em, models, subsets = ({}, {}, {}, {})

        for e, c in zip(elements, codes):
            label = ''.join(c)
            subdataset = combine_eem_datasets([split_set[i] for i in e])
            model_subdataset = PARAFAC(n_components=self.rank, non_negativity=self.non_negativity,
                                       tf_normalization=self.tf_normalization)
            model_subdataset.fit(subdataset)

            models[label] = model_subdataset
            subsets[label] = subdataset
        models = align_components_by_loadings(models, model_complete.ex_loadings, model_complete.em_loadings)
        self.eem_subsets = subsets
        self.subset_specific_models = models
        return self

    def compare(self):
        """
        Calculate the similarities of ex/em loadings between PARAFAC models established on sub-datasets.

        Returns
        -------
        similarities_ex: pandas.DataFrame
            Similarities in excitation loadings.
        similarities_em: pandas.DataFrame
            Similarities in emission loadings.
        """
        labels = sorted(self.subset_specific_models.keys())
        similarities_ex = {}
        similarities_em = {}
        for k in range(int(len(labels) / 2)):
            m1 = self.subset_specific_models[labels[k]]
            m2 = self.subset_specific_models[labels[-1 - k]]
            sims_ex = loadings_similarity(m1.ex_loadings, m2.ex_loadings).to_numpy().diagonal()
            sims_em = loadings_similarity(m1.em_loadings, m2.em_loadings).to_numpy().diagonal()
            pair_labels = '{m1} vs. {m2}'.format(m1=labels[k], m2=labels[-1 - k])
            similarities_ex[pair_labels] = sims_ex
            similarities_em[pair_labels] = sims_em
        similarities_ex = pd.DataFrame.from_dict(
            similarities_ex, orient='index',
            columns=['Similarities in C{i}-ex'.format(i=i + 1) for i in range(self.rank)]
        )
        similarities_ex.index.name_train = 'Test'
        similarities_em = pd.DataFrame.from_dict(
            similarities_em, orient='index',
            columns=['Similarities in C{i}-em'.format(i=i + 1) for i in range(self.rank)]
        )
        similarities_em.index.name_train = 'Test'
        return similarities_ex, similarities_em


class EEMNMF:
    """
    Non-negative matrix factorization (NMF) model for EEM dataset.

    Parameters
    ----------
    n_components: int
        The number of components
    solver: str, {'cd', 'mu', 'hals'}
        The numerical solver of NMF. 'cd' is a Coordinate Descent solver. 'mu' is a Multiplicative Update solver. 'hals'
        is a Hierarchical Alternating Least Squares solver.
    beta_loss: str, {‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’}
        Beta divergence to be minimized, measuring the distance between X and the dot product WH. Used only in 'mu'
        solver.
    alpha_component: float, default=0.0
        Strength of the elastic-net regularization on sample loadings.
    alpha_sample: float, default=0.0
        Strength of the elastic-net regularization on component loadings.
    l1_ratio: float, default=0.0
        The mixing ratio between L1 and L2 regularization:
            - 0.0 corresponds to pure L2 penalty,
            - 1.0 to pure L1,
            - values in between correspond to elastic-net.
    prior_dict_sample: dict, optional
        A dictionary mapping component indices (int) to prior vectors (1D array of length n_features) for regularizing
        sample loadings. The k-th loading of sample loadings will be penalized to be close to prior_dict_sample[k] if
        present. Only applied to 'hals' solver.
    prior_dict_component: dict, optional
        A dictionary mapping component indices (int) to prior vectors (1D array of length n_samples) for regularizing
        component loadings. The k-th loading of components loadings will be penalized to be close to
        prior_dict_component[k] if present. Only applied to 'hals' solver.
    gamma_sample: float, default=0
        Strength of the prior regularization on sample loadings. Only applied to 'hals' solver.
    gamma_component: float ,default=0
        Strength of the prior regularization on component loadings. Only applied to 'hals' solver.
    idx_top: list, optional
        List of indices of samples serving as numerators in ratio calculation.
    idx_bot: list, optional
        List of indices of samples serving as denominators in ratio calculation.
    lam: float, default=0
        Strength of the ratio regularization on sample loadings. Only applied to 'hals' solver.
    normalization: str, {'pixel_std'}:
        The normalization of EEMs before conducting NMF. 'pixel_std' normalizes the intensities of each pixel across
        all samples by standard deviation.
    sort_em: bool,
        Whether to sort components by emission peak position from lowest to highest. If False is passed, the
        components will be sorted by the contribution to the total variance.

    Attributes
    ----------
    fmax: pandas.DataFrame
        Fmax table calculated using the score matrix of NMF.
    nnls_fmax: pandas.DataFrame
        Fmax table calculated by fitting EEMs with NMF components using non-negative least square (NNLS).
    components: np.ndarray
        NMF components.
    eem_stack_train: np.ndarray
        EEMs used for PARAFAC model establishment.
    eem_stack_reconstructed: np.ndarray
        EEMs reconstructed by the established PARAFAC model.
    """

    def __init__(self, n_components, solver='cd', init='nndsvda', beta_loss='frobenius',
                 alpha_sample=0, alpha_component=0, l1_ratio=1,
                 prior_dict_sample=None, prior_dict_component=None,
                 gamma_sample=0, gamma_component=0, prior_ref_components=None,
                 idx_top=None, idx_bot=None, kw_top=None, kw_bot=None, lam=0,
                 normalization='pixel_std', sort_em=True, max_iter_als=100, max_iter_nnls=500, random_state=None):

        # -----------Parameters-------------
        self.n_components = n_components
        self.solver = solver
        self.init = init
        self.beta_loss = beta_loss
        self.alpha_sample = alpha_sample
        self.alpha_component = alpha_component
        self.l1_ratio = l1_ratio
        self.prior_dict_sample = prior_dict_sample
        self.prior_dict_component = prior_dict_component
        self.prior_ref_components = prior_ref_components
        self.gamma_sample = gamma_sample
        self.gamma_component = gamma_component
        self.idx_top = idx_top
        self.idx_bot = idx_bot
        self.kw_top = kw_top
        self.kw_bot = kw_bot
        self.lam = lam
        self.normalization = normalization
        self.sort_em = sort_em
        self.max_iter_als = max_iter_als
        self.max_iter_nnls = max_iter_nnls
        self.random_state = random_state

        # -----------Attributes-------------
        self.eem_stack_unfolded = None
        self.fmax = None
        self.nnls_fmax = None
        self.components = None
        self.decomposer = None
        self.normalization_factor_std = None
        self.normalization_factor_max = None
        self.reconstruction_error = None
        self.eem_stack_train = None
        self.eem_stack_reconstructed = None
        self.ex_range = None,
        self.em_range = None
        self.beta = None
        self.objective_function_error = None,

    def fit(self, eem_dataset):
        if self.kw_top is not None and self.kw_bot is not None:
            assert eem_dataset.index is not None
            self.idx_top = [i for i in range(len(eem_dataset.index)) if self.kw_top in eem_dataset.index[i]]
            self.idx_bot = [i for i in range(len(eem_dataset.index)) if self.kw_bot in eem_dataset.index[i]]
        if self.solver == 'cd' or self.solver == 'mu':
            if self.prior_dict_sample is not None or self.prior_dict_component is not None:
                raise ValueError("'cd' and 'mu' solver do not support prior knowledge input. Please use 'hals' solver "
                                 "instead")
            decomposer = NMF(
                n_components=self.n_components,
                solver=self.solver,
                init=self.init,
                beta_loss=self.beta_loss,
                alpha_W=self.alpha_sample,
                alpha_H=self.alpha_component,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state
            )
            eem_dataset.threshold_masking(0, 0, 'smaller', copy=False)
            n_samples = eem_dataset.eem_stack.shape[0]
            X = eem_dataset.eem_stack.reshape([n_samples, -1])
            X[np.isnan(X)] = 0
            if self.normalization == 'pixel_std':
                factor_std = np.std(X, axis=0)
                X = X / factor_std
                X[np.isnan(X)] = 0
                nmf_score = decomposer.fit_transform(X)
                components = decomposer.components_ * factor_std
            else:
                factor_std = None
                nmf_score = decomposer.fit_transform(X)
                components = decomposer.components_
            nmf_score = pd.DataFrame(nmf_score, index=eem_dataset.index,
                                     columns=["component {i} NMF-Fmax".format(i=i + 1) for i in
                                              range(self.n_components)])
            eem_stack_reconstructed = nmf_score @ components
        elif self.solver == 'hals':
            eem_dataset.threshold_masking(0, 0, 'smaller', copy=False)
            n_samples = eem_dataset.eem_stack.shape[0]
            X = eem_dataset.eem_stack.reshape([n_samples, -1])
            X[np.isnan(X)] = 0
            if self.normalization == 'pixel_std':
                factor_std = np.std(X, axis=0)
                X = X / factor_std
                X[np.isnan(X)] = 0
                if self.idx_top and self.idx_bot:
                    W, H, beta = nmf_hals_prior_ratio(
                        X,
                        rank=self.n_components,
                        init=self.init,
                        prior_dict_W=self.prior_dict_sample,
                        prior_dict_H=self.prior_dict_component,
                        alpha_W=self.alpha_sample,
                        alpha_H=self.alpha_component,
                        l1_ratio=self.l1_ratio,
                        gamma_W=self.gamma_sample,
                        gamma_H=self.gamma_component,
                        idx_top=self.idx_top,
                        idx_bot=self.idx_bot,
                        lam=self.lam,
                        max_iter_als=self.max_iter_als,
                        max_iter_nnls=self.max_iter_nnls,
                        prior_ref_components=self.prior_ref_components,
                        random_state=self.random_state
                    )
                    self.beta = beta
                else:
                    W, H = nmf_hals_prior(
                        X,
                        rank=self.n_components,
                        init=self.init,
                        prior_dict_W=self.prior_dict_sample,
                        prior_dict_H=self.prior_dict_component,
                        alpha_W=self.alpha_sample,
                        alpha_H=self.alpha_component,
                        l1_ratio=self.l1_ratio,
                        gamma_W=self.gamma_sample,
                        gamma_H=self.gamma_component,
                        max_iter_als=self.max_iter_als,
                        max_iter_nnls=self.max_iter_nnls,
                        prior_ref_components=self.prior_ref_components,
                        random_state=self.random_state
                    )
                nmf_score = W
                components = H * factor_std
            else:
                factor_std = None
                if self.idx_top and self.idx_bot:
                    W, H, beta = nmf_hals_prior_ratio(
                        X,
                        rank=self.n_components,
                        init=self.init,
                        prior_dict_W=self.prior_dict_sample,
                        prior_dict_H=self.prior_dict_component,
                        gamma_W=self.gamma_sample,
                        gamma_H=self.gamma_component,
                        alpha_W=self.alpha_sample,
                        alpha_H=self.alpha_component,
                        l1_ratio=self.l1_ratio,
                        idx_top=self.idx_top,
                        idx_bot=self.idx_bot,
                        lam=self.lam,
                        max_iter_als=self.max_iter_als,
                        max_iter_nnls=self.max_iter_nnls,
                        prior_ref_components=self.prior_ref_components,
                        random_state=self.random_state
                    )
                    self.beta = beta
                else:
                    W, H = nmf_hals_prior(
                        X,
                        rank=self.n_components,
                        init=self.init,
                        prior_dict_W=self.prior_dict_sample,
                        prior_dict_H=self.prior_dict_component,
                        gamma_W=self.gamma_sample,
                        gamma_H=self.gamma_component,
                        alpha_W=self.alpha_sample,
                        alpha_H=self.alpha_component,
                        l1_ratio=self.l1_ratio,
                        max_iter_als=self.max_iter_als,
                        max_iter_nnls=self.max_iter_nnls,
                        prior_ref_components=self.prior_ref_components,
                        random_state=self.random_state
                    )
                nmf_score = W
                components = H
            eem_stack_reconstructed = W @ H
            nmf_score = pd.DataFrame(nmf_score, index=eem_dataset.index,
                                     columns=["component {i} NMF-Fmax".format(i=i + 1) for i in
                                              range(self.n_components)])
        else:
            raise ValueError("Unknown solver name: choose 'mu', 'cd' or 'hals'.")
        factor_max = np.max(components, axis=1)
        components = components / factor_max[:, None]
        components = components.reshape([self.n_components, eem_dataset.eem_stack.shape[1],
                                         eem_dataset.eem_stack.shape[2]])
        nmf_score = nmf_score.mul(factor_max, axis=1)
        _, nnls_score, _ = eems_fit_components(eem_dataset.eem_stack, components,
                                                                     fit_intercept=False, positive=True)
        nnls_score = pd.DataFrame(nnls_score, index=eem_dataset.index,
                                  columns=["component {i} NNLS-Fmax".format(i=i + 1) for i in range(self.n_components)])
        if self.sort_em:
            em_peaks = []
            for i in range(self.n_components):
                flat_max_index = components[i].argmax()
                row_index, col_index = np.unravel_index(flat_max_index, components[i].shape)
                em_peaks.append(col_index)
            peak_rank = list(enumerate(stats.rankdata(em_peaks)))
            order = [i[0] for i in sorted(peak_rank, key=lambda x: x[1])]
            components = components[order]
            nmf_score = pd.DataFrame({'component {r} NMF-Fmax'.format(r=i + 1): nmf_score.iloc[:, order[i]]
                                      for i in range(self.n_components)})
            nnls_score = pd.DataFrame({'component {r} NNLS-Fmax'.format(r=i + 1): nnls_score.iloc[:, order[i]]
                                       for i in range(self.n_components)})
        self.fmax = nmf_score
        self.nnls_fmax = nnls_score
        self.components = components
        self.eem_stack_unfolded = X
        self.normalization_factor_std = factor_std
        self.normalization_factor_max = factor_max
        self.eem_stack_train = eem_dataset.eem_stack
        self.eem_stack_reconstructed = eem_stack_reconstructed.reshape(eem_dataset.eem_stack.shape)
        self.ex_range = eem_dataset.ex_range
        self.em_range = eem_dataset.em_range
        return self

    def component_peak_locations(self):
        """
        Get the ex/em of component peaks

        Returns
        -------
        max_exem: list
            A List of (ex, em) of component peaks.
        """
        max_exem = []
        for r in range(self.n_components):
            max_index = np.unravel_index(np.argmax(self.components[r, :, :]), self.components[r, :, :].shape)
            max_exem.append((self.ex_range[-(max_index[0] + 1)], self.em_range[max_index[1]]))
        return max_exem

    def residual(self):
        """
        Get the residual of the established PARAFAC model, i.e., the difference between the original EEM dataset and
        the reconstructed EEM dataset.

        Returns
        -------
        res: np.ndarray (3d)
            the residual
        """
        res = self.eem_stack_train - self.eem_stack_reconstructed
        return res

    def explained_variance(self):
        """
        Calculate the explained variance of the established NMF model

        Returns
        -------
        ev: float
            the explained variance
        """
        y_train = self.eem_stack_train.reshape(-1)
        y_pred = self.eem_stack_reconstructed.reshape(-1)
        ev = 100 * (1 - np.var(y_pred - y_train) / np.var(y_train))
        return ev

    def sample_rmse(self):
        """
        Calculate the root mean squared error (RMSE) of EEM of each sample.

        Returns
        -------
        sse: pandas.DataFrame
            Table of RMSE
        """
        res = self.residual()
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        rmse = pd.DataFrame(np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels), index=self.nnls_fmax.index,
                            columns=['RMSE'])
        return rmse

    def sample_normalized_rmse(self):
        """
        Calculate the normalized root mean squared error (normalized RMSE) of EEM of each sample. It is defined as the
        RMSE divided by the mean of original signal.

        Returns
        -------
        normalized_sse: pandas.DataFrame
            Table of normalized RMSE
        """
        res = self.residual()
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        normalized_sse = pd.DataFrame(sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels) /
                                      np.average(self.eem_stack_train, axis=(1, 2)),
                                      index=self.nnls_fmax.index, columns=['sample_normalized_rmse'])
        return normalized_sse

    def predict(self, eem_dataset: EEMDataset, fit_intercept=False, fit_beta=False, idx_top=None, idx_bot=None):
        """
        Predict the score and Fmax of a given EEM dataset using the component fitted. This method can be applied to a
        new EEM dataset independent of the one used in NMF model establishment.

        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset to be predicted.
        fit_intercept: bool
            Whether to calculate the intercept.
        fit_beta: bool
            Whether to fit the beta parameter (the proportions between "top" and "bot" samples).
        idx_top: list, optional
            List of indices of samples serving as numerators in ratio calculation.
        idx_bot: list, optional
            List of indices of samples serving as denominators in ratio calculation.

        Returns
        -------
        score_sample: pd.DataFrame
            The fitted score.
        fmax_sample: pd.DataFrame
            The fitted Fmax.
        eem_stack_pred: np.ndarray (3d)
            The EEM dataset reconstructed.
        """
        if not fit_beta:
            score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset.eem_stack, self.components,
                                                                            fit_intercept=fit_intercept)
        else:
            assert self.beta is not None, "Parameter beta must be provided through fitting."
            assert idx_top is not None and idx_bot is not None, "idx_top and idx_bot must be provided."
            max_values = np.amax(self.components, axis=(1, 2))
            score_sample = np.zeros([eem_dataset.eem_stack.shape[0], self.n_components])
            score_sample_bot = solve_W(
                X1=eem_stack_to_2d(eem_dataset.eem_stack)[idx_bot],
                X2=eem_stack_to_2d(eem_dataset.eem_stack)[idx_top],
                H=self.components.reshape([self.n_components, -1]),
                beta=self.beta
            )
            score_sample[idx_bot] = score_sample_bot
            score_sample[idx_top] = score_sample_bot * self.beta
            fmax_sample = score_sample * max_values
            eem_stack_pred = score_sample @ self.components.reshape([self.n_components, -1])
            eem_stack_pred = eem_stack_pred.reshape(eem_dataset.eem_stack.shape)
        score_sample = pd.DataFrame(score_sample, index=eem_dataset.index, columns=self.fmax.columns)
        fmax_sample = pd.DataFrame(fmax_sample, index=eem_dataset.index, columns=self.fmax.columns)
        return score_sample, fmax_sample, eem_stack_pred


class KMethod:
    """
    Conduct K-method, an EEM clustering algorithm aiming to minimize the general PARAFAC reconstruction error. The
    key hypothesis behind K-PARAFACs is that fitting EEMs with varied underlying chemical compositions using the same
    PARAFAC model could lead to a large reconstruction error. In contrast, samples with similar chemical composition
    can be incorporated into the same PARAFAC model with small reconstruction error (i.e., the root mean-squared
    error (RMSE) between the original EEM and its reconstruction derived from loadings and scores). if samples can be
    appropriately clustered by their chemical compositions, then the PARAFAC models established on individual
    clusters, i.e., the cluster-specific PARAFAC models, should exhibit reduced reconstruction error compared to the
    unified PARAFAC model. Based on this hypothesis, K-PARAFACs is proposed to search for an optimal clustering
    strategy so that the overall reconstruction error of cluster-specific PARAFAC models is minimized.

    Parameters
    -----------
    n_initial_splits: int
        Number of splits in clustering initialization (the first time that the EEM dataset is divided).
    max_iter: int
        Maximum number of iterations in a single run of base clustering.
    tol: float
        Tolerance in regard to the average Tucker's congruence between the cluster-specific PARAFAC models
        of two consecutive iterations to declare convergence. If the Tucker's congruence > 1-tol, then convergence is
        confirmed.
    elimination: 'default' or int
        The minimum number of EEMs in each cluster. During optimization, clusters with EEMs less than the specified
        number would be eliminated. If 'default' is passed, then the number is set to be the same as the number of
        components.

    Attributes
    ------------
    unified_model: PARAFAC
        Unified PARAFAC model.
    label_history: list
        A list of cluster labels after each run of clustering.
    error_history: list
        A list of average RMSE over all pixels after each run of clustering.
    labels: np.ndarray
        Final cluster labels.
    eem_clusters: dict
        EEM clusters.
    cluster_specific_models: dict
        Cluster-specific PARAFAC models.
    consensus_matrix: np.ndarray
        Consensus matrix.
    consensus_matrix_sorted: np.ndarray
        Sorted consensus matrix.
    """

    def __init__(self, base_model, n_initial_splits, distance_metric="reconstruction_error", max_iter=20, tol=0.001,
                 elimination='default', kw_top=None, kw_bot=None):

        # -----------Parameters-------------
        self.base_model = base_model
        self.n_initial_splits = n_initial_splits
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.tol = tol
        self.elimination = elimination
        self.kw_top = kw_top
        self.kw_bot = kw_bot
        self.subsampling_portion = None
        self.n_runs = None
        self.consensus_conversion_power = None

        # ----------Attributes-------------
        self.unified_model = None
        self.label_history = None
        self.error_history = None
        self.silhouette_score = None
        self.labels = None
        self.index_sorted = None
        self.ref_sorted = None
        self.threshold_r = None
        self.eem_clusters = None
        self.cluster_specific_models = None
        self.consensus_matrix = None
        self.distance_matrix = None
        self.linkage_matrix = None
        self.consensus_matrix_sorted = None

    def base_clustering(self, eem_dataset: EEMDataset):
        """
        Run clustering for a single time.

        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset to be clustered.

        Returns
        -------
        cluster_labels: np.ndarray
            Cluster labels.
        label_history: list
            Cluster labels in each iteration.
        error_history: list
            Average reconstruction error (RMSE) in each iteration.
        """

        # -------Generate a unified model as reference for ordering components--------

        unified_model = copy.deepcopy(self.base_model)
        unified_model.fit(eem_dataset)

        # -------Define functions for estimation and maximization steps-------

        def get_quenching_coef(fmax_tot, kw_o, kw_q):
            fmax_original = fmax_tot[fmax_tot.index.str.contains(kw_o)]
            fmax_quenched = fmax_tot[fmax_tot.index.str.contains(kw_q)]
            fmax_ratio = fmax_tot.copy()
            fmax_ratio[fmax_ratio.index.str.contains(kw_o)] = fmax_original.to_numpy() / fmax_quenched.to_numpy()
            fmax_ratio[fmax_ratio.index.str.contains(kw_q)] = fmax_original.to_numpy() / fmax_quenched.to_numpy()
            return fmax_ratio.to_numpy()

        def estimation(sub_datasets: dict):
            models = {}
            for label, d in sub_datasets.items():
                model = copy.deepcopy(self.base_model)
                model.fit(d)
                models[label] = model
            return models

        def maximization(models: dict):
            sample_error = []
            sub_datasets = {}
            for label, m in models.items():
                if self.distance_metric == "reconstruction_error_with_beta":
                    idx_top = [i for i in range(len(eem_dataset.index)) if self.kw_top in eem_dataset.index[i]]
                    idx_bot = [i for i in range(len(eem_dataset.index)) if self.kw_bot in eem_dataset.index[i]]
                    score_m, fmax_m, eem_stack_re_m = m.predict(
                        eem_dataset=eem_dataset,
                        fit_beta=True,
                        idx_bot=idx_bot,
                        idx_top=idx_top,
                    )
                    res = eem_dataset.eem_stack - eem_stack_re_m
                    res = np.sum(res**2, axis=(1, 2))
                    error_with_beta = np.zeros(fmax_m.shape[0])
                    error_with_beta[idx_top] = res[idx_top] + res[idx_bot]
                    error_with_beta[idx_bot] = res[idx_top] + res[idx_bot]
                    sample_error.append(error_with_beta)
                elif self.distance_metric == "reconstruction_error":
                    score_m, fmax_m, eem_stack_re_m = m.predict(eem_dataset)
                    res = eem_dataset.eem_stack - eem_stack_re_m
                    n_pixels = m.eem_stack_train.shape[1] * m.eem_stack_train.shape[2]
                    rmse = np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels)
                    sample_error.append(rmse)
                elif self.distance_metric == "quenching_coefficient":
                    if not all([self.kw_top, self.kw_bot]):
                        raise ValueError("Both kw_unquenched and kw_quenched must be passed.")
                    if type(m).__name__ == 'PARAFAC':
                        fmax_establishment = m.nnls_fmax
                    elif type(m).__name__ == 'EEMNMF':
                        fmax_establishment = m.nnls_fmax
                    else:
                        raise TypeError("Invalid base model type.")
                    quenching_coef_establishment = get_quenching_coef(fmax_establishment, self.kw_top,
                                                                      self.kw_bot)
                    quenching_coef_archetype = np.mean(quenching_coef_establishment, axis=0)

                    quenching_coef_test = get_quenching_coef(fmax_m, self.kw_top, self.kw_bot)

                    quenching_coef_diff = np.abs(quenching_coef_test - quenching_coef_archetype)
                    sample_error.append(np.sum(quenching_coef_diff ** 2, axis=1))
            best_model_idx = np.argmin(sample_error, axis=0)
            least_model_errors = np.min(sample_error, axis=0)
            for j, label in enumerate(models.keys()):
                eem_stack_j = eem_dataset.eem_stack[np.where(best_model_idx == j)]
                if eem_dataset.ref is not None:
                    ref_j = eem_dataset.ref.iloc[np.where(best_model_idx == j)]
                else:
                    ref_j = None
                if eem_dataset.index is not None:
                    index_j = [eem_dataset.index[k] for k, idx in enumerate(best_model_idx) if idx == j]
                else:
                    index_j = None
                sub_datasets[label] = EEMDataset(eem_stack=eem_stack_j, ex_range=eem_dataset.ex_range,
                                                 em_range=eem_dataset.em_range, ref=ref_j, index=index_j)
            best_model_label = np.array([list(models.keys())[idx] for idx in best_model_idx])
            return sub_datasets, best_model_label, least_model_errors

        # -------Define function for convergence detection-------

        def model_similarity(models_1: dict, models_2: dict):
            similarity = 0
            for label, m in models_1.items():
                similarity = component_similarity(m.components, models_2[label].components).to_numpy().diagonal()
            similarity = np.sum(similarity) / len(models_1)
            return similarity

        # -------Initialization--------
        label_history = []
        error_history = []
        sample_errors = []
        sub_datasets_n = {}
        if self.distance_metric == "reconstruction_error":
            initial_sub_eem_datasets = eem_dataset.splitting(n_split=self.n_initial_splits)
        elif self.distance_metric in ["quenching_coefficient", "reconstruction_error_with_beta"]:
            initial_sub_eem_datasets = []
            eem_dataset_unquenched, _ = eem_dataset.filter_by_index(self.kw_top, None, copy=True)
            initial_sub_eem_datasets_unquenched = eem_dataset_unquenched.splitting(n_split=self.n_initial_splits)
            eem_dataset_quenched, _ = eem_dataset.filter_by_index(self.kw_bot, None, copy=True)
            for subset in initial_sub_eem_datasets_unquenched:
                pos = [eem_dataset_unquenched.index.index(idx) for idx in subset.index]
                quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
                sub_eem_dataset_quenched, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
                subset.sort_by_index()
                sub_eem_dataset_quenched.sort_by_index()
                initial_sub_eem_datasets.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))

        for i, random_m in enumerate(initial_sub_eem_datasets):
            sub_datasets_n[i + 1] = random_m

        for n in range(self.max_iter):

            # -------Eliminate sub_datasets having EEMs less than the number of ranks--------
            cluster_label_to_remove = []
            for cluster_label, sub_dataset_i in sub_datasets_n.items():
                if self.elimination == 'default' and sub_dataset_i.eem_stack.shape[0] <= self.base_model.n_components:
                    cluster_label_to_remove.append(cluster_label)
                elif isinstance(self.elimination, int):
                    if self.elimination <= max(self.base_model.n_components, self.elimination):
                        cluster_label_to_remove.append(cluster_label)
            for l in cluster_label_to_remove:
                sub_datasets_n.pop(l)

            # -------The estimation step-------
            cluster_specific_models_new = estimation(sub_datasets_n)
            cluster_specific_models_new = align_components_by_components(
                cluster_specific_models_new,
                {f'component {i + 1}': unified_model.components[i] for i in range(unified_model.components.shape[0])},
                model_type='parafac' if isinstance(self.base_model, PARAFAC) else 'nmf'
            )

            # -------The maximization step--------
            sub_datasets_n, cluster_labels, fitting_errors = maximization(cluster_specific_models_new)
            label_history.append(cluster_labels)
            error_history.append(fitting_errors)

            # -------Detect convergence---------
            if 0 < n < self.max_iter - 1:
                if np.array_equal(label_history[-1], label_history[-2]):
                    break
                if len(cluster_specific_models_old) == len(cluster_specific_models_new):
                    if model_similarity(cluster_specific_models_new, cluster_specific_models_old) > 1 - self.tol:
                        break

            cluster_specific_models_old = cluster_specific_models_new

        label_history = pd.DataFrame(np.array(label_history).T, index=eem_dataset.index,
                                     columns=['iter_{i}'.format(i=i + 1) for i in range(n + 1)])
        error_history = pd.DataFrame(np.array(error_history).T, index=eem_dataset.index,
                                     columns=['iter_{i}'.format(i=i + 1) for i in range(n + 1)])
        self.label_history = [label_history]
        self.error_history = [error_history]
        self.unified_model = unified_model
        self.labels = cluster_labels
        self.eem_clusters = sub_datasets_n
        self.cluster_specific_models = cluster_specific_models_new

        return cluster_labels, label_history, error_history

    def calculate_consensus(self, eem_dataset: EEMDataset, n_base_clusterings: int, subsampling_portion: float):
        """
        Run the clustering for many times and combine the output of each run to obtain an optimal clustering.

        Parameters
        ----------
        eem_dataset: EEMDataset
            EEM dataset.
        n_base_clusterings: int
            Number of base clustering.
        subsampling_portion: float
            The portion of EEMs remained after subsampling.

        Returns
        -------
        self: object
            The established K-PARAFACs model
        """

        n_samples = eem_dataset.eem_stack.shape[0]
        co_label_matrix = np.zeros((n_samples, n_samples))
        co_occurrence_matrix = np.zeros((n_samples, n_samples))

        # ---------Repeat base clustering and generate consensus matrix---------

        n = 0
        label_history = []
        error_history = []

        if self.distance_metric == "quenching_coefficient":
            eem_dataset_unquenched, _ = eem_dataset.filter_by_index(self.kw_top, None, copy=True)
            eem_dataset_quenched, _ = eem_dataset.filter_by_index(self.kw_bot, None, copy=True)

        while n < n_base_clusterings:

            # ------Subsampling-------
            if self.distance_metric == "reconstruction_error":
                eem_dataset_n, selected_indices = eem_dataset.subsampling(portion=subsampling_portion)
            elif self.distance_metric == "quenching_coefficient":
                eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched.subsampling(
                    portion=subsampling_portion)
                pos = [eem_dataset_unquenched.index.index(idx) for idx in eem_dataset_new_uq.index]
                quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
                eem_dataset_new_q, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
                eem_dataset_n = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])
                eem_dataset_n.sort_by_index()
                selected_indices = [eem_dataset.index.index(idx) for idx in eem_dataset_n.index]
            n_samples_new = eem_dataset_n.eem_stack.shape[0]

            # ------Base clustering-------
            cluster_labels_n, label_history_n, error_history_n = self.base_clustering(eem_dataset_n)
            for j in range(n_samples_new):
                for k in range(n_samples_new):
                    co_occurrence_matrix[selected_indices[j], selected_indices[k]] += 1
                    if cluster_labels_n[j] == cluster_labels_n[k]:
                        co_label_matrix[selected_indices[j], selected_indices[k]] += 1
            label_history.append(label_history_n)
            error_history.append(error_history_n)

            # ----check if counting_matrix contains 0, meaning that not all sample pairs have been included in the
            # clustering. If this is the case, run more base clustering until all sample pairs are covered----
            if n == n_base_clusterings - 1 and np.any(co_occurrence_matrix == 0):
                warnings.warn(
                    'Not all sample pairs are covered. One extra clustering will be executed.')
            else:
                n += 1

        # ---------Obtain consensus matrix, distance matrix and linkage matrix----------
        consensus_matrix = co_label_matrix / co_occurrence_matrix

        self.n_runs = n_base_clusterings
        self.subsampling_portion = subsampling_portion
        self.label_history = label_history
        self.error_history = error_history
        self.consensus_matrix = consensus_matrix

        return consensus_matrix, label_history, error_history

    def hierarchical_clustering(self, eem_dataset, n_clusters, consensus_conversion_power=1):
        """

        Parameters
        ----------
        eem_dataset: EEMDataset
            EEM dataset to cluster.
        n_clusters: int
            Number of clusters.
        consensus_conversion_power: float
            The factor adjusting the conversion from consensus matrix (M) to distance matrix (D) used for hierarchical
            clustering. D_{i,j} = (1 - M_{i,j})^factor. This number influences the gradient of distance with respect
            to consensus. A smaller number will lead to shaper increase of distance at consensus close to 1.

        Returns
        -------

        """
        if self.consensus_matrix is None:
            raise ValueError('Consensus matrix is not defined.')
        distance_matrix = (1 - self.consensus_matrix) ** consensus_conversion_power
        linkage_matrix = linkage(squareform(distance_matrix), method='complete')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Find the minimum threshold distance for forming k clusters
        # max_d = 0
        # for i in range(linkage_matrix.shape[0] - n_clusters + 1):
        #     max_d = max(max_d, linkage_matrix[i, 2])

        linkage_matrix_sorted = linkage_matrix[linkage_matrix[:, 2].argsort()[::-1]]
        max_d = linkage_matrix_sorted[n_clusters - 2, 2]

        self.threshold_r = max_d

        sorted_indices = np.argsort(labels)
        consensus_matrix_sorted = self.consensus_matrix[sorted_indices][:, sorted_indices]
        if eem_dataset.index is not None:
            eem_index_sorted = [eem_dataset.index[i] for i in sorted_indices]
            self.index_sorted = eem_index_sorted
        if eem_dataset.ref is not None:
            eem_ref_sorted = eem_dataset.ref.iloc[sorted_indices, :]
            self.ref_sorted = eem_ref_sorted
        sc = silhouette_score(X=distance_matrix, labels=labels, metric='precomputed')
        self.silhouette_score = sc
        self.distance_matrix = distance_matrix
        self.linkage_matrix = linkage_matrix
        self.consensus_matrix_sorted = consensus_matrix_sorted

        # ---------Get final clusters and cluster-specific models-------
        clusters = {}
        cluster_specific_models = {}
        for j in set(list(labels)):
            eem_stack_j = eem_dataset.eem_stack[np.where(labels == j)]
            if eem_dataset.ref is not None:
                ref_j = eem_dataset.ref.iloc[np.where(labels == j)]
            else:
                ref_j = None
            if eem_dataset.index is not None:
                index_j = [eem_dataset.index[k] for k, idx in enumerate(labels) if idx == j]
            else:
                index_j = None
            cluster_j = [j] * eem_stack_j.shape[0]
            clusters[j] = EEMDataset(eem_stack=eem_stack_j, ex_range=eem_dataset.ex_range,
                                     em_range=eem_dataset.em_range, ref=ref_j, index=index_j, cluster=cluster_j)
            model = copy.deepcopy(self.base_model)
            # model = PARAFAC(rank=self.rank, non_negativity=self.non_negativity, init=self.init,
            #                 tf_normalization=self.tf_normalization,
            #                 loadings_normalization=self.loadings_normalization, sort_em=self.sort_em)
            model.fit(clusters[j])
            cluster_specific_models[j] = model

        self.labels = labels
        self.eem_clusters = clusters
        self.cluster_specific_models = cluster_specific_models

    def predict(self, eem_dataset: EEMDataset):
        """
        Fit the cluster-specific models to a given EEM dataset. Each EEM in the EEM dataset is fitted to the model that
        produce the least RMSE.

        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset to be predicted.

        Returns
        -------
        best_model_label: pd.DataFrame
            The best-fit model for every EEM.
        score_all: pd.DataFrame
            The score fitted with each cluster-specific model.
        fmax_all: pd.DataFrame
            The fmax fitted with each cluster-specific model.
        sample_error: pd.DataFrame
            The RMSE fitted with each cluster-specific model.
        """

        sample_error = []
        score_all = []
        fmax_all = []

        for label, m in self.cluster_specific_models.items():
            score_m, fmax_m, eem_stack_re_m = m.predict(eem_dataset)
            res = m.eem_stack_train - eem_stack_re_m
            n_pixels = m.eem_stack_train.shape[1] * m.eem_stack_train.shape[2]
            rmse = sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels)
            sample_error.append(rmse)
            score_all.append(score_m)
            fmax_all.append(fmax_m)

        score_all = pd.DataFrame(np.array(score_all), index=eem_dataset.index,
                                 columns=list(self.cluster_specific_models.keys()))
        fmax_all = pd.DataFrame(np.array(fmax_all), index=eem_dataset.index,
                                columns=list(self.cluster_specific_models.keys()))
        best_model_idx = np.argmin(sample_error, axis=0)
        # least_model_errors = np.min(sample_error, axis=0)
        # score_opt = np.array([score_all[i, j] for j, i in enumerate(best_model_idx)])
        # fmax_opt = np.array([fmax_all[i, j] for j, i in enumerate(best_model_idx)])
        best_model_label = np.array([list(self.cluster_specific_models.keys())[idx] for idx in best_model_idx])
        best_model_label = pd.DataFrame(best_model_label, index=eem_dataset.index, columns=['best-fit model'])
        sample_error = pd.DataFrame(np.array(sample_error), index=eem_dataset.index,
                                    columns=list(self.cluster_specific_models.keys()))

        return best_model_label, score_all, fmax_all, sample_error


# def hals_prior_nnls(
#         UtM,
#         UtU,
#         prior_dict=None,
#         V=None,
#         gamma=0,
#         alpha=0,
#         l1_ratio=0,
#         max_iter=500,
#         tol=1e-8,
#         eps=1e-8,
# ):
#     """
#     HALS-style non-negative least squares update for V in an NMF step,
#     with optional quadratic priors and elastic-net penalties on rows of V.
#
#     Solves: min_{V>=0} 0.5||M - U V||_F^2
#                          + sum_k (gamma/2)||V_k - p_k||^2
#                          + alpha*(l1_ratio*||V||_1 + 0.5*(1-l1_ratio)*||V||_F^2)
#     via alternating updates on each row V[k].
#
#     Parameters
#     ----------
#     UtM : array (r, n)
#         Precomputed U^T @ M.
#     UtU : array (r, r)
#         Precomputed U^T @ U.
#     prior_dict : dict {k: p_k}, optional
#         Priors for row k of V (p_k shape matches V[k, :]).
#     V : array (r, n), optional
#         Initial guess for V. If None, solves UtU V = UtM and clips.
#     gamma : float, optional
#         Quadratic prior weight (default 0 => no prior).
#     alpha : float, optional
#         Overall regularization weight for elastic-net penalty (default 0).
#     l1_ratio : float in [0,1], optional
#         The mix of L1 vs L2 in elastic-net (0 => pure L2, 1 => pure L1).
#     max_iter : int, optional
#         Maximum number of inner HALS iterations.
#     tol : float, optional
#         Convergence tolerance on row-wise updates.
#     eps : float, optional
#         Small constant to avoid divide by zero and ensure positivity.
#
#     Returns
#     -------
#     V : array (r, n)
#     """
#     r, n = tl.shape(UtM)
#     if prior_dict is None:
#         prior_dict = {}
#
#     # Initialize V
#     if V is None:
#         V_np = np.linalg.solve(np.asarray(UtU), np.asarray(UtM))
#         V = tl.tensor(np.clip(V_np, a_min=eps, a_max=None), dtype=float)
#         VVt = tl.dot(V, tl.transpose(V))
#         scale = tl.sum(UtM * V) / (tl.sum(UtU * VVt) + eps)
#         V = V * scale
#
#     # Precompute elastic-net constants
#     l2_pen = alpha * (1 - l1_ratio)
#     l1_pen = alpha * l1_ratio
#     prev_delta = None
#
#     for it in range(max_iter):
#         delta = 0.0
#         for k in range(r):
#             ukk = UtU[k, k]
#             if ukk < eps:
#                 continue
#
#             # Residual: UtM[k] - sum_{j!=k} UtU[k,j] * V[j]
#             Rk = UtM[k] - tl.dot(UtU[k], V) + ukk * V[k]
#
#             # Elastic-net base numerator/denominator
#             num = copy.copy(Rk)
#             # Highest-level denominator shape (vector) handles prior per-index
#             denom = np.full((n,), ukk + l2_pen, dtype=float)
#
#             # Quadratic prior if present and gamma > 0, ignoring NaNs
#             if k in prior_dict and gamma > 0:
#                 p_arr = np.asarray(prior_dict[k], dtype=float)
#                 mask = np.isfinite(p_arr).astype(float)
#                 mask_tl = tl.tensor(mask, dtype=float)
#                 p_tl = tl.tensor(np.nan_to_num(p_arr, nan=0.0), dtype=float)
#                 # apply prior on valid indices
#                 num = num + gamma * p_tl
#                 denom = denom + gamma * mask_tl
#
#             # apply L1 penalty (constant shift)
#             if l1_pen != 0:
#                 num = num - l1_pen
#
#             # compute update and enforce non-negativity
#             V_new = tl.clip(num / (denom + eps), a_min=eps)
#
#             # track change
#             delta += tl.norm(V[k] - V_new) ** 2
#             V[k] = V_new
#
#         # convergence check
#         if prev_delta is None:
#             prev_delta = delta
#         elif prev_delta > 0 and delta / prev_delta < tol:
#             break
#         prev_delta = delta
#
#     return V


def hals_prior_nnls(
        UtM,
        UtU,
        prior_dict=None,
        V=None,
        gamma=0,
        alpha=0,
        l1_ratio=0,
        r1_coef=0.0,            # weight of rank-one penalty
        mu=0.0,                  # weight of nuclear-norm penalty
        component_shape=None,       # tuple giving (b, c)
        max_iter=500,
        tol=1e-8,
        eps=1e-8,
):
    """
    HALS-style nonnegative least-squares update for V in an NMF step, with
    optional quadratic priors, elastic-net penalties, a rank-one deviation
    penalty, and a nuclear-norm penalty on each component.

    Solves for nonnegative V:
        min_{V >= 0} 0.5 * ||M - U V||_F^2
                       + (gamma/2) * sum_k ||V[k] - p_k||_F^2
                       + alpha * (l1_ratio * ||V||_1 + 0.5 * (1 - l1_ratio) * ||V||_F^2)
                       + r1_coef * sum_k (||M_k||_F^2 - sigma1(M_k)^2)
                       + mu * sum_k ||M_k||_*

    where:
      - U ∈ ℝ^{m×r}, M ∈ ℝ^{m×n}, V ∈ ℝ^{r×n}
      - M_k ∈ ℝ^{b×c} is row k of V reshaped to comp_shape = (b, c)
      - sigma1(M_k) is the leading singular value of M_k

    Updates are performed by alternating over rows k=0…r−1:
      1. HALS residual update with elastic-net and quadratic prior.
      2. Gradient-style correction for the rank-one deviation penalty.
      3. Proximal (soft-thresholding) update for the nuclear-norm penalty.
      4. Nonnegativity enforced by clipping to [eps, ∞).

    Parameters
    ----------
    UtM : array_like, shape (r, n)
        Precomputed U^T @ M.
    UtU : array_like, shape (r, r)
        Precomputed U^T @ U.
    prior_dict : dict, optional
        Mapping k → p_k array for quadratic priors on row V[k], same length n.
        If provided and gamma > 0, adds (gamma/2)*||V[k] - p_k||^2 penalty.
    V : array_like, shape (r, n), optional
        Initial guess for V; if None, a nonnegative least-squares solution
        of UtU V = UtM is used (then scaled and clipped).
    gamma : float, optional
        Weight of quadratic prior term (default 0: no prior).
    alpha : float, optional
        Overall weight for elastic-net penalty (default 0: no elastic-net).
    l1_ratio : float in [0,1], optional
        Mix between L1 and L2 in elastic-net (1 → pure L1, 0 → pure L2).
    r1_coef : float, optional
        Weight of the rank-one deviation penalty
        sum_k (||M_k||_F^2 - sigma1(M_k)^2) (default 0: none).
    mu : float, optional
        Weight of the nuclear-norm penalty sum_k ||M_k||_* (default 0: none).
    component_shape : tuple of ints (b, c), required if lambda_>0 or mu>0
        Shape to reshape each row V[k] into M_k of size (b, c).
    max_iter : int, optional
        Maximum number of HALS inner iterations (default 500).
    tol : float, optional
        Relative tolerance for convergence of row-wise updates (default 1e-8).
    eps : float, optional
        Small constant to avoid division by zero and enforce positivity (default 1e-8).

    Returns
    -------
    V : ndarray, shape (r, n)
        Updated nonnegative component matrix.

    Notes
    -----
    - The rank-one deviation penalty uses a “frozen” gradient that considers singular vectors as constants:
      ∇(||M_k||_F^2 - σ1^2) = 2 M_k - 2 σ1 u1 v1^T.
    - The nuclear-norm proximal step applies singular-value soft-thresholding:
      Prox_{τ||·||_*}(Y) = U diag(max(σ - τ, 0)) V^T.

    References
    ---------
    [1] Cai, Jian-Feng, Emmanuel J. Candès, and Zuowei Shen.
        "A singular value thresholding algorithm for matrix completion."
        SIAM Journal on optimization 20.4 (2010): 1956-1982.
    """
    r, n = tl.shape(UtM)
    b, c = component_shape
    if prior_dict is None:
        prior_dict = {}

    # Initialize V as before...
    if V is None:
        V_np = np.linalg.solve(np.asarray(UtU), np.asarray(UtM))
        V = tl.tensor(np.clip(V_np, a_min=eps, a_max=None), dtype=float)
        VVt = tl.dot(V, tl.transpose(V))
        scale = tl.sum(UtM * V) / (tl.sum(UtU * VVt) + eps)
        V = V * scale

    # Precompute elastic-net constants
    l2_pen = alpha * (1 - l1_ratio)
    l1_pen = alpha * l1_ratio
    prev_delta = None

    for it in range(max_iter):
        delta = 0.0

        for k in range(r):
            ukk = UtU[k, k]
            if ukk < eps:
                continue

            # Standard HALS residual
            Rk = UtM[k] - tl.dot(UtU[k], V) + ukk * V[k]

            # Base numerator/denominator (elastic-net + priors)
            num   = copy.copy(Rk)
            denom = np.full((n,), ukk + l2_pen, dtype=float)

            # Quadratic prior
            if k in prior_dict and gamma > 0:
                p_arr = np.asarray(prior_dict[k], dtype=float)
                mask  = np.isfinite(p_arr).astype(float)
                mask_tl = tl.tensor(mask, dtype=float)
                p_tl    = tl.tensor(np.nan_to_num(p_arr, nan=0.0), dtype=float)
                num   += gamma * p_tl
                denom += gamma * mask_tl

            # L1 shift
            if l1_pen != 0:
                num -= l1_pen

            # ─────────────────────────────────────────────
            # 1) Rank-one deviation penalty (gradient step)
            if r1_coef > 0:
                assert component_shape is not None
                # reshape row → M_k
                M_k = V[k].cpu().numpy().reshape(b, c)

                # top SVD (power‐method or full SVD)
                U1, S1, V1t = np.linalg.svd(M_k, full_matrices=False)
                sigma1 = S1[0]
                u1 = U1[:, 0]
                v1 = V1t.T[:, 0]

                # gradient of (‖M‖_F^2 - σ1^2) = 2*M - 2*σ1*u1*v1^T
                grad_lambda = 2*M_k - 2*sigma1 * np.outer(u1, v1)
                grad_flat   = grad_lambda.ravel()

                # incorporate gradient: numerator minus λ·grad, denominator plus 2λ
                num   = num - r1_coef * grad_flat
                denom = denom + 2 * r1_coef

            # 2) HALS‐style nonnegative update
            V_new = tl.clip(num / (denom + eps), a_min=eps)

            # ─────────────────────────────────────────────
            # 3) Nuclear‐norm proximal step (soft‐thresholding)
            if mu > 0:
                # step‐size scalar η = 1/(ukk + l2_pen + 2λ)  (approx)
                eta_k = 1.0 / (ukk + l2_pen + 2 * r1_coef + eps)

                # reshape interim back into matrix
                M_int = V_new.cpu().numpy().reshape(b, c)

                # SVD and singular‐value soft‐threshold τ = μ·η
                U, S, Vt = np.linalg.svd(M_int, full_matrices=False)
                tau = mu * eta_k
                S_thresh = np.maximum(S - tau, 0.0)

                # reconstruct and flatten
                M_prox = (U * S_thresh) @ Vt
                V_new = tl.tensor(np.clip(M_prox.ravel(), a_min=eps, a_max=None),
                                  dtype=float)

            # track change & write back
            delta += tl.norm(V[k] - V_new) ** 2
            V[k] = V_new

        # convergence check
        if prev_delta is not None and prev_delta > 0 and delta / prev_delta < tol:
            break
        prev_delta = delta

    return V

def nmf_hals_prior(
        X,
        rank,
        prior_dict_H=None,
        prior_dict_W=None,
        gamma_W=0,
        gamma_H=0,
        alpha_W=0,
        alpha_H=0,
        l1_ratio=0,
        max_iter_als=100,
        max_iter_nnls=500,
        tol=1e-6,
        eps=1e-8,
        init='random',
        custom_init=None,
        random_state=None,
        prior_ref_components=None,
):
    """
    Perform Non-negative Matrix Factorization (NMF) using the Hierarchical Alternating Least Squares (HALS) algorithm
    with optional prior guidance and elastic-net regularization.

    This function factorizes a non-negative matrix X into the product of two non-negative matrices W and H such that:
        X ≈ W @ H
    where W has shape (n_samples, rank), and H has shape (rank, n_features).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_pixels)
        Non-negative data matrix.
    rank : int
        Factorization rank r.
    prior_dict_H : dict {k: p_k}
        Priors for rows of H (zero-based indices), each p_k of shape (b,).
    prior_dict_W : dict {k: p_k}
        Priors for rows of W^T (i.e., columns of W), each p_k of shape (a,).
    gamma_W : float
        Prior regularization weight for W.
    gamma_H : float
        Prior regularization weight for H.
    alpha_W: float
        ElasticNet regularization weight for W.
    alpha_W: float
        ElasticNet regularization weight for H.
    l1_ratio: float between 0 and 1
        ElasticNet regularization mixing parameter
    max_iter_als : int
        Maximum number of outer ALS iterations.
    max_iter_nnls : int
        Maximum number of inner NNLS interations.
    tol : float
        Tolerance for convergence on reconstruction error.
    eps : float, optional
        Small constant to avoid divide by zero and ensure positivity.
    init : {'random', 'svd', 'nndsvd', 'nndsvda', 'nndsvdar', 'ordinary_nmf'}
        Initialization mode.
    custom_init: list [W_init, H_init]
        List of factors used for custom initialization
    random_state : int or None
        Seed for random initialization.
    prior_ref_components : dict, {k: ref_component of shape (n_pixels,)}
        Reference components to automatically reassign the keys of prior_dict_W according to the initialization.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
    H : array-like, shape (n_components, n_pixels)
    """
    X = tl.tensor(X, dtype=float)
    a, b = X.shape
    rng = np.random.RandomState(random_state)

    # Initialize W and H
    if init == 'random':
        W = tl.clip(rng.rand(a, rank), a_min=1e-6)
        H = tl.clip(rng.rand(rank, b), a_min=1e-6)
    elif init in ('svd', 'nndsvd', 'nndsvda', 'nndsvdar'):
        W, H = unfolded_eem_stack_initialization(X, rank=rank, method=init)
    elif init == 'ordinary_nmf':
        init_model = NMF(n_components=rank, init='nndsvd', random_state=random_state)
        W = init_model.fit_transform(X)
        H = init_model.components_
    elif init == 'custom':
        W, H = custom_init
    else:
        raise ValueError(f"Unknown init mode: {init}")

    # Default empty priors
    if prior_dict_H is None:
        prior_dict_H = {}
    if prior_dict_W is None:
        prior_dict_W = {}
    if prior_ref_components is not None:
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        print(cost_mat)
        # run Hungarian algorithm
        query_idx, ref_idx = linear_sum_assignment(cost_mat)
        H_new, W_new = np.zeros(H.shape), np.zeros(W.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, ref_idx):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
            r_list_query.pop(qi)
            r_list_ref.pop(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
        W, H = W_new, H_new

    prev_err = tl.norm(X - tl.dot(W, H))
    for it in range(max_iter_als):
        # Update H (rows) via HALS_NNLS with priors + elastic-net
        UtM_H = tl.dot(tl.transpose(W), X)
        UtU_H = tl.dot(tl.transpose(W), W)
        H = hals_prior_nnls(
            UtM=UtM_H,
            UtU=UtU_H,
            prior_dict=prior_dict_H,
            V=H,
            gamma=gamma_H,
            alpha=alpha_H,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls,
        )
        # Update W (columns) via HALS on W^T
        UtM_W = tl.dot(H, tl.transpose(X))
        UtU_W = tl.dot(H, tl.transpose(H))
        Wt = hals_prior_nnls(
            UtM=UtM_W,
            UtU=UtU_W,
            prior_dict=prior_dict_W,
            V=tl.transpose(W),
            gamma=gamma_W,
            alpha=alpha_W,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls,
        )
        W = tl.transpose(Wt)

        # Check convergence
        err = tl.norm(X - tl.dot(W, H))
        if it > 0:
            if abs(prev_err - err) / (prev_err + eps) < tol:
                break
        prev_err = err

    if prior_ref_components is not None:
        cost_mat = cdist(queries, H, metric='correlation')
        query_idx, ref_idx = linear_sum_assignment(cost_mat)
        H_new, W_new = np.zeros(H.shape), np.zeros(W.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, ref_idx):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
            r_list_query.pop(qi)
            r_list_ref.pop(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
        W, H = W_new, H_new

    return W, H


def cp_hals_prior(
        tensor,
        rank,
        prior_dict_A=None,
        prior_dict_B=None,
        prior_dict_C=None,
        gamma_A=0,
        gamma_B=0,
        gamma_C=0,
        prior_ref_components=None,
        alpha_A=0,
        alpha_B=0,
        alpha_C=0,
        l1_ratio=0,
        max_iter_als=200,
        max_iter_nnls=500,
        tol=1e-6,
        eps=1e-8,
        init='svd',
        custom_init=None,
        random_state=None
):
    """
    Perform non-negative PARAFAC/CP decomposition of a 3-way tensor using HALS with optional priors
    and elastic-net penalties on factor matrices A, B, C.

    Decomposes `tensor` of shape (I, J, K) into factors A (I x rank), B (J x rank), C (K x rank) such that:
        tensor ≈ [[A, B, C]]

    Parameters
    ----------
    tensor : array-like, shape (I, J, K)
        Input non-negative tensor.
    rank : int
        Number of components.
    prior_dict_A : dict {r: v_r}, optional
        Priors for columns of A: column r of A is penalized toward vector v_r.
    prior_dict_B : dict {r: v_r}, optional
        Priors for columns of B.
    prior_dict_C : dict {r: v_r}, optional
        Priors for columns of C.
    gamma_A, gamma_B, gamma_C : float, optional
        Quadratic prior weights for A, B, C.
    alpha_A, alpha_B, alpha_C : float, optional
        Elastic-net weights for A, B, C.
    l1_ratio : float in [0,1], optional
        Mix between L1 and L2 for elastic-net.
    max_iter_als : int, optional
        Maximum number of outer ALS iterations.
    max_iter_nnls : int, optional
        Maximum number of inner NNLS interations.
    tol : float, optional
        Convergence tolerance on reconstruction error.
    eps : float, optional
        Small constant to avoid zero division and ensure positivity.
    init : {'random', 'svd', 'nndsvd', 'nndsvda', 'nndsvdar'}, default 'random'
        Initialization scheme for factor matrices.
    random_state : int or None
        Random seed.

    Returns
    -------
    A : ndarray, shape (I, rank)
    B : ndarray, shape (J, rank)
    C : ndarray, shape (K, rank)
    """
    # Ensure tensor
    X = tl.tensor(tensor, dtype=float)
    I, J, K = X.shape
    rng = np.random.RandomState(random_state)

    # Initialize factors A, B, C
    if init == 'random':
        A = tl.clip(rng.rand(I, rank), a_min=eps)
        B = tl.clip(rng.rand(J, rank), a_min=eps)
        C = tl.clip(rng.rand(K, rank), a_min=eps)
    elif init in ('svd', 'nndsvd', 'nndsvda', 'nndsvdar'):
        # Use 2D initialization on each mode unfolding
        # Mode-0 init for A
        X1 = tl.unfold(X, mode=0)
        W1, _ = unfolded_eem_stack_initialization(tl.to_numpy(X1), rank, method=init)
        A = tl.tensor(np.clip(W1, a_min=eps, a_max=None), dtype=float)
        # Mode-1 init for B
        X2 = tl.unfold(X, mode=1)
        W2, _ = unfolded_eem_stack_initialization(tl.to_numpy(X2), rank, method=init)
        B = tl.tensor(np.clip(W2, a_min=eps, a_max=None), dtype=float)
        # Mode-2 init for C
        X3 = tl.unfold(X, mode=2)
        W3, _ = unfolded_eem_stack_initialization(tl.to_numpy(X3), rank, method=init)
        C = tl.tensor(np.clip(W3, a_min=eps, a_max=None), dtype=float)
    elif init == 'ordinary_cp':
        A, B, C = non_negative_parafac_hals(X, rank=rank, random_state=random_state)
    elif init == 'custom':
        A, B, C = custom_init
    else:
        raise ValueError(f"Unknown init mode: {init}")

    # Default empty priors
    if prior_dict_B is None:
        prior_dict_B = {}
    if prior_dict_C is None:
        prior_dict_C = {}
    if prior_dict_A is None:
        prior_dict_A = {}
    elif prior_ref_components is not None:
        H = np.zeros([rank, B.shape[0] * C.shape[0]])
        for r in range(rank):
            component = np.array([B[:, r]]).T.dot(np.array([C[:, r]]))
            H[r, :] = component.reshape(-1)
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, ref_idx = linear_sum_assignment(cost_mat)
        A_new, B_new, C_new = np.zeros(A.shape), np.zeros(B.shape), np.zeros(C.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, ref_idx):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
            r_list_query.pop(qi)
            r_list_ref.pop(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
        A, B, C = A_new, B_new, C_new

    prev_error = tl.norm(X - cp_to_tensor((None, [A, B, C])))
    for iteration in range(max_iter_als):
        # Update A:
        UtM = unfolding_dot_khatri_rao(tensor, (None, [A, B, C]), 0)
        UtU = (C.T @ C) * (B.T @ B)  # shape (rank, rank)
        A = hals_prior_nnls(
            UtM=UtM.T,  # shape (rank, I)
            UtU=UtU,
            prior_dict=prior_dict_A,
            V=A.T,
            gamma=gamma_A,
            alpha=alpha_A,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls
        )
        A = A.T

        # Update B:
        UtM = unfolding_dot_khatri_rao(tensor, (None, [A, B, C]), 1)
        UtU = (C.T @ C) * (A.T @ A)  # shape (rank, rank)
        B = hals_prior_nnls(
            UtM=UtM.T,  # shape (rank, J)
            UtU=UtU,
            prior_dict=prior_dict_B,
            V=B.T,
            gamma=gamma_B,
            alpha=alpha_B,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls
        )
        B = B.T

        # Update C:
        UtM = unfolding_dot_khatri_rao(tensor, (None, [A, B, C]), 2)
        UtU = (B.T @ B) * (A.T @ A)  # shape (rank, rank)
        C = hals_prior_nnls(
            UtM=UtM.T,  # shape (rank, K)
            UtU=UtU,
            prior_dict=prior_dict_C,
            V=C.T,
            gamma=gamma_C,
            alpha=alpha_C,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls
        )
        C = C.T

        # # Normalize factors for numerical stability (optional)
        # weights, [A, B, C] = cp_normalize((None, [A, B, C]))

        # Check convergence
        reconstructed = cp_to_tensor((None, [A, B, C]))
        err = tl.norm(X - reconstructed)
        if abs(prev_error - err) / (prev_error + eps) < tol:
            break
        prev_error = err

    return A, B, C


def cp_hals_prior_ratio(
        tensor,
        rank,
        prior_dict_A=None,
        prior_dict_B=None,
        prior_dict_C=None,
        gamma_A=0,
        gamma_B=0,
        gamma_C=0,
        prior_ref_components=None,
        alpha_A=0,
        alpha_B=0,
        alpha_C=0,
        l1_ratio=0,
        lam=0,
        idx_top=None,
        idx_bot=None,
        max_iter_als=200,
        max_iter_nnls=500,
        tol=1e-9,
        eps=1e-8,
        init='svd',
        custom_init=None,
        random_state=None
):
    """
    Perform non-negative PARAFAC/CP decomposition of a 3-way tensor using HALS with optional priors
    and elastic-net penalties on factor matrices A, B, C.

    Decomposes `tensor` of shape (I, J, K) into factors A (I x rank), B (J x rank), C (K x rank) such that:
        tensor ≈ [[A, B, C]]

    Parameters
    ----------
    tensor : array-like, shape (I, J, K)
        Input non-negative tensor.
    rank : int
        Number of components.
    prior_dict_A : dict {r: v_r}, optional
        Priors for columns of A: column r of A is penalized toward vector v_r.
    prior_dict_B : dict {r: v_r}, optional
        Priors for columns of B.
    prior_dict_C : dict {r: v_r}, optional
        Priors for columns of C.
    gamma_A, gamma_B, gamma_C : float, optional
        Quadratic prior weights for A, B, C.
    alpha_A, alpha_B, alpha_C : float, optional
        Elastic-net weights for A, B, C.
    l1_ratio : float in [0,1], optional
        Mix between L1 and L2 for elastic-net.
    max_iter_als : int, optional
        Maximum number of outer ALS iterations.
    max_iter_nnls : int, optional
        Maximum number of inner NNLS interations.
    tol : float, optional
        Convergence tolerance on reconstruction error.
    eps : float, optional
        Small constant to avoid zero division and ensure positivity.
    init : {'random', 'svd', 'nndsvd', 'nndsvda', 'nndsvdar'}, default 'random'
        Initialization scheme for factor matrices.
    random_state : int or None
        Random seed.

    Returns
    -------
    A : ndarray, shape (I, rank)
    B : ndarray, shape (J, rank)
    C : ndarray, shape (K, rank)
    beta: ndarray, shape (rank,)
    """
    # Ensure tensor
    X = tl.tensor(tensor, dtype=float)
    I, J, K = X.shape
    rng = np.random.RandomState(random_state)

    # Initialize factors A, B, C
    if init == 'random':
        A = tl.clip(rng.rand(I, rank), a_min=eps)
        B = tl.clip(rng.rand(J, rank), a_min=eps)
        C = tl.clip(rng.rand(K, rank), a_min=eps)
    elif init in ('svd', 'nndsvd', 'nndsvda', 'nndsvdar'):
        # Use 2D initialization on each mode unfolding
        # Mode-0 init for A
        X1 = tl.unfold(X, mode=0)
        W1, _ = unfolded_eem_stack_initialization(tl.to_numpy(X1), rank, method=init)
        A = tl.tensor(np.clip(W1, a_min=eps, a_max=None), dtype=float)
        # Mode-1 init for B
        X2 = tl.unfold(X, mode=1)
        W2, _ = unfolded_eem_stack_initialization(tl.to_numpy(X2), rank, method=init)
        B = tl.tensor(np.clip(W2, a_min=eps, a_max=None), dtype=float)
        # Mode-2 init for C
        X3 = tl.unfold(X, mode=2)
        W3, _ = unfolded_eem_stack_initialization(tl.to_numpy(X3), rank, method=init)
        C = tl.tensor(np.clip(W3, a_min=eps, a_max=None), dtype=float)
    elif init == 'ordinary_cp':
        A, B, C = non_negative_parafac_hals(X, rank=rank, random_state=random_state)[1]
    elif init == 'custom':
        A, B, C = custom_init
    else:
        raise ValueError(f"Unknown init mode: {init}")

    # Default empty priors
    if prior_dict_B is None:
        prior_dict_B = {}
    if prior_dict_C is None:
        prior_dict_C = {}
    if prior_dict_A is None:
        prior_dict_A = {}
    elif prior_ref_components is not None:
        H = np.zeros([rank, B.shape[0] * C.shape[0]])
        for r in range(rank):
            component = np.array([B[:, r]]).T.dot(np.array([C[:, r]]))
            H[r, :] = component.reshape(-1)
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, ref_idx = linear_sum_assignment(cost_mat)
        A_new, B_new, C_new = np.zeros(A.shape), np.zeros(B.shape), np.zeros(C.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, ref_idx):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
            r_list_query.pop(qi)
            r_list_ref.pop(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
        A, B, C = A_new, B_new, C_new

    beta = np.ones(rank, dtype=float)
    prev_error = tl.norm(X - cp_to_tensor((None, [A, B, C])))
    for iteration in range(max_iter_als):
        # Update B:
        UtM = unfolding_dot_khatri_rao(tensor, (None, [A, B, C]), 1)
        UtU = (C.T @ C) * (A.T @ A)  # shape (rank, rank)
        B = hals_prior_nnls(
            UtM=UtM.T,  # shape (rank, J)
            UtU=UtU,
            prior_dict=prior_dict_B,
            V=B.T,
            gamma=gamma_B,
            alpha=alpha_B,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls
        )
        B = B.T

        # Update C:
        UtM = unfolding_dot_khatri_rao(tensor, (None, [A, B, C]), 2)
        UtU = (B.T @ B) * (A.T @ A)  # shape (rank, rank)
        C = hals_prior_nnls(
            UtM=UtM.T,  # shape (rank, K)
            UtU=UtU,
            prior_dict=prior_dict_C,
            V=C.T,
            gamma=gamma_C,
            alpha=alpha_C,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls
        )
        C = C.T

        # --- Update W via ratio-aware HALS columns ---
        UtM = unfolding_dot_khatri_rao(tensor, (None, [A, B, C]), 0).T  # (r, m)
        UtU = (C.T @ C) * (B.T @ B)  # (r, r)
        for k in range(rank):
            Rk = UtM[k].copy()
            for j in range(rank):
                if j != k:
                    Rk -= UtU[k, j] * A[:, j]
            d = UtU[k, k]
            A[:, k] = hals_column_with_ratio(
                Rk=Rk,
                hk_norm2=d,
                beta_k=beta[k],
                lam=lam,
                k=k,
                prior_dict=prior_dict_A,
                gamma=gamma_A,
                alpha=alpha_A,
                l1_ratio=l1_ratio,
                idx_top=idx_top,
                idx_bot=idx_bot,
                eps=eps
            )

        # --- Beta‐step (closed form) ---
        beta = update_beta(A, idx_top=idx_top, idx_bot=idx_bot, eps=0)

        # # Normalize factors for numerical stability (optional)
        # weights, [A, B, C] = cp_normalize((None, [A, B, C]))

        # Check convergence
        reconstructed = cp_to_tensor((None, [A, B, C]))
        err = tl.norm(X - reconstructed)
        if abs(prev_error - err) / (prev_error + eps) < tol:
            break
        prev_error = err

    return A, B, C, beta


def unfolded_eem_stack_initialization(M, rank, method='nndsvd'):
    if method == 'ordinary_nmf':
        nmf_model = NMF(n_components=rank, init='nndsvd')
        W = nmf_model.fit_transform(M)
        H = nmf_model.components_
        return W, H
    # Step 1: Compute SVD of V
    U, S, VT = np.linalg.svd(M, full_matrices=False)  # SVD decomposition

    # Step 2: Keep the top-r components
    U_r = U[:, :rank]
    S_r = S[:rank]
    VT_r = VT[:rank, :]

    if method == 'svd':
        W = np.abs(U_r) * np.sqrt(S_r)[None, :]
        H = np.sqrt(S_r)[:, None] * np.abs(VT_r)
        W = np.clip(W, a_min=1e-6, a_max=None)
        H = np.clip(H, a_min=1e-6, a_max=None)

    else:
        # Step 3: Initialize W and H
        W = np.zeros((M.shape[0], rank))
        H = np.zeros((rank, M.shape[1]))

        for k in range(rank):
            u_k = U_r[:, k]
            v_k = VT_r[k, :]

            # Positive and negative parts
            u_k_pos = np.maximum(u_k, 0)
            u_k_neg = np.maximum(-u_k, 0)
            v_k_pos = np.maximum(v_k, 0)
            v_k_neg = np.maximum(-v_k, 0)

            # Normalize
            u_norm_pos = np.linalg.norm(u_k_pos)
            v_norm_pos = np.linalg.norm(v_k_pos)

            # Assign components
            if u_norm_pos * v_norm_pos > 0:
                W[:, k] = np.sqrt(S_r[k]) * (u_k_pos / u_norm_pos)
                H[k, :] = np.sqrt(S_r[k]) * (v_k_pos / v_norm_pos)
            else:
                W[:, k] = np.sqrt(S_r[k]) * (u_k_neg / np.linalg.norm(u_k_neg))
                H[k, :] = np.sqrt(S_r[k]) * (v_k_neg / np.linalg.norm(v_k_neg))

        # Step 4: Handle zero entries
        if method == 'nndsvd':
            pass
        if method == 'nndsvda':
            W[W == 0] = np.mean(M)
            H[H == 0] = np.mean(M)
        if method == 'nndsvdar':
            W[W == 0] = np.random.uniform(0, np.mean(M) / 100, W[W == 0].shape)
            H[H == 0] = np.random.uniform(0, np.mean(M) / 100, H[H == 0].shape)

    return W, H


def replace_factor_with_prior(factors, prior, replaced_mode, replaced_rank="best-fit", frozen_rank=None,
                              project_prior=True, X=None, show_replaced_rank=False):
    rank = factors[replaced_mode].shape[1]
    if replaced_rank == "best-fit":
        sim = -1
        replaced_rank = 0
        for j in [i for i in range(rank) if i != frozen_rank]:
            factor_init = factors[replaced_mode][:, j]
            r, _ = pearsonr(factor_init, prior)
            if r > sim:
                sim = r
                replaced_rank = j
        factors[replaced_mode][:, replaced_rank] = prior
    else:
        factors[replaced_mode][:, replaced_rank] = prior

    if project_prior and X is not None:
        residual = X
        for j in [i for i in range(rank) if i != replaced_rank]:
            residual -= np.outer(factors[0][:, j], factors[1][:, j])
        if replaced_mode == 0:
            E = residual.T
        elif replaced_mode == 1:
            E = residual
        projection = E @ prior / np.inner(prior, prior)
        projection[projection < 0] = 0
        factors[int(1 - replaced_mode)][:, replaced_rank] = projection

    if show_replaced_rank:
        return factors, replaced_rank
    else:
        return factors


def hals_column_with_ratio(
        Rk,
        hk_norm2,
        beta_k,
        lam,
        k,
        prior_dict=None,
        gamma=0.0,
        alpha=0.0,
        l1_ratio=0.0,
        idx_top=None,
        idx_bot=None,
        eps=1e-8
):
    """
    HALS‐style update for one W‐column w_k ∈ ℝ^m solving

        ½||Rk - d·w_k||²
      + (lam/2)*Σ_i (w[t_i] - β·w[b_i])²
      + (γ/2)*Σ_i (w_i - p_k[i])²
      + α[ℓ1||w_k||₁ + ((1-ℓ1)/2)||w_k||²]

    where:
      - d = ||h_k||²,
      - Rk = UᵀM - Σ_{j≠k}(UᵀU)_{kj} w_j,
      - lam = λ is the ratio penalty,
      - γ = gamma is the prior penalty,
      - α,ℓ1_mix = elastic-net weights,
      - prior_dict[k] = length-m vector p_k (with NaNs where no prior),
      - idx_top/idx_bot pair row-indices covering all rows.
    """
    m = Rk.shape[0]
    d = hk_norm2
    g = Rk.copy()

    # Elastic‐net
    l2_pen = alpha * (1 - l1_ratio)
    l1_pen = alpha * l1_ratio
    if l1_pen:
        g -= l1_pen

    # Prepare prior vector and mask
    if prior_dict is None:
        prior_dict = {}
    has_prior = (gamma > 0 and k in prior_dict)
    if has_prior:
        p_k = np.asarray(prior_dict[k], dtype=float)
        mask = np.isfinite(p_k)
        p_clean = np.nan_to_num(p_k, nan=0.0)
        # Add linear term once (will split per-entry below)
        g += gamma * p_clean

    w_new = np.empty_like(g)

    # Off-diagonal for ratio block
    off = -lam * beta_k

    # Solve each paired (t,b)
    for t, b in zip(idx_top, idx_bot):
        R1, R2 = g[t], g[b]
        # per-entry prior diag
        prior_t = gamma if (has_prior and mask[t]) else 0.0
        prior_b = gamma if (has_prior and mask[b]) else 0.0
        # build local diagonals
        a11 = d + lam + l2_pen + prior_t
        a22 = d + lam * beta_k ** 2 + l2_pen + prior_b
        det = a11 * a22 - off * off + eps

        # compute scalar updates
        w1 = (a22 * R1 - off * R2) / det
        w2 = (-off * R1 + a11 * R2) / det

        w_new[t] = float(max(eps, w1))
        w_new[b] = float(max(eps, w2))

    return w_new


def update_beta(
        W: np.ndarray,
        idx_top,
        idx_bot,
        eps: float = 1e-8,
        boundaries: tuple = (0.95, 1.4)
) -> np.ndarray:
    """
    Fit beta per component so that W[idx_top, j] ≈ beta[j] * W[idx_bot, j].

    Solves, for each component j,
        min_{beta_j} ∑_i (W_top[i,j] - beta_j * W_bot[i,j])^2
    which has the closed‐form
        beta_j = sum_i W_top[i,j] * W_bot[i,j]  /  (sum_i W_bot[i,j]^2).

    Parameters
    ----------
    W : np.ndarray, shape (m, r)
        Concentration matrix with m samples and r components.
    idx_top : sequence of ints
        Row indices in W corresponding to the “original” samples.
    idx_bot : sequence of ints
        Row indices in W corresponding to the “perturbed” samples.
        Must be the same length as idx_top.
    eps : float, optional
        Small constant to avoid division by zero when W_bot is nearly zero.
    boundaries : (min_beta, max_beta), optional
        Lower and upper bounds to clamp each estimated beta.

    Returns
    -------
    beta : np.ndarray, shape (r,)
        Estimated ratio for each of the r components, clamped to [min_beta, max_beta].
    """
    W = np.asarray(W, dtype=float)
    idx_top = np.asarray(idx_top, dtype=int)
    idx_bot = np.asarray(idx_bot, dtype=int)
    if idx_top.shape != idx_bot.shape:
        raise ValueError("`idx_top` and `idx_bot` must have the same length")

    # Extract the paired rows
    W_top = W[idx_top, :]  # shape (p, r)
    W_bot = W[idx_bot, :]  # shape (p, r)

    # Compute numerator and denominator for each component j:
    #   numerator_j   = sum_i W_top[i,j] * W_bot[i,j]
    #   denominator_j = sum_i W_bot[i,j]^2
    num = np.sum(W_top * W_bot, axis=0)
    den = np.sum(W_bot * W_bot, axis=0) + eps

    beta = num / den

    # Clamp into the desired interval
    beta_min, beta_max = boundaries
    return np.clip(beta, beta_min, beta_max)


def nmf_hals_prior_ratio(
        X,
        rank,
        idx_top,
        idx_bot,
        lam=0,
        prior_dict_H=None,
        prior_dict_W=None,
        prior_ref_components=None,
        gamma_W=0,
        gamma_H=0,
        alpha_W=0,
        alpha_H=0,
        l1_ratio=0,
        r1_coef=0,
        mu=0,
        component_shape=None,
        max_iter_als=100,
        max_iter_nnls=500,
        tol=1e-6,
        eps=0,
        init='random',
        custom_init=None,
        random_state=None
):
    """
    ALS‐NMF with three penalties:
      - Elastic-net on H and W (alpha_H, alpha_W).
      - Quadratic priors on H and W via prior_dict_H/prior_dict_W (gamma_H, gamma_W).
      - Ratio penalty on W: W[idx_top] ≈ beta * W[idx_bot] (lam=gamma_W).

    Parameters
    ----------
    X : array-like (m, n)
        Non-negative data.
    rank : int
        Number of components.
    idx_top, idx_bot : lists of int, length m/2
        Row‐index pairs covering all samples for the **ratio** penalty.
    prior_dict_H : dict {k: p_k}, optional
        Priors for H rows (length n, NaN to skip).
    prior_dict_W : dict {k: p_k}, optional
        Priors for W columns (length m, NaN to skip).
    lam : float
        Ratio penalty weight.
    gamma_W, gamma_H : float
        Quadratic prior weights.
    alpha_W, alpha_H : float
        Elastic-net weights for W and H.
    l1_ratio : float [0,1]
        Mix parameter for elastic-net.
    max_iter_als : int
        Outer ALS iterations.
    max_iter_nnls : int
        Inner HALS iterations for H.
    tol : float
        Convergence tolerance.
    eps : float
        Small positive floor.
    init : {'random','svd',...}, custom_init, random_state : as before.

    Returns
    -------
    W : ndarray (m, rank)
    H : ndarray (rank, n)
    beta : ndarray (rank,)
    """
    X_t = tl.tensor(X, dtype=float)
    m, n = X_t.shape
    rng = np.random.RandomState(random_state)

    # 1) Initialize W, H
    if init == 'random':
        W = np.clip(rng.rand(m, rank), eps, None)
        H = np.clip(rng.rand(rank, n), eps, None)
    elif init in ('svd', 'nndsvd', 'nndsvda', 'nndsvdar'):
        W, H = unfolded_eem_stack_initialization(X, rank, init)
    elif init == 'ordinary_nmf':
        model = NMF(n_components=rank, init='nndsvd', random_state=random_state)
        W = model.fit_transform(tl.to_numpy(X_t))
        H = model.components_
    elif init == 'custom':
        W, H = custom_init
    else:
        raise ValueError(f"Unknown init {init}")

    # Default empty priors
    if prior_dict_H is None:
        prior_dict_H = {}
    if prior_dict_W is None:
        prior_dict_W = {}
    if prior_ref_components is not None:
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, ref_idx = linear_sum_assignment(cost_mat)
        H_new, W_new = np.zeros(H.shape), np.zeros(W.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, ref_idx):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
            r_list_query.pop(qi)
            r_list_ref.pop(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
        W, H = W_new, H_new

    beta = np.ones(rank, dtype=float)
    prev_err = np.inf

    for _ in range(max_iter_als):
        # --- Update H via HALS with priors + elastic-net ---
        UtM_H = tl.dot(tl.transpose(W), X_t)
        UtU_H = tl.dot(tl.transpose(W), W)
        H = hals_prior_nnls(
            UtM=UtM_H, UtU=UtU_H,
            prior_dict=prior_dict_H,
            V=H,
            gamma=gamma_H, alpha=alpha_H,
            l1_ratio=l1_ratio,
            max_iter=max_iter_nnls,
            tol=tol, eps=eps
        )

        # --- Update W via ratio-aware HALS columns ---
        UtM_W = tl.to_numpy(tl.dot(H, tl.transpose(X_t)))  # (r, m)
        UtU_W = tl.to_numpy(tl.dot(H, tl.transpose(H)))  # (r, r)
        for k in range(rank):
            Rk = UtM_W[k].copy()
            for j in range(rank):
                if j != k:
                    Rk -= UtU_W[k, j] * W[:, j]
            d = UtU_W[k, k]
            W[:, k] = hals_column_with_ratio(
                Rk=Rk,
                hk_norm2=d,
                beta_k=beta[k],
                lam=lam,
                k=k,
                prior_dict=prior_dict_W,
                gamma=gamma_W,
                alpha=alpha_W,
                l1_ratio=l1_ratio,
                idx_top=idx_top,
                idx_bot=idx_bot,
                eps=eps
            )

        # --- Beta‐step (closed form) ---
        beta = update_beta(W, idx_top=idx_top, idx_bot=idx_bot, eps=0)

        # --- Convergence check ---
        err = tl.norm(X_t - tl.tensor(W) @ tl.tensor(H))
        if abs(prev_err - err) / (prev_err + eps) < tol:
            break
        prev_err = err

    if prior_ref_components is not None:
        cost_mat = cdist(queries, H, metric='correlation')
        query_idx, ref_idx = linear_sum_assignment(cost_mat)
        H_new, W_new = np.zeros(H.shape), np.zeros(W.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, ref_idx):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
            r_list_query.pop(qi)
            r_list_ref.pop(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
        W, H = W_new, H_new

    return W, H, beta


def solve_W(X1, H, X2=None, beta=None, reg=0.0, non_negativity=True):
    """
    Solve for W in the regression problem:
        loss = ||X1 - W @ H||_F^2
             + ||X2 - W @ diag(beta) @ H||_F^2  (optional if b is provided)

    If b is None, reduces to standard regression: minimize ||C - W @ H||_F^2.  In that case D and b are ignored.

    Arguments:
        X1 (ndarray): m x n matrix.
        X2 (ndarray, optional): m x n matrix.  Required if b is not None.
        H (ndarray): r x n matrix.
        beta (ndarray, optional): vector of length r.  If None, drop the second term.
        reg (float): optional regularization (ridge) parameter.
        non_negativity (bool): whether to apply non-negativity.

    Returns:
        W (ndarray): m x r solution matrix.
    """
    # Validate shapes
    m, n = X1.shape
    r, n_H = H.shape
    assert n_H == n, "H must be of shape (r, n)"
    if beta is not None:
        assert X2 is not None and X2.shape == (m, n), "D must match shape of C when b is provided"
        assert beta.shape[0] == r, "b must have length r"

    # Prepare design matrix and target for NNLS if needed
    if non_negativity:
        # Build block design and targets
        # A: (2n x r) or (n x r) if b is None
        A1 = H.T  # n x r
        Y_blocks = [X1]
        A_blocks = [A1]
        if beta is not None:
            A2 = (np.diag(beta) @ H).T  # n x r
            A_blocks.append(A2)
            Y_blocks.append(X2)
        A = np.vstack(A_blocks)
        # Solve row-wise
        W = np.zeros((m, r))
        for i in range(m):
            y = np.hstack([Y_blocks[j][i] for j in range(len(Y_blocks))])
            W[i], _ = nnls(A, y)
        return W

    # Numerator and Denominator
    if beta is None:
        # Standard regression: only C and H
        numerator = X1 @ H.T
        denominator = H @ H.T
    else:
        B = np.diag(beta)
        numerator = X1 @ H.T + X2 @ H.T @ B
        HHT = H @ H.T
        denominator = HHT + B @ HHT @ B

    # Add ridge regularization if requested
    if reg > 0:
        denominator = denominator + reg * np.eye(r)

    # Solve for W (avoid explicit inverse)
    # Solve (denominator.T) @ X = numerator.T  => X = W.T
    W = np.linalg.solve(denominator.T, numerator.T).T
    return W


def eem_stack_to_2d(eem_stack):
    return eem_stack.reshape([eem_stack.shape[0], -1])

#
# def hals_prior_nnls_torch(
#     UtM: torch.Tensor,
#     UtU: torch.Tensor,
#     prior_dict: dict = None,
#     V: torch.Tensor = None,
#     gamma: float = 0.0,
#     alpha: float = 0.0,
#     l1_ratio: float = 0.0,
#     max_iter: int = 500,
#     tol: float = 1e-8,
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     """
#     HALS-style non-negative least squares update (PyTorch) with priors & elastic-net.
#
#     UtM: (r, n), UtU: (r, r), V: (r, n)
#     prior_dict: {k: np.ndarray or torch.Tensor of length n}
#     Returns V updated as torch.Tensor.
#     """
#     device = UtM.device
#     dtype = UtM.dtype
#     r, n = UtM.shape
#     if prior_dict is None:
#         prior_dict = {}
#
#     # initialize V if needed
#     if V is None:
#         V = torch.linalg.solve(UtU, UtM).clamp(min=eps)
#         VVt = V @ V.T
#         scale = (UtM * V).sum() / ((UtU @ VVt).sum() + eps)
#         V = V * scale
#
#     l2_pen = alpha * (1 - l1_ratio)
#     l1_pen = alpha * l1_ratio
#     prev_delta = None
#
#     for _ in range(max_iter):
#         delta = 0.0
#         for k in range(r):
#             ukk = UtU[k, k]
#             if ukk < eps:
#                 continue
#             Rk = UtM[k] - UtU[k] @ V + ukk * V[k]
#             num = Rk.clone()
#             denom = torch.full((n,), ukk + l2_pen, device=device, dtype=dtype)
#             # apply prior if present
#             if k in prior_dict and gamma > 0:
#                 p = prior_dict[k]
#                 if not isinstance(p, torch.Tensor):
#                     p = torch.tensor(p, dtype=dtype, device=device)
#                 mask = torch.isfinite(p).float()
#                 p = torch.nan_to_num(p, nan=0.0)
#                 num += gamma * p
#                 denom += gamma * mask
#             if l1_pen != 0:
#                 num -= l1_pen
#             V_new = (num / (denom + eps)).clamp(min=eps)
#             delta += torch.norm(V[k] - V_new).item()**2
#             V[k] = V_new
#         if prev_delta is None:
#             prev_delta = delta
#         elif prev_delta > 0 and delta / prev_delta < tol:
#             break
#         prev_delta = delta
#     return V
#
#
# def nmf_hals_prior_torch(
#     X: np.ndarray,
#     rank: int,
#     prior_dict_H: dict = None,
#     prior_dict_W: dict = None,
#     gamma_W: float = 0.0,
#     gamma_H: float = 0.0,
#     alpha_W: float = 0.0,
#     alpha_H: float = 0.0,
#     l1_ratio: float = 0.0,
#     max_iter_als: int = 100,
#     max_iter_nnls: int = 500,
#     tol: float = 1e-6,
#     eps: float = 1e-8,
#     init: str = 'random',
#     random_state: int = None,
#     device_type: str = 'cpu',
# ) -> (np.ndarray, np.ndarray):
#     """
#     NMF via HALS (PyTorch backend). Inputs X and prior_dicts are numpy.
#     Returns W, H as numpy arrays.
#     """
#     if device_type == 'gpu':
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     else:
#         device = torch.device('cpu')
#     dtype = torch.float32
#
#     X_torch = torch.tensor(X, dtype=dtype, device=device)
#     a, b = X.shape
#     rng = np.random.RandomState(random_state)
#
#     # init W, H
#     if init == 'random':
#         W = torch.rand(a, rank, dtype=dtype, device=device).clamp(min=eps)
#         H = torch.rand(rank, b, dtype=dtype, device=device).clamp(min=eps)
#     else:
#         # use unfolded_eem_stack_initialization_torch on full X
#         W, H = unfolded_eem_stack_initialization(X, rank, method=init)
#         W = torch.tensor(W, dtype=torch.float32)
#         H = torch.tensor(H, dtype=torch.float32)
#         W = W.to(device).clamp(min=eps)
#         H = H.to(device).clamp(min=eps)
#
#     prior_dict_H = prior_dict_H or {}
#     prior_dict_W = prior_dict_W or {}
#
#     prev_err = torch.norm(X_torch - W @ H).item()
#     for _ in range(max_iter_als):
#         UtM_H = W.T @ X_torch
#         UtU_H = W.T @ W
#         H = hals_prior_nnls_torch(
#             UtM=UtM_H,
#             UtU=UtU_H,
#             prior_dict=prior_dict_H,
#             V=H,
#             gamma=gamma_H,
#             alpha=alpha_H,
#             l1_ratio=l1_ratio,
#             max_iter=max_iter_nnls,
#             tol=tol,
#             eps=eps,
#         )
#         UtM_W = H @ X_torch.T
#         UtU_W = H @ H.T
#         Wt = hals_prior_nnls_torch(
#             UtM=UtM_W,
#             UtU=UtU_W,
#             prior_dict=prior_dict_W,
#             V=W.T,
#             gamma=gamma_W,
#             alpha=alpha_W,
#             l1_ratio=l1_ratio,
#             max_iter=max_iter_nnls,
#             tol=tol,
#             eps=eps,
#         )
#         W = Wt.T
#
#         err = torch.norm(X_torch - W @ H).item()
#         if abs(prev_err - err) / (prev_err + eps) < tol:
#             break
#         prev_err = err
#
#     return W.cpu().numpy(), H.cpu().numpy()

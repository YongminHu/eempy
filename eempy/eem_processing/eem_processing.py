"""
Functions for EEM analysis
Author: Yongmin Hu (yongmin.hu@eawag.ch, yongminhu@outlook.com)
Last update: 2024-01-10
"""

from eempy.read_data import *
from eempy.utils import *
import scipy.stats as stats
import random
import pandas as pd
import numpy as np
import itertools
import string
import warnings
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
from tensorly.decomposition import parafac, non_negative_parafac
from tensorly.cp_tensor import cp_to_tensor
from tlviz.model_evaluation import core_consistency
from tlviz.outliers import compute_leverage
from tlviz.factor_tools import permute_cp_tensor
from pandas.plotting import register_matplotlib_converters
from scipy.sparse.linalg import ArpackError
from sklearn.ensemble import IsolationForest
from sklearn import svm
from typing import Optional

register_matplotlib_converters()


def process_eem_stack(eem_stack, f, *args, **kwargs):
    """
    Apply an EEM processing function across all EEMs in an EEM stack.

    Parameters
    ----------
    eem_stack: np.ndarray (3d)
        The EEM stack.
    f: callable
        The EEM processing function to appy. Available functions include all functions named in the format of
        "eem_xxx()".
    **kwargs: f function parameters
        The parameters of the EEM processing function.

    Returns
    -------
    processed_eem_stack: np.ndarray
        The processed EEM stack.
    other_outputs: tuple, optional
        If the EEM processing function have more than 1 returns, the rest of the returns will be stored in a tuple,
        where each element is the return of EEM processing function applied on one EEM.
    """
    processed_eem_stack = []
    other_outputs = []
    for i in range(eem_stack.shape[0]):
        f_output = f(eem_stack[i, :, :], *args, **kwargs)
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
    threshold：
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
    Apply Gaussian filtering to an EEM.

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


def eem_cutting(intensity, ex_range, em_range, em_min, em_max, ex_min, ex_max):
    """
    To cut the EEM.

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths.
    ex_min: float
        The lower boundary of excitation wavelength of the EEM after cutting.
    ex_max: float
        The upper boundary of excitation wavelength of the EEM after cutting.
    em_min: float
        The lower boundary of emission wavelength of the EEM after cutting.
    em_max: float
        The upper boundary of emission wavelength of the EEM after cutting.

    Returns
    -------
    intensity_cut: np.ndarray
        The cutted EEM.
    ex_range_cut: np.ndarray
        The cutted ex wavelengths.
    em_range_cut:np.ndarray
        The cutted em wavelengths.
    """
    em_min_idx = dichotomy_search(em_range, em_min)
    em_max_idx = dichotomy_search(em_range, em_max)
    ex_min_idx = dichotomy_search(ex_range, ex_min)
    ex_max_idx = dichotomy_search(ex_range, ex_max)
    intensity_cut = intensity[ex_range.shape[0]-ex_max_idx-1:ex_range.shape[0]-ex_min_idx, em_min_idx:em_max_idx+1]
    em_range_cut = em_range[em_min_idx:em_max_idx + 1]
    ex_range_cut = ex_range[ex_min_idx:ex_max_idx + 1]
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
    intensity_imputed = None
    if isinstance(fill_value, float):
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method, fill_value=fill_value)
    elif fill_value == 'linear_ex':
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method)
        for i in range(intensity_imputed.shape[1]):
            col = intensity_imputed[:, i]
            mask = np.isnan(col)
            if np.any(mask):
                interp_func = interp1d(np.flatnonzero(~mask), col[~mask], kind='linear',
                                       fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            intensity_imputed[:, i] = col
    elif fill_value == 'linear_em':
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method)
        for j in range(intensity_imputed.shape[0]):
            col = intensity_imputed[j, :]
            mask = np.isnan(col)
            if np.any(mask):
                interp_func = interp1d(np.flatnonzero(~mask), col[~mask], kind='linear',
                                       fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            intensity_imputed[j, :] = col
    return intensity_imputed


def eem_raman_normalization(intensity, blank=None, ex_range_blank=None, em_range_blank=None, from_blank=False,
                            integration_time=1, ex_lb=349, ex_ub=351, bandwidth=1800, bandwidth_type='wavenumber',
                            rsu_standard=20000, manual_rsu: Optional[float] = 1):
    """
    Normalize the EEM using the Raman scattering unit (RSU) given directly or calculated from a blank EEM.
    RSU_final = RSU_raw / (RSU_standard * integration_time).

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    blank: np.ndarray (2d)
        The blank EEM.
    ex_range_blank: np.ndarray (1d)
        The excitation wavelengths of blank.
    em_range_blank: np.ndarray (1d)
        The emission wavelengths of blank.
    from_blank: bool
        Whether to calculate the RSU from a blank. If False, manual_rsu will be used.
    integration_time: float
        The integration time of the blank measurement.
    ex_lb: float
        The lower boundary of excitation wavelength range within which the RSU is calculated.
    ex_ub: float
        The upper boundary of excitation wavelength range within which the RSU is calculated.
    bandwidth: float
        The bandwidth of Raman scattering peak.
    bandwidth_type: str, {"wavenumber", "wavelength"}
        The type of bandwidth. "wavenumber": (1/nm); "wavelength": (nm).
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
        ex_range_cut = ex_range_blank[(ex_range_blank >= ex_lb) & (ex_range_blank <= ex_ub)]
        rsu_tot = 0
        for ex in ex_range_cut.tolist():
            if bandwidth_type == 'wavenumber':
                em_target = -ex / (0.00036 * ex - 1)
                wn_target = 10000000 / em_target
                em_lb = 10000000 / (wn_target + bandwidth)
                em_rb = 10000000 / (wn_target - bandwidth)
                rsu, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
                                                     ex_min=ex, ex_max=ex, em_min=em_lb, em_max=em_rb)
            elif bandwidth_type == 'wavelength':
                rsu, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
                                                     ex_min=ex, ex_max=ex, em_min=ex - bandwidth, em_max=ex + bandwidth)
            else:
                raise ValueError("'bandwidth_type' should be either 'wavenumber' or 'wavelength'.")
            rsu_tot += rsu
    rsu_final = rsu_tot / (integration_time * rsu_standard)
    intensity_normalized = intensity / rsu_final
    return intensity_normalized, rsu_final


def eem_raman_masking(intensity, ex_range, em_range, width=5, method='linear', axis='grid'):
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
    method: str, {"linear", "cubic", "nan"}
        The method used to interpolate the Raman scattering.
    axis: str, {"ex", "em", "grid"}
        The axis along which the Raman scattering is interpolated. "ex": interpolation is conducted along the excitation
        wavelength; "em": interpolation is conducted along the emission wavelength; "grid": interpolation is conducted
        on the 2D grid of both excitation and emission wavelengths.

    Returns
    -------
    intensity_masked: np.ndarray
        The EEM with Raman scattering interpolated.
    raman_mask: np.ndarray
        Indicate the pixels that are interpolated. 0: pixel is interpolated; 1: pixel is not interpolated.
    """
    intensity_masked = np.array(intensity)
    raman_mask = np.ones(intensity.shape)
    lambda_em = -ex_range / (0.00036 * ex_range - 1)
    tol_emidx = int(np.round(width / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em[s] <= em_range[0] <= lambda_em[s] + width:
            emidx = dichotomy_search(em_range, lambda_em[s] + width)
            raman_mask[exidx, 0: emidx + 1] = 0
        elif lambda_em[s] - width <= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em[s])
            raman_mask[exidx, 0: emidx + tol_emidx + 1] = 0
        else:
            emidx = dichotomy_search(em_range, lambda_em[s] - width)
            raman_mask[exidx, emidx: emidx + 2 * tol_emidx + 1] = 0

    if method == 'nan':
        intensity_masked[np.where(raman_mask == 0)] = np.nan
    else:
        if axis == 'ex':
            for j in range(0, intensity.shape[1]):
                try:
                    x = np.flipud(ex_range)[np.where(raman_mask[:, j] == 1)]
                    y = intensity_masked[:, j][np.where(raman_mask[:, j] == 1)]
                    f1 = interp1d(x, y, kind=method, fill_value='extrapolate')
                    y_predict = f1(np.flipud(ex_range))
                    intensity_masked[:, j] = y_predict
                except ValueError:
                    continue

        if axis == 'em':
            for i in range(0, intensity.shape[0]):
                try:
                    x = em_range[np.where(raman_mask[i, :] == 1)]
                    y = intensity_masked[i, :][np.where(raman_mask[i, :] == 1)]
                    f1 = interp1d(x, y, kind=method, fill_value='extrapolate')
                    y_predict = f1(em_range)
                    intensity_masked[i, :] = y_predict
                except ValueError:
                    continue

        if axis == 'grid':
            old_nan = np.isnan(intensity)
            intensity_masked[np.where(raman_mask == 0)] = np.nan
            intensity_masked = eem_nan_imputing(intensity_masked, ex_range, em_range, method=method)
            # restore the nan values in non-raman-scattering region
            intensity_masked[old_nan] = np.nan
    return intensity_masked, raman_mask


def eem_rayleigh_masking(intensity, ex_range, em_range, width_o1=15, width_o2=15,
                         interpolation_axis_o1='grid', interpolation_axis_o2='grid', interpolation_method_o1='zero',
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
    interpolation_axis_o1: str, {"ex", "em", "grid"}
        The axis along which the 1st order Rayleigh scattering is interpolated. "ex": interpolation is conducted along
        the excitation wavelength; "em": interpolation is conducted along the emission wavelength; "grid": interpolation
        is conducted on the 2D grid of both excitation and emission wavelengths.
    interpolation_axis_o2: str, {"ex", "em", "grid"}
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
    lambda_em_o1 = ex_range
    tol_emidx_o1 = int(np.round(width_o1 / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em_o1[s] <= em_range[0] <= lambda_em_o1[s] + width_o1:
            emidx = dichotomy_search(em_range, lambda_em_o1[s] + width_o1)
            rayleigh_mask_o1[exidx, 0:emidx + 1] = 0
        elif lambda_em_o1[s] - width_o1 <= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em_o1[s])
            rayleigh_mask_o1[exidx, 0: emidx + tol_emidx_o1 + 1] = 0
        else:
            emidx = dichotomy_search(em_range, lambda_em_o1[s])
            rayleigh_mask_o1[exidx, emidx: emidx + tol_emidx_o1 + 1] = 0
            intensity_masked[exidx, 0: emidx] = 0
    lambda_em_o2 = ex_range * 2
    tol_emidx_o2 = int(np.round(width_o2 / (em_range[1] - em_range[0])))
    for s in range(0, intensity_masked.shape[0]):
        exidx = ex_range.shape[0] - s - 1
        if lambda_em_o2[s] <= em_range[0] <= lambda_em_o2[s] + width_o2:
            emidx = dichotomy_search(em_range, lambda_em_o2[s] + width_o2)
            rayleigh_mask_o2[exidx, 0:emidx + 1] = 0
        elif lambda_em_o2[s] - width_o2 <= em_range[0]:
            emidx = dichotomy_search(em_range, lambda_em_o2[s])
            rayleigh_mask_o2[exidx, 0: emidx + tol_emidx_o2 + 1] = 0
        else:
            emidx = dichotomy_search(em_range, lambda_em_o2[s] - width_o2)
            rayleigh_mask_o2[exidx, emidx: emidx + 2 * tol_emidx_o2 + 1] = 0

    for axis, itp, mask in zip([interpolation_axis_o1, interpolation_axis_o2],
                               [interpolation_method_o1, interpolation_method_o2],
                               [rayleigh_mask_o1, rayleigh_mask_o2]):
        if itp == 'zero':
            intensity_masked[np.where(mask == 0)] = 0
        elif itp == 'nan':
            intensity_masked[np.where(mask == 0)] = np.nan
        else:
            if axis == 'ex':
                for j in range(0, intensity.shape[1]):
                    try:
                        x = np.flipud(ex_range)[np.where(mask[:, j] == 1)]
                        y = intensity_masked[:, j][np.where(mask[:, j] == 1)]
                        f1 = interp1d(x, y, kind=itp, fill_value='extrapolate')
                        y_predict = f1(np.flipud(ex_range))
                        intensity_masked[:, j] = y_predict
                    except ValueError:
                        continue
            if axis == 'em':
                for i in range(0, intensity.shape[0]):
                    try:
                        x = em_range[np.where(mask[i, :] == 1)]
                        y = intensity_masked[i, :][np.where(mask[i, :] == 1)]
                        f1 = interp1d(x, y, kind=itp, fill_value='extrapolate')
                        y_predict = f1(em_range)
                        intensity_masked[i, :] = y_predict
                    except ValueError:
                        continue
            if axis == 'grid':
                old_nan = np.isnan(intensity)
                old_nan_o1 = np.isnan(intensity_masked)
                intensity_masked[np.where(mask == 0)] = np.nan
                intensity_masked = eem_nan_imputing(intensity_masked, ex_range, em_range, method=itp)
                # restore the nan values in non-raman-scattering region
                intensity_masked[old_nan] = np.nan
                intensity_masked[old_nan_o1] = np.nan
    return intensity_masked, rayleigh_mask_o1, rayleigh_mask_o2


def eem_ife_correction(intensity, ex_range, em_range, absorbance, ex_range_abs, cuvette_length=1):
    """
    Correct the inner filter effect (IFE).

    Parameters
    ----------
    intensity: np.ndarray (2d)
        The EEM.
    ex_range: np.ndarray (1d)
        The excitation wavelengths of EEM.
    em_range: np.ndarray (1d)
        The emission wavelengths of EEM.
    absorbance: np.ndarray (1d)
        The absorbance.
    ex_range_abs: np.ndarray (1d)
        The excitation wavelengths of absorbance.
    cuvette_length: float
        The length of cuvette in measurement.

    Returns
    -------
    intensity_corrected: np.ndarray
        The corrected EEM.
    """
    f1 = interp1d(ex_range_abs, absorbance, kind='linear', bounds_error=False, fill_value='extrapolate')
    absorbance_ex = np.fliplr(np.array([f1(ex_range)]))
    absorbance_em = np.array([f1(em_range)])
    ife_factors = 10 ** (cuvette_length * (absorbance_ex.T.dot(np.ones(absorbance_em.shape)) +
                                           np.ones(absorbance_ex.shape).T.dot(absorbance_em)))
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
    intensity_cut, em_range_cut, ex_range_cut = eem_cutting(intensity, ex_range, em_range,
                                                            em_min=em_min, em_max=em_max,
                                                            ex_min=ex_min, ex_max=ex_max)
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
    interp = RegularGridInterpolator((ex_range_old[::-1], em_range_old), intensity, method=method)
    x, y = np.meshgrid(em_range_new, ex_range_new[::-1])
    xx = x.flatten()
    yy = y.flatten()
    coordinates_new = np.concatenate([xx[:, np.newaxis], yy[:, np.newaxis]], axis=1)
    intensity_interpolated = interp(coordinates_new).reshape(ex_range_new.shape[0], em_range_new.shape[0])
    return intensity_interpolated


def eems_tf_normalization(eem_stack):
    """
    Normalize EEMs by the total fluorescence of each EEM.

    Parameters
    ----------
    eem_stack: np.ndarray (3d)
        The EEM stack.

    Returns
    -------
    eem_stack_normalized: np.ndarray
        The normalized EEM stack.
    weights: np.ndarray
        The total fluorescence of each EEM.
    """
    tf_list = []
    for i in range(eem_stack.shape[0]):
        tf = eem_stack[i].sum()
        tf_list.append(tf)
    weights = np.array(tf_list) / np.mean(tf_list)
    eem_stack_normalized = eem_stack / weights[:, np.newaxis, np.newaxis]
    return eem_stack_normalized, weights


def eems_outlier_detection_if(eem_stack, ex_range, em_range, tf_normalization=True, grid_size=(10, 10),
                              contamination=0.02):
    """
    tells whether it should be considered as an inlier according to the fitted model. +1: inlier; -1: outlier

    Parameters
    ----------
    eem_stack: np.ndarray (3d)
        The EEM stack.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths
    """
    if tf_normalization:
        eem_stack, _ = eems_tf_normalization(eem_stack)
    em_range_new = np.arange(em_range[0], em_range[-1], grid_size[1])
    ex_range_new = np.arange(ex_range[0], ex_range[-1], grid_size[0])
    eem_stack_interpolated = process_eem_stack(eem_stack, eem_interpolation, ex_range, em_range, ex_range_new,
                                               em_range_new)
    eem_stack_unfold = eem_stack_interpolated.reshape(eem_stack_interpolated.shape[0],
                                                      eem_stack_interpolated.shape[1] * eem_stack_interpolated.shape[2])
    eem_stack_unfold = np.nan_to_num(eem_stack_unfold)
    clf = IsolationForest(random_state=0, n_estimators=200, contamination=contamination)
    clf.fit(eem_stack_unfold)
    label = clf.predict(eem_stack_unfold)
    return label


def eems_outlier_detection_ocs(eem_stack, ex_range, em_range, tf_normalization=True, grid_size=(10, 10), nu=0.02,
                               kernel="rbf", gamma=10000):
    """
    tells whether it should be considered as an inlier according to the fitted model. +1: inlier; -1: outlier

    Parameters
    ----------
    eem_stack: np.ndarray (3d)
        The EEM stack.
    ex_range: np.ndarray (1d)
        The excitation wavelengths.
    em_range: np.ndarray (1d)
        The emission wavelengths

    """
    if tf_normalization:
        eem_stack, _ = eems_tf_normalization(eem_stack)
    em_range_new = np.arange(em_range[0], em_range[-1], grid_size[1])
    ex_range_new = np.arange(ex_range[0], ex_range[-1], grid_size[0])
    eem_stack_interpolated = process_eem_stack(eem_stack, eem_interpolation, ex_range, em_range, ex_range_new,
                                               em_range_new)
    eem_stack_unfold = eem_stack_interpolated.reshape(eem_stack_interpolated.shape[0],
                                                      eem_stack_interpolated.shape[1] * eem_stack_interpolated.shape[2])
    eem_stack_unfold = np.nan_to_num(eem_stack_unfold)
    clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(eem_stack_unfold)
    label = clf.predict(eem_stack_unfold)
    return label


def eems_fit_components(eem_stack, component_stack, fit_intercept=False):
    assert eem_stack.shape[1:] == component_stack.shape, "EEM and component have different shapes"
    score_sample = []
    fmax_sample = []
    max_values = np.amax(component_stack, axis=(1, 2))
    eem_stack_pred = np.zeros(eem_stack.shape)
    for i in range(eem_stack.shape[0]):
        y_true = eem_stack[i].reshape([-1])
        x = component_stack.reshape([component_stack.shape[0], -1]).T
        reg = LinearRegression(fit_intercept=fit_intercept)
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
    ref: np.ndarray (1d) or None
        Optional. The reference data, e.g., the COD of each sample. It should have a length equal to the number of
        samples in the eem_stack.
    index: list or None
        Optional. The index used to label each sample. The number of elements in the list should equal the number
        of samples in the eem_stack.
    """

    def __init__(self, eem_stack: np.ndarray, ex_range: np.ndarray, em_range: np.ndarray,
                 ref: Optional[np.ndarray] = None, index: Optional[list] = None):

        # ------------------parameters--------------------
        # The Em/Ex ranges should be sorted in ascending order
        self.eem_stack = eem_stack
        self.ex_range = ex_range
        self.em_range = em_range
        self.ref = ref
        self.index = index
        self.extent = (self.em_range.min(), self.em_range.max(), self.ex_range.min(), self.ex_range.max())

    # --------------------EEM dataset features--------------------
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
            qualified_pixel_proportion = np.count_nonzero(rel_std < threshold) / np.count_nonzero(~np.isnan(rel_std))
            print("The proportion of pixels with relative STD < {t}: ".format(t=threshold),
                  qualified_pixel_proportion)
        return rel_std

    def std(self):
        return np.std(self.eem_stack, axis=0)

    def total_fluorescence(self):
        return self.eem_stack.sum(axis=(1, 2))

    def regional_integration(self, em_boundary, ex_boundary):
        integrations, _ = process_eem_stack(self.eem_stack, eem_regional_integration, ex_range=self.ex_range,
                                            em_range=self.em_range, em_boundary=em_boundary, ex_boundary=ex_boundary)
        return integrations

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
        if self.index:
            fi = pd.DataFrame(fi, index=self.index)
        else:
            fi = pd.DataFrame(fi, index=np.arange(fi.shape[0]))
        ex_actual = self.ex_range[ex_idx]
        em_actual = self.em_range[em_idx]
        return fi, ex_actual, em_actual

    def correlation(self):
        """
        Analyze the correlation between reference and fluorescence intensity at each pair of ex/em.

        Returns
        -------
        corr_dict: dict
            A dictionary containing multiple correlation evaluation metrics.

        """
        m = self.eem_stack
        x = self.ref
        x = x.reshape(m.shape[0], 1)
        w, b, r2, pc, pc_p, sc, sc_p = [np.full((m.shape[1], m.shape[2]), fill_value=np.nan)] * 7
        e = np.full(m.shape, fill_value=np.nan)
        for i in range(m.shape[1]):
            for j in range(m.shape[2]):
                try:
                    y = (m[:, i, j])
                    reg = LinearRegression()
                    reg.fit(x, y)
                    w[i, j] = reg.coef_
                    b[i, j] = reg.intercept_
                    r2[i, j] = reg.score(x, y)
                    e[:, i, j] = reg.predict(x) - y
                    pc[i, j], pc_p[i, j] = stats.pearsonr(x, y)
                    sc[i, j], sc_p[i, j] = stats.spearmanr(x, y)
                except ValueError:
                    pass
        corr_dict = {'slope': w, 'intercept': b, 'r_square': r2, 'linear regression residual': e,
                     'Pearson corr. coef.': pc, 'Pearson corr. coef. p-value': pc_p, 'Spearman corr. coef.': sc,
                     'Spearman corr. coef. p-value': sc_p}
        return corr_dict

    # -----------------EEM dataset processing methods-----------------

    def threshold_masking(self, threshold, mask_type='greater', copy=True):
        eem_stack_masked, masks = process_eem_stack(self.eem_stack, eem_threshold_masking, threshold=threshold,
                                                    mask_type=mask_type)
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

    def nan_imputing(self, method='linear', fill_value='linear_ex', prior_mask=None, copy=True):
        eem_stack_imputed = process_eem_stack(self.eem_stack, eem_nan_imputing, ex_range=self.ex_range,
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

    def outlier_detection_if(self, tf_normalization=True, grid_size=(10, 10), contamination=0.02, deletion=False):
        labels = eems_outlier_detection_if(eem_stack=self.eem_stack, ex_range=self.ex_range, em_range=self.em_range,
                                           tf_normalization=tf_normalization, grid_size=grid_size,
                                           contamination=contamination)
        if deletion:
            self.eem_stack = self.eem_stack[labels != -1]
            self.ref = self.ref[labels != -1]
            self.index = [idx for i, idx in enumerate(self.index) if labels[i] != -1]
        return labels

    def outlier_detection_ocs(self, tf_normalization=True, grid_size=(10, 10), nu=0.02, kernel='rbf', gamma=10000,
                              deletion=False):
        labels = eems_outlier_detection_ocs(eem_stack=self.eem_stack, ex_range=self.ex_range, em_range=self.em_range,
                                            tf_normalization=tf_normalization, grid_size=grid_size, nu=nu,
                                            kernel=kernel, gamma=gamma)
        if deletion:
            self.eem_stack = self.eem_stack[labels != -1]
            self.ref = self.ref[labels != -1]
            self.index = [idx for i, idx in enumerate(self.index) if labels[i] != -1]
        return labels

    def splitting(self, n_split, rule: str = 'random'):
        """
        To split the EEM dataset and form multiple sub-datasets.

        Parameters
        ----------
        n_split: int
            The number of splits.
        rule: str, {'random', 'sequential'}
            If 'random' is passed, the split will be generated randomly. If 'sequential' is passed, the dataset will be
            split according to index order.

        Returns
        -------
        model_list: list
            A list of sub-datasets. Each of them is an EEMDataset object.
        """
        idx_eems = [i for i in range(self.eem_stack.shape[0])]
        model_list = []
        if rule == 'random':
            random.shuffle(idx_eems)
            idx_splits = np.array_split(idx_eems, n_split)
        elif rule == 'sequential':
            idx_splits = np.array_split(idx_eems, n_split)
        else:
            raise ValueError("'rule' should be either 'random' or 'sequential'")
        for split in idx_splits:
            if self.ref:
                ref = np.array([self.ref[i] for i in split])
            else:
                ref = None
            if self.index:
                index = [self.index[i] for i in split]
            else:
                index = None
            m = EEMDataset(eem_stack=np.array([self.eem_stack[i] for i in split]), ex_range=self.ex_range,
                           em_range=self.em_range, ref=ref, index=index)
            model_list.append(m)
        return model_list


def combine_eem_datasets(list_eem_datasets):
    eem_stack_combined = []
    ref_combined = []
    index_combined = []
    ex_range_0 = list_eem_datasets[0].ex_range
    em_range_0 = list_eem_datasets[0].em_range
    for d in list_eem_datasets:
        eem_stack_combined.append(d.eem_stack)
        ref_combined.append(d.ref)
        index_combined = index_combined + d.index
        if not np.array_equal(d.ex_range, ex_range_0) or not np.array_equal(d.em_range, em_range_0):
            warnings.warn(
                'ex_range and em_range of the datasets must be identical. If you want to combine EEM datasets '
                'having different ex/em ranges, please consider unify the ex/em ranges using the interpolation() '
                'method of EEMDataset object')
    eem_stack_combined = np.concatenate(eem_stack_combined, axis=0)
    ref_combined = np.concatenate(ref_combined, axis=0)
    eem_dataset_combined = EEMDataset(eem_stack=eem_stack_combined, ex_range=ex_range_0, em_range=em_range_0,
                                      ref=ref_combined, index=index_combined)
    return eem_dataset_combined


class PARAFAC:
    """
    PARAFAC model

    Parameters
    ----------
    rank: int
        The number of components
    non_negativity: bool
        Whether to apply the non-negativity constraint
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
    score
    ex_loadings
    em_loadings
    fmax
    component_stack
    cptensors
    eem_stack_train
    eem_stack_reconstructed
    ex_range
    em_range
    """

    def __init__(self, rank, non_negativity=True, init='svd', tf_normalization=True,
                 loadings_normalization: Optional[str] = 'sd', sort_em=True):

        # ----------parameters--------------
        self.rank = rank
        self.non_negativity = non_negativity
        self.init = init
        self.tf_normalization = tf_normalization
        self.loadings_normalization = loadings_normalization
        self.sort_em = sort_em

        # -----------attributes---------------
        self.score = None
        self.ex_loadings = None
        self.em_loadings = None
        self.fmax = None
        self.component_stack = None
        self.cptensors = None
        self.eem_stack_train = None
        self.eem_stack_reconstructed = None
        self.ex_range = None
        self.em_range = None

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
            _, tf_weights = eem_dataset.tf_normalization(copy=False)
        try:
            if not self.non_negativity:
                if np.isnan(eem_dataset.eem_stack).any():
                    mask = np.where(np.isnan(eem_dataset.eem_stack), 0, 1)
                    cptensors = parafac(eem_dataset.eem_stack, rank=self.rank, mask=mask, init=self.init)
                else:
                    cptensors = parafac(eem_dataset.eem_stack, rank=self.rank, init=self.init)
            else:
                if np.isnan(eem_dataset.eem_stack).any():
                    mask = np.where(np.isnan(eem_dataset.eem_stack), 0, 1)
                    cptensors = non_negative_parafac(eem_dataset.eem_stack, rank=self.rank, mask=mask, init=self.init)
                else:
                    cptensors = non_negative_parafac(eem_dataset.eem_stack, rank=self.rank, init=self.init)
        except ArpackError:
            print(
                "PARAFAC failed possibly due to the presence of patches of nan values. Please consider cut or "
                "interpolate the nan values.")
        a, b, c = cptensors[1]
        component_stack = np.zeros([self.rank, b.shape[0], c.shape[0]])
        for r in range(self.rank):

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
                maxc = c[:, r].min()
                b[:, r] = b[:, r] / maxb
                c[:, r] = c[:, r] / maxc
                a[:, r] = a[:, r] * maxb * maxc
            component = np.array([b[:, r]]).T.dot(np.array([c[:, r]]))
            component_stack[r, :, :] = component

        if self.tf_normalization:
            a = np.multiply(a, tf_weights[:, np.newaxis])
        score = pd.DataFrame(a)
        fmax = a * component_stack.max(axis=(1, 2))
        ex_loadings = pd.DataFrame(np.flipud(b), index=eem_dataset.ex_range)
        em_loadings = pd.DataFrame(c, index=eem_dataset.em_range)
        if self.sort_em:
            em_peaks = [c[1] for c in em_loadings.idxmax()]
            peak_rank = list(enumerate(stats.rankdata(em_peaks)))
            order = [i[0] for i in sorted(peak_rank, key=lambda x: x[1])]
            component_stack = component_stack[order]
            ex_loadings = pd.DataFrame({'component {r} ex loadings'.format(r=i + 1): ex_loadings.iloc[:, order[i]]
                                        for i in range(self.rank)})
            em_loadings = pd.DataFrame({'component {r} em loadings'.format(r=i + 1): em_loadings.iloc[:, order[i]]
                                        for i in range(self.rank)})
            score = pd.DataFrame({'component {r} score'.format(r=i + 1): score.iloc[:, order[i]]
                                  for i in range(self.rank)})
            fmax = pd.DataFrame({'component {r} fmax'.format(r=i + 1): fmax[:, order[i]]
                                 for i in range(self.rank)})
        else:
            column_labels = ['component {r}'.format(r=i + 1) for i in range(self.rank)]
            ex_loadings.columns = column_labels
            em_loadings.columns = column_labels
            score.columns = column_labels
            fmax = pd.DataFrame(fmax, columns=['component {r}'.format(r=i + 1) for i in range(self.rank)])

        ex_loadings.index = eem_dataset.ex_range.tolist()
        em_loadings.index = eem_dataset.em_range.tolist()

        if eem_dataset.index:
            score.index = eem_dataset.index
            fmax.index = eem_dataset.index
        else:
            score.index = [i + 1 for i in range(a.shape[0])]
            fmax.index = [i + 1 for i in range(a.shape[0])]

        self.score = score
        self.ex_loadings = ex_loadings
        self.em_loadings = em_loadings
        self.fmax = fmax
        self.component_stack = component_stack
        self.cptensors = cptensors
        self.eem_stack_train = eem_dataset.eem_stack
        self.ex_range = eem_dataset.ex_range
        self.em_range = eem_dataset.em_range
        self.eem_stack_reconstructed = cp_to_tensor(cptensors)
        return self

    def predict(self, eem_dataset: EEMDataset, fit_intercept=False):
        """
        Predict the score and Fmax of a given EEM dataset using the component fitted. This method can be applied to a
        new EEM dataset independent of the one used in PARAFAC model fitting.

        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM.
        fit_intercept: bool
            Whether to calculate the intercept.

        Returns
        -------
        score_sample: pd.DataFrame
            The fitted score.
        fmax_sample: pd.DataFrame
            The fitted Fmax.
        eem_stack_pred: np.ndarray (3d)
            The EEM dataset reconstructed.
        """
        score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset.eem_stack, self.component_stack,
                                                                        fit_intercept=fit_intercept)
        score_sample = pd.DataFrame(score_sample, index=eem_dataset.index, columns=self.score.columns)
        fmax_sample = pd.DataFrame(fmax_sample, index=eem_dataset.index, columns=self.fmax.columns)
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
        for r in range(self.rank):
            max_index = np.unravel_index(np.argmax(self.component_stack[r, :, :]), self.component_stack[r, :, :].shape)
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
        Calculate the explained variance of the established PARAFAC model

        Returns
        -------
        ev: float
            the explained variance
        """
        y_train = self.eem_stack_train.reshape(-1)
        y_pred = self.eem_stack_reconstructed.reshape(-1)
        ev = 100 * (1 - np.var(y_pred - y_train) / np.var(y_train))
        return ev

    def core_consistency(self):
        """
        Calculate the core consistency of the established PARAFAC model

        Returns
        -------
        ev: float
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
        elif mode == 'em':
            lvr = compute_leverage(self.em_loadings)
        elif mode == 'sample':
            lvr = compute_leverage(self.score)
        else:
            raise ValueError("'mode' should be 'ex' or 'em' or 'sample'.")
        lvr.index = lvr.index.set_levels(['leverage of {m}'.format(m=mode)] * len(lvr.index.levels[0]), level=0)
        return lvr

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
        rmse = pd.DataFrame(sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels), index=self.score.index)
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
                                      index=self.score.index)
        return normalized_sse

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
        summary = pd.concat([self.score, self.fmax, lvr, rmse, normalized_rmse], axis=1)
        return summary

    def export(self, filepath, name='', creator='', date='', email='', doi='', reference='', unit='', toolbox='',
               fluorometer='', nSample='', decomposition_method='', validation='', dataset_calibration='',
               preprocess='', sources='', description=''):
        """
        Export the PARAFAC model to a text file that can be uploaded to the online PARAFAC model database Openfluor
        (https://openfluor.lablicate.com/#).

        Parameters
        ----------
        filepath: str
            Location of the saved text file. Please specify the ".csv" extension.
        name
        creator
        date
        email
        doi
        reference
        unit
        toolbox
        fluorometer
        nSample
        decomposition_method
        validation
        dataset_calibration
        preprocess
        sources
        description

        Returns
        -------
        info_dict: dict
            A dictionary containing the information of the PARAFAC model.

        """
        info_dict = {'name': name, 'creator': creator, 'email': email, 'doi': doi, 'reference': reference,
                     'unit': unit, 'toolbox': toolbox, 'date': date, 'fluorometer': fluorometer, 'nSample': nSample,
                     'dateset_calibration': dataset_calibration, 'preprocess': preprocess,
                     'decomposition_method': decomposition_method,
                     'validation': validation, 'sources': sources, 'description': description}
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
                ex1_aligned, ex2_aligned = dynamic_time_warping(loadings1[:, n1], loadings2[:, n2])
            else:
                ex1_aligned, ex2_aligned = [loadings1[:, n1], loadings2[:, n2]]
            m_sim[n1, n2] = stats.pearsonr(ex1_aligned, ex2_aligned)[0]
    m_sim = pd.DataFrame(m_sim, index=['model1 C{i}'.format(i=i + 1) for i in range(loadings1.shape[1])],
                         columns=['model2 C{i}'.format(i=i + 1) for i in range(loadings2.shape[1])])
    return m_sim


def align_parafac_components(models_dict: dict, ex_ref: pd.DataFrame, em_ref: pd.DataFrame, wavelength_alignment=False):
    """
    Align the components of PARAFAC models according to given reference ex/em loadings so that similar components
    are labelled by the same name.

    Parameters
    ----------
    models_dict: dict
        Dictionary of PARAFAC object. The models to be aligned.
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
        m_sim_ex = loadings_similarity(model, ex_ref, wavelength_alignment=wavelength_alignment)
        m_sim_em = loadings_similarity(model, em_ref, wavelength_alignment=wavelength_alignment)
        m_sim = (m_sim_ex + m_sim_em) / 2
        ex_var, em_var = (model.ex_loadings, model.em_loadings)
        matched_index = []
        m_sim_copy = m_sim.copy()
        if ex_var.shape[1] <= ex_ref.shape[1]:
            for n_var in range(ex_var.shape[1]):
                max_index = np.argmax(m_sim[n_var, :])
                while max_index in matched_index:
                    m_sim_copy[n_var, max_index] = 0
                    max_index = np.argmax(m_sim_copy[n_var, :])
                matched_index.append(max_index)
            component_labels_var = [component_labels_ref[i] for i in matched_index]
            permutation = get_indices_smallest_to_largest(matched_index)
        else:
            for n_ref in range(ex_ref.shape[1]):
                max_index = np.argmax(m_sim[:, n_ref])
                while max_index in matched_index:
                    m_sim_copy[max_index, n_ref] = 0
                    max_index = np.argmax(m_sim_copy[:, n_ref])
                matched_index.append(max_index)
            non_ordered_index = list(set([i for i in range(ex_var.shape[1])]) - set(matched_index))
            permutation = matched_index + non_ordered_index
            component_labels_ref_extended = component_labels_ref + ['O{i}'.format(i=i + 1) for i in
                                                                    range(len(non_ordered_index))]
            component_labels_var = [0] * len(permutation)
            for i, nc in enumerate(permutation):
                component_labels_var[nc] = component_labels_ref_extended[i]
        model.score.columns, model.ex_loadings.columns, model.em_loadings.columns, model.fmax.columns = (
                [component_labels_var] * 4)
        model.score = model.score.iloc[:, permutation]
        model.ex_loadings = model.ex_loadings.iloc[:, permutation]
        model.em_loadings = model.em_loadings.iloc[:, permutation]
        model.fmax = model.fmax.iloc[:, permutation]
        model.component_stack = model.component_stack[permutation, :, :]
        model.cptensor = permute_cp_tensor(model.cptensor, permutation)
        models_dict_new[model_label] = model
        return models_dict_new


class SplitValidation:
    """
    Conduct PARAFAC model validation by evaluating the consistency of PARAFAC models established on EEM sub-datasets.
    """

    def __init__(self, rank, n_split, combination_size, rule, similarity_metric='TCC', non_negativity=True,
                 tf_normalization=True):
        """
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
        """
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
        model_complete = PARAFAC(rank=self.rank, non_negativity=self.non_negativity,
                                 tf_normalization=self.tf_normalization)
        model_complete.fit(eem_dataset=eem_dataset)
        sims_ex, sims_em, models, subsets = ({}, {}, {}, {})

        for e, c in zip(elements, codes):
            label = ''.join(c)
            subdataset = combine_eem_datasets([split_set[i] for i in e])
            model_subdataset = PARAFAC(rank=self.rank, non_negativity=self.non_negativity,
                                       tf_normalization=self.tf_normalization)
            model_subdataset.fit(subdataset)

            models[label] = model_subdataset
            subsets[label] = subdataset
        models = align_parafac_components(models, model_complete.ex_loadings, model_complete.em_loadings)

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
        similarities_ex = pd.DataFrame(similarities_ex, columns=['C{i}'.format(i=i + 1) for i in range(self.rank)])
        similarities_em = pd.DataFrame(similarities_em, columns=['C{i}'.format(i=i + 1) for i in range(self.rank)])
        return similarities_ex, similarities_em


class KPARAFACs:

    def __init__(self, rank, n_clusters, dropout_rate=0.8, max_iter=20, tol=0.001, non_negativity=True, init='svd',
                 tf_normalization=True, loadings_normalization: Optional[str] = 'sd', sort_em=True):

        # -----------Parameters-------------
        self.rank = rank
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.dropout_rate = dropout_rate
        self.tol = tol
        self.non_negativity = non_negativity
        self.init = init
        self.tf_normalization = tf_normalization
        self.loadings_normalization = loadings_normalization
        self.sort_em = sort_em

        # ----------Attributes-------------
        self.label_history = None
        self.error_history = None
        self.consensus_matrix = None

    def fit(self, eem_dataset: EEMDataset):

        # -------Define functions for each step-------
        def estimation(sub_datasets: dict):
            cluster_specific_models ={}
            for label, d in sub_datasets.items():
                model = PARAFAC(rank=self.rank, non_negativity=self.non_negativity, init=self.init,
                                  tf_normalization=self.tf_normalization,
                                  loadings_normalization=self.loadings_normalization, sort_em=self.sort_em)
                model.fit(d)
                cluster_specific_models[label] = model
            return cluster_specific_models

        def maximization(cluster_specific_models: dict):
            sample_error = []
            sub_datasets = {}
            for label, m in cluster_specific_models.items():
                score_m, fmax_m, eem_stack_re_m = m.predict(eem_dataset)
                res = m.eem_stack_train - eem_stack_re_m
                n_pixels = m.eem_stack_train.shape[1] * m.eem_stack_train.shape[2]
                rmse = sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels)
                sample_error.append(rmse)
            best_model_idx = np.argmin(sample_error, axis=0)
            least_model_errors = np.min(sample_error, axis=0)
            for j, label in enumerate(cluster_specific_models.keys()):
                eem_stack_j = eem_dataset.eem_stack[np.where(best_model_idx==j)]
                if eem_dataset.ref:
                    ref_j = eem_dataset.ref[np.where(best_model_idx==j)]
                else:
                    ref_j = None
                if eem_dataset.index:
                    index_j = [eem_dataset.index[k] for k, idx in enumerate(best_model_idx) if idx == j]
                else:
                    index_j = None
                sub_dataset = EEMDataset(eem_stack=eem_stack_j, ex_range=eem_dataset.ex_range,
                                         em_range=eem_dataset.em_range, ref=ref_j, index=index_j)
                sub_datasets[label] = sub_dataset
            return sub_datasets, best_model_idx, least_model_errors

        # -------Initialization--------
        cluster_history = []
        error_history = []
        initial_sub_datasets = {}
        sub_eem_datasets = eem_dataset.splitting(n_split=self.n_clusters)
        for i, random_m in enumerate(sub_eem_datasets):
            initial_sub_datasets[i+1] = random_m

        for n in range(self.max_iter):
            # ---------The estimation step----------

            # ---------The maximization step----------




        def parafac_em_clustering(eem_stack, em_range, ex_range, rank, index, n_splits, n_iterations, plot_errors=True,
                                  metric='mse'):
            # random initialization
            sub_eem_stacks, index_classes = eem_stack_spliting(eem_stack, index, n_splits, rule='random')
            # The EM
            error_classes_iterations = []
            for n in range(n_iterations):
                component_stacks = parafac_estimation(sub_eem_stacks, em_range, ex_range, rank, index_classes)
                sub_eem_stacks, index_classes, error_classes, coef_classes = parafac_maximization(eem_stack,
                                                                                                  component_stacks,
                                                                                                  index, metric=metric)
                # remove class with the number of samples less than the number of ranks
                idx_to_remove = []
                if n < n_iterations - 1:
                    for i in range(len(index_classes)):
                        if len(index_classes[i]) < rank + 1:
                            idx_to_remove.append(i)
                    for j in sorted(idx_to_remove, reverse=True):
                        del (sub_eem_stacks[j])
                        del (index_classes[j])
                error_classes_iterations.append(error_classes)
                if n > 0:
                    if index_classes_prev == index_classes:
                        break
                index_classes_prev = index_classes.copy()
            for c in range(len(sub_eem_stacks)):
                tbl = pd.DataFrame({"labels": np.full((len(error_classes[c]),), c + 1), "metric": error_classes[c]},
                                   index=index_classes[c])
                if c > 0:
                    tbl = pd.concat([tbl_old, tbl])
                tbl_old = tbl
            tbl = tbl.sort_index()
            tbl.index.name = 'Time'
            if plot_errors:
                error_mean_classes = [[] for i in range(len(sub_eem_stacks) + 1)]
                for error_classes in error_classes_iterations:
                    for i in range(len(sub_eem_stacks)):
                        mean_error_class = np.mean(error_classes[i])
                        error_mean_classes[i].append(mean_error_class)
                    error_classes_flat = [item for sublist in error_classes for item in sublist]
                    error_mean_classes[-1].append(np.mean(error_classes_flat))
                for i in range(len(sub_eem_stacks)):
                    plt.plot(error_mean_classes[i])
                    plt.title('cluster {i}'.format(i=i))
                    plt.show()
                plt.plot(error_mean_classes[-1])
                plt.title("all samples")
                plt.show()
            return component_stacks, sub_eem_stacks, index_classes, error_classes_iterations, tbl
        return



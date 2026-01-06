"""
Functions for EEM preprocessing and post-processing
Author: Yongmin Hu (yongminhu@outlook.com)
Last update: 2025-12
"""

from eempy.utils import *
import scipy.stats as stats
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tensorly.tenalg import unfolding_dot_khatri_rao
from tlviz.factor_tools import permute_cp_tensor
from typing import Optional


def process_eem_stack(eem_stack, f, **kwargs):
    """
    Apply an EEM processing function across all EEMs in an EEM stack.

    Parameters
    ----------
    eem_stack : np.ndarray (3d)
        The EEM stack.
    f : callable
        The EEM processing function to apply. Available functions include all functions named in the format of
        "eem_xxx()".
    **kwargs : f function parameters
        The parameters of the EEM processing function.

    Returns
    -------
    processed_eem_stack : np.ndarray
        The processed EEM stack.
    other_outputs : tuple, optional
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
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    threshold : float or int
        The intensity threshold value.
    fill : float or int
        The value to use for masked data points.
    mask_type : str, {"greater", "smaller"}
        Specifies whether to mask intensities 'greater' than or 'smaller' than the threshold.

    Returns
    -------
    intensity_masked : np.ndarray
        The EEM with masked values replaced by the `fill` value.
    mask : np.ndarray
        The mask matrix. 1.0: unmasked area; np.nan: masked area.
    """
    intensity_masked = intensity.copy().astype(float)
    if mask_type == 'smaller':
        condition = intensity < threshold
    elif mask_type == 'greater':
        condition = intensity > threshold
    else:
        raise ValueError("mask_type must be 'greater' or 'smaller'")
    mask = np.ones(intensity.shape)
    mask[condition] = np.nan
    intensity_masked[condition] = fill
    return intensity_masked, mask


def eem_region_masking(intensity, ex_range, em_range, ex_min=230, ex_max=500, em_min=250, em_max=810, fill='nan'):
    """
    Mask the fluorescence intensities in a specified rectangular region.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    ex_min : float
        The lower boundary of the excitation wavelength for the masked area.
    ex_max : float
        The upper boundary of the excitation wavelength for the masked area.
    em_min : float
        The lower boundary of the emission wavelength for the masked area.
    em_max : float
        The upper boundary of the emission wavelength for the masked area.
    fill : str, {"nan", "zero"}
        Specifies how to fill the masked area.

    Returns
    -------
    intensity_masked : np.ndarray
        The masked EEM.
    mask : np.ndarray
        The mask matrix. 1.0: unmasked area; 0.0: masked area.
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
    else:
        raise ValueError(f"fill option '{fill}' not recognized. Use 'nan' or 'zero'.")
    return intensity_masked, mask


def eem_gaussian_filter(intensity, sigma=1, truncate=3):
    """
    Apply Gaussian filtering to an EEM. Reference: scipy.ndimage.gaussian_filter

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    sigma : float
        Standard deviation for Gaussian kernel.
    truncate : float
        Truncate the filter at this many standard deviations.

    Returns
    -------
    intensity_filtered : np.ndarray
        The filtered EEM.
    """
    intensity_filtered = gaussian_filter(intensity, sigma=sigma, truncate=truncate)
    return intensity_filtered


def eem_median_filter(intensity, window_size=(3, 3), mode='reflect'):
    """
    Apply a median filter to an EEM. This is a wrapper for scipy.ndimage.median_filter.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    window_size : tuple of two integers
        The shape of the filter window, e.g., (height, width). For example,
        (3, 5) specifies a window of 3 pixels along axis 0 and 5 pixels along axis 1.
    mode : str, {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
        Determines how the array is extended beyond its boundaries.
        The default is 'reflect'.

    Returns
    -------
    intensity_filtered : np.ndarray
        The filtered EEM.
    """
    intensity_filtered = median_filter(intensity, size=window_size, mode=mode)
    return intensity_filtered


def eem_cutting(intensity, ex_range_old, em_range_old, ex_min_new, ex_max_new, em_min_new, em_max_new):
    """
    To cut the EEM.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    ex_min_new : float
        The lower boundary of excitation wavelength of the EEM after cutting.
    ex_max_new : float
        The upper boundary of excitation wavelength of the EEM after cutting.
    em_min_new : float
        The lower boundary of emission wavelength of the EEM after cutting.
    em_max_new : float
        The upper boundary of emission wavelength of the EEM after cutting.

    Returns
    -------
    intensity_cut : np.ndarray
        The cut EEM.
    ex_range_cut : np.ndarray
        The cut ex wavelengths.
    em_range_cut : np.ndarray
        The cut em wavelengths.
    """
    em_min_idx = dichotomy_search(em_range_old, em_min_new)
    em_max_idx = dichotomy_search(em_range_old, em_max_new)
    ex_min_idx = dichotomy_search(ex_range_old, ex_min_new)
    ex_max_idx = dichotomy_search(ex_range_old, ex_max_new)
    intensity_cut = intensity[
                    ex_range_old.shape[0] - ex_max_idx - 1:ex_range_old.shape[0] - ex_min_idx,
                    em_min_idx:em_max_idx + 1
                    ]
    em_range_cut = em_range_old[em_min_idx:em_max_idx + 1]
    ex_range_cut = ex_range_old[ex_min_idx:ex_max_idx + 1]
    return intensity_cut, ex_range_cut, em_range_cut


def eem_nan_imputing(intensity, ex_range, em_range, method: str = 'linear', fill_value: str = 'linear_ex'):
    """
    Impute the NaN values in an EEM using 2D interpolation and 1D extrapolation.
    This function first uses 2D interpolation ('linear' or 'cubic') for points
    within the convex hull of the data. Then, it handles points outside this area
    (extrapolation) using the method specified by `fill_value`.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    method : str, {"linear", "cubic"}
        The method for the primary 2D interpolation. Passed to `scipy.griddata`.
    fill_value : float or str, {"linear_ex", "linear_em"}
        Controls how to fill values outside the non-NaN data region.
        - float: Fills with the given constant value.
        - 'linear_ex': Performs 1D linear extrapolation along each column (excitation axis).
        - 'linear_em': Performs 1D linear extrapolation along each row (emission axis).

    Returns
    -------
    intensity_imputed : np.ndarray
        The imputed EEM with NaN values filled.
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
                interp_func = interp1d(np.flatnonzero(~mask), col[~mask], kind='linear', fill_value='extrapolate')
                col[mask] = interp_func(np.flatnonzero(mask))
            intensity_imputed[:, i] = col
    elif fill_value == 'linear_em':
        intensity_imputed = griddata((xx, yy), zz, (x, y), method=method)
        for j in range(intensity_imputed.shape[0]):
            col = intensity_imputed[j, :]
            mask = np.isnan(col)
            if np.any(mask) and np.any(~mask):
                interp_func = interp1d(np.flatnonzero(~mask), col[~mask], kind='linear', fill_value='extrapolate')
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
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    blank : np.ndarray (2d) or np.ndarray (3d)
        The blank. If this function is called by implementing process_eem_stack(), a 3d array with length n (the number
        of samples) at axis 0 may be passed. In this case, each EEM will be normalized with a sample-specific blank.
    ex_range_blank : np.ndarray (1d)
        The excitation wavelengths of blank.
    em_range_blank : np.ndarray (1d)
        The emission wavelengths of blank.
    from_blank : bool
        Whether to calculate the RSU from a blank. If False, manual_rsu will be used.
    integration_time : float
        The integration time of the blank measurement.
    ex_target : float
        The excitation wavelength at which the RSU is calculated.
    bandwidth : float
        The bandwidth of Raman scattering peak.
    rsu_standard : float
        A factor used to divide the raw RSU. This is used to control the magnitude of RSU so that the normalized
        intensity of EEM would not be numerically too high or too low.
    manual_rsu : float
        If from_blank = False, this will be used as the RSU_final directly.

    Returns
    -------
    intensity_normalized : np.ndarray
        The normalized EEM.
    rsu_final : float
        The Final RSU used in normalization.
    """
    if not from_blank:
        return intensity / manual_rsu, manual_rsu
    else:
        em_target = -ex_target / (0.00036 * ex_target - 1)
        rsu, _ = eem_regional_integration(blank, ex_range_blank, em_range_blank,
                                          ex_min=ex_target, ex_max=ex_target, em_min=em_target - bandwidth / 2,
                                          em_max=em_target + bandwidth / 2)
    rsu_final = rsu / (integration_time * rsu_standard)
    intensity_normalized = intensity / rsu_final
    return intensity_normalized, rsu_final


def eem_raman_scattering_removal(intensity, ex_range, em_range, width=5, interpolation_method='linear',
                                 interpolation_dimension='2d', recover_original_nan=True):
    """
    Remove and interpolate the first-order Raman scattering peak of water.
    This function identifies the Raman peak based on the excitation wavelength
    (approximating a Raman shift of ~3400-3600 cm⁻¹ for water), masks this
    region, and then fills the masked values using a specified method.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    width : float
        The total width (in nm) of the Raman scattering band to remove.
    interpolation_method : str, {"linear", "cubic", "nan", "zero"}
        The method used to fill the masked Raman scattering region.
    interpolation_dimension : str, {"1d-ex", "1d-em", "2d"}
        The axis for interpolation. '1d-ex' interpolates along the excitation
        axis, '1d-em' along the emission axis, and '2d' uses 2D interpolation.
        This parameter is ignored if `interpolation_method` is 'nan' or 'zero'.#
    recover_original_nan : bool
        If True (default), the nan values existing before scattering removal will be preserved as nan.

    Returns
    -------
    intensity_filled : np.ndarray
        The EEM with the Raman scattering region removed and filled.
    raman_mask : np.ndarray
        A mask indicating the Raman scattering region.
        0: masked pixel; 1: unmasked pixel.
    """
    intensity_filled = np.array(intensity)
    lambda_em = -ex_range / (0.00036 * ex_range - 1)
    half_width = width / 2
    # Calculate the distance of every point in the EEM grid from its theoretical Raman peak.
    # The use of np.newaxis broadcasts the arrays to create a 2D comparison grid.
    distance_from_peak = np.abs(em_range[np.newaxis, :] - lambda_em[:, np.newaxis])
    # Create a boolean mask where the distance is within the half-width
    raman_mask_bool = distance_from_peak <= half_width
    # The intensity matrix has a reversed excitation axis, so we flip the mask
    # and convert it to a float mask (True->0.0, False->1.0).
    raman_mask = (~np.flipud(raman_mask_bool)).astype(float)
    original_nan = np.isnan(intensity)
    if interpolation_method == 'nan':
        intensity_filled[raman_mask == 0] = np.nan
    elif interpolation_method == 'zero':
        intensity_filled[raman_mask == 0] = 0
    else:
        if interpolation_dimension == '1d-ex':
            for j in range(intensity.shape[1]):
                col_mask = raman_mask[:, j]
                if np.all(col_mask == 0) or np.all(col_mask == 1): continue
                known_x = np.flipud(ex_range)[col_mask == 1]
                known_y = intensity_filled[:, j][col_mask == 1]
                f1 = interp1d(known_x, known_y, kind=interpolation_method, fill_value='extrapolate')
                intensity_filled[:, j] = f1(np.flipud(ex_range))
        elif interpolation_dimension == '1d-em':
            for i in range(intensity.shape[0]):
                row_mask = raman_mask[i, :]
                if np.all(row_mask == 0) or np.all(row_mask == 1): continue
                known_x = em_range[row_mask == 1]
                known_y = intensity_filled[i, :][row_mask == 1]
                f1 = interp1d(known_x, known_y, kind=interpolation_method, fill_value='extrapolate')
                intensity_filled[i, :] = f1(em_range)
        elif interpolation_dimension == '2d':
            intensity_filled[raman_mask == 0] = np.nan
            intensity_filled = eem_nan_imputing(intensity_filled, ex_range, em_range, method=interpolation_method)
    if recover_original_nan:
        # Restore original NaNs that were not part of the Raman mask
        intensity_filled[original_nan] = np.nan
    return intensity_filled, raman_mask


def eem_rayleigh_scattering_removal(intensity, ex_range, em_range, width_o1=20, width_o2=30,
                                    interpolation_dimension_o1='2d', interpolation_dimension_o2='2d',
                                    interpolation_method_o1='zero',
                                    interpolation_method_o2='linear',
                                    recover_original_nan=True):
    """
    Remove and interpolate first and second-order Rayleigh scattering.
    This function first zeroes out the physically meaningless region where the
    emission wavelength is less than or equal to the excitation wavelength
    (Em <= Ex). It then masks and fills the first and second-order scattering bands.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    width_o1 : float
        The total width (in nm) of the 1st order Rayleigh scattering (Em = Ex).
    width_o2 : float
        The total width (in nm) of the 2nd order Rayleigh scattering (Em = 2*Ex).
    interpolation_dimension_o1 : str, {"1d-ex", "1d-em", "2d"}
        The axis for interpolating the 1st order scattering. '1d-ex' for
        excitation axis, '1d-em' for emission axis, '2d' for 2D grid.
    interpolation_dimension_o2 : str, {"1d-ex", "1d-em", "2d"}
        The axis for interpolating the 2nd order scattering.
    interpolation_method_o1 : str, {"linear", "cubic", "nan", "zero", "none"}
        The method for filling the 1st order scattering region.
    interpolation_method_o2 : str, {"linear", "cubic", "nan", "zero", "none"}
        The method for filling the 2nd order scattering region.
    recover_original_nan : bool
        If True (default), the nan values existing before scattering removal will be preserved as nan.

    Returns
    -------
    intensity_filled : np.ndarray
        The EEM with Rayleigh scattering regions removed and filled.
    rayleigh_mask_o1 : np.ndarray
        A mask indicating the 1st order Rayleigh scattering region.
        0: masked pixel; 1: unmasked pixel.
    rayleigh_mask_o2 : np.ndarray
        A mask indicating the 2nd order Rayleigh scattering region.
        0: masked pixel; 1: unmasked pixel.
    """
    intensity_masked = np.array(intensity)
    # --- Step 1: Zero out the physically meaningless region (Em <= Ex) ---
    # Create a 2D boolean mask where the condition Em <= Ex is true
    invalid_region_mask = em_range[np.newaxis, :] <= ex_range[:, np.newaxis]
    # The intensity matrix has a reversed excitation axis
    invalid_region_mask_flipped = np.flipud(invalid_region_mask)
    # Apply the mask to zero out the invalid region
    intensity_masked[invalid_region_mask_flipped] = 0
    # --- Step 2: Vectorized Mask Creation for 1st Order Rayleigh Scattering ---
    half_width_o1 = width_o1 / 2
    dist_o1 = np.abs(em_range[np.newaxis, :] - ex_range[:, np.newaxis])
    mask_bool_o1 = dist_o1 <= half_width_o1
    rayleigh_mask_o1 = (~np.flipud(mask_bool_o1)).astype(float)
    # --- Step 3: Vectorized Mask Creation for 2nd Order Rayleigh Scattering ---
    half_width_o2 = width_o2 / 2
    dist_o2 = np.abs(em_range[np.newaxis, :] - (ex_range * 2)[:, np.newaxis])
    mask_bool_o2 = dist_o2 <= half_width_o2
    rayleigh_mask_o2 = (~np.flipud(mask_bool_o2)).astype(float)
    # --- Step 4: Interpolation Loop ---
    for axis, itp, mask in zip([interpolation_dimension_o1, interpolation_dimension_o2],
                               [interpolation_method_o1, interpolation_method_o2],
                               [rayleigh_mask_o1, rayleigh_mask_o2]):
        old_nan = np.isnan(intensity)
        if itp == 'zero':
            intensity_masked[mask == 0] = 0
        elif itp == 'nan':
            intensity_masked[mask == 0] = np.nan
        elif itp == 'none':
            pass
        else:  # Handle 'linear' or 'cubic'
            if axis == '1d-ex':
                for j in range(intensity.shape[1]):
                    col_mask = mask[:, j]
                    if np.all(col_mask == 0) or np.all(col_mask == 1): continue
                    x = np.flipud(ex_range)[col_mask == 1]
                    y = intensity_masked[:, j][col_mask == 1]
                    f1 = interp1d(x, y, kind=itp, fill_value='extrapolate')
                    intensity_masked[:, j] = f1(np.flipud(ex_range))
            elif axis == '1d-em':
                for i in range(intensity.shape[0]):
                    row_mask = mask[i, :]
                    if np.all(row_mask == 0) or np.all(row_mask == 1): continue
                    x = em_range[row_mask == 1]
                    y = intensity_masked[i, :][row_mask == 1]
                    f1 = interp1d(x, y, kind=itp, fill_value='extrapolate')
                    intensity_masked[i, :] = f1(em_range)
            elif axis == '2d':
                intensity_masked[mask == 0] = np.nan
                intensity_masked = eem_nan_imputing(intensity_masked, ex_range, em_range, method=itp,
                                                    fill_value='linear_ex')
        if recover_original_nan:
            intensity_masked[old_nan] = np.nan
    return intensity_masked, rayleigh_mask_o1, rayleigh_mask_o2


def eem_ife_correction(intensity, ex_range_eem, em_range_eem, absorbance, ex_range_abs):
    """
    Correct the inner filter effect (IFE) using absorbance data.

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range_eem : np.ndarray
        A 1D NumPy array of the EEM's excitation wavelengths.
    em_range_eem : np.ndarray
        A 1D NumPy array of the EEM's emission wavelengths.
    absorbance : np.ndarray
        A 1D NumPy array of the absorbance spectrum corresponding to a single sample.
    ex_range_abs : np.ndarray
        A 1D NumPy array of the wavelengths for the absorbance spectrum.

    Returns
    -------
    intensity_corrected : np.ndarray
        The IFE-corrected EEM.
    """
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
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    ex_min : float
        The lower boundary of excitation wavelengths of the integrated region.
    ex_max : float
        The upper boundary of excitation wavelengths of the integrated region.
    em_min : float
        The lower boundary of emission wavelengths of the integrated region.
    em_max : float
        The upper boundary of emission wavelengths of the integrated region.

    Returns
    -------
    integration : float
        The RFI.
    avg_regional_intensity :
        The average fluorescence intensity in the region.
    """
    ex_range_interpolated = np.sort(np.unique(np.concatenate([ex_range, [ex_min, ex_max]])))
    em_range_interpolated = np.sort(np.unique(np.concatenate([em_range, [em_min, em_max]])))
    intensity_interpolated = eem_interpolation(intensity, ex_range, em_range, ex_range_interpolated,
                                               em_range_interpolated, method='linear')
    intensity_cut, ex_range_cut, em_range_cut = eem_cutting(
        intensity_interpolated, ex_range_interpolated, em_range_interpolated,
        ex_min_new=ex_min, ex_max_new=ex_max,
        em_min_new=em_min, em_max_new=em_max
    )
    if intensity_cut.shape[0] == 1:
        integration = np.trapz(intensity_cut, em_range_cut, axis=1)
    elif intensity_cut.shape[1] == 1:
        integration = np.trapz(intensity_cut, ex_range_cut, axis=0)
    else:
        result_x = np.trapz(intensity_cut, np.flip(ex_range_cut), axis=0)
        integration = np.absolute(np.trapz(result_x, em_range_cut, axis=0))
    integration_area = (ex_max - ex_min) * (em_max - em_min)
    if integration_area > 0:
        avg_regional_intensity = integration / integration_area
    else:
        avg_regional_intensity = 0.0
    return integration, avg_regional_intensity


def eem_interpolation(intensity, ex_range_old, em_range_old, ex_range_new, em_range_new, method: str = 'linear'):
    """
    Interpolate an EEM onto a new grid of excitation and emission wavelengths.
    This function is typically used to standardize the axes of multiple EEMs.
    It assumes the rows of the intensity matrix correspond to descending excitation
    wavelengths. It may not be able to interpolate an EEM containing NaN
    values. For NaN value imputation, please consider eem_nan_imputing().

    Parameters
    ----------
    intensity : np.ndarray
        A 2D NumPy array representing the Excitation-Emission Matrix.
    ex_range_old : np.ndarray
        A 1D array of the original excitation wavelengths (must be monotonic).
    em_range_old : np.ndarray
        A 1D array of the original emission wavelengths (must be monotonic).
    ex_range_new : np.ndarray
        A 1D array of the target excitation wavelengths.
    em_range_new : np.ndarray
        A 1D array of the target emission wavelengths.
    method : str, {"linear", "nearest", "slinear", "cubic", "quintic"}
        The interpolation method, passed to `scipy.RegularGridInterpolator`.

    Returns
    -------
    intensity_interpolated : np.ndarray
        The interpolated EEM on the new wavelength grid.
    """
    interp = RegularGridInterpolator((ex_range_old[::-1], em_range_old), intensity, method=method, bounds_error=False)
    x, y = np.meshgrid(ex_range_new[::-1], em_range_new, indexing='ij')
    intensity_interpolated = interp((x, y)).reshape(ex_range_new.shape[0], em_range_new.shape[0])
    return intensity_interpolated


def eems_tf_normalization(eem_stack):
    """
    Normalize a stack of EEMs by their relative total fluorescence.
    Each EEM in the stack is divided by its total fluorescence, which has been
    normalized to the mean total fluorescence of the entire stack.

    Parameters
    ----------
    eem_stack : np.ndarray
        The 3D EEM stack, with shape (n_samples, n_ex_wavelengths, n_em_wavelengths).

    Returns
    -------
    intensity_normalized : np.ndarray
        The normalized EEM stack.
    weights : np.ndarray
        The normalization factor for each EEM. This is the total fluorescence
        of each EEM divided by the mean total fluorescence of the stack.
    """
    total_fluorescences = eem_stack.sum(axis=(1, 2))
    weights = total_fluorescences / np.mean(total_fluorescences)
    intensity_normalized = eem_stack / weights[:, np.newaxis, np.newaxis]
    return intensity_normalized, weights


def eems_fit_components(eem_stack, components, fit_intercept=False, positive=True):
    """
    Fit each EEM in a stack as a linear combination of a set of components.
    This function performs linear regression for each EEM sample against
    the provided spectral components. It works on copies of the input data
    and does not modify the original arrays.

    Parameters
    ----------
    eem_stack : np.ndarray
        The 3D EEM stack, with shape (n_samples, n_ex_wavelengths, n_em_wavelengths).
    components : np.ndarray
        A 3D array of spectral components (n_components, n_ex, n_em).
    fit_intercept : bool, optional
        Whether to calculate an intercept for the linear model. Defaults to False.
    positive : bool, optional
        When set to True, forces the coefficients (scores) to be non-negative.
        Defaults to True.

    Returns
    -------
    score_sample : np.ndarray
        A 2D array (n_samples, n_components) of the fitted scores
        for each component in each sample.
    fmax_sample : np.ndarray
        A 2D array (n_samples, n_components) of the Fmax (scores multiplied by the
        maximum intensity of each component).
    eem_stack_pred : np.ndarray
        The 3D stack of reconstructed EEMs based on the model fit.
    """
    assert eem_stack.shape[1:] == components.shape[1:], "EEM and component have different shapes"
    eem_stack_copy = np.nan_to_num(eem_stack)
    components_copy = np.nan_to_num(components)
    score_sample = []
    fmax_sample = []
    max_values = np.amax(components_copy, axis=(1, 2))
    eem_stack_pred = np.zeros(eem_stack_copy.shape)
    component_matrix = components_copy.reshape([components_copy.shape[0], -1]).T
    for i in range(eem_stack_copy.shape[0]):
        y_true = eem_stack_copy[i].reshape([-1])
        reg = LinearRegression(fit_intercept=fit_intercept, positive=positive)
        reg.fit(component_matrix, y_true)
        y_pred = reg.predict(component_matrix)
        eem_stack_pred[i, :, :] = y_pred.reshape(eem_stack_copy.shape[1:])
        score_sample.append(reg.coef_)
        fmax_sample.append(reg.coef_ * max_values)
    score_sample = np.array(score_sample)
    fmax_sample = np.array(fmax_sample)
    return score_sample, fmax_sample, eem_stack_pred


def loadings_similarity(loadings1: pd.DataFrame, loadings2: pd.DataFrame, wavelength_alignment=False, dtw=False):
    """
    Calculate the Tucker's congruence between each pair of components of two loadings (of excitation or emission).
    Parameters
    ----------
    loadings1 : pandas.DataFrame
        The first loadings. Each column of the table corresponds to one component.
    loadings2 : pandas.DataFrame
        The second loadings. Each column of the table corresponds to one component.
    wavelength_alignment : bool
        Align the ex/em ranges of the components. This is useful if the PARAFAC models have different ex/em wavelengths.
        Note that ex/em will be aligned according to the ex/em ranges with the lower intervals between the two PARAFAC
        models.
    dtw : bool
        Apply dynamic time warping (DTW) to align the component loadings before calculating the similarity. This is
        useful for matching loadings with similar but shifted shapes.
    Returns
    -------
    m_sim : pandas.DataFrame
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
    """
    Calculate the Pearson correlation coefficient between each pair of components of two PARAFAC or NMF models.
    Parameters:
    ----------
    components1 : np.ndarray
        The first set of components. Each component is a 3D array (n_components, ex, em).
    components2 : np.ndarray
        The second set of components. Each component is a 3D array (n_components, ex, em).
    Returns:
    -------
    m_sim : pandas.DataFrame
        The table of component similarities between each pair of components.
    """
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
    models_dict : dict
        Dictionary of PARAFAC objects, the models to be aligned.
    ex_ref : pandas.DataFrame
        Ex loadings of the reference
    em_ref : pandas.DataFrame
        Em loadings of the reference
    wavelength_alignment : bool
        Align the ex/em ranges of the components. This is useful if the PARAFAC models have different ex/em wavelengths.
        Note that ex/em will be aligned according to the ex/em ranges with the lower intervals between the two PARAFAC
        models.
    Returns
    -------
    models_dict_new : dict
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
        model.beta = model.beta[permutation] if model.beta is not None else None
        models_dict_new[model_label] = model
    return models_dict_new


def align_components_by_components(models_dict: dict, components_ref: dict):
    """
    Align the components of PARAFAC or NMF models according to given reference components so that similar components
    are labelled by the same name.
    Parameters
    ----------
    models_dict : dict
        Dictionary of PARAFAC objects, the models to be aligned.
    components_ref : dict
        Dictionary where each item is a reference component. The keys are the component labels, the values are the
        components (np.ndarray).
    Returns
    -------
    models_dict_new : dict
        Dictionary of the aligned PARAFAC object.
    """
    component_labels_ref = list(components_ref.keys())
    components_stack_ref = np.array([c.reshape(-1) for c in components_ref.values()])
    models_dict_new = {}
    for model_label, model in models_dict.items():
        comp = model.components.reshape([model.n_components, -1])
        cost_mat = cdist(components_stack_ref, comp, metric='correlation')
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        # permutation = list(row_ind)
        # matched_index = list(col_ind)
        pairs = [(i, j) for i, j in zip(row_ind, col_ind) if i < cost_mat.shape[0]]
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        row_ind, col_ind = zip(*sorted_pairs)
        col_ind = list(col_ind)
        non_ordered_index = list(set([i for i in range(cost_mat.shape[1])]) - set(col_ind))
        permutation = col_ind + non_ordered_index
        if non_ordered_index:
            component_labels_ref_extended = component_labels_ref + ['O{i}'.format(i=i + 1) for i in
                                                                    range(len(non_ordered_index))]
        else:
            component_labels_ref_extended = component_labels_ref
        component_labels_var = [0] * len(permutation)
        for i, nc in enumerate(permutation):
            component_labels_var[nc] = component_labels_ref_extended[i]
        model.fmax.columns, model.nnls_fmax.columns = (
                [component_labels_var] * 2)
        model.fmax = model.fmax.iloc[:, permutation]
        model.nnls_fmax = model.nnls_fmax.iloc[:, permutation]
        model.components = model.components[permutation, :, :]
        model.beta = model.beta[permutation] if model.beta is not None else None
        if getattr(model, 'ex_loadings', None) and getattr(model, 'em_loadings', None):
            model.ex_loadings.columns, model.em_loadings.columns = (
                    [component_labels_var] * 2)
        if getattr(model, 'score', None):
            model.score.columns = component_labels_var
        models_dict_new[model_label] = model
    return models_dict_new




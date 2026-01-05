import numpy as np
import pandas as pd

import warnings
import copy
import json
import random
from scipy import stats
from typing import Optional

from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from eempy.utils import dichotomy_search
from .basic import (
    process_eem_stack,
    eem_threshold_masking,
    eem_gaussian_filter,
    eem_region_masking,
    eem_median_filter,
    eem_cutting,
    eem_nan_imputing,
    eem_raman_normalization,
    eems_tf_normalization,
    eem_rayleigh_scattering_removal,
    eem_raman_scattering_removal,
    eem_ife_correction,
    eem_interpolation,
    eem_regional_integration,
)


class EEMDataset:
    """
    Build an EEM dataset.

    Parameters
    ----------
    eem_stack : np.ndarray
        The 3D EEM stack, with shape (n_samples, n_ex_wavelengths, n_em_wavelengths).
    ex_range : np.ndarray
        A 1D NumPy array of the excitation wavelengths.
    em_range : np.ndarray
        A 1D NumPy array of the emission wavelengths.
    index : list or None
        Optional. The name used to label each sample. The number of elements in the list should equal the number
        of samples in the eem_stack (with the same sample order).
    ref : pd.DataFrame or None
        Optional. The reference data, e.g., the contaminant concentrations in each sample.
        It should have a length equal to the number of samples in the eem_stack.
        The index of each sample should be the name given in parameter "index".
        It is possible to have more than one column.
        NaN is allowed (for example, if contaminant concentrations in specific samples are unknown).
    cluster : list or None
        Optional. The classification of samples, e.g., the output of EEM clustering algorithms.
        The number of elements in the list should equal the number
        of samples in the eem_stack (with the same sample order).
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
        """        Compute the fluorescence index (FI) for each sample.

        FI is computed as intensity(ex=370 nm, em=470 nm) divided by intensity(ex=370 nm, em=520 nm).

        Returns
        -------
        fi: pandas.DataFrame
            Fluorescence index values. Note: the current implementation labels the output column as "BIX" even though the
            values correspond to FI."""
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

        Parameters
        ----------
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
    def threshold_masking(self, threshold, fill, mask_type='greater', inplace=True):
        """
        Mask fluorescence intensity values above or below a threshold across all samples.

        Parameters
        ----------
        threshold: float or int
            Intensity threshold.
        fill: float or int
            Value used to replace masked pixels.
        mask_type: str, {"greater", "smaller"}, default="greater"
            Whether to mask values greater than or smaller than `threshold`.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset with threshold masking applied.
        """
        eem_stack_masked, masks = process_eem_stack(self.eem_stack, eem_threshold_masking, threshold=threshold,
                                                    fill=fill, mask_type=mask_type)
        if inplace:
            self.eem_stack = eem_stack_masked
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_masked
            return eem_dataset_new

    def gaussian_filter(self, sigma=1, truncate=3, inplace=True):
        """
        Apply Gaussian filtering to every EEM in the dataset.

        Parameters
        ----------
        sigma: float, default=1
            Standard deviation of the Gaussian kernel.
        truncate: float, default=3
            Truncate the filter at this many standard deviations.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset with Gaussian filtering applied.
        """
        eem_stack_filtered = process_eem_stack(self.eem_stack, eem_gaussian_filter, sigma=sigma, truncate=truncate)
        if inplace:
            self.eem_stack = eem_stack_filtered
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_filtered
            return eem_dataset_new

    def median_filter(self, window_size=(3, 3), mode='reflect', inplace=True):
        """
        Apply median filtering to an EEM.

        Parameters
        ----------
        window_size: tuple of two integers
            Gives the shape that is taken from the input array, at every element position, to define the input to the filter
            function.
        mode: str, {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
            The mode parameter determines how the input array is extended beyond its boundaries.
        inplace: bool
            if True, overwrite the EEMDataset object with the processed EEMs.

        Returns
        -------
        eem_dataset_new: EEMDataset
            The processed EEM dataset.
        """
        eem_stack_filtered = process_eem_stack(self.eem_stack, eem_median_filter, window_size=window_size, mode=mode)
        if inplace:
            self.eem_stack = eem_stack_filtered
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_filtered
            return eem_dataset_new

    def region_masking(self, ex_min, ex_max, em_min, em_max, fill_value='nan', inplace=True):
        """
        Mask a rectangular excitation/emission region in every EEM in the dataset.

        Parameters
        ----------
        ex_min: float, default=230
            Lower bound of the excitation wavelength window to mask (nm).
        ex_max: float, default=500
            Upper bound of the excitation wavelength window to mask (nm).
        em_min: float, default=250
            Lower bound of the emission wavelength window to mask (nm).
        em_max: float, default=810
            Upper bound of the emission wavelength window to mask (nm).
        fill_value: str, {"nan", "zero"}, default="nan"
            How to fill the masked region.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset with regional masking applied.
        """
        eem_stack_masked, _ = process_eem_stack(
            self.eem_stack, eem_region_masking, ex_range=self.ex_range,
            em_range=self.em_range, ex_min=ex_min, ex_max=ex_max, em_min=em_min,
            em_max=em_max, fill_value=fill_value
        )
        if inplace:
            self.eem_stack = eem_stack_masked
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_masked
            return eem_dataset_new

    def cutting(self, ex_min, ex_max, em_min, em_max, inplace=True):
        """
        Cut every EEM in the dataset to a new excitation/emission window.

        Parameters
        ----------
        ex_min: float
            Lower bound of the excitation wavelength window to keep (nm).
        ex_max: float
            Upper bound of the excitation wavelength window to keep (nm).
        em_min: float
            Lower bound of the emission wavelength window to keep (nm).
        em_max: float
            Upper bound of the emission wavelength window to keep (nm).
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset after cutting. The dataset's `ex_range` and `em_range` are updated accordingly.
        """
        eem_stack_cut, new_ranges = process_eem_stack(
            self.eem_stack, eem_cutting, ex_range_old=self.ex_range,
            em_range_old=self.em_range,
            ex_min_new=ex_min, ex_max_new=ex_max, em_min_new=em_min,
            em_max_new=em_max
        )
        if inplace:
            self.eem_stack = eem_stack_cut
            self.ex_range = new_ranges[0][0]
            self.em_range = new_ranges[0][1]
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_cut
            eem_dataset_new.ex_range = new_ranges[0][0]
            eem_dataset_new.em_range = new_ranges[0][1]
            return eem_dataset_new

    def nan_imputing(self, method='linear', fill_value='linear_ex', inplace=True):
        """
        Impute NaN pixels in every EEM in the dataset.

        Parameters
        ----------
        method: str, {"linear", "cubic"}, default="linear"
            2D interpolation method passed to `scipy.interpolate.griddata`.
        fill_value: float or str, {"linear_ex", "linear_em"}, default="linear_ex"
            How to fill pixels outside the convex hull of non-NaN data.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset with NaN pixels filled.
        """
        eem_stack_imputed = process_eem_stack(self.eem_stack, eem_nan_imputing, ex_range=self.ex_range,
                                              em_range=self.em_range, method=method, fill_value=fill_value)
        if inplace:
            self.eem_stack = eem_stack_imputed
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_imputed
            return eem_dataset_new

    def raman_normalization(self, ex_range_blank=None, em_range_blank=None, blank=None, from_blank=False,
                            integration_time=1, ex_target=350, bandwidth=5,
                            rsu_standard=20000, manual_rsu=1, inplace=True):
        """
        Normalize every EEM in the dataset by a Raman scattering unit (RSU). RSU can be supplied directly (
        `from_blank=False`) or calculated from blank EEM data (`from_blank=True`). The normalization factor is
        RSU_raw divided by (`rsu_standard` * `integration_time`).

        Parameters
        ----------
        blank: np.ndarray, optional
            Blank EEM(s) used to estimate RSU when `from_blank=True`.
        ex_range_blank: np.ndarray, optional
            Excitation wavelength axis for the blank EEM(s).
        em_range_blank: np.ndarray, optional
            Emission wavelength axis for the blank EEM(s).
        from_blank: bool, default=False
            If True, calculate RSU from the provided blank EEM(s).
        integration_time: float, default=1
            Integration time used for the blank measurement.
        ex_target: float, default=350
            Excitation wavelength (nm) at which RSU is computed.
        bandwidth: float, default=5
            Raman peak bandwidth (nm) used for regional integration.
        rsu_standard: float, default=20000
            Scaling factor applied to RSU to control the magnitude of normalized intensities.
        manual_rsu: float, default=1
            RSU used directly when `from_blank=False`.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            Raman-normalized EEM dataset.
        """
        eem_stack_normalized, rsu = process_eem_stack(
            self.eem_stack, eem_raman_normalization, ex_range_blank=ex_range_blank,
            em_range_blank=em_range_blank, blank=blank, from_blank=from_blank,
            integration_time=integration_time, ex_target=ex_target,
            bandwidth=bandwidth, rsu_standard=rsu_standard, manual_rsu=manual_rsu
        )
        if inplace:
            self.eem_stack = eem_stack_normalized
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_normalized
            return eem_dataset_new

    def tf_normalization(self, inplace=True):
        """
        Normalize every EEM by its total fluorescence. Each sample is divided by its total fluorescence, normalized
        to the mean total fluorescence across the dataset.

        Parameters
        ----------
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            Total-fluorescence-normalized EEM dataset.
        weights: np.ndarray
            Per-sample normalization factors (total fluorescence divided by the dataset mean).
        """
        eem_stack_normalized, weights = eems_tf_normalization(self.eem_stack)
        if inplace:
            self.eem_stack = eem_stack_normalized
            return self, weights
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_normalized
            return eem_dataset_new, weights

    def raman_scattering_removal(self, width=5, interpolation_method='linear', interpolation_dimension='2d',
                                 inplace=True, recover_original_nan=True):
        """
        Remove the first-order Raman scattering band and fill the masked region.

        Parameters
        ----------
        width: float, default=5
            Total width (nm) of the Raman scattering band to mask.
        interpolation_method: str, {"linear", "cubic", "nan", "zero"}, default="linear"
            Method used to fill the masked region.
        interpolation_dimension: str, {"1d-ex", "1d-em", "2d"}, default="2d"
            Interpolation axis/dimension used when `interpolation_method` is not "nan" or "zero".
        recover_original_nan: bool, default=True
            If True, preserve NaN pixels that existed before scattering removal.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset with Raman scattering removed and filled.
        """
        eem_stack_masked, _ = process_eem_stack(
            self.eem_stack, eem_raman_scattering_removal, ex_range=self.ex_range,
            em_range=self.em_range, width=width,
            interpolation_method=interpolation_method,
            interpolation_dimension=interpolation_dimension,
            recover_original_nan=recover_original_nan
        )
        if inplace:
            self.eem_stack = eem_stack_masked
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_masked
            return eem_dataset_new

    def rayleigh_scattering_removal(self, width_o1=15, width_o2=15, interpolation_dimension_o1='2d',
                                    interpolation_dimension_o2='2d', interpolation_method_o1='zero',
                                    interpolation_method_o2='linear', inplace=True, recover_original_nan=True):
        """
        Remove first- and second-order Rayleigh scattering bands and fill the masked regions.

        Parameters
        ----------
        width_o1: float, default=15
            Total width (nm) of the first-order Rayleigh band (Em = Ex).
        width_o2: float, default=15
            Total width (nm) of the second-order Rayleigh band (Em = 2*Ex).
        interpolation_dimension_o1: str, {"1d-ex", "1d-em", "2d"}, default="2d"
            Interpolation axis/dimension for the first-order band.
        interpolation_dimension_o2: str, {"1d-ex", "1d-em", "2d"}, default="2d"
            Interpolation axis/dimension for the second-order band.
        interpolation_method_o1: str, {"linear", "cubic", "nan", "zero", "none"}, default="zero"
            Fill method for the first-order band.
        interpolation_method_o2: str, {"linear", "cubic", "nan", "zero", "none"}, default="linear"
            Fill method for the second-order band.
        recover_original_nan: bool, default=True
            If True, preserve NaN pixels that existed before scattering removal.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset with Rayleigh scattering removed and filled.
        """
        eem_stack_masked, _ = process_eem_stack(
            self.eem_stack, eem_rayleigh_scattering_removal, ex_range=self.ex_range,
            em_range=self.em_range, width_o1=width_o1,
            width_o2=width_o2,
            interpolation_dimension_o1=interpolation_dimension_o1,
            interpolation_dimension_o2=interpolation_dimension_o2,
            interpolation_method_o1=interpolation_method_o1,
            interpolation_method_o2=interpolation_method_o2,
            recover_original_nan=recover_original_nan
        )
        if inplace:
            self.eem_stack = eem_stack_masked
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_masked
            return eem_dataset_new

    def ife_correction(self, absorbance, ex_range_abs, inplace=True):
        """
        Apply inner filter effect (IFE) correction to every EEM using absorbance spectra.

        Parameters
        ----------
        absorbance: np.ndarray
            Absorbance spectra stack (n_samples, n_abs_wavelengths).
        ex_range_abs: np.ndarray
            Wavelength axis (nm) for the absorbance spectra.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            IFE-corrected EEM dataset.
        """
        eem_stack_corrected = process_eem_stack(
            self.eem_stack, eem_ife_correction, ex_range_eem=self.ex_range,
            em_range_eem=self.em_range, absorbance=absorbance,
            ex_range_abs=ex_range_abs
        )
        if inplace:
            self.eem_stack = eem_stack_corrected
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_corrected
            return eem_dataset_new

    def interpolation(self, ex_range_new, em_range_new, method, inplace=True):
        """
        Interpolate every EEM onto a new excitation/emission wavelength grid.

        Parameters
        ----------
        ex_range_new: np.ndarray
            Target excitation wavelength axis (nm).
        em_range_new: np.ndarray
            Target emission wavelength axis (nm).
        method: str, {"linear", "nearest", "slinear", "cubic", "quintic"}
            Interpolation method passed to `scipy.interpolate.RegularGridInterpolator`.
        inplace: bool, default=True
            If True, overwrite `self` and return it. If False, return a new EEMDataset instance.

        Returns
        -------
        eem_dataset_new: EEMDataset
            EEM dataset interpolated to the new wavelength grid. The dataset's `ex_range` and `em_range` are updated accordingly.
        """
        eem_stack_interpolated = process_eem_stack(
            self.eem_stack, eem_interpolation, ex_range_old=self.ex_range,
            em_range_old=self.em_range, ex_range_new=ex_range_new,
            em_range_new=em_range_new, method=method
        )
        if inplace:
            self.eem_stack = eem_stack_interpolated
            self.ex_range = ex_range_new
            self.em_range = em_range_new
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_interpolated
            eem_dataset_new.ex_range = ex_range_new
            eem_dataset_new.em_range = em_range_new
            return eem_dataset_new

    def splitting(self, n_split, rule: str = 'random', random_state=None,
                  kw_top=None, kw_bot=None, idx_top=None, idx_bot=None):
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
        if kw_top is not None and kw_bot is not None:
            assert self.index is not None, "EEMDataset index is not specified."
            idx_top = [i for i in range(len(self.index)) if kw_top in self.index[i]]
            idx_bot = [i for i in range(len(self.index)) if kw_bot in self.index[i]]
        if idx_top is None and idx_bot is None:
            idx_eems = [i for i in range(self.eem_stack.shape[0])]
            if rule == 'random':
                if random_state is not None:
                    random.seed(random_state)
                random.shuffle(idx_eems)
                idx_splits = np.array_split(idx_eems, n_split)
            elif rule == 'sequential':
                idx_splits = np.array_split(idx_eems, n_split)
            else:
                raise ValueError("'rule' should be either 'random' or 'sequential'")
        elif idx_top is not None and idx_bot is not None:
            assert len(idx_top) == len(idx_bot), "'idx_top' must have the same length as 'idx_bot'"
            if rule == 'random':
                if random_state is not None:
                    random.seed(random_state)
                shuffle_order = np.random.permutation(len(idx_top))
                idx_splits_top = np.array_split([idx_top[i] for i in shuffle_order], n_split)
                idx_splits_bot = np.array_split([idx_bot[i] for i in shuffle_order], n_split)
                idx_splits = []
                for s_top, s_bot in zip(idx_splits_top, idx_splits_bot):
                    idx_splits.append(np.concatenate([s_top, s_bot], axis=0))
            elif rule == 'sequential':
                idx_splits_top = np.array_split(idx_top, n_split)
                idx_splits_bot = np.array_split(idx_bot, n_split)
                idx_splits = []
                for s_top, s_bot in zip(idx_splits_top, idx_splits_bot):
                    idx_splits.append(np.concatenate([s_top, s_bot], axis=0))
            else:
                raise ValueError("'rule' should be either 'random' or 'sequential'")
        else:
            raise ValueError("only one of 'idx_top' and 'idx_bot' is defined.")
        subset_list = []
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

    def subsampling(self, portion=0.8, inplace=True):
        """
        Randomly select a portion of the EEM.

        Parameters
        ----------
        portion: float
            The portion.
        inplace: bool
            if True, overwrite the EEMDataset object.

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
        if inplace:
            self.eem_stack = eem_stack_new
            self.index = index_new
            self.ref = ref_new
            self.cluster = cluster_new
            return self.sort_by_index(inplace=True), sorted(selected_indices)
        else:
            eem_dataset_sub = copy.deepcopy(self)
            eem_dataset_sub.eem_stack = eem_stack_new
            eem_dataset_sub.index = index_new
            eem_dataset_sub.ref = ref_new
            eem_dataset_sub.cluster = cluster_new
            return eem_dataset_sub.sort_by_index(inplace=True), sorted(selected_indices)

    def sort_by_index(self, inplace=True):
        """
        Sort the sample order of eem_stack, index and reference (if exists) by the index.

        Parameters
        -------
        inplace: bool
            If True, overwrite the EEMDataset object.

        Returns
        -------
        eem_dataset_new: EEMDataset
            The processed EEM dataset.
        """
        sorted_indices = sorted(range(len(self.index)), key=lambda i: self.index[i])
        if inplace:
            self.index = sorted(self.index)
            self.eem_stack = self.eem_stack[sorted_indices]
            if self.ref is not None:
                self.ref = self.ref.iloc[sorted_indices]
            if self.cluster is not None:
                self.cluster = [self.cluster[i] for i in sorted_indices]
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.index = sorted(self.index)
            eem_dataset_new.eem_stack = self.eem_stack[sorted_indices]
            if self.ref is not None:
                eem_dataset_new.ref = self.ref.iloc[sorted_indices]
            if self.cluster is not None:
                eem_dataset_new.cluster = [self.cluster[i] for i in sorted_indices]
            return eem_dataset_new

    def filter_by_index(self, mandatory_keywords, optional_keywords, inplace=True):
        """
        Select the samples whose indexes contain the given keyword.

        Parameters
        -------
        mandatory_keywords: str or list of str
            Keywords for selecting samples whose indexes contain all the mandatory keywords.
        optional_keywords: str or list of str
            Keywords for selecting samples whose indexes contain any of the optional keywords.
        inplace: bool
            if True, overwrite the EEMDataset object.

        Returns
        -------
        eem_dataset_new: EEMDataset
            The filtered EEM dataset.
        """
        if mandatory_keywords is None and optional_keywords is None:
            return self.eem_stack, self.index, self.ref, self.cluster, []
        if self.index is None or not self.index:
            raise ValueError('index is not defined or empty in EEMDataset')
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
        if inplace:
            self.eem_stack = eem_stack_filtered
            self.index = index_filtered
            self.ref = ref_filtered
            self.cluster = cluster_filtered
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_filtered
            eem_dataset_new.index = index_filtered
            eem_dataset_new.ref = ref_filtered
            eem_dataset_new.cluster = cluster_filtered
            return eem_dataset_new

    def filter_by_cluster(self, cluster_names, inplace=True):
        """
        Select the samples belong to certain cluster(s).

        Parameters
        -------
        cluster_names: int/float/str or list of int/float/str
            cluster names.
        inplace: bool
            if False, overwrite the EEMDataset object.

        Returns
        -------
        eem_dataset_new: EEMDataset
            The filtered EEM dataset.
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
        if inplace:
            self.eem_stack = eem_stack_filtered
            self.index = index_filtered
            self.ref = ref_filtered
            self.cluster = cluster_filtered
            return self
        else:
            eem_dataset_new = copy.deepcopy(self)
            eem_dataset_new.eem_stack = eem_stack_filtered
            eem_dataset_new.index = index_filtered
            eem_dataset_new.ref = ref_filtered
            eem_dataset_new.cluster = cluster_filtered
            return eem_dataset_new

    def to_json(self, filepath=None):
        eem_dataset_json_dict = {
            'eem_stack': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for sublist in
                          self.eem_stack.tolist()],
            'ex_range': self.ex_range.tolist(),
            'em_range': self.em_range.tolist(),
            'index': self.index,
            'ref': [self.ref.columns.tolist()] + self.ref.values.tolist() if self.ref is not None else None,
            'cluster': self.cluster,
        }
        if filepath is not None:
            with open(filepath, 'w') as f:
                json.dump(eem_dataset_json_dict, f, indent=4)
        return eem_dataset_json_dict


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

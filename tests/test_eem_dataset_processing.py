import copy
from pathlib import Path

import numpy as np
import pytest

from eempy.eem_processing.eem_dataset import EEMDataset
from eempy.read_data.read_data import read_abs_dataset, read_eem_dataset
from eempy.utils import dichotomy_search


@pytest.fixture(scope="module")
def sample_dataset():
    data_dir = Path(__file__).resolve().parent / "sample_data"
    eem_files = [
        "2021-02-01-14-00_R1PEM.dat",
        "2021-02-01-16-00_R1PEM.dat",
    ]
    abs_files = [
        "2021-02-01-14-00_R1ABS.dat",
        "2021-02-01-16-00_R1ABS.dat",
    ]
    eem_stack, ex_range, em_range, _ = read_eem_dataset(
        str(data_dir),
        custom_filename_list=eem_files,
        file_first_row="ex",
    )
    abs_stack, ex_range_abs, _ = read_abs_dataset(
        str(data_dir),
        custom_filename_list=abs_files,
    )
    index = [Path(name).stem for name in eem_files]
    dataset = EEMDataset(eem_stack=eem_stack, ex_range=ex_range, em_range=em_range, index=index)
    return dataset, abs_stack, ex_range_abs


def test_dataset_construction(sample_dataset):
    dataset, _, _ = sample_dataset
    assert dataset.eem_stack.shape[0] == len(dataset.index)
    assert dataset.eem_stack.shape[1] == dataset.ex_range.shape[0]
    assert dataset.eem_stack.shape[2] == dataset.em_range.shape[0]
    assert np.all(np.diff(dataset.ex_range) > 0)
    assert np.all(np.diff(dataset.em_range) > 0)


def test_threshold_masking(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    threshold = np.nanpercentile(ds.eem_stack, 99)
    masked = ds.threshold_masking(threshold=threshold, fill=-999.0, mask_type="greater", inplace=False)
    assert np.any(masked.eem_stack == -999.0)
    assert not np.any(ds.eem_stack == -999.0)
    mask = ds.eem_stack > threshold
    assert np.all(masked.eem_stack[mask] == -999.0)


def test_gaussian_and_median_filters(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    gaussian = ds.gaussian_filter(sigma=1, truncate=2, inplace=False)
    assert gaussian.eem_stack.shape == ds.eem_stack.shape
    assert not np.allclose(gaussian.eem_stack, ds.eem_stack)
    median = ds.median_filter(window_size=(3, 3), mode="reflect", inplace=False)
    assert median.eem_stack.shape == ds.eem_stack.shape
    assert not np.allclose(median.eem_stack, ds.eem_stack)


def test_region_masking_and_nan_imputing(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    ex_min = ds.ex_range[5]
    ex_max = ds.ex_range[10]
    em_min = ds.em_range[5]
    em_max = ds.em_range[10]
    masked = ds.region_masking(
        ex_min=ex_min,
        ex_max=ex_max,
        em_min=em_min,
        em_max=em_max,
        fill_value="nan",
        inplace=False,
    )
    assert np.isnan(masked.eem_stack).any()
    imputed = masked.nan_imputing(method="linear", fill_value="linear_ex", inplace=False)
    assert not np.isnan(imputed.eem_stack).any()


def test_cutting_and_interpolation(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    ex_min = ds.ex_range[5]
    ex_max = ds.ex_range[15]
    em_min = ds.em_range[5]
    em_max = ds.em_range[15]
    cut = ds.cutting(ex_min=ex_min, ex_max=ex_max, em_min=em_min, em_max=em_max, inplace=False)
    assert cut.eem_stack.shape[1] == cut.ex_range.shape[0]
    assert cut.eem_stack.shape[2] == cut.em_range.shape[0]
    assert cut.ex_range[0] >= ex_min
    assert cut.ex_range[-1] <= ex_max
    assert cut.em_range[0] >= em_min
    assert cut.em_range[-1] <= em_max
    ex_new = cut.ex_range[::2]
    em_new = cut.em_range[::2]
    interp = cut.interpolation(ex_range_new=ex_new, em_range_new=em_new, method="linear", inplace=False)
    assert interp.eem_stack.shape[1] == ex_new.shape[0]
    assert interp.eem_stack.shape[2] == em_new.shape[0]


def test_raman_and_tf_normalization(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    raman = ds.raman_normalization(from_blank=False, manual_rsu=2, inplace=False)
    assert np.allclose(raman.eem_stack, ds.eem_stack / 2)
    tf_dataset, weights = ds.tf_normalization(inplace=False)
    assert weights.shape[0] == ds.eem_stack.shape[0]
    expected_total = np.mean(ds.eem_stack.sum(axis=(1, 2)))
    assert np.allclose(tf_dataset.eem_stack.sum(axis=(1, 2)), expected_total)


def test_scattering_removals(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    raman_removed = ds.raman_scattering_removal(
        width=5,
        interpolation_method="zero",
        interpolation_dimension="2d",
        inplace=False,
    )
    assert raman_removed.eem_stack.shape == ds.eem_stack.shape
    assert not np.isnan(raman_removed.eem_stack).any()
    rayleigh_removed = ds.rayleigh_scattering_removal(
        width_o1=10,
        width_o2=10,
        interpolation_method_o1="zero",
        interpolation_method_o2="zero",
        inplace=False,
    )
    assert rayleigh_removed.eem_stack.shape == ds.eem_stack.shape
    assert not np.isnan(rayleigh_removed.eem_stack).any()


def test_ife_correction(sample_dataset):
    dataset, abs_stack, ex_range_abs = sample_dataset
    ds = copy.deepcopy(dataset)
    corrected = ds.ife_correction(absorbance=abs_stack, ex_range_abs=ex_range_abs, inplace=False)
    assert corrected.eem_stack.shape == ds.eem_stack.shape
    assert not np.allclose(corrected.eem_stack, ds.eem_stack)


def test_region_masking_geometry(sample_dataset):
    dataset, _, _ = sample_dataset
    ds = copy.deepcopy(dataset)
    ex_min = ds.ex_range[3]
    ex_max = ds.ex_range[7]
    em_min = ds.em_range[4]
    em_max = ds.em_range[9]
    masked = ds.region_masking(
        ex_min=ex_min,
        ex_max=ex_max,
        em_min=em_min,
        em_max=em_max,
        fill_value="zero",
        inplace=False,
    )
    em_min_idx = dichotomy_search(ds.em_range, em_min)
    em_max_idx = dichotomy_search(ds.em_range, em_max)
    ex_min_idx = dichotomy_search(ds.ex_range, ex_min)
    ex_max_idx = dichotomy_search(ds.ex_range, ex_max)
    masked_region = masked.eem_stack[:, ex_min_idx:ex_max_idx + 1, em_min_idx:em_max_idx + 1]
    assert np.all(masked_region == 0)

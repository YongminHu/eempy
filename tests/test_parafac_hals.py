from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorly")
pytest.importorskip("tlviz")

from eempy.eem_processing.eem_dataset import EEMDataset
from eempy.eem_processing.parafac import PARAFAC
from eempy.read_data.read_data import get_filelist, read_eem_dataset


@pytest.fixture(scope="module")
def preprocessed_dataset():
    data_dir = Path(__file__).resolve().parent / "sample_data"
    mandatory_keywords = ["PEM"]
    optional_keywords = []

    filename_list = sorted(get_filelist(str(data_dir), mandatory_keywords, optional_keywords))[:12]
    eem_stack, ex_range, em_range, _ = read_eem_dataset(
        str(data_dir),
        custom_filename_list=filename_list,
        file_first_row="ex",
    )
    index = [Path(name).stem for name in filename_list]
    dataset = EEMDataset(eem_stack=eem_stack, ex_range=ex_range, em_range=em_range, index=index)

    dataset = dataset.threshold_masking(
        threshold=0,
        fill=0,
        mask_type="smaller",
        inplace=False,
    )
    dataset = dataset.rayleigh_scattering_removal(
        width_o1=10,
        width_o2=10,
        interpolation_method_o1="zero",
        interpolation_method_o2="zero",
        inplace=False,
    )
    dataset = dataset.raman_scattering_removal(
        width=5,
        interpolation_method="zero",
        interpolation_dimension="2d",
        inplace=False,
    )
    assert not np.isnan(dataset.eem_stack).any()
    return dataset


def test_parafac_hals_fit(preprocessed_dataset):
    dataset = preprocessed_dataset
    model = PARAFAC(
        n_components=2,
        solver="hals",
        init="svd",
        max_iter_als=30,
        random_state=0,
    )
    model.fit(dataset)

    assert model.components.shape[0] == 2
    assert model.components.shape[1] == dataset.eem_stack.shape[1]
    assert model.components.shape[2] == dataset.eem_stack.shape[2]
    assert model.score.shape[0] == dataset.eem_stack.shape[0]
    assert model.score.shape[1] == 2
    assert 0 <= model.variance_explained() <= 100


def test_parafac_tf_normalization_keeps_original_scale(preprocessed_dataset):
    n_subset = 6
    dataset = EEMDataset(
        eem_stack=preprocessed_dataset.eem_stack[:n_subset].copy(),
        ex_range=preprocessed_dataset.ex_range.copy(),
        em_range=preprocessed_dataset.em_range.copy(),
        index=preprocessed_dataset.index[:n_subset],
    )
    total_before = dataset.total_fluorescence().copy()

    model = PARAFAC(
        n_components=2,
        solver="hals",
        init="svd",
        max_iter_als=5,
        random_state=0,
        tf_normalization=True,
    )
    model.fit(dataset)

    np.testing.assert_allclose(dataset.total_fluorescence(), total_before)
    np.testing.assert_allclose(model.eem_stack_train, dataset.eem_stack)
    np.testing.assert_allclose(model.fmax.to_numpy(), model.nnls_fmax.to_numpy(), rtol=1e-5, atol=1e-3)

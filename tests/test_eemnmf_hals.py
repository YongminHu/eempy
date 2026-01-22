from pathlib import Path

import numpy as np
import pytest

from eempy.eem_processing.eem_dataset import EEMDataset
from eempy.eem_processing.eemnmf import EEMNMF
from eempy.read_data.read_data import get_filelist, read_eem_dataset


@pytest.fixture(scope="module")
def preprocessed_dataset():
    data_dir = Path(__file__).resolve().parent / "sample_data"
    mandatory_keywords = ["PEM"]
    optional_keywords = ["2021-02-01"]

    filename_list = get_filelist(str(data_dir), mandatory_keywords, optional_keywords)
    eem_stack, ex_range, em_range, _ = read_eem_dataset(
        str(data_dir),
        mandatory_keywords=mandatory_keywords,
        optional_keywords=optional_keywords,
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


def test_eemnmf_hals_fit(preprocessed_dataset):
    dataset = preprocessed_dataset
    model = EEMNMF(
        n_components=2,
        solver="hals",
        init="nndsvda",
        max_iter_als=30,
        random_state=0,
    )
    model.fit(dataset)

    assert model.components.shape[0] == 2
    assert model.components.shape[1] == dataset.eem_stack.shape[1]
    assert model.components.shape[2] == dataset.eem_stack.shape[2]
    assert model.fmax.shape[0] == dataset.eem_stack.shape[0]
    assert model.fmax.shape[1] == 2
    assert 0 <= model.variance_explained() <= 100

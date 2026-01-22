from pathlib import Path

import numpy as np
import pytest

from eempy.eem_processing.eem_dataset import EEMDataset
from eempy.eem_processing.eemnmf import EEMNMF
from eempy.eem_processing.validation import SplitValidation
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


def test_split_validation_components(preprocessed_dataset):
    dataset = preprocessed_dataset
    base_model = EEMNMF(
        n_components=2,
        solver="hals",
        init="nndsvda",
        max_iter_als=20,
        random_state=0,
    )
    validator = SplitValidation(base_model=base_model, n_splits=2, combination_size="half", rule="random", random_state=0)
    validator.fit(dataset)

    similarities = validator.compare_components()
    assert similarities.shape[0] == 1
    assert similarities.shape[1] == base_model.n_components
    assert np.isfinite(similarities.to_numpy()).all()


def test_split_validation_parafac_only_guard(preprocessed_dataset):
    dataset = preprocessed_dataset
    base_model = EEMNMF(n_components=2, solver="hals", init="nndsvda", max_iter_als=10, random_state=0)
    validator = SplitValidation(base_model=base_model, n_splits=2, combination_size="half", rule="random", random_state=0)
    validator.fit(dataset)
    with pytest.raises(ValueError):
        validator.compare_parafac_loadings()

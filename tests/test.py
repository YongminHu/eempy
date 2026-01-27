from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorly")
pytest.importorskip("tlviz")

from eempy.eem_processing.eem_dataset import EEMDataset
from eempy.eem_processing.parafac import PARAFAC
from eempy.read_data.read_data import get_filelist, read_eem_dataset
from eempy.eem_processing import eem_raman_scattering_removal
import matplotlib.pyplot as plt

data_dir = "C:/PhD/eempy/tests/sample_data"
mandatory_keywords = ["PEM"]
optional_keywords = ["2021-02-02"]

filename_list = get_filelist(str(data_dir), mandatory_keywords, optional_keywords)
eem_stack, ex_range, em_range, _ = read_eem_dataset(
    str(data_dir),
    mandatory_keywords=mandatory_keywords,
    optional_keywords=optional_keywords,
    file_first_row="ex",
)
eem_dataset = EEMDataset(eem_stack, ex_range, em_range)
eem_dataset.rayleigh_scattering_removal(
    width_o1=30,
    width_o2=30,
    interpolation_dimension_o1="2d",
    interpolation_method_o1="zero",
    interpolation_dimension_o2="2d",
    interpolation_method_o2="linear",
    inplace=True,
)
eem_dataset.raman_scattering_removal(
    width=20,
    interpolation_method="linear",
    inplace=True,
)
parafac_model = PARAFAC(n_components=3, random_state=42)
parafac_model.fit(eem_dataset) 

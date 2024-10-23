import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import (EEMDataset, PARAFAC, EEMNMF, eem_raman_normalization, eem_cutting, eem_interpolation,
                                  SplitValidation, loadings_similarity, align_components_by_loadings)
from eempy.plot import plot_eem, plot_loadings, plot_score
import re


# eem_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIPEM.dat'
# blank_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIBEM.dat'
eem_dataset_path = 'C:/PhD\Fluo-detect/_data/_greywater/2024_quenching_nmf/279_cut_raman_interpolated.json'
#

eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
sv = SplitValidation(rank=5)
sv.fit(eem_dataset)
similarities_ex, similarities_em = sv.compare()
print(similarities_ex)
print(similarities_em)

import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd

# from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
# from eempy.eem_processing import (EEMDataset, PARAFAC, EEMNMF, KMethod, eem_cutting, eem_interpolation,
#                                   SplitValidation, loadings_similarity, align_components_by_loadings)
# from eempy.plot import plot_eem, plot_loadings, plot_fmax
# import re
#
#
# # eem_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIPEM.dat'
# # blank_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIBEM.dat'
# eem_dataset_path = 'C:/PhD/Fluo-detect/_data/_greywater/2024_quenching_nmf/324_ex_274_em_300_raman_15_interpolated_gaussian.json'
# #
#
# eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
# eem_dataset.filter_by_index(['2024-10-22', 'M3'], None, copy=False)
#
# base_model = PARAFAC(n_components=3)
# kmethod = KMethod(base_model=base_model, n_initial_splits=3, max_iter=3)
# consensus_matrix, label_history, error_history = kmethod.calculate_consensus(eem_dataset, 3, 0.8)
# print(error_history)
# print(label_history)


# import os
#
# def add_text_to_filenames(folder_path, text_to_add):
#     # Loop through all files in the folder
#     for filename in os.listdir(folder_path):
#         if '2024-10-' in filename:
#             # Construct full file path
#             full_file_path = os.path.join(folder_path, filename)
#             # Check if it is a file (not a directory)
#             if os.path.isfile(full_file_path):
#                 # Create new filename by adding the text to the original filename
#                 new_filename = filename[:24] + filename[26:]
#                 # Construct full path for the new file
#                 new_full_file_path = os.path.join(folder_path, new_filename)
#                 # Rename the file
#                 os.rename(full_file_path, new_full_file_path)
#                 print(f'Renamed: {filename} to {new_filename}')
#
# # Example usage
# folder_path = 'C:/PhD/Fluo-detect/_data/_greywater/2024_quenching_nmf'  # Replace with the path to your folder
# text_to_add = 'M3'  # Replace with the text you want to add
# add_text_to_filenames(folder_path, text_to_add)

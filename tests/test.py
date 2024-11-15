import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import (EEMDataset, PARAFAC, EEMNMF, KMethod, eem_cutting, eem_interpolation,
                                  SplitValidation, loadings_similarity, align_components_by_loadings,
                                  eems_fit_components)
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from sklearn.metrics import mean_squared_error
import re
#
#
# # eem_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIPEM.dat'
# # blank_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIBEM.dat'


# ---------------Effect of random initialization-----------------

#       --------PARAFAC--------
eem_dataset_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
true_components = np.array([eem_dataset.eem_stack[-5], eem_dataset.eem_stack[0]])
_, fmax_measured, _ = eems_fit_components(eem_dataset.eem_stack, true_components)
fmax_measured = pd.DataFrame(fmax_measured, index=eem_dataset.index)
components = []
fmaxs = []
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
least_conc_error_index = 0
least_eem_error_index = 0
for i in range(50):
    parafac_model = PARAFAC(n_components=2, init='random', tol=1e-09, n_iter_max=500)
    parafac_model.fit(eem_dataset)
    fmax = parafac_model.fmax
    components.append(parafac_model.components)
    fmax = pd.concat([fmax, fmax_measured], axis=1)
    fmax_0gL = fmax[fmax.index.str.contains('0gL')]
    fmax_0gL['E. coli (million #/ mL)'] = [0, 2.75, 1.375, 4.125, 5.5]
    fmax_0gL['BSA (mg/L)'] = [1.66, 0.83, 1.245, 0.415, 0]
    fmax_0gL = fmax_0gL.sort_values(by='E. coli (million #/ mL)')
    fmaxs.append(fmax_0gL)
    ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmax_0gL.iloc[:, 0], '-.', color='grey')
    ax2.plot(fmax_0gL['BSA (mg/L)'], fmax_0gL.iloc[:, 1], '-.', color='grey')
    rmse_ecoli = mean_squared_error(fmax_0gL.iloc[:, 2], fmax_0gL.iloc[:, 0])
    rmse_bsa = mean_squared_error(fmax_0gL.iloc[:, 3], fmax_0gL.iloc[:, 1])
    if i > 0:
        if rmse_bsa + rmse_ecoli < rmse_avg:
            rmse_avg = rmse_bsa + rmse_ecoli
            least_conc_error_index = i
        if parafac_model.sample_rmse().sum().sum() < recon_error:
            recon_error = parafac_model.sample_rmse().sum().sum()
            least_eem_error_index = i
    else:
        rmse_avg = rmse_bsa + rmse_ecoli
        recon_error = parafac_model.sample_rmse()
        recon_error = recon_error.sum().sum()
ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmax_0gL.iloc[:, 2], '-o', color='red', label='Measured Fmax')
ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_conc_error_index].iloc[:, 0], '-o', color='blue', label='Conc. best fit Fmax')
ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_eem_error_index].iloc[:, 0], '-o', color='green', label='EEM best fit Fmax')
ax2.plot(fmax_0gL['BSA (mg/L)'], fmax_0gL.iloc[:, 3], '-o', color='red', label='Measured Fmax')
ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_conc_error_index].iloc[:, 1], '-o', color='blue', label='Conc. best fit Fmax')
ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_eem_error_index].iloc[:, 1], '-o', color='green', label='EEM best fit Fmax')
fig1.legend(bbox_to_anchor=(0.9, 0.3))
fig1.suptitle('PARAFAC Fmax of E. coli')
fig2.legend(bbox_to_anchor=(0.9, 0.3))
fig2.suptitle('PARAFAC Fmax of BSA')
fig1.show()
fig2.show()
plot_eem(components[least_conc_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit E. coli PARAFAC component')
plot_eem(components[least_conc_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit E. coli PARAFAC component')
plot_eem(components[least_eem_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit BSA PARAFAC component')
plot_eem(components[least_eem_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit BSA PARAFAC component')

# #       --------NMF--------
# components = []
# fmaxs = []
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# least_conc_error_index = 0
# least_eem_error_index = 0
# for i in range(1000):
#     nmf_model = EEMNMF(n_components=2, init='random',sort_em=True)
#     nmf_model.fit(eem_dataset)
#     fmax = nmf_model.nnls_fmax
#     components.append(nmf_model.components)
#     fmax = pd.concat([fmax, fmax_measured], axis=1)
#     fmax_0gL = fmax[fmax.index.str.contains('0gL')]
#     fmax_0gL['E. coli (million #/ mL)'] = [0, 2.75, 1.375, 4.125, 5.5]
#     fmax_0gL['BSA (mg/L)'] = [1.66, 0.83, 1.245, 0.415, 0]
#     fmax_0gL = fmax_0gL.sort_values(by='E. coli (million #/ mL)')
#     fmaxs.append(fmax_0gL)
#     ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmax_0gL.iloc[:, 0], '-.', color='grey')
#     ax2.plot(fmax_0gL['BSA (mg/L)'], fmax_0gL.iloc[:, 1], '-.', color='grey')
#     rmse_ecoli = mean_squared_error(fmax_0gL.iloc[:, 2], fmax_0gL.iloc[:, 0])
#     rmse_bsa = mean_squared_error(fmax_0gL.iloc[:, 3], fmax_0gL.iloc[:, 1])
#     if i > 0:
#         if rmse_bsa + rmse_ecoli < rmse_avg:
#             rmse_avg = rmse_bsa + rmse_ecoli
#             least_conc_error_index = i
#         if nmf_model.sample_rmse().sum().sum() < recon_error:
#             recon_error = nmf_model.sample_rmse().sum().sum()
#             least_eem_error_index = i
#     else:
#         rmse_avg = rmse_bsa + rmse_ecoli
#         recon_error = nmf_model.sample_rmse()
#         recon_error = recon_error.sum().sum()
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmax_0gL.iloc[:, 2], '-o', color='red', label='Measured Fmax')
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_conc_error_index].iloc[:, 0], '-o', color='blue', label='Conc. best fit Fmax')
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_eem_error_index].iloc[:, 0], '-o', color='green', label='EEM best fit Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmax_0gL.iloc[:, 3], '-o', color='red', label='Measured Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_conc_error_index].iloc[:, 1], '-o', color='blue', label='Conc. best fit Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_eem_error_index].iloc[:, 1], '-o', color='green', label='EEM best fit Fmax')
# fig1.legend(bbox_to_anchor=(0.9, 0.3))
# fig1.suptitle('NMF Fmax of E. coli')
# fig2.legend(bbox_to_anchor=(0.9, 0.3))
# fig2.suptitle('NMF Fmax of BSA')
# fig1.show()
# fig2.show()
# plot_eem(components[least_conc_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit E. coli NMF component')
# plot_eem(components[least_conc_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit E. coli NMF component')
# plot_eem(components[least_eem_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit BSA NMF component')
# plot_eem(components[least_eem_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit BSA NMF component')


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

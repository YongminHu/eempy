from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
import re


#
#
# # eem_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIPEM.dat'
# # blank_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIBEM.dat'


eem_dataset_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_up = eem_dataset.eem_stack[eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKI-reS')]
eem_bot = eem_dataset.eem_stack[eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKISYM.')]
eem_new = np.concatenate([eem_up[:-8], eem_bot[-8:]], axis=0)
# plot_eem(eem_up, eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=1500)
# plot_eem(eem_new, eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=1500)
eem_stack_new = np.delete(eem_dataset.eem_stack,
                          eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKI-reS'),
                          axis=0)
eem_dataset.index.remove('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKI-reS')
replaced_idx = eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKISYM.')
eem_stack_new[replaced_idx] = eem_new
eem_dataset.eem_stack = eem_stack_new

true_components = np.array([eem_dataset.eem_stack[-5], eem_dataset.eem_stack[0]])
_, fmax_measured, _ = eems_fit_components(eem_dataset.eem_stack, true_components)
fmax_measured = pd.DataFrame(fmax_measured, index=eem_dataset.index,
                             columns=['component 1 measured fmax', 'component 2 measured fmax'])
# ref = pd.DataFrame(
#     {
#         'E. coli (million #/ mL)': [0, 2.75, 1.375, 4.125, 5.5],
#         'BSA (mg/L)': [1.66, 0.83, 1.245, 0.415, 0]
#     },
#     index=eem_dataset.index
# )

# -------------Compare PARAFAC components and actual components--------------

# parafac_model = PARAFAC(n_components=2, init='svd', tol=1e-09, n_iter_max=500)
# parafac_model.fit(eem_dataset)
# parafac_components = parafac_model.components
# normalized_true_components = true_components / np.max(true_components, axis=(1, 2)).reshape((-1, 1, 1))
# normalized_parafac_components = parafac_components / np.max(parafac_components, axis=(1, 2)).reshape((-1, 1, 1))
# fig_ecoli, ax_ecoli = plot_eem(normalized_parafac_components[0],
#                                ex_range=eem_dataset.ex_range,
#                                em_range=eem_dataset.em_range,
#                                plot_tool='matplotlib',
#                                vmin=0,
#                                vmax=1,
#                                auto_intensity_range=False,
#                                figure_size=(6, 6),
#                                fix_aspect_ratio=False,
#                                display=False,
#                                axis_label_font_size=18,
#                                axis_ticks_font_size=14,
#                                cbar_font_size=14,
#                                cbar_fraction=0.02,
#                                # title="Fingerprint residual in BSA component"
#                                )
# fig_ecoli.tight_layout()
# fig_ecoli.show()
# fig_bsa, ax_bsa = plot_eem(normalized_parafac_components[1],
#                            ex_range=eem_dataset.ex_range,
#                            em_range=eem_dataset.em_range,
#                            plot_tool='matplotlib',
#                            vmin=0,
#                            vmax=1,
#                            auto_intensity_range=False,
#                            figure_size=(6, 6),
#                            fix_aspect_ratio=False,
#                            display=False,
#                            axis_label_font_size=18,
#                            axis_ticks_font_size=14,
#                            cbar_font_size=14,
#                            cbar_fraction=0.02,
#                            # title="Fingerprint residual in BSA component"
#                            )
# fig_bsa.tight_layout()
# fig_bsa.show()

# # --------------Ideal Fmax vs. measured Fmax vs. modelled Fmax---------------
#
fmax_ideal_ecoli = np.array([0, 2.75, 1.375, 4.125, 5.5])/5.5*np.max(true_components[0])
fmax_ideal_bsa = np.array([1.66, 0.83, 1.245, 0.415, 0])/1.66*np.max(true_components[1])
parafac_model = PARAFAC(n_components=2, init='svd', tol=1e-09, n_iter_max=500, tf_normalization=False)
parafac_model.fit(eem_dataset)
fmax = parafac_model.fmax
fmax.columns = ['component 1 modelled fmax', 'component 2 modelled fmax']

fmax = pd.concat([fmax, fmax_measured], axis=1)
fmax_0gL = fmax[fmax.index.str.contains('0gL')]
fmax_0gL['component 1 ideal fmax'] = fmax_ideal_ecoli
fmax_0gL['component 2 ideal fmax'] = fmax_ideal_bsa
fmax_0gL = fmax_0gL.sort_values(by='component 1 ideal fmax')

fig, ax = plt.subplots()
ax.plot(
    fmax_0gL['component 1 ideal fmax'],
    fmax_0gL['component 2 ideal fmax'],
    '--o',
    color='black',
    label='Ideal Fmax'
)
ax.plot(
    fmax_0gL['component 1 measured fmax'],
    fmax_0gL['component 2 measured fmax'],
    '-o',
    color='black',
    label='Fmax fitted with actual fingerprint'
)
# ax.plot(
#     fmax_0gL['component 1 modelled fmax'],
#     fmax_0gL['component 2 modelled fmax'],
#     '-o',
#     color='red',
#     label='Fmax fitted with PARAFAC'
# )
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('E. coli component Fmax', fontsize=14)
ax.set_ylabel('BSA component Fmax', fontsize=14)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88), fontsize=12)
fig.show()


# # --------------Stern-Volmer equation validation----------
#
fmax_ideal_ecoli = np.array([0, 2.75, 1.375, 4.125, 5.5])/5.5*np.max(true_components[0])
fmax_ideal_bsa = np.array([1.66, 0.83, 1.245, 0.415, 0])/1.66*np.max(true_components[1])
parafac_model = PARAFAC(n_components=2, init='svd', tol=1e-10, n_iter_max=1000)
parafac_model.fit(eem_dataset)
# fmax = parafac_model.fmax

# nmf_model = EEMNMF(n_components=2, init='nndsvd', sort_em=True)
# nmf_model.fit(eem_dataset)
# fmax = nmf_model.nnls_fmax

fmax = fmax_measured

fmax_original = fmax[fmax.index.str.contains('0gL')]
means_ecoli = []
mins_ecoli = []
maxs_ecoli = []
std_ecoli = []
means_bsa = []
mins_bsa = []
maxs_bsa = []
std_bsa = []
for c_key in ["0gL", "1_25gL", "2_5gL", "3_75gL", "+5gL"]:
    c_key = re.escape(c_key)
    fmax_quenched = fmax[fmax.index.str.contains(c_key)]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    coef_ecoli = fmax_ratio[1:, 0]
    coef_bsa = fmax_ratio[0:-1, 1]
    means_ecoli.append(np.mean(coef_ecoli))
    mins_ecoli.append(np.min(coef_ecoli))
    maxs_ecoli.append(np.max(coef_ecoli))
    std_ecoli.append(np.std(coef_ecoli))
    means_bsa.append(np.mean(coef_bsa))
    mins_bsa.append(np.min(coef_bsa))
    maxs_bsa.append(np.max(coef_bsa))
    std_bsa.append(np.std(coef_bsa))

x = np.array([0, 1.25, 2.5, 3.75, 5])

slope_bsa, intercept_bsa = np.polyfit(x, means_bsa, 1)
regression_line_bsa = slope_bsa * x + intercept_bsa
slope_ecoli, intercept_ecoli = np.polyfit(x, means_ecoli, 1)
regression_line_ecoli = slope_ecoli * x + intercept_ecoli
# Plot
fig, ax = plt.subplots()

ax.plot(x, means_bsa, '-o', color='black', markersize=10, label='BSA component', zorder=3)
lower_error = [mean - min_val for mean, min_val in zip(means_bsa, mins_bsa)]
upper_error = [max_val - mean for max_val, mean in zip(maxs_bsa, means_bsa)]
ax.errorbar(
    x, means_bsa,
    # yerr=[lower_error, upper_error],
    yerr=[std_bsa, std_bsa],
    fmt='none',  # Do not plot markers/line
    ecolor='black',
    capsize=5,
    capthick=2,
    # label='Min/Max Range'
)
ax.plot(
    x, regression_line_bsa,
    linestyle='--',
    color='grey',
    label='reg. BSA component'
    # label=f'Linear Regression: y = {slope_bsa:.2f}x + {intercept_bsa:.2f}'
)
ax.plot(x, means_ecoli, '-o', markeredgecolor='black', markerfacecolor='white', color='black',
        markersize=10, label='E. coli component', zorder=3)
lower_error = [mean - min_val for mean, min_val in zip(means_ecoli, mins_ecoli)]
upper_error = [max_val - mean for max_val, mean in zip(maxs_ecoli, means_ecoli)]
ax.errorbar(
    x, means_ecoli,
    # yerr=[lower_error, upper_error],
    yerr=[std_ecoli, std_ecoli],
    fmt='none',  # Do not plot markers/line
    ecolor='black',
    capsize=5,
    capthick=2,
    # label='Min/Max Range'
)
ax.plot(
    x, regression_line_ecoli,
    linestyle='--',
    color='grey',
    # label=f'Linear Regression: y = {slope_ecoli:.2f}x + {intercept_ecoli:.2f}'
)
# Customize the plot
ax.set_xticks(x)
ax.set_ylim([0.75, 3])
ax.set_xlim([0, 5.2])
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('KI concentration (g/L)', fontsize=18)
ax.set_ylabel(r"$F_{0}/F$", fontsize=18)
# ax.set_title('Mean with Min/Max Range')
ax.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # ---------------Effect of initialization-----------------
# # #     --------PARAFAC--------
# components = []
# fmaxs = []
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# least_conc_error_index = 0
# least_eem_error_index = 0
# for i in range(1):
#     parafac_model = PARAFAC(n_components=2, init='svd', tol=1e-09, n_iter_max=500)
#     parafac_model.fit(eem_dataset)
#     fmax = parafac_model.fmax
#     fmax.columns = ['component 1 modelled fmax', 'component 2 modelled fmax']
#     components.append(parafac_model.components)
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
#         if parafac_model.sample_rmse().sum().sum() < recon_error:
#             recon_error = parafac_model.sample_rmse().sum().sum()
#             least_eem_error_index = i
#     else:
#         rmse_avg = rmse_bsa + rmse_ecoli
#         recon_error = parafac_model.sample_rmse()
#         recon_error = recon_error.sum().sum()
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmax_0gL['component 1 measured fmax'], '-o', color='red', label='Measured Fmax')
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_conc_error_index]['component 1 modelled fmax'], '-o', color='blue', label='Conc. best fit Fmax')
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_eem_error_index]['component 1 modelled fmax'], '-o', color='green', label='EEM best fit Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmax_0gL['component 2 measured fmax'], '-o', color='red', label='Measured Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_conc_error_index]['component 2 modelled fmax'], '-o', color='blue', label='Conc. best fit Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_eem_error_index]['component 2 modelled fmax'], '-o', color='green', label='EEM best fit Fmax')
# fig1.legend(bbox_to_anchor=(0.9, 0.3))
# fig1.suptitle('PARAFAC Fmax of E. coli')
# fig2.legend(bbox_to_anchor=(0.9, 0.3))
# fig2.suptitle('PARAFAC Fmax of BSA')
# fig1.show()
# fig2.show()
# plot_eem(components[least_conc_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit E. coli PARAFAC component')
# plot_eem(components[least_conc_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit E. coli PARAFAC component')
# plot_eem(components[least_eem_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit BSA PARAFAC component')
# plot_eem(components[least_eem_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit BSA PARAFAC component')


# # #       --------NMF--------
# components = []
# fmaxs = []
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# least_conc_error_index = 0
# least_eem_error_index = 0
# for i in range(1):
#     nmf_model = EEMNMF(n_components=2, init='nndsvd',sort_em=True)
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

# ------------------CorrPARAFAC---------
# components = []
# fmaxs = []
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# least_conc_error_index = 0
# least_eem_error_index = 0
# for i in range(1):
#     parafac_model = CorrPARAFAC(n_components=2, init='svd', tol=1e-09, n_outer_iter_max=500, n_inner_iter=1,
#                                 loadings_normalization='sd', tf_normalization=True)
#     parafac_model.fit(eem_dataset)
#     fmax = parafac_model.fmax
#     components.append(parafac_model.components)
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
#         if parafac_model.sample_rmse().sum().sum() < recon_error:
#             recon_error = parafac_model.sample_rmse().sum().sum()
#             least_eem_error_index = i
#     else:
#         rmse_avg = rmse_bsa + rmse_ecoli
#         recon_error = parafac_model.sample_rmse()
#         recon_error = recon_error.sum().sum()
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmax_0gL.iloc[:, 2], '-o', color='red', label='Measured Fmax')
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_conc_error_index].iloc[:, 0], '-o', color='blue', label='Conc. best fit Fmax')
# ax1.plot(fmax_0gL['E. coli (million #/ mL)'], fmaxs[least_eem_error_index].iloc[:, 0], '-o', color='green', label='EEM best fit Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmax_0gL.iloc[:, 3], '-o', color='red', label='Measured Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_conc_error_index].iloc[:, 1], '-o', color='blue', label='Conc. best fit Fmax')
# ax2.plot(fmax_0gL['BSA (mg/L)'], fmaxs[least_eem_error_index].iloc[:, 1], '-o', color='green', label='EEM best fit Fmax')
# fig1.legend(bbox_to_anchor=(0.9, 0.3))
# fig1.suptitle('PARAFAC Fmax of E. coli')
# fig2.legend(bbox_to_anchor=(0.9, 0.3))
# fig2.suptitle('PARAFAC Fmax of BSA')
# fig1.show()
# fig2.show()
# plot_eem(components[least_conc_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit E. coli PARAFAC component')
# plot_eem(components[least_conc_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit E. coli PARAFAC component')
# plot_eem(components[least_eem_error_index][0], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='Conc. best fit BSA PARAFAC component')
# plot_eem(components[least_eem_error_index][1], ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range, title='EEM. best fit BSA PARAFAC component')

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# ---------Rename files-----------
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


# -----------SVD initialization

# l1, l2, l3 = eems_decomposition_initialization(eem_dataset.eem_stack, rank=2)
# print(l1)

# -----------CorrPARAFAC

# corr_parfac = CorrPARAFAC(n_components=2)
# corr_parfac.fit(eem_dataset)
# plot_eem(corr_parfac.components[0], ex_range=corr_parfac.ex_range, em_range=corr_parfac.em_range)
# plot_eem(corr_parfac.components[1], ex_range=corr_parfac.ex_range, em_range=corr_parfac.em_range)


eem_stack, ex_range, em_range, indexes = read_eem_dataset(
    folder_path='C:/PhD/Fluo-detect/_data/20250327_BSA_Ecoli_HA',
    mandatory_keywords='SYM',
    wavelength_alignment=True,
)

abs_stack, ex_range, indexes = read_abs_dataset(
    folder_path='C:/PhD/Fluo-detect/_data/20250327_BSA_Ecoli_HA',
    wavelength_alignment=True,
)

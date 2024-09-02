import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem
from eempy.eem_processing import (EEMDataset, PARAFAC, EEMNMF, eem_raman_normalization, eem_cutting, eem_interpolation,
                                  SplitValidation, loadings_similarity, align_parafac_components)
from eempy.plot import plot_eem, plot_loadings, plot_score
import re


# eem_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIPEM.dat'
# blank_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240215_NEST_M3/B1S12024-02-12-M3+0gLKIBEM.dat'
# folder_path = 'C:/PhD\Fluo-detect/_data/_greywater/20240706_online_M3_full/20240706_online_full.json'
#
# with open(folder_path, 'r') as file:
#     eem_dataset_dict = json.load(file)
#
# print(eem_dataset_dict['index'])

a = pd.DataFrame(np.arange(9).reshape([3,3]), columns=['a', 'b', 'c'])
print(np.array(a.loc[:,['a']]))

# intensity, ex_range, em_range, index = read_eem(eem_path)
# blank, ex_range_blank, em_range_blank, _ = read_eem(blank_path)
# em_range_blank_interpolated = np.unique(np.sort(np.concatenate([em_range_blank, [399.5, 401.5]])))
# blank_interpolated = eem_interpolation(blank, ex_range_blank, em_range_blank, ex_range_blank, em_range_blank_interpolated)
# plot_eem(blank, ex_range_blank, em_range_blank)
# plot_eem(blank_interpolated, ex_range_blank, em_range_blank_interpolated)
# blank_cut, ex_range_cut, em_range_cut = eem_cutting(intensity, ex_range, em_range, 349, 351, 400, 410)
# print(blank_cut)

# intensity_n, _ = eem_raman_normalization(intensity, from_blank=True, blank=blank, ex_range_blank=ex_range_blank,
#                                          em_range_blank=em_range_blank, ex_lb=349, ex_ub=351, bandwidth=5)

#
# eem_stack, ex_range, em_range, index = read_eem_dataset(folder_path=folder_path,
#                                                         mandatory_keywords=['07-11', 'SYM'])

# blank_stack, ex_range_blank, em_range_blank, _ = read_eem_dataset(folder_path=folder_path,
#                                                         mandatory_keywords='BEM.dat')
# intensity_normalized, rsu_final = eem_raman_normalization(eem_stack[0], blank_stack[0], ex_range_blank, em_range_blank, from_blank=True)

# eem_stack[eem_stack<0] = 0

# eem_dataset1 = EEMDataset(eem_stack, ex_range, em_range)
# # eem_dataset2 = EEMDataset(eem_stack[10:20], ex_range, em_range)
# eem_dataset1.cutting(ex_min=270, ex_max=400, em_min=310, em_max=500, copy=False)
# eem_dataset1.rayleigh_scattering_removal(copy=False, interpolation_method_o1='zero', interpolation_method_o2='nan')
# eem_dataset1.raman_scattering_removal(copy=False, interpolation_method='nan', width=10)
#
# eem_dataset_json_dict = {
#     'eem_stack': eem_dataset1.eem_stack.tolist(),
#     'ex_range': eem_dataset1.ex_range.tolist(),
#     'em_range': eem_dataset1.em_range.tolist(),
#     'index': eem_dataset1.index,
#     'ref': eem_dataset1.ref.tolist() if eem_dataset1.ref is not None else None
# }
#
# eem_dataset = EEMDataset(
#     eem_stack=np.array(eem_dataset_json_dict['eem_stack']),
#     ex_range=np.array(eem_dataset_json_dict['ex_range']),
#     em_range=np.array(eem_dataset_json_dict['em_range']),
#     index=eem_dataset_json_dict['index']
# )
#
# nmf = EEMNMF(n_components=3)
#
# nmf.fit(eem_dataset)
# plot_eem(nmf.components[0], ex_range=eem_dataset1.ex_range, em_range=eem_dataset1.em_range, auto_intensity_range=True)

# eem_dataset2.rayleigh_scattering_removal(copy=False)
# parafac_model1 = PARAFAC(rank=3, non_negativity=True)
# parafac_model2 = PARAFAC(rank=3)
# parafac_model1.fit(eem_dataset1)
# parafac_model2.fit(eem_dataset2)

# sim = loadings_similarity(parafac_model1.ex_loadings, parafac_model2.ex_loadings)
# print(sim)
#
# models_dict_new = align_parafac_components({'model1': parafac_model1, 'model2': parafac_model2}, parafac_model1.ex_loadings, parafac_model1.em_loadings)
# print(models_dict_new)

# # eem_dataset.raman_normalization(ex_range_blank, em_range_blank, blank_stack, from_blank=True, copy=False)
# # print(eem_dataset.eem_stack.shape)
#
# parafac_model = PARAFAC(rank=3)

# parafac_model.fit(eem_dataset)
#
# print(parafac_model.leverage('sample'))

# split_validation = SplitValidation(rank=3)
# split_validation.fit(eem_dataset1)
# subset_specific_models = split_validation.subset_specific_models
#
# labels = sorted(subset_specific_models.keys())
# similarities_ex = {}
# similarities_em = {}
# for k in range(int(len(labels) / 2)):
#     m1 = subset_specific_models[labels[k]]
#     print(m1)
#     m2 = subset_specific_models[labels[-1 - k]]
#     print(m2)
#     sims_ex = loadings_similarity(m1.ex_loadings, m2.ex_loadings).to_numpy().diagonal()
#     print(sims_ex)
#     sims_em = loadings_similarity(m1.em_loadings, m2.em_loadings).to_numpy().diagonal()
#     print(sims_em)
#     pair_labels = '{m1} vs. {m2}'.format(m1=labels[k], m2=labels[-1 - k])
#     similarities_ex[pair_labels] = sims_ex
#     print(similarities_ex)
#     similarities_em[pair_labels] = sims_em
#     print(similarities_em)
# similarities_ex = pd.DataFrame.from_dict(similarities_ex, orient='index', columns=['C{i}'.format(i=i + 1) for i in range(3)])
# similarities_em = pd.DataFrame.from_dict(similarities_em, orient='index', columns=['C{i}'.format(i=i + 1) for i in range(3)])
#
# print(similarities_ex)
# print(similarities_em)


# similarities_ex, similarities_em = split_validation.compare()
#
# print(subset_specific_models)
# print(similarities_ex)



# abs_stack, ex_range_abs, _, _ = read_abs_dataset(folder_path=folder_path, index_pos=(0, -7))
#
# fig = plot_eem(eem_stack[0], ex_range, em_range, auto_intensity_range=False, vmin=0, vmax=2000, plot_tool='plotly',
#                rotate=False, fix_aspect_ratio=True, title='2024-05-13-06')

# eem_dataset_1 = EEMDataset(eem_stack[0:5], ex_range, em_range, index=index[0:5], ref=np.arange(5))
# eem_dataset_2 = EEMDataset(eem_stack[5:10], ex_range, em_range, index=index[5:10], ref=np.arange(10))
# eem_dataset_1.cutting(270,350,310,600, copy=False)
# eem_dataset_1.rayleigh_masking(width_o1=20, width_o2=20, copy=False, interpolation_method_o1='linear',
#                              interpolation_method_o2='linear',
#                              interpolation_axis_o2='grid')
# eem_dataset_1.raman_masking(width=20, copy=False, interpolation_method='linear')
# parafac1 = PARAFAC(rank=3)
# parafac1.fit(eem_dataset_1)
#
# eem_dataset_2.cutting(270,350,310,600, copy=False)
# eem_dataset_2.rayleigh_masking(width_o1=20, width_o2=20, copy=False, interpolation_method_o1='linear',
#                              interpolation_method_o2='linear',
#                              interpolation_axis_o2='grid')
# eem_dataset_2.raman_masking(width=20, copy=False, interpolation_method='linear')
# parafac2 = PARAFAC(rank=4)
# parafac2.fit(eem_dataset_2)
#
# fig = plot_loadings({'model1': parafac1, 'model2': parafac2}, plot_tool='plotly')

# c_q = []
# time = []
#
# for i in index:
#     time.append(re.search(r'S1(.+?)\+', i).group(1))
#     m = re.search(r'\+(.+?)gLKI', i)
#
#     if m:
#         c = m.group(1)
#         c_q.append(float(c.replace("_", ".")))
#
# raw_data = {}
# for t in list(set(time)):
#     raw_data[t] = {"eem_stack": [], "ref": []}
#
# for i, (c, t) in enumerate(zip(c_q, time)):
#     raw_data[t]['eem_stack'].append(eem_stack[i])
#     raw_data[t]['ref'].append(c)
#
# data = {}
# for key, value in raw_data.items():
#     eem_dataset = EEMDataset(eem_stack=np.array(value['eem_stack']), ref=np.array(value['ref']),
#                              ex_range=ex_range, em_range=em_range)
#     eem_dataset.sort_by_ref()
#     eem_dataset.ife_correction(abs_stack, ex_range_abs, copy=False)
#     eem_dataset.gaussian_filter(sigma=1, copy=False)
#     eem_dataset.cutting(200,400,310,500, copy=False)
#     eem_dataset.rayleigh_masking(width_o1=10, width_o2=10, copy=False, interpolation_method_o2='nan')
#     eem_dataset.raman_masking(width=10, copy=False, interpolation_method='nan')
#     # for i in range(eem_dataset.eem_stack.shape[0]):
#         # plot_eem(eem_dataset.eem_stack[i], eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=4000,
#         #          auto_intensity_range=False, figure_size=(15, 7))
#         # plt.title(key + ': ' + str(eem_dataset.ref[i]))
#         # plt.show()
#     eem_dataset.eem_stack = eem_dataset.eem_stack[0] / eem_dataset.eem_stack - 1
#     # for i in range(eem_dataset.eem_stack.shape[0]):
#         # plot_eem(eem_dataset.eem_stack[i], eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=1.5,
#         #          auto_intensity_range=False, figure_size=(15, 7))
#         # plt.title(key + ': ' + str(eem_dataset.ref[i]))
#         # plt.show()
#
#     corr_dict = eem_dataset.correlation(fit_intercept=False)
#     for k in corr_dict.keys():
#         if k == 'intercept':
#             plot_eem(corr_dict[k], eem_dataset.ex_range, eem_dataset.em_range, vmin=-0.2, vmax=0.2,
#                      auto_intensity_range=False, figure_size=(15, 7))
#             plt.title(key + k)
#             plt.show()
#         elif k == 'slope':
#             plot_eem(corr_dict[k], eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=0.4,
#                      auto_intensity_range=False, figure_size=(15, 7))
#             plt.title(key + k)
#             plt.show()
#         elif k == 'r_square':
#             plot_eem(1-corr_dict[k], eem_dataset.ex_range, eem_dataset.em_range, vmin=0.0001, vmax=0.1, scale_type='log',
#                      auto_intensity_range=False, figure_size=(15, 7), n_cbar_ticks=4)
#             plt.title(key +  k)
#             plt.show()

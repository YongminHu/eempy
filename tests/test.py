import numpy as np
from matplotlib import pyplot as plt

from eempy.read_data import read_eem_dataset, read_abs_dataset
from eempy.eem_processing import EEMDataset, PARAFAC
from eempy.plot import plot_eem
import re


folder_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240304_BSA'

eem_stack, ex_range, em_range, index = read_eem_dataset(folder_path=folder_path, index_pos=(0, -7))
abs_stack, ex_range_abs, _, _ = read_abs_dataset(folder_path=folder_path, index_pos=(0, -7))
c_q = []
time = []

for i in index:
    time.append(re.search(r'S1(.+?)\+', i).group(1))
    m = re.search(r'\+(.+?)gLKI', i)

    if m:
        c = m.group(1)
        c_q.append(float(c.replace("_", ".")))

raw_data = {}
for t in list(set(time)):
    raw_data[t] = {"eem_stack": [], "ref": []}

for i, (c, t) in enumerate(zip(c_q, time)):
    raw_data[t]['eem_stack'].append(eem_stack[i])
    raw_data[t]['ref'].append(c)

data = {}
for key, value in raw_data.items():
    eem_dataset = EEMDataset(eem_stack=np.array(value['eem_stack']), ref=np.array(value['ref']),
                             ex_range=ex_range, em_range=em_range)
    eem_dataset.sort_by_ref()
    # eem_dataset.ife_correction(abs_stack, ex_range_abs, copy=False)
    # eem_dataset.gaussian_filter(sigma=1, copy=False)
    eem_dataset.cutting(200,400,310,500, copy=False)
    eem_dataset.rayleigh_masking(width_o1=10, width_o2=10, copy=False, interpolation_method_o2='nan')
    # eem_dataset.raman_masking(width=10, copy=False, interpolation_method='nan')
    eem_dataset.eem_stack = eem_dataset.eem_stack[0] / eem_dataset.eem_stack - 1
    for i in range(eem_dataset.eem_stack.shape[0]):
        plot_eem(eem_dataset.eem_stack[i], eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=1.5,
                 auto_intensity_range=False, figure_size=(12, 7))
        plt.title(key + ': ' + str(eem_dataset.ref[i]))
        plt.show()

# corr_dict = eem_dataset.correlation(fit_intercept=False)
# for k in corr_dict.keys():
#     if k == 'intercept':
#         plot_eem(corr_dict[k], eem_dataset.ex_range, eem_dataset.em_range, vmin=-0.2, vmax=0.2,
#                  auto_intensity_range=False, figure_size=(15, 7))
#         plt.title(k)
#         plt.show()
#     elif k == 'slope':
#         plot_eem(corr_dict[k], eem_dataset.ex_range, eem_dataset.em_range, vmin=0.15, vmax=0.25,
#                  auto_intensity_range=False, figure_size=(15, 7))
#         plt.title(k)
#         plt.show()
#     elif k == 'r_square':
#         plot_eem(1-corr_dict[k], eem_dataset.ex_range, eem_dataset.em_range, vmin=0.0001, vmax=0.1, scale_type='log',
#                  auto_intensity_range=False, figure_size=(15, 7), n_cbar_ticks=4)
#         plt.title(k)
#         plt.show()



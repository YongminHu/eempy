import numpy as np
from matplotlib import pyplot as plt

from eempy.read_data import read_eem_dataset, read_abs_dataset
from eempy.eem_processing import EEMDataset, PARAFAC
from eempy.plot import plot_eem
import re


folder_path = 'C:/PhD/Fluo-detect/_data/_greywater/20240117_GW_M3'

eem_stack, ex_range, em_range, index = read_eem_dataset(folder_path=folder_path, index_pos=(4, -8))
abs_stack, ex_range_abs, _, _ = read_abs_dataset(folder_path=folder_path, index_pos=(4, -8))
c_q = []
time = []

for i in index:
    time.append(i[:16])
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
    eem_dataset.ife_correction(abs_stack, ex_range_abs, copy=False)
    eem_dataset.gaussian_filter(sigma=1, copy=False)
    eem_dataset.cutting(275,400,275,500, copy=False)
    eem_dataset.rayleigh_masking(width_o1=15, width_o2=15, copy=False)
    eem_dataset.raman_masking(width=8, copy=False)
    plot_eem(eem_dataset.eem_stack[0], eem_dataset.ex_range, eem_dataset.em_range)
    plt.show()


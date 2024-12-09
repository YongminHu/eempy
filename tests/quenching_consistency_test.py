import numpy as np

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching_nmf/sample_300_ex_274_em_300_raman_15_interpolated_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)

def get_q_coef(eem_dataset, model, kw_o, kw_q):
    m = model
    m.fit(eem_dataset)
    fmax_tot = m.fmax
    ref_tot = eem_dataset.ref
    fmax_original = fmax_tot[fmax_tot.index.str.contains(kw_o)]
    fmax_quenched = fmax_tot[fmax_tot.index.str.contains(kw_q)]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    ref_original = ref_tot[ref_tot.index.str.contains(kw_o)]
    corr, p = pearsonr(ref_original, fmax_original)
    return fmax_ratio, corr, p

kw_dict = {
    'normal_july': [['M3'], ['2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17']],
    'stagnation_july': [['M3'], ['2024-07-18', '2024-07-19']],
    'G1': [['G1'], []],
    'G2': [['G2'], []],
    'G3': [['G3'], []],
    'normal_october': [['M3'], ['2024-10-16', '2024-10-22']],
    'high_flow': [['M3'], ['2024-10-16', '2024-10-17']],
    'stagnation_october': [['M3'], ['2024-10-16', '2024-10-18']],
    'breakthrough': [['M3'], ['2024-10-16', '2024-10-21']],
}

model = PARAFAC(n_components=4)

fig1, ax1 = plt.subplots()
for name, kw in kw_dict.items():
    eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
        eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
    eem_dataset_specific = EEMDataset(eem_stack_filtered,
                                      ex_range=eem_dataset.ex_range,
                                      em_range=eem_dataset.em_range,
                                      index=index_filtered,
                                      ref=ref_filtered)
    fmax_ratio, corr, p = get_q_coef(eem_dataset_specific, model, kw[0], kw[1])
    ratio_std = np.std(fmax_ratio)
    ax1.plot(ratio_std, corr, label=name)
fig1.legend()

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching_nmf/sample_300_ex_274_em_300_raman_15_interpolated_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)

def gest_q_coef(eem_dataset, model, kw_o, kw_q):
    m = model
    m.fit(eem_dataset)
    fmax_tot = m.fmax
    ref_tot = eem_dataset.ref
    fmax_original = fmax_tot[fmax_tot.index.str.contains(kw_o)]
    fmax_quenched = fmax_tot[fmax_tot.index.str.contains(kw_q)]
    fmax_ratio = fmax_original / fmax_quenched
    ref_original = ref_tot[ref_tot.index.str.contains(kw_o)]
    corr, p = pearsonr(ref_original, fmax_original)
    return fmax_ratio, corr, p
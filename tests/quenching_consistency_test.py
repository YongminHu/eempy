import numpy as np

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_300_ex_274_em_300_raman_15_interpolated_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)

def get_q_coef(eem_dataset, model, kw_o, kw_q, fmax_col):
    m = model
    m.fit(eem_dataset)
    fmax_tot = m.fmax.iloc[:, fmax_col]
    ref_tot = eem_dataset.ref['TCC']
    fmax_original = fmax_tot[fmax_tot.index.str.contains(kw_o)]
    fmax_quenched = fmax_tot[fmax_tot.index.str.contains(kw_q)]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    nan_rows = ref_tot[ref_tot.isna()].index
    ref_dropna = ref_tot.drop(nan_rows)
    fmax_dropna = fmax_tot.drop(nan_rows)
    cor, p = pearsonr(ref_dropna, fmax_dropna)
    return fmax_ratio, cor, p


def leave_one_out_test(eem_dataset: EEMDataset, model, kw_o, kw_q, fmax_col):
    eem_stack_full = eem_dataset.eem_stack
    index_full = eem_dataset.index
    ref_full = eem_dataset.ref
    ex_range = eem_dataset.ex_range
    em_range = eem_dataset.em_range
    fmax_ratio_full, cor_full, p_full = get_q_coef(eem_dataset, model, kw_o, kw_q, fmax_col)
    ratio_std_full = np.std(fmax_ratio_full)
    ratio_std_diff, cor_diff, p_diff = [], [], []
    for i in range(int(eem_stack_full.shape[0]/2)):
        eem_stack_test = np.delete(eem_stack_full, [2*i, 2*i+1], 0)
        index_test = [idx for pos, idx in enumerate(index_full) if pos not in [2*i, 2*i+1]]
        ref_test = ref_full.drop(index_full[2*i: 2*i+2])
        eem_dataset_test = EEMDataset(eem_stack_test,
                                      ex_range=ex_range,
                                      em_range=em_range,
                                      ref=ref_test,
                                      index=index_test)
        fmax_ratio_test, cor_test, p_test = get_q_coef(eem_dataset_test, model, kw_o, kw_q, fmax_col)
        ratio_std_test = np.std(fmax_ratio_test)
        ratio_std_diff.append(ratio_std_test - ratio_std_full)
        cor_diff.append(cor_test - cor_full)
        p_diff.append(p_test - p_full)
    return ratio_std_diff, cor_diff, p_diff


kw_dict = {
    'normal_july': [['M3'], ['2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17']],
    'stagnation_july': [['M3'], ['2024-07-18', '2024-07-19']],
    'G1': [['G1'], None],
    'G2': [['G2'], None],
    'G3': [['G3'], None],
    'normal_october': [['M3'], ['2024-10-16', '2024-10-22']],
    'high_flow': [['M3'], ['2024-10-17']],
    'stagnation_october': [['M3'], ['2024-10-18']],
    'breakthrough': [['M3'], ['2024-10-21']],
}

kw_dict_trial = {
    'high_flow': [['M3'], ['2024-10-17']],
    'stagnation_october': [['M3'], ['2024-10-18']],
    'breakthrough': [['M3'], ['2024-10-21']],
}

model = PARAFAC(n_components=4)


# # -------------Correlation between Pearson-r and std(F0/F) in different operating conditions---------
# fig_corr = go.Figure()
# fig_box = go.Figure()
# for name, kw in kw_dict.items():
#     eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
#         eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#     eem_dataset_specific = EEMDataset(eem_stack_filtered,
#                                       ex_range=eem_dataset.ex_range,
#                                       em_range=eem_dataset.em_range,
#                                       index=index_filtered,
#                                       ref=ref_filtered)
#     fmax_ratio, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 0)
#     ratio_std = np.std(fmax_ratio)
#     fig_corr.add_trace(go.Scatter(
#         x=[cor],
#         y=[ratio_std],
#         mode='markers',  # Use 'lines', 'markers+lines', etc., for other styles
#         marker=dict(
#             size=12,  # Marker size
#             symbol='circle',  # Marker symbol
#             opacity=0.8  # Transparency
#         ),
#         name=name  # Legend name for this trace
#     ))
#     fig_box.add_trace(go.Box(y=fmax_ratio, name=f"{name}", width=0.5))
#
# fig_corr.update_layout(
#     title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title="Pearson r",
#     yaxis_title="std(F0/F)",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
# fig_corr.show()
#
# fig_box.update_layout(
#     title="Boxplot for Iterated Arrays",
#     xaxis_title="Scenario",
#     yaxis_title="F0/F",
#     boxmode="group"  # Group the boxes together
# )
# # Show the plot
# fig_box.show()


# --------------Leave-one-out-test in different operating conditions---------------------
n_cols = 3
n_rows = len(kw_dict)//n_cols + 1
fig_change = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(kw_dict.keys()))

for i, (name, kw) in enumerate(kw_dict.items()):
    eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
        eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
    eem_dataset_specific = EEMDataset(eem_stack_filtered,
                                      ex_range=eem_dataset.ex_range,
                                      em_range=eem_dataset.em_range,
                                      index=index_filtered,
                                      ref=ref_filtered)
    ratio_std_change, cor_change, p_change = leave_one_out_test(eem_dataset_specific, model, 'B1C1', 'B1C2', 0)
    fig_change.add_trace(
        go.Scatter(x=ratio_std_change, y=cor_change, text=name, mode='markers'),
        row=i//n_cols + 1, col=i % n_cols + 1
    )
    fig_change.update_xaxes(title_text='change in std(F0/F)')
    fig_change.update_yaxes(title_text='change in Pearson r')
fig_change.show()


import numpy as np

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_298_ex_274_em_310_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_stack, index, ref, cluster, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)
eem_dataset = EEMDataset(eem_stack, eem_dataset.ex_range, eem_dataset.em_range, index, ref, cluster)

# eem_dataset_ref_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
# eem_dataset_ref = read_eem_dataset_from_json(eem_dataset_ref_path)
# standard_bacteria_component = eem_dataset_ref.eem_stack[-5]


def get_q_coef(eem_dataset, model, kw_o, kw_q, fmax_col, ref_name, standard_component):
    m = model
    m.fit(eem_dataset)
    if type(model).__name__ == 'PARAFAC':
        fmax_tot = m.fmax
    elif type(model).__name__ == 'EEMNMF':
        fmax_tot = m.nnls_fmax
    else:
        return
    ref_tot = eem_dataset.ref[ref_name]
    nan_rows = ref_tot[ref_tot.isna()].index
    ref_dropna = ref_tot.drop(nan_rows)
    fmax_dropna = fmax_tot.drop(nan_rows)

    cor = -2
    p = 1
    if fmax_col == 'best_corr':
        for i in range(fmax_dropna.shape[1]):
            fmax_i = fmax_dropna.iloc[:, i]
            cor_i, p_i = pearsonr(ref_dropna, fmax_i)
            if cor_i > cor:
                cor = cor_i
                fmax_col = i
                p = p_i
    elif fmax_col == 'best_component':
        similarity = -2
        for i in range(m.components.shape[0]):
            fmax_i = fmax_dropna.iloc[:, i]
            cor_i, p_i = pearsonr(ref_dropna, fmax_i)
            c1_unfolded, c2_unfolded = [m.components[i].reshape(-1), standard_component.reshape(-1)]
            similarity_i = stats.pearsonr(c1_unfolded, c2_unfolded)[0]
            if similarity_i > similarity:
                cor = cor_i
                fmax_col = i
                p = p_i
                similarity = similarity_i
    else:
        cor, p = pearsonr(ref_dropna, fmax_dropna.iloc[:, fmax_col])

    fmax_slice = fmax_tot.iloc[:, fmax_col]
    fmax_original = fmax_slice[fmax_slice.index.str.contains(kw_o)]
    fmax_quenched = fmax_slice[fmax_slice.index.str.contains(kw_q)]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()

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
    'normal_july': [['M3'], ['2024-07-12', '2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17']],
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

n_components = 4
model = PARAFAC(n_components=n_components)
model.fit(eem_dataset)
standard_bacteria_component = model.components[0]


# -------------Correlation between Pearson-r and std(F0/F) in different operating conditions---------


# --------for the most correlated component
fig_sq_corr = go.Figure()
fig_sq_box = go.Figure()
for name, kw in kw_dict.items():
    eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
        eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
    eem_dataset_specific = EEMDataset(eem_stack_filtered,
                                      ex_range=eem_dataset.ex_range,
                                      em_range=eem_dataset.em_range,
                                      index=index_filtered,
                                      ref=ref_filtered)
    fmax_ratio, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 'best_component', 'TCC',
                                    standard_bacteria_component)
    ratio_std = np.std(fmax_ratio)
    fig_sq_corr.add_trace(go.Scatter(
        x=[cor],
        y=[ratio_std],
        mode='markers',  # Use 'lines', 'markers+lines', etc., for other styles
        marker=dict(
            size=12,  # Marker size
            symbol='circle',  # Marker symbol
            opacity=0.8  # Transparency
        ),
        name=name  # Legend name for this trace,
    ))
    fig_sq_box.add_trace(go.Box(y=fmax_ratio, name=f"{name}", width=0.5))
fig_sq_corr.show()
fig_sq_box.show()

# # --------iterate through all components--------
# n_cols = 2
# n_rows = len(kw_dict)//n_cols + 1
# fig_sq_corr = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'component {i}' for i in range(n_components)])
# fig_sq_box = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'component {i}' for i in range(n_components)])
# for i in range(n_components):
#     for name, kw in kw_dict.items():
#         eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
#             eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#         eem_dataset_specific = EEMDataset(eem_stack_filtered,
#                                           ex_range=eem_dataset.ex_range,
#                                           em_range=eem_dataset.em_range,
#                                           index=index_filtered,
#                                           ref=ref_filtered)
#         _, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 'best', 'TCC')
#         fmax_ratio, _, _ = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', i, 'TCC')
#         ratio_std = np.std(fmax_ratio)
#         fig_sq_corr.add_trace(go.Scatter(
#             x=[cor],
#             y=[ratio_std],
#             mode='markers',  # Use 'lines', 'markers+lines', etc., for other styles
#             marker=dict(
#                 size=12,  # Marker size
#                 symbol='circle',  # Marker symbol
#                 opacity=0.8  # Transparency
#             ),
#             name=name  # Legend name for this trace,
#         ),
#             row=i // n_cols + 1, col=i % n_cols + 1)
#         fig_sq_box.add_trace(go.Box(y=fmax_ratio, name=f"{name}", width=0.5), row=i // n_cols + 1, col=i % n_cols + 1)
#
# fig_sq_corr.update_layout(
#     title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title="Pearson r",
#     yaxis_title="std(F0/F)",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
#
# fig_sq_box.update_layout(
#     title="Boxplot for Iterated Arrays",
#     xaxis_title="Scenario",
#     yaxis_title="F0/F",
#     boxmode="group"  # Group the boxes together
# )
#
# fig_sq_corr.show()
# fig_sq_box.show()


# # --------------Leave-one-out-test in different operating conditions---------------------
# n_cols = 3
# n_rows = len(kw_dict)//n_cols + 1
# fig_loo = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(kw_dict.keys()))
#
# for i, (name, kw) in enumerate(kw_dict.items()):
#     eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
#         eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#     eem_dataset_specific = EEMDataset(eem_stack_filtered,
#                                       ex_range=eem_dataset.ex_range,
#                                       em_range=eem_dataset.em_range,
#                                       index=index_filtered,
#                                       ref=ref_filtered)
#     ratio_std_change, cor_change, p_change = leave_one_out_test(eem_dataset_specific, model, 'B1C1', 'B1C2', 0)
#     fig_loo.add_trace(
#         go.Scatter(x=ratio_std_change, y=cor_change, text=name, mode='markers'),
#         row=i//n_cols + 1, col=i % n_cols + 1
#     )
#     fig_loo.update_xaxes(title_text='change in std(F0/F)')
#     fig_loo.update_yaxes(title_text='change in Pearson r')
# fig_loo.show()

# # -------------Leave-scenario-out-test--------------------
# fmax_ratio_full, cor_full, p_full = get_q_coef(eem_dataset, model, 'B1C1', 'B1C2', 0)
# ratio_std_full = np.std(fmax_ratio_full)
# ratio_std_diff, cor_diff, p_diff = [], [], []
# fig_lso_corr = go.Figure()
# fig_lso_box = go.Figure()
# fig_lso_diff = go.Figure()
# for name, kw in kw_dict.items():
#     eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
#         eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#     positions = [eem_dataset.index.index(value) for value in index_filtered]
#     eem_stack_test = np.delete(eem_dataset.eem_stack, positions, 0)
#     index_test = [idx for pos, idx in enumerate(eem_dataset.index) if pos not in positions]
#     ref_test = eem_dataset.ref.drop(index_filtered)
#     eem_dataset_test = EEMDataset(eem_stack_test,
#                                   ex_range=eem_dataset.ex_range,
#                                   em_range=eem_dataset.em_range,
#                                   ref=ref_test,
#                                   index=index_test)
#     fmax_ratio, cor_test, p_test = get_q_coef(eem_dataset_test, model, 'B1C1', 'B1C2', 0)
#     ratio_std_test = np.std(fmax_ratio)
#     fig_lso_corr.add_trace(go.Scatter(
#         x=[cor_test],
#         y=[ratio_std_test],
#         mode='markers',  # Use 'lines', 'markers+lines', etc., for other styles
#         marker=dict(
#             size=12,  # Marker size
#             symbol='circle',  # Marker symbol
#             opacity=0.8  # Transparency
#         ),
#         name=name  # Legend name for this trace
#     ))
#     fig_lso_box.add_trace(go.Box(y=fmax_ratio, name=f"{name}", width=0.5))
#     ratio_std_diff.append(ratio_std_test-ratio_std_full)
#     cor_diff.append(cor_test - cor_full)
#     p_diff.append(p_test - p_full)
#     fig_lso_diff.add_trace(
#         go.Scatter(x=[ratio_std_test-ratio_std_full], y=[cor_test - cor_full], name=f"{name}", mode='markers'),
#     )
#
# fig_lso_diff.update_layout(
#     title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title="change in std(F0/F)",
#     yaxis_title="change in Pearson r",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
# fig_lso_diff.show()
#
# fig_lso_corr.update_layout(
#     title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title="Pearson r",
#     yaxis_title="std(F0/F)",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
# fig_lso_corr.show()
#
# fig_lso_box.update_layout(
#     title="Boxplot for Iterated Arrays",
#     xaxis_title="Scenario",
#     yaxis_title="F0/F",
#     boxmode="group"  # Group the boxes together
# )
# # Show the plot
# fig_lso_box.show()

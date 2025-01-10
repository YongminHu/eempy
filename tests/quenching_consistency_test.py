import numpy as np

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_282_ex_274_em_310_mfem_7_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)


# eem_dataset_ref_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
# eem_dataset_ref = read_eem_dataset_from_json(eem_dataset_ref_path)
# standard_bacteria_component = eem_dataset_ref.eem_stack[-5]


def get_q_coef(eem_dataset, model, kw_o, kw_q, fmax_col, ref_name, standard_component=None):
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
        cor = [pearsonr(ref_dropna, fmax_dropna.iloc[:, c])[0] for c in fmax_col]
        p = [pearsonr(ref_dropna, fmax_dropna.iloc[:, c])[1] for c in fmax_col]

    fmax_slice = fmax_tot.iloc[:, fmax_col]
    fmax_original = fmax_slice[fmax_slice.index.str.contains(kw_o)]
    fmax_quenched = fmax_slice[fmax_slice.index.str.contains(kw_q)]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    fmax_ratio_df = pd.DataFrame(fmax_ratio, index=fmax_original.index)
    return fmax_ratio, fmax_ratio_df, cor, p


def leave_one_out_test(eem_dataset: EEMDataset, model, kw_o, kw_q, fmax_col, ref_name, standard_component=None):
    eem_stack_full = eem_dataset.eem_stack
    index_full = eem_dataset.index
    ref_full = eem_dataset.ref
    ex_range = eem_dataset.ex_range
    em_range = eem_dataset.em_range
    fmax_ratio_full, _, cor_full, p_full = get_q_coef(eem_dataset, model, kw_o, kw_q, fmax_col, ref_name,
                                                      standard_component)
    ratio_std_full = np.std(fmax_ratio_full)
    ratio_std_diff, cor_diff, p_diff = [], [], []
    for i in range(int(eem_stack_full.shape[0] / 2)):
        eem_stack_test = np.delete(eem_stack_full, [2 * i, 2 * i + 1], 0)
        index_test = [idx for pos, idx in enumerate(index_full) if pos not in [2 * i, 2 * i + 1]]
        ref_test = ref_full.drop(index_full[2 * i: 2 * i + 2])
        eem_dataset_test = EEMDataset(eem_stack_test,
                                      ex_range=ex_range,
                                      em_range=em_range,
                                      ref=ref_test,
                                      index=index_test)
        fmax_ratio_test, _, cor_test, p_test = get_q_coef(eem_dataset_test, model, kw_o, kw_q, fmax_col, ref_name,
                                                          standard_component)
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
eem_dataset_standard, _ = eem_dataset.filter_by_index(kw_dict['normal_july'][0], kw_dict['normal_july'][1], copy=True)
model_standard = PARAFAC(n_components=4)
model_standard.fit(eem_dataset_standard)
standard_bacteria_component = model_standard.components[0]

# # ------------Boxplots of F0/F for each operating condition------------
#
# n_cols = 3
# n_rows = len(kw_dict) // n_cols + 1
# fig_box_for_each_operating_condition = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(kw_dict.keys()))
#
# for i, (name, kw) in enumerate(kw_dict.items()):
#     eem_dataset_specific, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
#     fmax_ratio, fmax_ratio_df, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 'best_component', 'TCC',
#                                                    standard_bacteria_component)
#     for j in range(n_components):
#         fmax_ratio, fmax_ratio_df, cor_tcc, p_tcc = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2',
#                                                                j, 'TCC')
#         fmax_ratio, fmax_ratio_df, cor_doc, p_doc = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2',
#                                                                j, 'DOC')
#         ratio_std = np.std(fmax_ratio)
#         fig_box_for_each_operating_condition.add_trace(
#             go.Box(
#                 y=fmax_ratio,
#                 name=f"component {j}",
#                 width=1,
#                 showlegend=False,
#             ),
#             row=i // n_cols + 1, col=i % n_cols + 1)
#         fig_box_for_each_operating_condition.add_trace(
#             go.Scatter(
#                 x=[f"component {j}"],  # Position the text at the x location of the box
#                 y=[np.mean(fmax_ratio)],  # Position the text near the mean of the box values
#                 text=[f'std={ratio_std:.2g}<br>cor_tcc={cor_tcc:.2g}<br>cor_doc={cor_doc:.2g}'],  # Custom text
#                 mode='text',  # Only display text
#                 textposition="top right",  # Position text relative to the point
#                 showlegend=False  # Hide legend for this trace
#             ),
#             row=i // n_cols + 1, col=i % n_cols + 1
#         )
# fig_box_for_each_operating_condition.update_layout(width=2500, height=2500)
# fig_box_for_each_operating_condition.show()

# -------------Correlation between Pearson-r and std(F0/F) in different operating conditions---------

# # --------for the most correlated component
# fig_sq_corr = go.Figure()
# fig_sq_box = go.Figure()
# fmax_ratios = []
# for name, kw in kw_dict.items():
#     eem_dataset_specific, _ = (eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#     fmax_ratio, fmax_ratio_df, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 'best_component', 'TCC',
#                                                    standard_bacteria_component)
#     fmax_ratios.append(fmax_ratio_df)
#     ratio_std = np.std(fmax_ratio)
#     fig_sq_corr.add_trace(go.Scatter(
#         x=[cor],
#         y=[ratio_std],
#         mode='markers+text',  # Use 'lines', 'markers+lines', etc., for other styles
#         text=[f'p={p:.2g}'],
#         textposition='top right',
#         marker=dict(
#             size=16,  # Marker size
#             symbol='circle',  # Marker symbol
#             opacity=0.8  # Transparency
#         ),
#         name=name  # Legend name for this trace,
#     ))
#     fig_sq_box.add_trace(go.Box(y=fmax_ratio, name=f"{name}", width=0.5))
# fig_sq_corr.show()
# fig_sq_box.show()

# # --------iterate through all components--------
# n_cols = 2
# n_rows = len(kw_dict)//n_cols + 1
# fig_sq_corr = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'component {i}' for i in range(n_components)])
# fig_sq_box = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f'component {i}' for i in range(n_components)])
# for i in range(n_components):
#     eem_component_standard = model_standard.components[i]
#     for name, kw in kw_dict.items():
#         eem_stack_filtered, index_filtered, ref_filtered, cluster_filtered, sample_number_all_filtered = (
#             eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#         eem_dataset_specific = EEMDataset(eem_stack_filtered,
#                                           ex_range=eem_dataset.ex_range,
#                                           em_range=eem_dataset.em_range,
#                                           index=index_filtered,
#                                           ref=ref_filtered)
#         _, _, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 'best_component', 'TCC', standard_bacteria_component)
#         fmax_ratio, _, _, _ = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', 'best_component', 'TCC', eem_component_standard)
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


# --------------Greedy deletion for individual dataset------------------

# -------Deletion by biggest std reduction-------

eem_dataset_test, _ = eem_dataset.filter_by_index(
    None,
    ["2024-10-16", "2024-10-17", "2024-10-18", "2024-10-22"],
    copy=True
)
fmax_ratio_init, _, cor_init, p_init = get_q_coef(eem_dataset_test, model, 'B1C1', 'B1C2', 'best_component', 'TCC',
                                                  standard_bacteria_component)
ratio_std_init = np.std(fmax_ratio_init)
# n_iter = int(eem_dataset_test.eem_stack.shape[0] * 0.5)
n_iter = 30
ratio_std_trajectory = [ratio_std_init]
cor_trajectory = [cor_init]
p_trajectory = [p_init]
index_removed_trajectory = ['Full']
for i in range(n_iter):
    ratio_std_diff, cor_diff, p_diff = leave_one_out_test(eem_dataset_test, model, 'B1C1', 'B1C2', 'best_component',
                                                          'TCC', standard_bacteria_component)
    biggest_std_diff = min(ratio_std_diff)
    idx_biggest_std_diff = ratio_std_diff.index(biggest_std_diff)
    ratio_std_trajectory.append(ratio_std_trajectory[-1] + biggest_std_diff)
    cor_trajectory.append(cor_trajectory[-1] + cor_diff[idx_biggest_std_diff])
    p_trajectory.append(p_trajectory[-1] + p_diff[idx_biggest_std_diff])
    index_removed_trajectory.append(eem_dataset_test.index[2*idx_biggest_std_diff])
    index_remained = [idx for pos, idx in enumerate(eem_dataset_test.index) if pos not in [2 * idx_biggest_std_diff, 2 * idx_biggest_std_diff + 1]]
    eem_dataset_test.filter_by_index(None, index_remained, copy=False)
fig_greedy = go.Figure()
fig_greedy.add_trace(
    go.Scatter(
        mode="markers+lines",
        x=ratio_std_trajectory,
        y=cor_trajectory,
        text=index_removed_trajectory
    )
)
fig_greedy.update_layout(
    # title="Correlation between Pearson r and std(F0/F)",
    xaxis_title="std(F0/F)",
    yaxis_title="Pearson r",
    showlegend=True,  # Show legend
    template="plotly_white"  # Use a clean template
)
fig_greedy.show()

# # ---------Deletion by most deviated ratio----------
#
# # eem_dataset_test, _ = eem_dataset.filter_by_index(None, ['2024-10-17', '2024-10-18'], copy=True)
#
# eem_dataset_unquenched, _ = eem_dataset.filter_by_index('B1C1', None, copy=True)
# eem_dataset_quenched, _ = eem_dataset.filter_by_index('B1C2', None, copy=True)
# eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched.subsampling(portion=0.35)
# pos = [eem_dataset_unquenched.index.index(idx) for idx in eem_dataset_new_uq.index]
# quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
# eem_dataset_new_q, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
# eem_dataset_test = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])
# eem_dataset_test.sort_by_index()
#
# n_iter = int(eem_dataset_test.eem_stack.shape[0]/2 * 0.5)
# # n_iter = 3
# # n_iter = int(eem_dataset_test.eem_stack.shape[0]/2) - 5
# ratio_std_trajectory = []
# ratio_mean_trajectory = []
# cor_trajectory = []
# p_trajectory = []
# index_removed_trajectory = []
#
# for i in range(n_iter):
#     fmax_ratio, _, cor, p = get_q_coef(eem_dataset_test, model, 'B1C1', 'B1C2', [0,1,2,3], 'TCC',
#                                        standard_bacteria_component)
#     ratio_std = np.std(fmax_ratio, axis=0)
#     ratio_mean = np.mean(fmax_ratio, axis=0)
#     ratio_deviation = np.abs(fmax_ratio - ratio_mean)
#     idx_most_deviated_ratio = np.argmax(np.mean(ratio_deviation, axis=1))
#     ratio_std_trajectory.append(ratio_std[0])
#     ratio_mean_trajectory.append(ratio_mean[0])
#     cor_trajectory.append(cor[0])
#     p_trajectory.append(p[0])
#
#     # ratio_std = np.std(fmax_ratio)
#     # ratio_mean = np.mean(fmax_ratio)
#     # ratio_deviation = np.abs(fmax_ratio - ratio_mean)
#     # idx_most_deviated_ratio = np.argmax(ratio_deviation)
#     # ratio_std_trajectory.append(ratio_std)
#     # ratio_mean_trajectory.append(ratio_mean)
#     # cor_trajectory.append(cor)
#     # p_trajectory.append(p)
#
#     index_removed_trajectory.append(eem_dataset_test.index[2*idx_most_deviated_ratio])
#     index_remained = [idx for pos, idx in enumerate(eem_dataset_test.index) if pos not in [2 * idx_most_deviated_ratio, 2 * idx_most_deviated_ratio + 1]]
#     eem_dataset_test.filter_by_index(None, index_remained, copy=False)
# fig_greedy = go.Figure()
# fig_greedy.add_trace(
#     go.Scatter(
#         mode='markers+lines',
#         x=ratio_std_trajectory,
#         y=cor_trajectory,
#         text=index_removed_trajectory
#     )
# )
# fig_greedy.update_layout(
#     # title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title="std(F0/F)",
#     yaxis_title="Pearson r",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
# fig_greedy.show()

# -------------Model testing------------

# eem_dataset_unquenched, _ = eem_dataset.filter_by_index('B1C1', None, copy=True)
# eem_dataset_quenched, _ = eem_dataset.filter_by_index('B1C2', None, copy=True)
# eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched.subsampling(portion=0.35)
# pos = [eem_dataset_unquenched.index.index(idx) for idx in eem_dataset_new_uq.index]
# quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
# eem_dataset_new_q, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
# eem_dataset_test = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])
# eem_dataset_test.sort_by_index()

# eem_dataset_test, _ = eem_dataset.filter_by_index(['2024-10-'], None, copy=True)
# eem_dataset_train, _ = eem_dataset.filter_by_index(['2024-07-'], None, copy=True)
# model_established = PARAFAC(n_components=4)
# fmax_ratio_established, _, cor_established, p_established = get_q_coef(eem_dataset_train, model_established, 'B1C1',
#                                                                        'B1C2', 'best_component', 'TCC',
#                                                                        standard_bacteria_component)
# model_established.fit(eem_dataset_train)
# _, fmax_test, eem_stack_pred = model_established.predict(eem_dataset_test)
# fmax_test_original = fmax_test[fmax_test.index.str.contains('B1C1')]
# fmax_test_quenched = fmax_test[fmax_test.index.str.contains('B1C2')]
# fmax_test_ratio = fmax_test_original.to_numpy() / fmax_test_quenched.to_numpy()
# ref_test = eem_dataset_test.ref['TCC']
# nan_rows = ref_test[ref_test.isna()].index
# ref_dropna = ref_test.drop(nan_rows)
# fmax_dropna = fmax_test.drop(nan_rows)
#
#
# n_iter = int(eem_dataset_test.eem_stack.shape[0]/2 * 0.75)
# # n_iter = 3
# # n_iter = int(eem_dataset_test.eem_stack.shape[0]/2) - 5
# ratio_std_trajectory = []
# ratio_mean_trajectory = []
# cor_trajectory = []
# p_trajectory = []
# index_removed_trajectory = []
#
# for i in range(n_iter):
#     cor, p = pearsonr(fmax_dropna.iloc[:, 0], ref_dropna)
#     ratio_std = np.std(fmax_test_ratio[:, 0], axis=0)
#     ratio_mean = np.mean(fmax_test_ratio[:, 0], axis=0)
#     ratio_deviation = np.abs(fmax_test_ratio - ratio_mean)
#     idx_most_deviated_ratio = np.argmax(np.mean(ratio_deviation, axis=1))
#     ratio_std_trajectory.append(ratio_std)
#     ratio_mean_trajectory.append(ratio_mean)
#     cor_trajectory.append(cor)
#     p_trajectory.append(p)
#     index_removed_trajectory.append(eem_dataset_test.index[2*idx_most_deviated_ratio])
#     fmax_dropna = fmax_dropna.drop(index=fmax_dropna.index[2*idx_most_deviated_ratio: 2*idx_most_deviated_ratio+2])
#     ref_dropna = ref_dropna.drop(index=ref_dropna.index[2 * idx_most_deviated_ratio: 2 * idx_most_deviated_ratio + 2])
#     fmax_test_ratio = np.delete(fmax_test_ratio, idx_most_deviated_ratio, axis=0)
#
# fig_greedy = go.Figure()
# fig_greedy.add_trace(
#     go.Scatter(
#         mode='markers+lines',
#         x=ratio_std_trajectory,
#         y=cor_trajectory,
#         text=index_removed_trajectory
#     )
# )
# fig_greedy.update_layout(
#     # title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title="std(F0/F)",
#     yaxis_title="Pearson r",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
# fig_greedy.show()


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


# ----------------test QPARAFACs----------------

# eem_dataset_test, _ = eem_dataset.filter_by_index(['2024-10-'], None, copy=True)
# base_model = PARAFAC(n_components=4)
# qparafacs = KMethod(base_model, n_initial_splits=3, error_calculation="quenching_coefficient", max_iter=10, tol=0.0001,
#                     elimination='default', kw_unquenched='B1C1', kw_quenched='B1C2')
# # cluster_labels, label_history, error_history = qparafacs.base_clustering(eem_dataset_test)
# consensus_matrix, label_history, error_history = qparafacs.calculate_consensus(eem_dataset_test, 5, 0.8)
#
# # eem_dataset_test = combine_eem_datasets([eem_dataset.filter_by_index(['2024-10-17'], None, copy=True)[0],
# #                                          eem_dataset.filter_by_index(['2024-10-18'], None, copy=True)[0]],)
# # print(eem_dataset_test.index)
# # print(eem_dataset_test.ref)

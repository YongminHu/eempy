from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from scipy.stats import pearsonr
from matplotlib.colors import TABLEAU_COLORS
from datetime import datetime
import seaborn as sns

colors = list(TABLEAU_COLORS.values())
# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_266_ex_274_em_310_mfem_3.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)


# eem_dataset_ref_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
# eem_dataset_ref = read_eem_dataset_from_json(eem_dataset_ref_path)
# standard_bacteria_component = eem_dataset_ref.eem_stack[-5]


def get_q_coef(eem_dataset, model, kw_o, kw_q, fmax_col, ref_name, standard_component=None):
    m = model
    m.fit(eem_dataset)
    if type(model).__name__ == 'PARAFAC':
        fmax_tot = m.nnls_fmax
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


# kw_dict = {
#     'normal_jul': [['M3'], ['2024-07-12', '2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17'], 3],
#     'stagnation_jul': [['M3'], ['2024-07-18', '2024-07-19'], 4],
#     'normal_oct': [['M3'], ['2024-10-16', '2024-10-22'], 4],
#     'stagnation_oct': [['M3'], ['2024-10-18'], 3],
#     'high_flow': [['M3'], ['2024-10-17'], 3],
#     'shortcut 1': [['G3'], None, 4],
#     'shortcut 2': [['G2'], None, 4],
#     'shortcut 3': [['G1'], None, 4],
#     'cross-connection': [['M3'], ['2024-10-21'], 4],
#     # 'all': [['2024'], None, 4]
# }

kw_dict = {
    'nj': [['M3'], ['2024-07-12', '2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17'], 4],
    'sj': [['M3'], ['2024-07-18', '2024-07-19'], 4],
    'no': [['M3'], ['2024-10-16', '2024-10-22'], 4],
    'so': [['M3'], ['2024-10-18'], 3],
    'hf': [['M3'], ['2024-10-17'], 3],
    'sc1': [['G3'], None, 4],
    'sc2': [['G2'], None, 4],
    'sc3': [['G1'], None, 4],
    'cc': [['M3'], ['2024-10-21'], 4],
    # 'all': [['2024'], None, 4]
}

indices_scenarios = {}
for name, kw in kw_dict.items():
    eem_dataset_train, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
    indices_scenarios[name] = eem_dataset_train.index

scenarios_colors = {
    'nj': colors[0],
    'sj': colors[1],
    'no1': colors[2],
    'no2': colors[2],
    'so': colors[3],
    'hf': colors[4],
    'sc1': colors[5],
    'sc2': colors[6],
    'sc3': colors[7],
    'cc': colors[8],
    # 'all': colors[9]
}


kw_dict_type1 = {
    'stagnation_jul': [['M3'], ['2024-07-18', '2024-07-19'], 4],
    'normal_oct': [['M3'], ['2024-10-16', '2024-10-22'], 4],
    'stagnation_oct': [['M3'], ['2024-10-18'], 3],
}

# kw_dict_trial = {
#     'high_flow': [['M3'], ['2024-10-17']],
#     'stagnation_october': [['M3'], ['2024-10-18']],
#     'breakthrough': [['M3'], ['2024-10-21']],
# }

# n_components = 4
# model = PARAFAC(n_components=n_components)
# eem_dataset_standard, _ = eem_dataset.filter_by_index(kw_dict['normal_october'][0], kw_dict['normal_october'][1],
#                                                       copy=True)
# model_standard = PARAFAC(n_components=4)
# model_standard.fit(eem_dataset_standard)
# standard_bacteria_component = model_standard.components[0]

# --------------F0/F vs. TCC vs. DOC-------------

# # ----------using the full model---------
model = PARAFAC(n_components=4)
model.fit(eem_dataset)
fmax = model.nnls_fmax.iloc[:, 0]
tcc = eem_dataset.ref['TCC'].dropna()
doc = eem_dataset.ref['DOC'].dropna()
fmax_original = fmax[fmax.index.str.contains('B1C1')]
fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
for name, idx in indices_scenarios.items():
    sample_number = [i for i in range(len(tcc.index)) if tcc.index[i] in idx]
    x = tcc[sample_number] /0.06/1e6
    time = [datetime.strptime(t[0:16], '%Y-%m-%d-%H-%M') for t in x.index]
    y = fmax_ratio[sample_number]
    plt.plot(x, y, 'o', color=scenarios_colors[name])
plt.show()

for name, kw in kw_dict.items():
    model = PARAFAC(n_components=kw[2])
    eem_dataset_scenario, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
    model.fit(eem_dataset_scenario)
    fmax = model.nnls_fmax.iloc[:, 0]
    tcc = eem_dataset_scenario.ref['TCC'].dropna()
    doc = eem_dataset_scenario.ref['DOC'].dropna()
    fmax_original = fmax[fmax.index.str.contains('B1C1')]
    fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    x = tcc /0.06/1e6
    time = [datetime.strptime(t[0:16], '%Y-%m-%d-%H-%M') for t in x.index]
    ticks = pd.date_range(start=min(time), end=max(time), periods=5)
    y = fmax_ratio
    plt.figure()
    plt.plot(time, y, '-o', color=scenarios_colors[name])
    plt.ylim([0.95, 1.35])
    plt.title(name, fontsize=16)
    plt.xticks(ticks)
    plt.xlabel("Time")
    plt.ylabel("F0/F")
    plt.show()

# # ------------Timeseries for all scenarios------------
#
# n_cols = 2
# n_rows = len(kw_dict) // n_cols + 1
# fig_box_for_each_operating_condition = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(kw_dict.keys()))
#
# for i, (name, kw) in enumerate(kw_dict.items()):
#     eem_dataset_specific, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
#     n_components = kw[2]
#     model = PARAFAC(n_components=n_components)
#     fmax_ratio, fmax_ratio_df, cor_tcc, p_tcc = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2',
#                                                            [k for k in range(n_components)], 'TCC')
#     # if name == "all":
#     #     colors = ['black' if any(kw in idx for kw in ['G1', 'G2', 'G3']) else 'blue' for idx in fmax_ratio_df.index]
#     for j in range(n_components):
#         ratio_std = np.std(fmax_ratio[:, j])
#         fig_box_for_each_operating_condition.add_trace(
#             go.Box(
#                 y=fmax_ratio[:, j].reshape(-1),
#                 name=f"component {j+1}",
#                 width=0.5,
#                 showlegend=False,
#                 boxpoints='all',
#                 text=list(fmax_ratio_df.index),  # Assign custom labels to each point
#                 hoverinfo="y+text",  # Show both value and custom index
#             ),
#             row=i // n_cols + 1, col=i % n_cols + 1)
#         axis_name = f"yaxis{i + 1}"  # yaxis1, yaxis2, yaxis3...
#         fig_box_for_each_operating_condition.update_layout({axis_name: dict(range=[0.95, 1.5])})
# fig_box_for_each_operating_condition.update_layout(width=2000, height=3000)
#
# fig_box_for_each_operating_condition.show()


# # -----------------Predictive power of PARAFAC models--------------
#
# fig = go.Figure()
# template = pd.DataFrame(np.full([len(kw_dict_type1), len(kw_dict)], np.nan),
#                      index=list(kw_dict_type1.keys()),
#                      columns=list(kw_dict.keys()))
# table = {param: template.copy() for param in [
#     'Mean(F0/F)',
#     'Mean(F0/F)diff',
#     'STD(F0/F)',
#     'STD(F0/F)diff',
#     'r_TCC',
#     'p_TCC',
#     'RMSE_TCC',
#     # 'r_DOC',
#     # 'p_DOC',
#     # 'RMSE_DOC'
# ]}
#
# for i, (name_train, kw_train) in enumerate(kw_dict_type1.items()):
#     eem_dataset_train, _ = eem_dataset.filter_by_index(kw_train[0], kw_train[1], copy=True)
#     n_components = kw_train[2]
#     model = PARAFAC(n_components=n_components)
#     model.fit(eem_dataset_train)
#     fmax = model.fmax.iloc[:, 0]
#     fmax_tcc_train = pd.concat([eem_dataset_train.ref['TCC'], fmax], axis=1)
#     fmax_tcc_train = fmax_tcc_train.dropna()
#     cor = -2
#     p = 1
#     cor_tcc_train, p_tcc_train = pearsonr(fmax_tcc_train.iloc[:, 0], fmax_tcc_train.iloc[:, 1])
#     fmax_original = fmax[fmax.index.str.contains('B1C1')]
#     fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
#     fmax_ratio_train = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#     ratio_mean_train = np.mean(fmax_ratio_train)
#     ratio_std_train = np.std(fmax_ratio_train)
#
#     lr = LinearRegression()
#     lr.fit(X=fmax_tcc_train.iloc[:, 0].to_numpy().reshape(-1, 1), y=fmax_tcc_train.iloc[:, 1].to_numpy().reshape(-1, 1))
#     coef = lr.coef_[0]
#     intercept = lr.intercept_[0]
#
#     for j, (name_test, kw_test) in enumerate(kw_dict.items()):
#         if name_test != name_train:
#             eem_dataset_test, _ = eem_dataset.filter_by_index(kw_test[0], kw_test[1], copy=True)
#             _, fmax_test, eem_stack_pred = model.predict(eem_dataset_test)
#             fmax = fmax_test.iloc[:, 0]
#             fmax_tcc_test = pd.concat([eem_dataset_test.ref['TCC'], fmax], axis=1)
#             fmax_tcc_test = fmax_tcc_test.dropna()
#             fmax_original = fmax[fmax.index.str.contains('B1C1')]
#             fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
#             fmax_ratio_test = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#             ratio_mean_test = np.mean(fmax_ratio_test)
#             ratio_std_test = np.std(fmax_ratio_test)
#             cor_tcc_test, p_tcc_test = pearsonr(fmax_tcc_test.iloc[:, 0], fmax_tcc_test.iloc[:, 1])
#             pred_tcc = (fmax - intercept)/coef
#             rmse = np.sqrt(np.mean((fmax_tcc_test.iloc[:, 0] - pred_tcc) ** 2))
#             table['Mean(F0/F)'].iloc[i, j] = ratio_mean_test
#             table['Mean(F0/F)diff'].iloc[i, j] = ratio_mean_train - ratio_mean_test
#             table['STD(F0/F)'].iloc[i, j] = ratio_std_test
#             table['STD(F0/F)diff'].iloc[i, j] = ratio_std_train - ratio_std_test
#             table['r_TCC'].iloc[i, j] = cor_tcc_test
#             table['p_TCC'].iloc[i, j] = p_tcc_test
#             table['RMSE_TCC'].iloc[i, j] = rmse

# for training_dataset in list(kw_dict_type1.keys()):
# #     x_param = 'Mean(F0/F)diff'
# #     y_param = 'r_TCC'
# #
# #     x = table[x_param].loc[training_dataset]
# #     y = table[y_param].loc[training_dataset]
# #
# #     fig = plt.figure()
# #     for i in range(x.size):
# #       if x.iloc[i] is not np.nan and y.iloc[i] is not np.nan:
# #         plt.scatter(x.iloc[i], y.iloc[i], c=colors[i], label=x.index[i])
# #     plt.xlabel(x_param)
# #     plt.ylabel(y_param)
# #     plt.title(training_dataset)
# #     plt.legend()
# #     plt.show()


# # ------------Boxplots of F0/F of all components for each operating condition------------
#
# n_cols = 2
# n_rows = len(kw_dict) // n_cols + 1
# fig_box_for_each_operating_condition = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(kw_dict.keys()))
#
# for i, (name, kw) in enumerate(kw_dict.items()):
#     eem_dataset_specific, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
#     n_components = kw[2]
#     model = PARAFAC(n_components=n_components)
#     fmax_ratio, fmax_ratio_df, cor_tcc, p_tcc = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2',
#                                                            [k for k in range(n_components)], 'TCC')
#     # if name == "all":
#     #     colors = ['black' if any(kw in idx for kw in ['G1', 'G2', 'G3']) else 'blue' for idx in fmax_ratio_df.index]
#     for j in range(n_components):
#         ratio_std = np.std(fmax_ratio[:, j])
#         fig_box_for_each_operating_condition.add_trace(
#             go.Box(
#                 y=fmax_ratio[:, j].reshape(-1),
#                 name=f"component {j+1}",
#                 width=0.5,
#                 showlegend=False,
#                 boxpoints='all',
#                 text=list(fmax_ratio_df.index),  # Assign custom labels to each point
#                 hoverinfo="y+text",  # Show both value and custom index
#             ),
#             row=i // n_cols + 1, col=i % n_cols + 1)
#         axis_name = f"yaxis{i + 1}"  # yaxis1, yaxis2, yaxis3...
#         fig_box_for_each_operating_condition.update_layout({axis_name: dict(range=[0.95, 1.5])})
# fig_box_for_each_operating_condition.update_layout(width=2000, height=3000)
#
# fig_box_for_each_operating_condition.show()


# # -------------Pairwise plots of F0/F----------
#
# from sklearn.cluster import DBSCAN
# from itertools import combinations
#
# # Simulate a dataset with 100 samples and 5 features
#
# fmax_ratio, fmax_ratio_df, cor_tcc, p_tcc = get_q_coef(eem_dataset, model, 'B1C1', 'B1C2',
#                                                        [k for k in range(n_components)], 'TCC')
# feature_names = [f'C{i+1}' for i in range(n_components)]
#
# # # Perform DBSCAN clustering
# # dbscan = DBSCAN(eps=0.09, min_samples=5)
# # labels = dbscan.fit_predict(fmax_ratio)
#
# # # Label the data by conditions
# labels = list(fmax_ratio_df.index)
# for i, (name, kw) in enumerate(kw_dict.items()):
#     eem_dataset_o, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
#     _, filtered_idx = eem_dataset_o.filter_by_index(kw[0], kw[1], copy=True)
#     labels = [name if i in filtered_idx else labels[i] for i in range(len(labels))]
#
# # Get unique clusters, including noise (-1)
# unique_clusters = np.unique(labels)
# colors = {label: f'rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})'
#           for label in unique_clusters}
#
# # Generate all feature combinations
# feature_combinations = list(combinations(range(fmax_ratio.shape[1]), 2))
# num_plots = len(feature_combinations)
#
# # Create subplots layout
# rows = 2  # Adjust number of rows if needed
# cols = 5  # Number of columns for 10 plots
#
# fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{feature_names[i]} vs {feature_names[j]}'
#                                                              for i, j in feature_combinations])
#
# # Add scatter plots for each pair of features
# for idx, (i, j) in enumerate(feature_combinations):
#     row = idx // cols + 1
#     col = idx % cols + 1
#
#     for label in unique_clusters:
#         # cluster_mask = labels == label
#         cluster_mask = [i for i in range(len(labels)) if labels[i] == label]
#         fig.add_trace(
#             go.Scatter(
#                 x=fmax_ratio[cluster_mask, i],
#                 y=fmax_ratio[cluster_mask, j],
#                 mode='markers',
#                 marker=dict(color=colors[label], size=6),
#                 name=f'Cluster {label}' if label != -1 else 'Noise',
#                 legendgroup=f'Cluster {label}',
#                 text=fmax_ratio_df.index[cluster_mask],
#                 hoverinfo="text"
#             ),
#             row=row, col=col
#         )
#
# # Update layout
# fig.update_layout(
#     height=800, width=1500,
#     title_text="Pairwise Feature Plots for DBSCAN Clustering",
#     showlegend=True
# )
#
# fig.show()


# -------------Correlation between Pearson-r and std(F0/F) in different operating conditions---------

# # --------for the most correlated component
# fig_sq_corr = go.Figure()
# fmax_ratios = []
# reference_param = 'DOC'
# for name, kw in kw_dict.items():
#     n_components = kw[2]
#     model = PARAFAC(n_components=n_components)
#     eem_dataset_specific, _ = (eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#     fmax_ratio, fmax_ratio_df, cor, p = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', [0], reference_param,
#                                                    standard_bacteria_component)
#     fmax_ratios.append(fmax_ratio_df)
#     ratio_std = np.std(fmax_ratio)
#     fig_sq_corr.add_trace(go.Scatter(
#         x=cor,
#         y=[ratio_std],
#         mode='markers+text',  # Use 'lines', 'markers+lines', etc., for other styles
#         text=[f'p={p[0]:.2g}'],
#         textposition='top right',
#         marker=dict(
#             size=16,  # Marker size
#             symbol='circle',  # Marker symbol
#             opacity=0.8  # Transparency
#         ),
#         name=name  # Legend name for this trace,
#     ))
# fig_sq_corr.update_layout(
#     # title="Correlation between Pearson r and std(F0/F)",
#     xaxis_title=f"Pearson r between {reference_param} and Fmax",
#     yaxis_title="std(Fmax0/Fmax)",
#     showlegend=True,  # Show legend
#     template="plotly_white"  # Use a clean template
# )
# fig_sq_corr.show()


# # ----------F0/F boxplots with x-axis positioned according to r_TCC--------
#
# fmax_ratios = []
# r_tcc_scenarios = []
#
# for c, (name, kw) in enumerate(kw_dict.items()):
#     n_components = kw[2]
#     model = PARAFAC(n_components=n_components)
#     eem_dataset_train, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
#
#     model.fit(eem_dataset_train)
#     fmax = model.fmax.iloc[:, 0]
#     fmax_tcc_train = pd.concat([eem_dataset_train.ref['TCC'], fmax], axis=1)
#     fmax_tcc_train = fmax_tcc_train.dropna()
#     fmax_doc_train = pd.concat([eem_dataset_train.ref['DOC'], fmax], axis=1)
#     fmax_doc_train = fmax_doc_train.dropna()
#     cor_tcc, p_tcc = pearsonr(fmax_tcc_train.iloc[:, 0], fmax_tcc_train.iloc[:, 1])
#     cor_doc, p_doc = pearsonr(fmax_doc_train.iloc[:, 0], fmax_doc_train.iloc[:, 1])
#
#     fmax_original = fmax[fmax.index.str.contains('B1C1')]
#     fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
#     fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#     fmax_ratios.append(fmax_ratio)
#     r_tcc_scenarios.append(cor_tcc)
#
# fmax_ratio_means = [np.mean(fmax_ratio) for fmax_ratio in fmax_ratios]
#
# # Create figure
# plt.figure(figsize=(8, 6))
#
# boxplot = plt.boxplot(
#     fmax_ratios,
#     positions=r_tcc_scenarios,  # Actual x-axis positions
#     widths=0.1,
#     patch_artist=True,
#     medianprops={'color': 'black'},
#     boxprops={'facecolor': 'grey'},
#     manage_ticks=False  # <--- THIS IS CRUCIAL
# )
#
# # Add red dots for means
# plt.scatter(r_tcc_scenarios, fmax_ratio_means, color='red', zorder=3, s=40)
#
# # Regression line
# slope, intercept = np.polyfit(r_tcc_scenarios, fmax_ratio_means, 1)
# x_regression = np.array([min(r_tcc_scenarios), max(r_tcc_scenarios)])
# y_regression = slope * x_regression + intercept
# plt.plot(x_regression, y_regression)
#
# # Set custom x-axis ticks
# plt.xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])  # Only these ticks will appear
# plt.xlim(-1, 1)  # Force full axis range
#
# # Customize plot
# plt.xlabel('$r_{TCC}$', fontsize=20)
# plt.ylabel('$F_{0}/F$', fontsize=20)
# plt.xlim([-1, 1])
# plt.tick_params(labelsize=18)
# plt.legend()
# plt.tight_layout()
# plt.show()


# -----------r-TCC vs. r-DOC--------------

##------plotly--------
# fig_sq_corr = go.Figure()
# fmax_ratios = []
# for c, (name, kw) in enumerate(kw_dict.items()):
#     n_components = kw[2]
#     model = PARAFAC(n_components=n_components)
#     eem_dataset_specific, _ = (eem_dataset.filter_by_index(kw[0], kw[1], copy=True))
#     fmax_ratio, fmax_ratio_df, cor_tcc, p_tcc = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', [0], "TCC",
#                                                            None)
#     _, _, cor_doc, p_doc = get_q_coef(eem_dataset_specific, model, 'B1C1', 'B1C2', [0], "DOC",
#                                                            None)
#     fmax_ratios.append(fmax_ratio_df)
#     ratio_std = np.std(fmax_ratio)
#     fig_sq_corr.add_trace(go.Scatter(
#         x=cor_tcc,
#         y=cor_doc,
#         mode='markers',  # Use 'lines', 'markers+lines', etc., for other styles
#         textposition='top right',
#         marker=dict(
#             color=colors[c],
#             size=800*ratio_std,  # Marker size
#             symbol='circle',  # Marker symbol
#             opacity=0.8,  # Transparency
#         ),
#         name=name  # Legend name for this trace,
#     ))
# fig_sq_corr.update_layout(
#     height=600,
#     width=800,
#     xaxis=dict(range=[-1, 1]),
#     yaxis=dict(range=[-1, 1]),
#     xaxis_title="Pearson r between TCC and Fmax",
#     yaxis_title="Pearson r between DOC and Fmax",
#     showlegend=True,  # Show legend
# )
# fig_sq_corr.show()


# #-------matplotlib--------
# from matplotlib.font_manager import FontProperties
#
# # Initialize the figure
# fig, ax = plt.subplots(figsize=(9, 8))
#
# for spine in ax.spines.values():
#     spine.set_linewidth(2.5)  # Set the linewidth to make the frame bold
#
# fmax_ratios = []
# scatter_plots = []  # To store scatter plot objects for legend
#
# # Define a set of example sizes for the legend
# example_sizes = [0.01, 0.03, 0.05, 0.07]  # Example standard deviations
# size_labels = [f"{s}" for s in example_sizes]  # Labels for the legend
#
# for c, (name, kw) in enumerate(kw_dict.items()):
#     n_components = kw[2]
#     model = PARAFAC(n_components=n_components)
#     eem_dataset_train, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
#
#     model.fit(eem_dataset_train)
#     fmax = model.fmax.iloc[:, 0]
#     fmax_tcc_train = pd.concat([eem_dataset_train.ref['TCC'], fmax], axis=1)
#     fmax_tcc_train = fmax_tcc_train.dropna()
#     fmax_doc_train = pd.concat([eem_dataset_train.ref['DOC'], fmax], axis=1)
#     fmax_doc_train = fmax_doc_train.dropna()
#
#     # lr = LinearRegression(fit_intercept=True)
#     # lr.fit(X=fmax_tcc_train.iloc[:, 0].to_numpy().reshape(-1, 1),
#     #        y=fmax_tcc_train.iloc[:, 1].to_numpy().reshape(-1, 1))
#     # coef = lr.coef_[0]
#     # intercept = lr.intercept_[0]
#     # cor_tcc = lr.score(X=fmax_tcc_train.iloc[:, 0].to_numpy().reshape(-1, 1),
#     #                    y=fmax_tcc_train.iloc[:, 1].to_numpy().reshape(-1, 1))
#     # lr.fit(X=fmax_doc_train.iloc[:, 0].to_numpy().reshape(-1, 1),
#     #        y=fmax_doc_train.iloc[:, 1].to_numpy().reshape(-1, 1))
#     # cor_doc = lr.score(X=fmax_doc_train.iloc[:, 0].to_numpy().reshape(-1, 1),
#     #                    y=fmax_doc_train.iloc[:, 1].to_numpy().reshape(-1, 1))
#     cor_tcc, p_tcc = pearsonr(fmax_tcc_train.iloc[:, 0], fmax_tcc_train.iloc[:, 1])
#     cor_doc, p_doc = pearsonr(fmax_doc_train.iloc[:, 0], fmax_doc_train.iloc[:, 1])
#
#     fmax_original = fmax[fmax.index.str.contains('B1C1')]
#     fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
#     fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#
#     ratio_std = np.std(fmax_ratio)
#     print(ratio_std)
#
#     # Plot the scatter plot and store the scatter object
#     scatter = ax.scatter(
#         cor_tcc,
#         cor_doc,
#         s=5000 * ratio_std,  # Marker size
#         edgecolor='black',
#         facecolor=colors[c],
#         linewidth=1,
#         alpha=0.8,  # Transparency
#         label=name  # Legend name for this trace (used for color legend)
#     )
#     scatter_plots.append(scatter)
#
# # Set plot limits and labels
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
# ax.axhline(0, color='black', linewidth=1.5)
# ax.axvline(0, color='black', linewidth=1.5)
# ax.tick_params(axis='both', which='major', labelsize=18)
# ax.set_xlabel(r"$r_{TCC}$", fontsize=28)
# ax.set_ylabel(r"$r_{DOC}$", fontsize=28)
#
# name_legend = ax.legend(loc="lower left", title="Dataset", labelspacing=0.8, fontsize=16, ncol=3,
#                         bbox_to_anchor=(0, 0.2), columnspacing=0.75,
#                         title_fontproperties=FontProperties(size=18, weight='bold'))
# ax.add_artist(name_legend)  # Add the color legend to the plot
#
# # Create a custom legend for marker sizes
# size_handles = [
#     plt.scatter([], [], s=5000 * s, edgecolor='black', facecolor='white', linewidth=1, alpha=0.8, label=size_labels[i])
#     for i, s in enumerate(example_sizes)
# ]
# size_legend = ax.legend(handles=size_handles, title="Std(F0/F)", loc="lower left",
#                         bbox_to_anchor=(0, 0), labelspacing=0.8, fontsize=16, ncol=2,
#                         title_fontproperties=FontProperties(size=18, weight='bold'))
# ax.add_artist(size_legend)  # Add the size legend to the plot
#
# # Show the plot
# plt.tight_layout()
# plt.show()


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

# --------Fig 1: Nc vs. r_tcc and mean(F0/F)--------
# --------Histplots of model on testing dataset: effect of N components-------
dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   # '2024-07-13',
                                                   # '2024-07-15',
                                                   # '2024-07-16',
                                                   # '2024-07-17',
                                                   # '2024-07-18',
                                                   # '2024-07-19',
                                                   # '2024-10-17',
                                                   # '2024-10-21',
                                                   # 'G1',
                                                   # 'G2',
                                                   # 'G3',
                                                   '2024-'
                                               ]
                                               )

r_list = [3, 4, 5, 6]
n_outliers = 40
fmax_col = 2
target_name = 'DOC'
hist_fmax_ratio = plt.figure()
for r_i in r_list:
    model = PARAFAC(n_components=r_i, init='svd', non_negativity=True,
                    tf_normalization=True, sort_em=True, loadings_normalization='maximum')
    model.fit(dataset_train)
    target_test_re = dataset_train.ref[target_name]
    valid_indices_test_re = target_test_re.index[~target_test_re.isna()]
    target_test_re = target_test_re.dropna().to_numpy()
    fmax_test_re = model.nnls_fmax
    fmax_original_test_re = fmax_test_re[fmax_test_re.index.str.contains('B1C1')]
    mask_test_re = fmax_original_test_re.index.isin(valid_indices_test_re)
    fmax_original_test_re = fmax_original_test_re[mask_test_re]
    fmax_quenched_test_re = fmax_test_re[fmax_test_re.index.str.contains('B1C2')]
    fmax_quenched_test_re = fmax_quenched_test_re[mask_test_re]
    fmax_ratio_test_re = fmax_original_test_re.to_numpy() / fmax_quenched_test_re.to_numpy()
    fmax_ratio_target_test_re = fmax_ratio_test_re[:, fmax_col]
    r_target_test_re, p_target_test_re = pearsonr(target_test_re, fmax_original_test_re.iloc[:, fmax_col])
    sns.histplot(fmax_ratio_target_test_re, binwidth=0.0125, binrange=(0.85, 1.325), kde=True, stat='density',
                 alpha=0.3, label=f'{r_i}-component, r={r_target_test_re:.2f}')
plt.legend()
plt.tight_layout()
plt.show()

#--------Fig 2: applying established model to quantify new samples: outlier detection-------

dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   # '2024-07-13',
                                                   # '2024-07-15',
                                                   # '2024-07-16',
                                                   # '2024-07-17',
                                                   # '2024-07-18',
                                                   # '2024-07-19',
                                                   # '2024-10-17',
                                                   # '2024-10-21',
                                                   # 'G1',
                                                   # 'G2',
                                                   # 'G3',
                                                   '2024-'
                                               ]
                                               )

dataset_test, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   # '2024-07-15',
                                                   # '2024-07-16',
                                                   # '2024-07-17',
                                                   # '2024-07-18',
                                                   # '2024-07-19',
                                                   '2024-10-'
                                                   # '2024-10-17',
                                                   # '2024-10-21',
                                                   # 'G1',
                                                   # 'G2',
                                                   # 'G3',
                                               ]
                                               )

indices_test_in_scenarios = {}
for name, kw in kw_dict.items():
    dataset_test_filtered, _ = dataset_test.filter_by_index(kw[0], kw[1], copy=True)
    indices_test_in_scenarios[name] = dataset_test_filtered.index

r = 5
n_outliers = 40
fmax_col = 2
target_name = 'DOC'
model = PARAFAC(n_components=r, init='svd', non_negativity=True,
                tf_normalization=True, sort_em=True, loadings_normalization='maximum')
model.fit(dataset_train)
target_train = dataset_train.ref[target_name]
valid_indices_train = target_train.index[~target_train.isna()]
target_train = target_train.dropna().to_numpy()
fmax_train = model.nnls_fmax
fmax_original_train = fmax_train[fmax_train.index.str.contains('B1C1')]
mask_train = fmax_original_train.index.isin(valid_indices_train)
fmax_original_train = fmax_original_train[mask_train]
fmax_quenched_train = fmax_train[fmax_train.index.str.contains('B1C2')]
fmax_quenched_train = fmax_quenched_train[mask_train]
fmax_ratio_train = fmax_original_train.to_numpy() / fmax_quenched_train.to_numpy()
fmax_ratio_target_train = fmax_ratio_train[:, fmax_col]
r_target_train, p_target_train = pearsonr(target_train, fmax_original_train.iloc[:, fmax_col])
slope_train, intercept_train = np.polyfit(target_train, fmax_original_train.iloc[:, fmax_col], deg=1)

target_test_true = dataset_test.ref[target_name]
valid_indices_test = target_test_true.index[~target_test_true.isna()]
target_test_true = target_test_true.dropna().to_numpy()
_, fmax_test, _ = model.predict(eem_dataset=dataset_test)
fmax_original_test = fmax_test[fmax_test.index.str.contains('B1C1')]
mask_test = fmax_original_test.index.isin(valid_indices_test)
fmax_original_test = fmax_original_test[mask_test]
fmax_quenched_test = fmax_test[fmax_test.index.str.contains('B1C2')]
fmax_quenched_test = fmax_quenched_test[mask_test]
fmax_ratio_test = fmax_original_test.to_numpy() / fmax_quenched_test.to_numpy()
fmax_ratio_target_test = fmax_ratio_test[:, fmax_col]
r_target_test, p_target_test = pearsonr(target_test_true, fmax_original_test.iloc[:, fmax_col])
slope_test, intercept_test = np.polyfit(target_test_true, fmax_original_test.iloc[:, fmax_col], deg=1)
target_test_pred = (fmax_original_test.iloc[:, fmax_col] - intercept_train) / slope_train

absolute_error = target_test_true - target_test_pred
relative_error = target_test_true / target_test_pred
partitioned_indices = np.argpartition(fmax_ratio_target_test, -n_outliers)[-n_outliers:]


# ---------Histplot of F0/F for training and testing, with outliers labeled---------

threshold = 1.27 # 1.225 for DOC, 1.15 for TCC
binrange = (1, 1.4)
plt.figure()
sns.histplot(fmax_ratio_target_train, binwidth=0.01, binrange=binrange, kde=True, stat='density', color="blue",
             alpha=0.5, label='training')
ax = sns.histplot(fmax_ratio_target_test, binwidth=0.01, binrange=binrange, kde=True, stat='density', color="orange",
             alpha=0.5, label='testing')
sns.histplot([0], binwidth=0.01, binrange=binrange, kde=True, stat='density', color="white",
             alpha=0.3, label='outliers', hatch='////', edgecolor='red')
for bar in ax.patches:
    # Calculate the midpoint of the bin
    bin_left = bar.get_x()
    bin_width = bar.get_width()
    bin_mid = bin_left + bin_width / 2

    # Check if the midpoint is above the threshold
    if bin_mid >= threshold:
        # Add hatch pattern and change color
        bar.set_hatch("////")      # Hatch pattern (e.g., "////", "xxx", "..")
        bar.set_edgecolor('red')
plt.xlim(binrange)
plt.xlabel("$F_{0}/F$", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=18)
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()

# ---------Fmax vs. TCC or DOC in training and testing------------

plt.figure()
plt.scatter(target_train, fmax_original_train.iloc[:, fmax_col], label='training', color='blue')
plt.scatter(target_test_true[fmax_ratio_target_test <= threshold], fmax_original_test.iloc[fmax_ratio_target_test <= threshold, fmax_col],
            label='testing', color='green')
plt.scatter(target_test_true[fmax_ratio_target_test > threshold], fmax_original_test.iloc[fmax_ratio_target_test > threshold, fmax_col],
            label='outliers', color='red')
plt.xlabel('TCC (#/mL)')
plt.ylabel('C1 Fmax')
# plt.xlabel('DOC (mg/L)')
# plt.ylabel('C3 Fmax')
plt.legend()
plt.show()

# ---------Histplot of outlier rates in each scenario---------
outlier_indices = fmax_original_test.index[fmax_ratio_target_test >= threshold]
outlier_rates = {}
for name, indices in indices_test_in_scenarios.items():
    n_outliers = 0
    for idx in indices:
        if idx in outlier_indices:
            n_outliers += 1
    outlier_rates[name] = (n_outliers / len(indices)*100*2) if indices else 0
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(list(outlier_rates.keys())[2:], list(outlier_rates.values())[2:], color='red')
ax.set_ylim([0, 100])
ax.set_ylabel('Outlier rate (%)', fontsize=18)
ax.tick_params(labelsize=20)
fig.tight_layout()
fig.show()

#---------Fig 3: timeseries of anomalies detected-----------

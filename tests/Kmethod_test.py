import matplotlib.pyplot as plt

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from scipy.stats import pearsonr
from matplotlib.colors import TABLEAU_COLORS
from scipy.stats import ks_2samp
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import pickle
import seaborn as sns


colors = list(TABLEAU_COLORS.values())


# ------------Read EEM dataset-------------
def get_eem_dataset_stats(eem_dataset, base_model):
    base_model.fit(eem_dataset)
    fmax = base_model.nnls_fmax.iloc[:, 0]
    fmax_tcc = pd.concat([eem_dataset.ref['TCC (million #/mL)'], fmax], axis=1)
    fmax_tcc = fmax_tcc.dropna()
    fmax_doc = pd.concat([eem_dataset.ref['DOC (mg/L)'], fmax], axis=1)
    fmax_doc = fmax_doc.dropna()
    tcc_mean = np.mean(fmax_tcc['TCC (million #/mL)'])
    tcc_std = np.std(fmax_tcc['TCC (million #/mL)'])
    doc_mean = np.mean(fmax_doc['DOC (mg/L)'])
    doc_std = np.std(fmax_doc['DOC (mg/L)'])
    if fmax.shape[0] > 2:
        cor_tcc, p_tcc = pearsonr(fmax_tcc.iloc[:, 0], fmax_tcc.iloc[:, 1])
        cor_doc, p_doc = pearsonr(fmax_doc.iloc[:, 0], fmax_doc.iloc[:, 1])
    else:
        cor_tcc, p_tcc = None, None
        cor_doc, p_doc = None, None
    fmax_original = fmax[fmax.index.str.contains('B1C1')]
    fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    ratio_mean = np.mean(fmax_ratio)
    ratio_median = np.median(fmax_ratio)
    ratio_std = np.std(fmax_ratio)
    stats = {
        'n_samples': fmax.shape[0],
        'Mean(F0/F)': ratio_mean,
        'Median (F0/F)': ratio_median,
        'Std(F0/F)': ratio_std,
        'Mean(TCC)': tcc_mean,
        'Std(TCC)': tcc_std,
        'r_TCC': cor_tcc,
        'p_TCC': p_tcc,
        'Mean(DOC)': doc_mean,
        'Std(DOC)': doc_std,
        'r_DOC': cor_doc,
        'p_DOC': p_doc,
    }
    return stats


def k_method_quenching(eem_dataset, base_model, n_clusters, minimum_dataset_size, depth, consensus_only=True,
                       n_base_clusterings=30, subsampling_portion=0.8):
    clustered_datasets = {"0": eem_dataset}
    cluster_stats = get_eem_dataset_stats(eem_dataset, base_model)
    cluster_stats = pd.DataFrame(cluster_stats, index=['0'])
    consensus_matrices = {}
    for i in range(depth):
        clustered_datasets_old = copy.deepcopy(clustered_datasets)
        for mother_code, mother_dataset in clustered_datasets_old.items():
            if len(mother_code) == i + 1:
                base_model = PARAFAC(n_components=4)
                kmodel = KMethod(
                    base_model=base_model,
                    n_initial_splits=max(n_clusters),
                    distance_metric="quenching_coefficient",
                    max_iter=10,
                    tol=0.005,
                    kw_top='B1C1',
                    kw_bot='B1C2'
                )
                kmodel.calculate_consensus(mother_dataset, n_base_clusterings=n_base_clusterings,
                                           subsampling_portion=subsampling_portion)
                consensus_matrices[mother_code] = kmodel.consensus_matrix
            if consensus_only:
                return consensus_matrices, cluster_stats
            else:
                best_score = 0
                eem_clusters = None
                for n in n_clusters:
                    kmodel.hierarchical_clustering(mother_dataset, n_clusters=n)
                    score = kmodel.silhouette_score
                    cluster_sizes = [m.eem_stack.shape[0] for m in kmodel.eem_clusters.values()]
                    if min(cluster_sizes) <= minimum_dataset_size:
                        break
                    if score > best_score:
                        best_score = score
                        eem_clusters = kmodel.eem_clusters
                if eem_clusters is not None:
                    for cluster_number, cluster_dataset in eem_clusters.items():
                        children_code = mother_code + str(cluster_number)
                        clustered_datasets[children_code] = cluster_dataset
                        cluster_i_stats = get_eem_dataset_stats(cluster_dataset, base_model)
                        cluster_stats.loc[children_code] = cluster_i_stats
    return clustered_datasets, cluster_stats


def k_method_consensus_to_clusters_stats(eem_dataset_combinations, consensus_matrices_combos, base_model, n_clusters):
    cluster_stats_all_combos = {}
    clustered_dataset_all_combos = {}
    for combo_code, combo_consensus in consensus_matrices_combos.items():
        kmodel = KMethod(
            base_model=base_model,
            n_initial_splits=5,
            distance_metric="quenching_coefficient",
            max_iter=10,
            tol=0.005,
            kw_top='B1C1',
            kw_bot='B1C2'
        )
        dataset_0 = eem_dataset_combinations[combo_code]
        o_sample_indices = [index for index, value in enumerate(dataset_0.index) if 'B1C1' in value]
        # q_sample_indices = [index for index, value in enumerate(dataset_0.index) if 'B1C2' in value]
        consensus = combo_consensus['0'][o_sample_indices, :]
        consensus = consensus[:, o_sample_indices]
        dataset_0_o, _ = dataset_0.filter_by_index(['B1C1'], None, copy=True)
        dataset_0_q, _ = dataset_0.filter_by_index(['B1C2'], None, copy=True)
        kmodel.consensus_matrix = consensus
        kmodel.hierarchical_clustering(dataset_0_o, n_clusters=n_clusters)
        clustered_dataset_combo = {}
        for c, d in kmodel.eem_clusters.items():
            idx_q = [dataset_0_q.index[i] for i in range(len(dataset_0_o.index)) if dataset_0_o.index[i] in d.index]
            dataset_cluster = combine_eem_datasets(
                [
                    d,
                    dataset_0.filter_by_index(None, idx_q, copy=True)[0]
                ]
            )
            clustered_dataset_combo['0' + str(c)] = dataset_cluster

        cluster_stats = get_eem_dataset_stats(dataset_0, base_model)
        cluster_stats = pd.DataFrame(cluster_stats, index=['0'])
        for cluster_code, cluster_dataset in clustered_dataset_combo.items():
            cluster_i_stats = get_eem_dataset_stats(cluster_dataset, base_model)
            cluster_stats.loc[cluster_code] = cluster_i_stats
        cluster_stats_all_combos[combo_code] = cluster_stats
        clustered_dataset_all_combos[combo_code] = clustered_dataset_combo
    return cluster_stats_all_combos, clustered_dataset_all_combos


# ------------Conduct clustering on EEM dataset based on F0/F-------------

# # -------step 1: calculate consensus----------

eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_260_ex_274_em_310_mfem_3.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)

consensus_matrices, cluster_stats = k_method_quenching(
    eem_dataset=eem_dataset,
    base_model=PARAFAC(n_components=4),
    n_clusters=[5],
    minimum_dataset_size=10,
    depth=1,
    consensus_only=True,
    n_base_clusterings=100,
    subsampling_portion=0.8
)

# with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/consensus_matrices_all.pkl",
#           'wb') as file:
#     pickle.dump(consensus_matrices, file)


# # --------step 2: hierarchical clustering------------

with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/consensus_matrices_all.pkl",
          'rb') as file:
    consensus_matrices = pickle.load(file)

cluster_stats_all_combos, clustered_dataset_all_combos = k_method_consensus_to_clusters_stats(
    eem_dataset_combinations={'all': eem_dataset},
    consensus_matrices_combos={'all': consensus_matrices},
    base_model=PARAFAC(n_components=4),
    n_clusters=2
)

# #--------step 3: more in-depth clustering------------

eem_cluster1 = clustered_dataset_all_combos['all']['02']
eem_cluster2 = clustered_dataset_all_combos['all']['01']

consensus_matrices, cluster_stats = k_method_quenching(
    eem_dataset=eem_cluster2,
    base_model=PARAFAC(n_components=4),
    n_clusters=[5],
    minimum_dataset_size=10,
    depth=1,
    consensus_only=True,
    n_base_clusterings=100,
    subsampling_portion=0.8
)
# with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/consensus_matrices_cluster1.pkl",
#           'wb') as file:
#     pickle.dump(consensus_matrices, file)


# -------------------

with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/consensus_matrices_cluster1.pkl",
          'rb') as file:
    consensus_matrices = pickle.load(file)

cluster_stats_all_combos, clustered_dataset_all_combos = k_method_consensus_to_clusters_stats(
    eem_dataset_combinations={'cluster1': eem_cluster2},
    consensus_matrices_combos={'cluster1': consensus_matrices},
    base_model=PARAFAC(n_components=4),
    n_clusters=2
)
eem_cluster2 = clustered_dataset_all_combos['cluster1']['01']
eem_cluster3 = clustered_dataset_all_combos['cluster1']['02']


# consensus_matrices, cluster_stats = k_method_quenching(
#     eem_dataset=eem_cluster1,
#     base_model=PARAFAC(n_components=4),
#     n_clusters=[4],
#     minimum_dataset_size=5,
#     depth=1,
#     consensus_only=True,
#     n_base_clusterings=30,
#     subsampling_portion=0.8
# )
#
# cluster_stats_all_combos, clustered_dataset_all_combos = k_method_consensus_to_clusters_stats(
#     eem_dataset_combinations={'cluster1': eem_cluster1},
#     consensus_matrices_combos={'cluster1': consensus_matrices},
#     base_model=PARAFAC(n_components=4),
#     n_clusters=2
# )
#
# eem_cluster1 = clustered_dataset_all_combos['cluster1']['01']
# eem_cluster2 = clustered_dataset_all_combos['cluster1']['02']

with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/all_5clusters.pkl",
          'rb') as file:
    list_5clusters = pickle.load(file)

d1d2 = combine_eem_datasets([list_5clusters[0], list_5clusters[1]])
list_4clusters = [d1d2, list_5clusters[2], list_5clusters[4], list_5clusters[3]]

# cluster_labels = []
# for l in eem_dataset.index:
#     if l in list_4clusters[0].index:
#         cluster_labels.append(1)
#     elif l in list_4clusters[1].index:
#         cluster_labels.append(2)
#     elif l in list_4clusters[2].index:
#         cluster_labels.append(3)
#     elif l in list_4clusters[3].index:
#         cluster_labels.append(4)
#
sorted_indices = np.argsort(labels)
matrix = consensus_matrices['0']
# Step 2: Rearrange matrix and labels
sorted_matrix = matrix[sorted_indices, :][:, sorted_indices]
sorted_labels = np.array(labels)[sorted_indices]

plt.imshow(sorted_matrix, cmap='Reds')
plt.show()

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# Convert to condensed form
condensed = squareform(1-matrix)

# Hierarchical clustering
Z = linkage(condensed, method='average')  # or 'ward', 'single', etc.

# Get cluster labels for a given number of clusters, e.g., k=5
labels = fcluster(Z, t=8, criterion='maxclust')

# #---------step 3: analysis and plotting

# colors = ['darkgoldenrod', 'olivedrab', 'royalblue']
# eem_cluster1 = clustered_dataset_all_combos['all']['01']
# eem_cluster2 = clustered_dataset_all_combos['all']['02']
# eem_cluster3 = combine_eem_datasets([clustered_dataset_all_combos['all']['03'],
#                                      clustered_dataset_all_combos['all']['04'],
#                                      ])
sfc_pair1 = [0, 'TCC (million #/mL)']
sfc_pair2 = [1, 'DOC (mg/L)']
sfc_pair3 = [2, 'DOC (mg/L)']
sfc_pair4 = [3, 'DOC (mg/L)']

def calculate_f0f_sfc(dataset, fmax_col, target_ref):
    model = PARAFAC(n_components=4)
    model.fit(dataset)
    fmax = model.nnls_fmax
    target = cluster.ref[target_ref]
    valid_indices = target.index[~target.isna()]
    target = target.dropna().to_numpy()
    fmax_original = fmax[fmax.index.str.contains('B1C1')]
    mask = fmax_original.index.isin(valid_indices)
    fmax_original = fmax_original[mask]
    fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
    fmax_quenched = fmax_quenched[mask]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    sfc =  target / fmax_original.iloc[:, fmax_col].to_numpy()
    return fmax_ratio[:, fmax_col], sfc

# plt.figure()
# for i, cluster in enumerate([eem_cluster1, eem_cluster2, eem_cluster3]):
#     model = PARAFAC(n_components=4)
#     model.fit(cluster)
#     fmax = model.fmax
#     # ------sfc1-------
#     _, sfc1 = calculate_f0f_sfc(cluster, sfc_pair1[0], sfc_pair1[1])
#     # --------sfc2------
#     _, sfc2 = calculate_f0f_sfc(cluster, sfc_pair2[0], sfc_pair2[1])
#     # ---------plot-------
#     plt.plot(sfc1, sfc2, 'o', markersize=6, markeredgecolor='black', color=colors[i])
#     plt.xlabel(f'C{sfc_pair1[0]+1} Fmax/{sfc_pair1[1][0:3]}')
#     plt.ylabel(f'C{sfc_pair2[0] + 1} Fmax/{sfc_pair2[1][0:3]}')
# plt.show()

#----------boxplots--------
# Define colors for the boxes
# colors = ['orangered', 'gold', 'olivedrab', 'royalblue']
colors = ['red', 'red', 'dimgrey', 'dimgrey']

f0fs = []
sfcs = []
for pair in [sfc_pair1, sfc_pair2, sfc_pair3, sfc_pair4]:
    f0fs_r = []
    sfcs_r = []
    for i, cluster in enumerate(list_4clusters[::-1]):
        f0f, sfc = calculate_f0f_sfc(cluster, pair[0], pair[1])
        f0fs_r.append(f0f)
        sfcs_r.append(sfc)
    f0fs.append(f0fs_r)
    sfcs.append(sfcs_r)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)
for i, ax in enumerate(axes):
    box = ax.boxplot(f0fs[i], vert=False, patch_artist=True)  # Horizontal boxplot with colors
    # Apply colors to boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_yticks((4,3,2,1))  # Positions of the groups (1, 2, 3, 4)
    ax.set_yticklabels(['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4'], fontsize=12)  # Labels for the groups
    ax.set_xlabel(f"C{i+1} apparent " + "$F_{0}/F$", fontsize=14)
    ax.set_xlim([0.98, 1.45])
    ax.tick_params(labelsize=10)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)

for i, ax in enumerate(axes):
    sns.violinplot(data=f0fs[i][::-1], orient="h", ax=ax, palette=colors[::-1],  cut=0, bw_adjust=0.5)  # Horizontal violin plot
    ax.set_yticks([0, 1, 2, 3])  # Adjust positions of the groups
    ax.set_yticklabels(['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4'], fontsize=12)  # Labels for the groups
    ax.set_xlabel(f"C{i+1} apparent " + "$F_{0}/F$", fontsize=14)
    ax.set_xlim([0.98, 1.45])
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()


# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)
# for i, ax in enumerate(axes):
#     box = ax.boxplot(f0fs[i][:-1], vert=False, patch_artist=True)  # Horizontal boxplot with colors
#     # Apply colors to boxes
#     for patch, color in zip(box['boxes'], ['red', 'black', 'black']):
#         patch.set_facecolor(color)
#     ax.set_yticks((3,2,1))  # Positions of the groups (1, 2, 3, 4)
#     ax.set_yticklabels(['cluster 1', 'cluster 2', 'cluster 3'], fontsize=12)  # Labels for the groups
#     ax.set_xlabel(f"C{i+1} " + "$F_{0}/F$", fontsize=14)
# plt.tick_params(labelsize=12)
# plt.tight_layout()
# plt.show()

xlabels = [
    'TCC/C1 $F_{max}$ (million #/mL/A.U.)',
    'DOC/C2 $F_{max}$ (mg/L/A.U.)',
    'DOC/C3 $F_{max}$ (mg/L/A.U.)',
    'DOC/C4 $F_{max}$ (mg/L/A.U.)',
]

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)
for i, ax in enumerate(axes):
    box = ax.boxplot(sfcs[i], vert=False, patch_artist=True)  # Horizontal boxplot with colors
    # Apply colors to boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_yticks((4, 3, 2, 1))  # Positions of the groups (1, 2, 3, 4)
    ax.set_yticklabels(['all', 'cluster 1', 'cluster 2', 'cluster 3'], fontsize=12)  # Labels for the groups
    # ax.set_xlabel(f"C{i+1} " + "Fmax/" + [sfc_pair1, sfc_pair2, sfc_pair3, sfc_pair4][i][1], fontsize=14)
    ax.set_xlabel(xlabels[i], fontsize=14)
    if i > 0:
        ax.set_xlim([0, 0.01])
    ax.tick_params(labelsize=14)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)

for i, ax in enumerate(axes):
    sns.violinplot(data=sfcs[i][::-1], orient="h", ax=ax, palette=colors[::-1], cut=0, bw_adjust=0.5)  # Horizontal violin plot
    ax.set_yticks([0, 1, 2, 3])  # Adjusted to match violin plot indexing
    ax.set_yticklabels(['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4'], fontsize=12)  # Labels for the groups
    ax.set_xlabel(xlabels[i], fontsize=14)

    # if i > 0:
    #     ax.set_xlim([0, 0.01])  # Apply x-axis limit conditionally
    # else:
    #     ax.set_xlim([])

    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()


#
# # ----------cluster vs. sample categories--------
colors = ['royalblue', 'olivedrab', 'gold', 'orangered']

categories_kw_dict = {
    # 'Jul-low MBR influence': [['2024-07'], None],
    # 'Oct-low MBR influence': [['2024-10', 'M3'], ['2024-10-16', '2024-10-18', '2024-10-22']],
    # 'Oct-high MBR influence': [['2024-10'], ['G1', 'G2', 'G3', '2024-10-17', '2024-10-21']]
    'July-all': [['2024-07'], None],
    'October-all': [['2024-10'], None],
    'High flow': [['2024-10-17'], None],
    'Simulated\ncross-connection': [['2024-10-21'], None],
    'BAC top': [['G1'], None],
    'BAC middle': [['G2'], None],
    'BAC bottom': [['G3'], None],
    'October-others$*$': [['M3'], ['2024-10-16', '2024-10-22']]
}

categories_indices = {}
for name, kw in categories_kw_dict.items():
    eem_dataset_filtered, _ = eem_dataset.filter_by_index(kw[0], kw[1])
    categories_indices[name] = eem_dataset_filtered.index

# cluster_categories_counts = {}
# for category, indices in categories_indices.items():
#     counts_list = []
#     for cluster in [eem_cluster1, eem_cluster2, eem_cluster3]:
#         counts = len(set(indices) & set(cluster.index))
#         counts_list.append(counts/2)
#     cluster_categories_counts[category] = counts_list

cluster_categories_counts = {}
for cluster_data, cluster_name in zip(list_4clusters, ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4']):
    counts_list = []
    for category, indices in categories_indices.items():
        counts = len(set(indices) & set(cluster_data.index))
        counts_list.append(counts/len(indices)*100)
    cluster_categories_counts[cluster_name] = counts_list


fig, ax = plt.subplots(figsize=(6, 8))
bottom = np.zeros(8)
# colors_bar = ['#1f77b4', '#2ca02c', '#ff7f0e']

for i, (name, weight_count) in enumerate(cluster_categories_counts.items()):
    p = ax.barh(
        list(categories_kw_dict.keys()),
        # ['cluster 1', 'cluster 2', 'cluster 3'],
        weight_count, 0.3,
        label=name, left=bottom, color=colors[i]
    )
    bottom += weight_count

ax.tick_params(labelsize=16, axis='both', rotation=0)
ax.legend(loc="best", fontsize=16, bbox_to_anchor=(1.05, -0.1), ncol=2, handlelength=0.5)
ax.set_xlabel('Share (%)', fontsize=16)
ax.set_xlim([0, 100])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
#
#
#
# ------------Fmax vs. TCC with or without outlier cluster------

model_full = PARAFAC(n_components=4).fit(eem_dataset)
dataset_outlier_removed = combine_eem_datasets([eem_cluster1, eem_cluster2])
model_outlier_removed = PARAFAC(n_components=4).fit(dataset_outlier_removed)
model_in_use = model_full
dataset_in_use = eem_dataset

fmax = model_in_use.fmax
target = dataset_in_use.ref['TCC (million #/mL)']
valid_indices = target.index[~target.isna()]
target = target.dropna().to_numpy()
fmax_original = fmax[fmax.index.str.contains('B1C1')]
mask = fmax_original.index.isin(valid_indices)
fmax_original = fmax_original[mask]

outlier_mask = [True if idx in eem_cluster3.index else False for idx in fmax_original.index]

plt.figure()
# a, b = np.polyfit(target, fmax_original.iloc[:, 0], deg=1)
a, b = np.polyfit(target, fmax_original.iloc[:, 0], deg=1)
plt.plot(
    [-1, 10],
    a * np.array([-1, 10]) + b,
    '--',
    color='grey',
    label='reg.'
)
fmax_original_target_numpy = fmax_original.to_numpy()

plt.scatter(target[[not b for b in outlier_mask]], fmax_original_target_numpy[[not b for b in outlier_mask]][:, 0], label='other clusters', color='black', alpha=0.6)
plt.scatter(target[outlier_mask], fmax_original_target_numpy[outlier_mask][:, 0], label='outlier cluster(s)', color='red', alpha=0.6)

plt.xlabel('TCC (million #/mL)', fontsize=20)
plt.ylabel(f'C1 Fmax', fontsize=20)
plt.legend(
    # bbox_to_anchor=[1.02, 0.37],
    # bbox_to_anchor=[0.58, 0.63],
    fontsize=16
)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.xlim([0, 2.5])
plt.ylim([0, 2500])
plt.show()

print(pearsonr(fmax_original_target_numpy[:, 0], target))

# --------------Outlier removal--------------

def outlier_removal(eem_dataset, clustered_datasets, base_model, target_depth, n_steps='max', method='ks',
                    set_point=None):
    stats_full_dataset = get_eem_dataset_stats(eem_dataset, base_model)
    outlier_removal_stats = pd.DataFrame(stats_full_dataset, index=['0'])
    clustered_datasets_filtered = {key: value for key, value in clustered_datasets.items() if
                                   len(key) == target_depth + 1}
    if n_steps == 'max':
        n_steps = len(clustered_datasets_filtered) - 1
    for n in range(n_steps):
        distance_stats = {}
        # Compare each group to the pooled others
        for i, (code, dataset_excluded) in enumerate(clustered_datasets_filtered.items()):
            eem_dataset_remained = combine_eem_datasets(
                [list(clustered_datasets_filtered.values())[j] for j in range(len(clustered_datasets_filtered)) if
                 j != i]
            )
            base_model.fit(eem_dataset_remained)
            fmax_remained = base_model.nnls_fmax.iloc[:, 0]
            fmax_original = fmax_remained[fmax_remained.index.str.contains('B1C1')]
            fmax_quenched = fmax_remained[fmax_remained.index.str.contains('B1C2')]
            fmax_ratio_remained = fmax_original.to_numpy() / fmax_quenched.to_numpy()
            base_model.fit(dataset_excluded)
            fmax_excluded = base_model.nnls_fmax.iloc[:, 0]
            fmax_original = fmax_excluded[fmax_excluded.index.str.contains('B1C1')]
            fmax_quenched = fmax_excluded[fmax_excluded.index.str.contains('B1C2')]
            fmax_ratio_excluded = fmax_original.to_numpy() / fmax_quenched.to_numpy()

            # Compute KS statistic
            if method == 'ks':
                ks, ks_p_value = ks_2samp(fmax_ratio_excluded, fmax_ratio_remained)
                distance_stats[code] = ks
            elif method == 'set_point':
                if set_point is None:
                    raise ValueError("set_point is not defined")
                else:
                    diff = np.abs(np.mean(fmax_ratio_remained) - set_point)
                    distance_stats[code] = diff

        if method == 'ks':
            outlier_code = max(distance_stats, key=lambda k: distance_stats[k])
        elif method == 'set_point':
            outlier_code = min(distance_stats, key=lambda k: distance_stats[k])
        clustered_datasets_filtered = {key: value for key, value in clustered_datasets_filtered.items() if
                                       key != outlier_code}
        eem_dataset_cleaned = combine_eem_datasets(
            [list(clustered_datasets_filtered.values())[j] for j in range(len(clustered_datasets_filtered))]
        )
        if len(eem_dataset_cleaned.index) > 9:
            stats_cleaned_dataset = get_eem_dataset_stats(eem_dataset_cleaned, base_model)
            outlier_removal_stats.loc['-' + outlier_code] = stats_cleaned_dataset
    return outlier_removal_stats


# ---------------go through combinations--------------
#
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_260_ex_274_em_310_mfem_3.json"
eem_dataset_work = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset_work, _ = eem_dataset_work.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)

# kw_dict_conditions = {
#     'normal_jul': [['M3'], ['2024-07-12', '2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17'], 3],
#     'stagnation_jul': [['M3'], ['2024-07-18', '2024-07-19'], 4],
#     'normal_oct': [['M3'], ['2024-10-16', '2024-10-22'], 4],
#     'stagnation_oct': [['M3'], ['2024-10-18'], 3],
#     'high_flow': [['M3'], ['2024-10-17'], 3],
#     'shortcut 1': [['G3'], None, 4],
#     'shortcut 2': [['G2'], None, 4],
#     'shortcut 3': [['G1'], None, 4],
#     'cross-connection': [['M3'], ['2024-10-21'], 4],
# }
#
# eem_dataset_pool = {}
# for name, kw in kw_dict_conditions.items():
#     eem_dataset_conditioned, _ = eem_dataset_work.filter_by_index(kw[0], kw[1], copy=True)
#     eem_dataset_pool[name] = eem_dataset_conditioned
#
# code_combinations = itertools.combinations(eem_dataset_pool.keys(), 5)
# eem_dataset_combinations = {
#     combo: combine_eem_datasets([eem_dataset_pool[code] for code in combo]) for combo in code_combinations
# }

kw_dict_types = {
    'type 1': [['2024-07-'], None],
    'other types': [['2024-10'], None]
}

eem_dataset_pool = {}
for name, kw in kw_dict_types.items():
    eem_dataset_conditioned, _ = eem_dataset_work.filter_by_index(kw[0], kw[1], copy=True)
    eem_dataset_pool[name] = eem_dataset_conditioned
# type1_proportion = 36 / eem_dataset_pool['type 1'].eem_stack.shape[0]
other_type_proportion = 120 / eem_dataset_pool['other types'].eem_stack.shape[0]
eem_dataset_combinations = {}
for i in range(30):
    # eem_dataset_unquenched_type1, _ = eem_dataset_pool['type 1'].filter_by_index('B1C1', None, copy=True)
    # eem_dataset_quenched_type1, _ = eem_dataset_pool['type 1'].filter_by_index('B1C2', None, copy=True)
    # eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched_type1.subsampling(portion=type1_proportion)
    # pos = [eem_dataset_unquenched_type1.index.index(idx) for idx in eem_dataset_new_uq.index]
    # quenched_index = [eem_dataset_quenched_type1.index[idx] for idx in pos]
    # eem_dataset_new_q, _ = eem_dataset_pool['type 1'].filter_by_index(None, quenched_index, copy=True)
    # eem_dataset_type_1 = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])

    eem_dataset_unquenched_other, _ = eem_dataset_pool['other types'].filter_by_index('B1C1', None, copy=True)
    eem_dataset_quenched_other, _ = eem_dataset_pool['other types'].filter_by_index('B1C2', None, copy=True)
    eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched_other.subsampling(portion=other_type_proportion)
    pos = [eem_dataset_unquenched_other.index.index(idx) for idx in eem_dataset_new_uq.index]
    quenched_index = [eem_dataset_quenched_other.index[idx] for idx in pos]
    eem_dataset_new_q, _ = eem_dataset_pool['other types'].filter_by_index(None, quenched_index, copy=True)
    eem_dataset_other_type = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])

    # combined_data = combine_eem_datasets([eem_dataset_type_1, eem_dataset_other_type])
    # eem_dataset_combinations[i] = combined_data
    eem_dataset_combinations[i] = eem_dataset_other_type

outlier_removal_stats_combos = {}
cluster_stats_combos = {}
consensus_matrices_combos = {}
run_time = 0
for code, eem_dataset_combo in eem_dataset_combinations.items():
    print(run_time)
    consensus_matrices, cluster_stats = k_method_quenching(
        eem_dataset=eem_dataset_combo,
        base_model=PARAFAC(n_components=4),
        n_clusters=[2, 3, 4, 5],
        minimum_dataset_size=10,
        depth=1,
        consensus_only=True
    )
    consensus_matrices_combos[code] = consensus_matrices
    cluster_stats_combos[code] = cluster_stats
    run_time += 1

with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/consensus_matrices_combos_0_percent_jul.pkl",
          'wb') as file:
    pickle.dump(consensus_matrices_combos, file)
with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/eem_dataset_combinations_0_percent_jul.pkl",
          'wb') as file:
    pickle.dump(eem_dataset_combinations, file)

#     #--------------------------
# clustered_datasets_combo, cluster_stats_combo = k_method_quenching(
#     eem_dataset=eem_dataset_combo,
#     base_model=PARAFAC(n_components=4),
#     n_clusters=[2, 3, 4, 5],
#     minimum_dataset_size=10,
#     depth=2
# )
# outlier_removal_stats = outlier_removal(
#     eem_dataset=eem_dataset_combo,
#     clustered_datasets=clustered_datasets_combo,
#     base_model=PARAFAC(n_components=4),
#     target_depth=2,
#     n_steps=3
# )
# outlier_removal_stats_combos[code] = outlier_removal_stats


with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/consensus_matrices_combos_0_percent_jul.pkl",
          'rb') as file:
    consensus_matrices_combos = pickle.load(file)
with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/eem_dataset_combinations_0_percent_jul.pkl",
          'rb') as file:
    eem_dataset_combinations = pickle.load(file)

outlier_removal_stats_combos = {}
for combo_code, combo_consensus in consensus_matrices_combos.items():
    print(combo_code)
    kmodel = KMethod(
        base_model=PARAFAC(n_components=4),
        n_initial_splits=5,
        distance_metric="quenching_coefficient",
        max_iter=10,
        tol=0.005,
        kw_top='B1C1',
        kw_bot='B1C2'
    )
    dataset_0 = eem_dataset_combinations[combo_code]
    o_sample_indices = [index for index, value in enumerate(dataset_0.index) if 'B1C1' in value]
    # q_sample_indices = [index for index, value in enumerate(dataset_0.index) if 'B1C2' in value]
    consensus = combo_consensus['0'][o_sample_indices, :]
    consensus = consensus[:, o_sample_indices]
    dataset_0_o, _ = dataset_0.filter_by_index(['B1C1'], None, copy=True)
    dataset_0_q, _ = dataset_0.filter_by_index(['B1C2'], None, copy=True)
    kmodel.consensus_matrix = consensus
    kmodel.hierarchical_clustering(dataset_0_o, n_clusters=5)
    clustered_dataset_combo = {}
    for c, d in kmodel.eem_clusters.items():
        idx_q = [dataset_0_q.index[i] for i in range(len(dataset_0_o.index)) if dataset_0_o.index[i] in d.index]
        dataset_cluster = combine_eem_datasets(
            [
                d,
                dataset_0.filter_by_index(None, idx_q, copy=True)[0]
            ]
        )
        clustered_dataset_combo['0' + str(c)] = dataset_cluster

    outlier_removal_stats = outlier_removal(
        dataset_0,
        clustered_dataset_combo,
        PARAFAC(n_components=4),
        target_depth=1,
        n_steps='max',
        method='set_point',
        set_point=1
    )
    outlier_removal_stats_combos[combo_code] = outlier_removal_stats

# with open("C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/outlier_removal_stats_combos_0_percent_jul_5_splits.pkl",
#           'wb') as file:
#     pickle.dump(outlier_removal_stats_combos, file)


with open(
        "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/outlier_removal_stats_combos_0_percent_jul_5_splits.pkl",
        'rb') as file:
    outlier_removal_stats_combos = pickle.load(file)

# ------------robustness check: plot r before and after outlier removal---------

norm = Normalize(vmin=0, vmax=60)
cmap = plt.cm.viridis  # Choose a colormap
# Define boundaries and extract colors from viridis at midpoints
bounds = [0, 10, 20, 30, 40, 50, 60]
n_intervals = len(bounds) - 1  # 5 intervals
midpoints = np.linspace(0, 1, n_intervals * 2 + 1)[1::2]  # Midpoints in [0,1] for viridis
colors = plt.cm.viridis(midpoints)  # Extract colors from viridis
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

fig_dot_r_r, ax_dot_r_r = plt.subplots()
codes_type_1 = ['stagnation_jul', 'normal_oct', 'stagnation_oct']
code_stagnation_jul = ['stagnation_jul']
n_common_type = 0
n_success = 0
n_failure = 0

for code, outlier_removal_stats in outlier_removal_stats_combos.items():
    if True:
        mean_fmax_ratio = outlier_removal_stats['Mean(F0/F)'].to_numpy()
        std_fmax_ratio = outlier_removal_stats['Std(F0/F)'].to_numpy()
        r_tcc = outlier_removal_stats['r_TCC'].to_numpy()
        p_tcc = outlier_removal_stats['p_TCC'].to_numpy()
        n_samples = outlier_removal_stats['n_samples'].to_numpy()
        n_samples = n_samples / np.max(n_samples)
        outlier_rate = (1 - n_samples) * 100

        if mean_fmax_ratio[0] > 1.1:
            if mean_fmax_ratio[-1] <= 1.1:
                index_cut = next((i for i, v in enumerate(mean_fmax_ratio) if v <= 1.1), None)
            else:
                index_cut = r_tcc.shape[0] - 1

            # Scatter with color determined by outlier_rate, using cmap and norm
            if r_tcc[0] < r_tcc[index_cut]:
                n_success += 1
            else:
                n_failure += 1

            ax_dot_r_r.scatter(
                r_tcc[0], r_tcc[index_cut],
                c=outlier_rate[index_cut],  # Pass the value directly
                cmap=cmap, norm=norm,  # Use the discrete colormap and norm
                edgecolors='k', linewidths=0.5,  # Optional: Add marker borders
                s=100
            )
            if p_tcc[0] < 0.05:
                print(p_tcc[0])
        else:
            ax_dot_r_r.scatter(
                r_tcc[0], r_tcc[0],
                c=0,  # Pass the value directly
                cmap=cmap, norm=norm,  # Use the discrete colormap and norm
                edgecolors='k', linewidths=0.5,  # Optional: Add marker borders
                s=100
            )

# Create the colorbar with interval labels
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax_dot_r_r)
cb.set_label('Sample removal rate (%)', fontsize=20)

# Set ticks and labels at midpoints of each interval
tick_positions = [bounds[i] for i in range(len(bounds))]
tick_labels = [f'{bounds[i]}' for i in range(len(bounds))]
cb.set_ticks(tick_positions)
cb.ax.tick_params(labelsize=16)
cb.set_ticklabels(tick_labels)

# Plot settings (unchanged from your original code)
ax_dot_r_r.set_ylabel(r"$r_{TCC}$ after outlier removal", fontsize=20)
ax_dot_r_r.set_xlabel(r"$r_{TCC}$ before outlier removal", fontsize=20)
ax_dot_r_r.tick_params(axis='both', labelsize=16)
ax_dot_r_r.plot([-1, 1], [-1, 1], '--', color='black', linewidth=2, alpha=0.5)
# ax_dot_r_r.plot([-1, 1], [-0.9, 1.1], '--', color='black', linewidth=2, alpha=0.5)
ax_dot_r_r.plot([-1, 1], [-0.8, 1.2], '--', color='black', linewidth=2, alpha=0.5)
# ax_dot_r_r.plot([-1, 1], [-0.7, 1.3], '--', color='black', linewidth=2, alpha=0.5)
ax_dot_r_r.plot([-1, 1], [-0.6, 1.4], '--', color='black', linewidth=2, alpha=0.5)
ax_dot_r_r.plot([-1, 1], [-0.4, 1.6], '--', color='black', linewidth=2, alpha=0.5)
ax_dot_r_r.set_xlim([-0.25, 1])
ax_dot_r_r.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_dot_r_r.set_ylim([-0.25, 1])
ax_dot_r_r.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

plt.tight_layout()
plt.show()
#
# # -----------------Check TCC vs. Fmax vs. F0/F----------------
# eem_dataset_combinations_test = {code: eem_dataset_combinations[code]
#                                  for code in [
#                                      ('normal_jul', 'stagnation_oct', 'high_flow', 'shortcut 1', 'shortcut 2')
#                                  ]
#                                  }
#
# fig_dot_fmax_tcc, ax_dot_fmax_tcc = plt.subplots()
# for code, eem_dataset_combo in eem_dataset_combinations_test.items():
#     model = PARAFAC(n_components=4)
#     model.fit(eem_dataset_combo)
#     fmax = model.fmax.iloc[:, 0]
#     fmax_original = fmax[fmax.index.str.contains('B1C1')]
#     fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
#     fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#     tcc = eem_dataset_combo.ref[fmax.index.str.contains('B1C1')]['TCC'] / 0.06
#     scatter = ax_dot_fmax_tcc.scatter(tcc, fmax_original, c=fmax_ratio, cmap='viridis')
#
# cbar = fig_dot_fmax_tcc.colorbar(scatter, ax=ax_dot_fmax_tcc, label='Feature Value')
# ax_dot_fmax_tcc.set_ylabel("Fmax")
# ax_dot_fmax_tcc.set_xlabel("TCC (#/ mL)")
# plt.show()
#
# cluster_stats_all_combos = k_method_consensus_to_clusters_stats(eem_dataset_combinations,
#                                                                 consensus_matrices_combos,
#                                                                 PARAFAC(n_components=4),
#                                                                 n_clusters=5
#                                                                 )


# ----------

from sklearn.cluster import KMeans

colors = ['royalblue','olivedrab','gold']
colors = ['grey', 'grey', 'red']

categories_kw_dict = {
    # 'Jul-low MBR influence': [['2024-07'], None],
    # 'Oct-low MBR influence': [['2024-10', 'M3'], ['2024-10-16', '2024-10-18', '2024-10-22']],
    # 'Oct-high MBR influence': [['2024-10'], ['G1', 'G2', 'G3', '2024-10-17', '2024-10-21']]
    'July-all': [['2024-07'], None],
    'October-all': [['2024-10'], None],
    'High flow': [['2024-10-17'], None],
    'Simulated\ncross-connection': [['2024-10-21'], None],
    'BAC top': [['G1'], None],
    'BAC middle': [['G2'], None],
    'BAC bottom': [['G3'], None],
    'October-others$*$': [['M3'], ['2024-10-16', '2024-10-22']]
}

model = PARAFAC(n_components=4)
model.fit(eem_dataset)
fmax = model.nnls_fmax
fmax_original = fmax[fmax.index.str.contains('B1C1')]
fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(fmax_ratio)
labels = kmeans.labels_

f0fs = []
sfcs = []
for i in range(4):
    f0f_r = []
    sfc_r = []
    target_name = 'TCC (million #/mL)' if i == 0 else 'DOC (mg/L)'
    for j in [2, 0, 1]:
        f0f_r.append(fmax_ratio[labels==j, i])
        target_j = eem_dataset.ref[target_name]
        target_j = target_j[target_j.index.str.contains('B1C1')]
        target_j = target_j.iloc[labels==j].to_numpy()
        fmax_original_j = fmax_original.iloc[labels==j, i].to_numpy()
        sfc_r.append(np.where(np.isnan(target_j), np.nan, target_j / fmax_original_j))
    # f0f_r.append(fmax_ratio[:, i])
    # target_j = eem_dataset.ref[target_name]
    # target_j = target_j[target_j.index.str.contains('B1C1')].to_numpy()
    # fmax_original_j = fmax_original.iloc[:, i].to_numpy()
    # sfc_r.append(np.where(np.isnan(target_j), np.nan, target_j / fmax_original_j))
    sfcs.append(sfc_r)
    f0fs.append(f0f_r)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)
for i, ax in enumerate(axes):
    sns.violinplot(data=f0fs[i], orient="h", ax=ax, cut=0, bw_adjust=0.3, palette=colors)  # Horizontal violin plot
    ax.set_yticks([0, 1, 2])  # Adjust positions of the groups
    ax.set_yticklabels(['cluster 1', 'cluster 2', 'cluster 3'], fontsize=12)  # Labels for the groups
    ax.set_xlabel(f"C{i+1} apparent " + "$F_{0}/F$", fontsize=14)
    ax.set_xlim([0.98, 1.45])
    ax.tick_params(labelsize=10)
plt.tight_layout()
plt.show()

xlabels = [
    'TCC/C1 $F_{max}$ (million #/mL/A.U.)',
    'DOC/C2 $F_{max}$ (mg/L/A.U.)',
    'DOC/C3 $F_{max}$ (mg/L/A.U.)',
    'DOC/C4 $F_{max}$ (mg/L/A.U.)',
]

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 7), sharex=False)
for i, ax in enumerate(axes):
    sns.violinplot(data=sfcs[i], orient="h", ax=ax, cut=0, bw_adjust=0.3, palette=colors)  # Horizontal violin plot
    ax.set_yticks([0, 1, 2])  # Adjusted to match violin plot indexing
    ax.set_yticklabels(['cluster 1', 'cluster 2', 'cluster 3'], fontsize=12)  # Labels for the groups
    ax.set_xlabel(xlabels[i], fontsize=14)
    if i > 0:
        ax.set_xlim([0, 0.01])  # Apply x-axis limit conditionally
    else:
        ax.set_xticks([0, 0.001, 0.002])

    ax.tick_params(labelsize=10)
plt.tight_layout()
plt.show()

cluster_indices = [[fmax_original.index[j] for j in range(labels.shape[0]) if labels[j]==i] for i in [2, 0, 1]]
cluster_categories_counts = {}
for cluster_index, cluster_name in zip(cluster_indices, ['cluster 1', 'cluster 2', 'cluster 3']):
    counts_list = []
    for category, indices in categories_indices.items():
        counts = len(set(indices) & set(cluster_index))
        counts_list.append(counts/len(indices)*200)
    cluster_categories_counts[cluster_name] = counts_list


fig, ax = plt.subplots(figsize=(6, 8))
bottom = np.zeros(8)
# colors_bar = ['#1f77b4', '#2ca02c', '#ff7f0e']

for i, (name, weight_count) in enumerate(cluster_categories_counts.items()):
    p = ax.barh(
        list(categories_kw_dict.keys()),
        # ['cluster 1', 'cluster 2', 'cluster 3'],
        weight_count, 0.3,
        label=name, left=bottom, color=colors[i]
    )
    bottom += weight_count

ax.tick_params(labelsize=16, axis='both', rotation=0)
ax.legend(loc="best", fontsize=16, bbox_to_anchor=(1.05, -0.1), ncol=3, handlelength=0.5)
ax.set_xlabel('Share (%)', fontsize=16)
ax.set_xlim([0, 100])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



#----------------------

eem_dataset.cluster = [item for item in labels for _ in range(2)]
eem_cluster1, _ = eem_dataset.filter_by_cluster(2, copy=True)
eem_cluster2, _ = eem_dataset.filter_by_cluster(0, copy=True)
eem_cluster3, _ = eem_dataset.filter_by_cluster(1, copy=True)

model_full = PARAFAC(n_components=4).fit(eem_dataset)
dataset_outlier_removed = combine_eem_datasets([eem_cluster1, eem_cluster2])
model_outlier_removed = PARAFAC(n_components=4).fit(dataset_outlier_removed)
model_in_use = model_outlier_removed
dataset_in_use = dataset_outlier_removed

fmax = model_in_use.fmax
target = dataset_in_use.ref['TCC (million #/mL)']
valid_indices = target.index[~target.isna()]
target = target.dropna().to_numpy()
fmax_original = fmax[fmax.index.str.contains('B1C1')]
mask = fmax_original.index.isin(valid_indices)
fmax_original = fmax_original[mask]

outlier_mask = labels==1

plt.figure()
# a, b = np.polyfit(target, fmax_original.iloc[:, 0], deg=1)
a, b = np.polyfit(target, fmax_original.iloc[:, 0], deg=1)
plt.plot(
    [-1, 10],
    a * np.array([-1, 10]) + b,
    '--',
    color='grey',
    label='reg.'
)
fmax_original_target_numpy = fmax_original.to_numpy()

plt.scatter(target, fmax_original_target_numpy[:, 0], label='other clusters', color='black', alpha=0.6)
# plt.scatter(target[[not b for b in outlier_mask]], fmax_original_target_numpy[[not b for b in outlier_mask]][:, 0], label='other clusters', color='black', alpha=0.6)
# plt.scatter(target[outlier_mask], fmax_original_target_numpy[outlier_mask][:, 0], label='outlier cluster(s)', color='red', alpha=0.6)

plt.xlabel('TCC (million #/mL)', fontsize=20)
plt.ylabel(f'C1 Fmax', fontsize=20)
plt.legend(
    # bbox_to_anchor=[1.02, 0.37],
    # bbox_to_anchor=[0.58, 0.63],
    fontsize=16
)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.xlim([0, 2.5])
plt.ylim([0, 2500])
plt.show()

print(pearsonr(fmax_original_target_numpy[:, 0], target))


#-----------------------

def outlier_removal_simple(eem_dataset_work, n_clusters, set_point, r):
    model = PARAFAC(n_components=4)
    model.fit(eem_dataset_work)
    fmax = model.nnls_fmax
    fmax_original = fmax[fmax.index.str.contains('B1C1')]
    fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    target = eem_dataset_work.ref['TCC (million #/mL)']
    target = target.dropna().to_numpy()
    r_before, p_before = pearsonr(fmax_original.iloc[:, r], target)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(fmax_ratio)
    labels = kmeans.labels_
    eem_dataset_work.cluster = [item for item in labels for _ in range(2)]
    fmax_ratio_r = fmax_ratio[:, r]
    mean_fmax_ratios = [np.mean(fmax_ratio_r[labels == j]) for j in range(n_clusters)]
    qualified_clusters = [index for index, value in enumerate(mean_fmax_ratios) if value <= set_point]
    qualified_dataset, _ = eem_dataset_work.filter_by_cluster(qualified_clusters, copy=True)
    model.fit(qualified_dataset)
    fmax_after = model.nnls_fmax
    fmax_after_original = fmax_after[fmax_after.index.str.contains('B1C1')]
    target_after = qualified_dataset.ref['TCC (million #/mL)']
    target_after = target_after.dropna().to_numpy()
    r_after, p_after = pearsonr(fmax_after_original.iloc[:, r], target_after)
    outlier_rate = fmax_after_original.shape[0] / fmax_original.shape[0]  * 100
    return r_before, r_after, p_before, p_after, outlier_rate


kw_dict_types = {
    'type 1': [['2024-07-'], None],
    'other types': [['2024-10'], None]
}

eem_dataset_pool = {}
for name, kw in kw_dict_types.items():
    eem_dataset_conditioned, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
    eem_dataset_pool[name] = eem_dataset_conditioned
# type1_proportion = 36 / eem_dataset_pool['type 1'].eem_stack.shape[0]
other_type_proportion = 120 / eem_dataset_pool['other types'].eem_stack.shape[0]
eem_dataset_combinations = {}
for i in range(10):
    # eem_dataset_unquenched_type1, _ = eem_dataset_pool['type 1'].filter_by_index('B1C1', None, copy=True)
    # eem_dataset_quenched_type1, _ = eem_dataset_pool['type 1'].filter_by_index('B1C2', None, copy=True)
    # eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched_type1.subsampling(portion=type1_proportion)
    # pos = [eem_dataset_unquenched_type1.index.index(idx) for idx in eem_dataset_new_uq.index]
    # quenched_index = [eem_dataset_quenched_type1.index[idx] for idx in pos]
    # eem_dataset_new_q, _ = eem_dataset_pool['type 1'].filter_by_index(None, quenched_index, copy=True)
    # eem_dataset_type_1 = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])

    eem_dataset_unquenched_other, _ = eem_dataset_pool['other types'].filter_by_index('B1C1', None, copy=True)
    eem_dataset_quenched_other, _ = eem_dataset_pool['other types'].filter_by_index('B1C2', None, copy=True)
    eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched_other.subsampling(portion=other_type_proportion)
    pos = [eem_dataset_unquenched_other.index.index(idx) for idx in eem_dataset_new_uq.index]
    quenched_index = [eem_dataset_quenched_other.index[idx] for idx in pos]
    eem_dataset_new_q, _ = eem_dataset_pool['other types'].filter_by_index(None, quenched_index, copy=True)
    eem_dataset_other_type = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])

    # combined_data = combine_eem_datasets([eem_dataset_type_1, eem_dataset_other_type])
    # eem_dataset_combinations[i] = combined_data
    eem_dataset_combinations[i] = eem_dataset_other_type

rs_before = []
rs_after = []
ps_before = []
ps_after = []
outlier_rates = []
for code, dataset in eem_dataset_combinations.items():
    r_before, r_after, p_before, p_after, outlier_rate = outlier_removal_simple(dataset, n_clusters=3, set_point=1.1,
                                                                                r=0)
    rs_before.append(r_before)
    rs_after.append(r_after)
    ps_before.append(p_before)
    ps_after.append(p_after)
    outlier_rates.append(outlier_rate)

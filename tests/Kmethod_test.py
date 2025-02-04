import numpy as np

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import TABLEAU_COLORS
from scipy.stats import ks_2samp

colors = list(TABLEAU_COLORS.values())
# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_250_ex_274_em_310_mfem_7_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)


def k_method_quenching(eem_dataset, base_model, n_clusters, minimum_dataset_size, depth):
    clustered_datasets = {"0": eem_dataset}
    base_model.fit(eem_dataset)
    fmax = base_model.fmax.iloc[:, 0]
    fmax_tcc = pd.concat([eem_dataset.ref['TCC'], fmax], axis=1)
    fmax_tcc = fmax_tcc.dropna()
    fmax_doc = pd.concat([eem_dataset.ref['DOC'], fmax], axis=1)
    fmax_doc = fmax_doc.dropna()
    tcc_mean = np.mean(fmax_tcc['TCC'])
    tcc_std = np.std(fmax_tcc['TCC'])
    doc_mean = np.mean(fmax_doc['DOC'])
    doc_std = np.std(fmax_doc['DOC'])
    cor_tcc, p_tcc = pearsonr(fmax_tcc.iloc[:, 0], fmax_tcc.iloc[:, 1])
    cor_doc, p_doc = pearsonr(fmax_doc.iloc[:, 0], fmax_doc.iloc[:, 1])
    fmax_original = fmax[fmax.index.str.contains('B1C1')]
    fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    ratio_mean = np.mean(fmax_ratio)
    ratio_std = np.std(fmax_ratio)
    cluster_stats = pd.DataFrame({
        'n_samples': fmax.shape[0],
        'Mean(F0/F)': ratio_mean,
        'Std(F0/F)': ratio_std,
        'Mean(TCC)': tcc_mean,
        'Std(TCC)': tcc_std,
        'r_TCC': cor_tcc,
        'p_TCC': p_tcc,
        'Mean(DOC)': doc_mean,
        'Std(DOC)': doc_std,
        'r_DOC': cor_doc,
        'p_DOC': p_doc,
    }, index=['0'])

    for i in range(depth):
        clustered_datasets_old = copy.deepcopy(clustered_datasets)
        for mother_code, mother_dataset in clustered_datasets_old.items():
            if len(mother_code) == i + 1:
                base_model = PARAFAC(n_components=4)
                kmodel = KMethod(
                    base_model=base_model,
                    n_initial_splits=max(n_clusters),
                    error_calculation="quenching_coefficient",
                    max_iter=10,
                    tol=0.005,
                    kw_unquenched='B1C1',
                    kw_quenched='B1C2'
                )
                kmodel.calculate_consensus(mother_dataset, n_base_clusterings=10, subsampling_portion=1)
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
                        base_model = PARAFAC(n_components=4)
                        base_model.fit(cluster_dataset)
                        fmax = base_model.fmax.iloc[:, 0]
                        fmax_tcc = pd.concat([cluster_dataset.ref['TCC'], fmax], axis=1)
                        fmax_tcc = fmax_tcc.dropna()
                        fmax_doc = pd.concat([cluster_dataset.ref['DOC'], fmax], axis=1)
                        fmax_doc = fmax_doc.dropna()
                        tcc_mean = np.mean(fmax_tcc['TCC'])
                        tcc_std = np.std(fmax_tcc['TCC'])
                        doc_mean = np.mean(fmax_doc['DOC'])
                        doc_std = np.std(fmax_doc['DOC'])
                        cor_tcc, p_tcc = pearsonr(fmax_tcc.iloc[:, 0], fmax_tcc.iloc[:, 1])
                        cor_doc, p_doc = pearsonr(fmax_doc.iloc[:, 0], fmax_doc.iloc[:, 1])
                        fmax_original = fmax[fmax.index.str.contains('B1C1')]
                        fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
                        fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
                        ratio_mean = np.mean(fmax_ratio)
                        ratio_std = np.std(fmax_ratio)
                        cluster_stats.loc[children_code] = [
                            fmax.shape[0],
                            ratio_mean,
                            ratio_std,
                            tcc_mean,
                            tcc_std,
                            cor_tcc,
                            p_tcc,
                            doc_mean,
                            doc_std,
                            cor_doc,
                            p_doc
                        ]
    return clustered_datasets, cluster_stats


clustered_datasets, cluster_stats = k_method_quenching(
    eem_dataset=eem_dataset,
    base_model=PARAFAC(n_components=4),
    n_clusters=[2, 3, 4, 5],
    minimum_dataset_size=10,
    depth=2
)


# --------------Outlier removal--------------

def outlier_removal(eem_dataset, clustered_datasets, base_model, target_depth, n_steps):
    base_model.fit(eem_dataset)
    fmax = base_model.fmax.iloc[:, 0]
    fmax_tcc = pd.concat([eem_dataset.ref['TCC'], fmax], axis=1)
    fmax_tcc = fmax_tcc.dropna()
    fmax_doc = pd.concat([eem_dataset.ref['DOC'], fmax], axis=1)
    fmax_doc = fmax_doc.dropna()
    tcc_mean = np.mean(fmax_tcc['TCC'])
    tcc_std = np.std(fmax_tcc['TCC'])
    doc_mean = np.mean(fmax_doc['DOC'])
    doc_std = np.std(fmax_doc['DOC'])
    cor_tcc, p_tcc = pearsonr(fmax_tcc.iloc[:, 0], fmax_tcc.iloc[:, 1])
    cor_doc, p_doc = pearsonr(fmax_doc.iloc[:, 0], fmax_doc.iloc[:, 1])
    fmax_original = fmax[fmax.index.str.contains('B1C1')]
    fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
    ratio_mean = np.mean(fmax_ratio)
    ratio_std = np.std(fmax_ratio)
    outlier_removal_stats = pd.DataFrame({
        'n_samples': fmax.shape[0],
        'Mean(F0/F)': ratio_mean,
        'Std(F0/F)': ratio_std,
        'Mean(TCC)': tcc_mean,
        'Std(TCC)': tcc_std,
        'r_TCC': cor_tcc,
        'p_TCC': p_tcc,
        'Mean(DOC)': doc_mean,
        'Std(DOC)': doc_std,
        'r_DOC': cor_doc,
        'p_DOC': p_doc,
    }, index=['0'])
    ks_stats = {}
    clustered_datasets_filtered = {key: value for key, value in clustered_datasets.items() if
                                   len(key) == target_depth + 1}
    for n in range(n_steps):
        # Compare each group to the pooled others
        for i, (code, dataset_excluded) in enumerate(clustered_datasets_filtered.items()):
            eem_dataset_remained = combine_eem_datasets(
                [list(clustered_datasets_filtered.values())[j] for j in range(len(clustered_datasets_filtered)) if
                 j != i]
            )
            base_model.fit(eem_dataset_remained)
            fmax_remained = base_model.fmax.iloc[:, 0]
            fmax_original = fmax_remained[fmax_remained.index.str.contains('B1C1')]
            fmax_quenched = fmax_remained[fmax_remained.index.str.contains('B1C2')]
            fmax_ratio_remained = fmax_original.to_numpy() / fmax_quenched.to_numpy()
            base_model.fit(dataset_excluded)
            fmax_excluded = base_model.fmax.iloc[:, 0]
            fmax_original = fmax_excluded[fmax_excluded.index.str.contains('B1C1')]
            fmax_quenched = fmax_excluded[fmax_excluded.index.str.contains('B1C2')]
            fmax_ratio_excluded = fmax_original.to_numpy() / fmax_quenched.to_numpy()

            # Compute KS statistic
            ks, ks_p_value = ks_2samp(fmax_ratio_excluded, fmax_ratio_remained)
            ks_stats[code] = ks

        outlier_code = max(ks_stats, key=lambda k: ks_stats[k])
        clustered_datasets_filtered = {key: value for key, value in clustered_datasets_filtered.items() if
                                       key != outlier_code}
        eem_dataset_cleaned = combine_eem_datasets(
            [list(clustered_datasets_filtered.values())[j] for j in range(len(clustered_datasets_filtered))]
        )
        base_model.fit(eem_dataset_cleaned)
        fmax = base_model.fmax.iloc[:, 0]
        fmax_tcc = pd.concat([eem_dataset_cleaned.ref['TCC'], fmax], axis=1)
        fmax_tcc = fmax_tcc.dropna()
        fmax_doc = pd.concat([eem_dataset_cleaned.ref['DOC'], fmax], axis=1)
        fmax_doc = fmax_doc.dropna()
        tcc_mean = np.mean(fmax_tcc['TCC'])
        tcc_std = np.std(fmax_tcc['TCC'])
        doc_mean = np.mean(fmax_doc['DOC'])
        doc_std = np.std(fmax_doc['DOC'])
        cor_tcc, p_tcc = pearsonr(fmax_tcc.iloc[:, 0], fmax_tcc.iloc[:, 1])
        cor_doc, p_doc = pearsonr(fmax_doc.iloc[:, 0], fmax_doc.iloc[:, 1])
        fmax_original = fmax[fmax.index.str.contains('B1C1')]
        fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
        fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
        ratio_mean = np.mean(fmax_ratio)
        ratio_std = np.std(fmax_ratio)
        outlier_removal_stats.loc['-' + outlier_code] = [
            fmax.shape[0],
            ratio_mean,
            ratio_std,
            tcc_mean,
            tcc_std,
            cor_tcc,
            p_tcc,
            doc_mean,
            doc_std,
            cor_doc,
            p_doc,
        ]
    return outlier_removal_stats


# ---------------go through combinations--------------

kw_dict = {
    'normal_jul': [['M3'], ['2024-07-12', '2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17'], 3],
    'stagnation_jul': [['M3'], ['2024-07-18', '2024-07-19'], 4],
    'normal_oct': [['M3'], ['2024-10-16', '2024-10-22'], 4],
    'stagnation_oct': [['M3'], ['2024-10-18'], 3],
    'high_flow': [['M3'], ['2024-10-17'], 3],
    'shortcut 1': [['G3'], None, 4],
    'shortcut 2': [['G2'], None, 4],
    'shortcut 3': [['G1'], None, 4],
    'cross-connection': [['M3'], ['2024-10-21'], 4],
}

eem_dataset_pool = {}
for name, kw in kw_dict.items():
    eem_dataset_conditioned, _ = eem_dataset.filter_by_index(kw[0], kw[1], copy=True)
    eem_dataset_pool[name] = eem_dataset_conditioned

code_combinations = itertools.combinations(eem_dataset_pool.keys(), 5)
eem_dataset_combinations = {
    combo: combine_eem_datasets([eem_dataset_pool[code] for code in combo]) for combo in code_combinations
}

outlier_removal_stats_combos = {}
for code, eem_dataset_combo in eem_dataset_combinations.items():
    clustered_datasets_combo, cluster_stats_combo = k_method_quenching(
        eem_dataset=eem_dataset_combo,
        base_model=PARAFAC(n_components=4),
        n_clusters=[2, 3, 4, 5],
        minimum_dataset_size=10,
        depth=2
    )
    outlier_removal_stats = outlier_removal(
        eem_dataset=eem_dataset_combo,
        clustered_datasets=clustered_datasets_combo,
        base_model=PARAFAC(n_components=4),
        target_depth=2,
        n_steps=3
    )
    outlier_removal_stats_combos[code] = outlier_removal_stats

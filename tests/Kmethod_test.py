from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import TABLEAU_COLORS

colors = list(TABLEAU_COLORS.values())
# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_250_ex_274_em_310_mfem_7_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)
n_clusters = [2, 3, 4, 5]
minimum_dataset_size = 10

clustered_datasets = {"0": eem_dataset}
total_levels = 3

base_model = PARAFAC(n_components=4)
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

for i in range(total_levels):
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
            kmodel.calculate_consensus(mother_dataset, n_base_clusterings=100, subsampling_portion=0.8)
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
print(cluster_stats)

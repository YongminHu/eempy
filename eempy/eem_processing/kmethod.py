import numpy as np
import pandas as pd

import copy
import warnings

from math import sqrt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .eem_dataset import EEMDataset, combine_eem_datasets
from .basic import component_similarity, align_components_by_components


class KMethod:
    """
    K-method (e.g., K-PARAFACs or K-NMFs) for EEM clustering by minimizing reconstruction error (Hu et al.,
    Water Research, 2025).

    This class implements the K-method family of clustering algorithms for excitation–emission matrix (EEM) datasets.
    The key hypothesis is that fitting EEMs with high chemical composition variability using a single, unified set of
    components (e.g., one PARAFAC or NMF model) can lead to over-generalized component formation and large
    reconstruction error. In contrast, EEMs sharing similar chemical compositions can be clustered and represented by
    cluster-specific component sets, resulting in a number of unique component sets that better capture the
    variability in chemical composition between clusters and reduce overall reconstruction error.

    Based on this hypothesis, K-method searches for a clustering strategy that minimizes the overall reconstruction
    error by iterating between:
    - **Estimation**: fit a base decomposition model (``base_model``) separately on each current cluster to obtain
      cluster-specific models.
    - **Maximization / Assignment**: assign each sample to the cluster whose model yields the smallest distance
      (e.g., reconstruction RMSE), forming updated clusters.

    Repeating this procedure yields cluster-specific PARAFAC/NMF models that (ideally) reconstruct the dataset better
    than a single unified model.

    In addition, K-method can be run multiple times with subsampling to form a **consensus matrix** and then derive a
    final clustering using hierarchical clustering on a distance matrix computed from consensus values.

    Parameters
    ----------
    base_model : object
        Base decomposition model used within each cluster (e.g., an instance of ``PARAFAC`` or ``EEMNMF``). Before
        passing to ``KMethod``, the base model should be properly configured (e.g., number of components,
        regularizations to be implemented, etc.).
    n_initial_splits : int
        Number of splits used in initialization (the first partition of the dataset before iterative refinement).
    distance_metric : {'reconstruction_error', 'reconstruction_error_with_beta', 'quenching_coefficient'}, default 'reconstruction_error'
        Criterion used for assignment in the maximization step.
        - ``'reconstruction_error'``: assign each sample to the model with the smallest per-sample RMSE.
        - ``'reconstruction_error_with_beta'``: like reconstruction error, but pairs samples into top/bot groups and
          uses beta-based reconstruction (requires ``kw_top`` and ``kw_bot`` and a base model that supports beta logic).
        - ``'quenching_coefficient'``: assign samples based on similarity of estimated quenching coefficients derived
          from paired top/bot samples (requires ``kw_top`` and ``kw_bot``).
    max_iter : int, default 20
        Maximum number of K-method iterations in a single base clustering run.
    tol : float, default 0.001
        Convergence tolerance based on similarity between cluster-specific models of two consecutive iterations.
        If the average Tucker’s congruence (or component similarity proxy) exceeds ``1 - tol``, convergence is declared.
    elimination : {'default'} or int, default 'default'
        Minimum allowed cluster size during optimization. Clusters with fewer samples than the threshold are removed.
        - ``'default'``: use ``base_model.n_components`` as the minimum cluster size.
        - ``int``: explicit minimum cluster size.
    kw_top : str, optional
        Keyword used to identify "top" samples from ``eem_dataset.index`` when a paired-top/bot distance metric is used.
        Samples are selected by substring matching against entries of ``eem_dataset.index``.
    kw_bot : str, optional
        Keyword used to identify "bot" samples from ``eem_dataset.index`` when a paired-top/bot distance metric is used.
        Samples are selected by substring matching against entries of ``eem_dataset.index``.

    Attributes
    ----------
    unified_model : object or None
        Unified model fitted once on the full dataset (a deep copy of ``base_model``). Used as a reference for
        aligning components and for some distance calculations.
    label_history : list or None
        History of cluster assignments. For base clustering runs, this is typically a list containing a DataFrame
        with per-sample labels across iterations.
    error_history : list or None
        History of per-sample distances/errors (e.g., RMSE) across iterations, typically stored as DataFrames.
    silhouette_score : float or None
        Silhouette score computed on the final distance matrix during hierarchical clustering (when available).
    labels : numpy.ndarray or None
        Final cluster labels for each sample. Labels are cluster IDs returned by hierarchical clustering
        (typically 1..K), or by base clustering when used directly.
    index_sorted : list or None
        Dataset index reordered by the final hierarchical clustering labels (when available).
    ref_sorted : pandas.DataFrame or None
        Reference table reordered by the final hierarchical clustering labels (when available).
    threshold_r : float or None
        Distance threshold used for hierarchical clustering cut (derived from the linkage matrix).
    eem_clusters : dict or None
        Mapping from cluster label to an ``EEMDataset`` containing the EEMs assigned to that cluster.
    cluster_specific_models : dict or None
        Mapping from cluster label to the fitted cluster-specific model (deep copies of ``base_model`` fitted on each
        cluster).
    consensus_matrix : numpy.ndarray or None
        Consensus matrix ``M`` with shape ``(n_samples, n_samples)``, where ``M[i, j]`` is the fraction of base runs
        in which sample i and j co-occur in the same cluster.
    distance_matrix : numpy.ndarray or None
        Distance matrix derived from consensus, typically ``D[i, j] = (1 - M[i, j])**p`` (see
        ``consensus_conversion_power``).
    linkage_matrix : numpy.ndarray or None
        Hierarchical clustering linkage matrix computed from the consensus-derived distance matrix.
    consensus_matrix_sorted : numpy.ndarray or None
        Consensus matrix reordered by the final cluster labels for visualization.

    References
    ----------
    [1]  Hu, Yongmin, Eberhard Morgenroth, and Céline Jacquin. "Online monitoring of greywater reuse system using
         excitation-emission matrix (EEM) and K-PARAFACs." Water Research 268 (2025): 122604.
    """
    def __init__(self, base_model, n_initial_splits, distance_metric="reconstruction_error", max_iter=20, tol=0.001,
                 elimination='default', kw_top=None, kw_bot=None):
        # -----------Parameters-------------
        self.base_model = base_model
        self.n_initial_splits = n_initial_splits
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.tol = tol
        self.elimination = elimination
        self.kw_top = kw_top
        self.kw_bot = kw_bot
        self.subsampling_portion = None
        self.n_runs = None
        self.consensus_conversion_power = None
        # ----------Attributes-------------
        self.unified_model = None
        self.label_history = None
        self.error_history = None
        self.silhouette_score = None
        self.labels = None
        self.index_sorted = None
        self.ref_sorted = None
        self.threshold_r = None
        self.eem_clusters = None
        self.cluster_specific_models = None
        self.consensus_matrix = None
        self.distance_matrix = None
        self.linkage_matrix = None
        self.consensus_matrix_sorted = None


    def base_clustering(self, eem_dataset: EEMDataset):
        """
        Run clustering for a single time.
        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset to be clustered.
        Returns
        -------
        cluster_labels: np.ndarray
            Cluster labels.
        label_history: list
            Cluster labels in each iteration.
        error_history: list
            Average reconstruction error (RMSE) in each iteration.
        """
        # -------Generate a unified model as reference for ordering components--------
        unified_model = copy.deepcopy(self.base_model)
        unified_model.fit(eem_dataset)
        # -------Define functions for estimation and maximization steps-------
        def get_quenching_coef(fmax_tot, kw_o, kw_q):
            fmax_original = fmax_tot[fmax_tot.index.str.contains(kw_o)]
            fmax_quenched = fmax_tot[fmax_tot.index.str.contains(kw_q)]
            fmax_ratio = fmax_tot.copy()
            fmax_ratio[fmax_ratio.index.str.contains(kw_o)] = fmax_original.to_numpy() / fmax_quenched.to_numpy()
            fmax_ratio[fmax_ratio.index.str.contains(kw_q)] = fmax_original.to_numpy() / fmax_quenched.to_numpy()
            return fmax_ratio.to_numpy()
        def estimation(sub_datasets: dict):
            models = {}
            for label, d in sub_datasets.items():
                model = copy.deepcopy(self.base_model)
                model.fit(d)
                models[label] = model
            return models
        def maximization(models: dict):
            sample_error = []
            sub_datasets = {}
            for label, m in models.items():
                if self.distance_metric == "reconstruction_error_with_beta":
                    idx_top = [i for i in range(len(eem_dataset.index)) if self.kw_top in eem_dataset.index[i]]
                    idx_bot = [i for i in range(len(eem_dataset.index)) if self.kw_bot in eem_dataset.index[i]]
                    score_m, fmax_m, eem_stack_re_m = m.predict(
                        eem_dataset=eem_dataset,
                        fit_beta=True,
                        idx_bot=idx_bot,
                        idx_top=idx_top,
                    )
                    res = eem_dataset.eem_stack - eem_stack_re_m
                    res = np.sum(res ** 2, axis=(1, 2))
                    error_with_beta = np.zeros(fmax_m.shape[0])
                    error_with_beta[idx_top] = res[idx_top] + res[idx_bot]
                    error_with_beta[idx_bot] = res[idx_top] + res[idx_bot]
                    sample_error.append(error_with_beta)
                elif self.distance_metric == "reconstruction_error":
                    score_m, fmax_m, eem_stack_re_m = m.predict(eem_dataset)
                    res = eem_dataset.eem_stack - eem_stack_re_m
                    n_pixels = m.eem_stack_train.shape[1] * m.eem_stack_train.shape[2]
                    rmse = np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels)
                    sample_error.append(rmse)
                elif self.distance_metric == "quenching_coefficient":
                    if not all([self.kw_top, self.kw_bot]):
                        raise ValueError("Both kw_unquenched and kw_quenched must be passed.")
                    if type(m).__name__ == 'PARAFAC':
                        fmax_establishment = m.nnls_fmax
                    elif type(m).__name__ == 'EEMNMF':
                        fmax_establishment = m.nnls_fmax
                    else:
                        raise TypeError("Invalid base model type.")
                    quenching_coef_establishment = get_quenching_coef(fmax_establishment, self.kw_top,
                                                                      self.kw_bot)
                    quenching_coef_archetype = np.mean(quenching_coef_establishment, axis=0)
                    quenching_coef_test = get_quenching_coef(fmax_m, self.kw_top, self.kw_bot)
                    quenching_coef_diff = np.abs(quenching_coef_test - quenching_coef_archetype)
                    sample_error.append(np.sum(quenching_coef_diff ** 2, axis=1))
            best_model_idx = np.argmin(sample_error, axis=0)
            least_model_errors = np.min(sample_error, axis=0)
            for j, label in enumerate(models.keys()):
                eem_stack_j = eem_dataset.eem_stack[np.where(best_model_idx == j)]
                if eem_dataset.ref is not None:
                    ref_j = eem_dataset.ref.iloc[np.where(best_model_idx == j)]
                else:
                    ref_j = None
                if eem_dataset.index is not None:
                    index_j = [eem_dataset.index[k] for k, idx in enumerate(best_model_idx) if idx == j]
                else:
                    index_j = None
                sub_datasets[label] = EEMDataset(eem_stack=eem_stack_j, ex_range=eem_dataset.ex_range,
                                                 em_range=eem_dataset.em_range, ref=ref_j, index=index_j)
            best_model_label = np.array([list(models.keys())[idx] for idx in best_model_idx])
            return sub_datasets, best_model_label, least_model_errors
        # -------Define function for convergence detection-------
        def model_similarity(models_1: dict, models_2: dict):
            similarity = 0
            for label, m in models_1.items():
                similarity = component_similarity(m.components, models_2[label].components).to_numpy().diagonal()
            similarity = np.sum(similarity) / len(models_1)
            return similarity
        # -------Initialization--------
        label_history = []
        error_history = []
        sample_errors = []
        sub_datasets_n = {}
        if self.distance_metric == "reconstruction_error":
            initial_sub_eem_datasets = eem_dataset.splitting(n_split=self.n_initial_splits)
        elif self.distance_metric in ["quenching_coefficient", "reconstruction_error_with_beta"]:
            initial_sub_eem_datasets = []
            eem_dataset_unquenched = eem_dataset.filter_by_index(self.kw_top, None, inplace=False)
            initial_sub_eem_datasets_unquenched = eem_dataset_unquenched.splitting(n_split=self.n_initial_splits)
            eem_dataset_quenched = eem_dataset.filter_by_index(self.kw_bot, None, inplace=False)
            for subset in initial_sub_eem_datasets_unquenched:
                pos = [eem_dataset_unquenched.index.index(idx) for idx in subset.index]
                quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
                sub_eem_dataset_quenched = eem_dataset.filter_by_index(None, quenched_index, inplace=False)
                subset.sort_by_index()
                sub_eem_dataset_quenched.sort_by_index()
                initial_sub_eem_datasets.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))
        for i, random_m in enumerate(initial_sub_eem_datasets):
            sub_datasets_n[i + 1] = random_m
        for n in range(self.max_iter):
            # -------Eliminate sub_datasets having EEMs less than the number of ranks--------
            cluster_label_to_remove = []
            for cluster_label, sub_dataset_i in sub_datasets_n.items():
                if self.elimination == 'default' and sub_dataset_i.eem_stack.shape[0] <= self.base_model.n_components:
                    cluster_label_to_remove.append(cluster_label)
                elif isinstance(self.elimination, int):
                    if self.elimination <= max(self.base_model.n_components, self.elimination):
                        cluster_label_to_remove.append(cluster_label)
            for l in cluster_label_to_remove:
                sub_datasets_n.pop(l)
            # -------The estimation step-------
            cluster_specific_models_new = estimation(sub_datasets_n)
            cluster_specific_models_new = align_components_by_components(
                cluster_specific_models_new,
                {f'component {i + 1}': unified_model.components[i] for i in range(unified_model.components.shape[0])},
            )
            # -------The maximization step--------
            sub_datasets_n, cluster_labels, fitting_errors = maximization(cluster_specific_models_new)
            label_history.append(cluster_labels)
            error_history.append(fitting_errors)
            # -------Detect convergence---------
            if 0 < n < self.max_iter - 1:
                if np.array_equal(label_history[-1], label_history[-2]):
                    break
                if len(cluster_specific_models_old) == len(cluster_specific_models_new):
                    if model_similarity(cluster_specific_models_new, cluster_specific_models_old) > 1 - self.tol:
                        break
            cluster_specific_models_old = cluster_specific_models_new
        label_history = pd.DataFrame(np.array(label_history).T, index=eem_dataset.index,
                                     columns=['iter_{i}'.format(i=i + 1) for i in range(n + 1)])
        error_history = pd.DataFrame(np.array(error_history).T, index=eem_dataset.index,
                                     columns=['iter_{i}'.format(i=i + 1) for i in range(n + 1)])
        self.label_history = [label_history]
        self.error_history = [error_history]
        self.unified_model = unified_model
        self.labels = cluster_labels
        self.eem_clusters = sub_datasets_n
        self.cluster_specific_models = cluster_specific_models_new
        return cluster_labels, label_history, error_history


    def calculate_consensus(self, eem_dataset: EEMDataset, n_base_clusterings: int, subsampling_portion: float):
        """
        Run the clustering for many times and combine the output of each run to obtain an optimal clustering.
        Parameters
        ----------
        eem_dataset: EEMDataset
            EEM dataset.
        n_base_clusterings: int
            Number of base clustering.
        subsampling_portion: float
            The portion of EEMs remained after subsampling.
        Returns
        -------
        self: object
            The established K-PARAFACs model
        """
        n_samples = eem_dataset.eem_stack.shape[0]
        co_label_matrix = np.zeros((n_samples, n_samples))
        co_occurrence_matrix = np.zeros((n_samples, n_samples))
        # ---------Repeat base clustering and generate consensus matrix---------
        n = 0
        label_history = []
        error_history = []
        if self.distance_metric == "quenching_coefficient":
            eem_dataset_unquenched = eem_dataset.filter_by_index(self.kw_top, None, inplace=False)
            eem_dataset_quenched = eem_dataset.filter_by_index(self.kw_bot, None, inplace=False)
        while n < n_base_clusterings:
            # ------Subsampling-------
            if self.distance_metric == "reconstruction_error":
                eem_dataset_n, selected_indices = eem_dataset.subsampling(portion=subsampling_portion, inplace=False)
            elif self.distance_metric == "quenching_coefficient":
                eem_dataset_new_uq, selected_indices_uq = eem_dataset_unquenched.subsampling(
                    portion=subsampling_portion, inplace=False)
                pos = [eem_dataset_unquenched.index.index(idx) for idx in eem_dataset_new_uq.index]
                quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
                eem_dataset_new_q = eem_dataset.filter_by_index(None, quenched_index, inplace=False)
                eem_dataset_n = combine_eem_datasets([eem_dataset_new_uq, eem_dataset_new_q])
                eem_dataset_n.sort_by_index()
                selected_indices = [eem_dataset.index.index(idx) for idx in eem_dataset_n.index]
            n_samples_new = eem_dataset_n.eem_stack.shape[0]
            # ------Base clustering-------
            cluster_labels_n, label_history_n, error_history_n = self.base_clustering(eem_dataset_n)
            for j in range(n_samples_new):
                for k in range(n_samples_new):
                    co_occurrence_matrix[selected_indices[j], selected_indices[k]] += 1
                    if cluster_labels_n[j] == cluster_labels_n[k]:
                        co_label_matrix[selected_indices[j], selected_indices[k]] += 1
            label_history.append(label_history_n)
            error_history.append(error_history_n)
            # ----check if counting_matrix contains 0, meaning that not all sample pairs have been included in the
            # clustering. If this is the case, run more base clustering until all sample pairs are covered----
            if n == n_base_clusterings - 1 and np.any(co_occurrence_matrix == 0):
                warnings.warn(
                    'Not all sample pairs are covered. One extra clustering will be executed.')
            else:
                n += 1
        # ---------Obtain consensus matrix, distance matrix and linkage matrix----------
        consensus_matrix = co_label_matrix / co_occurrence_matrix
        self.n_runs = n_base_clusterings
        self.subsampling_portion = subsampling_portion
        self.label_history = label_history
        self.error_history = error_history
        self.consensus_matrix = consensus_matrix
        return consensus_matrix, label_history, error_history


    def hierarchical_clustering(self, eem_dataset, n_clusters, consensus_conversion_power=1):
        """
        Parameters
        ----------
        eem_dataset: EEMDataset
            EEM dataset to cluster.
        n_clusters: int
            Number of clusters.
        consensus_conversion_power: float
            The factor adjusting the conversion from consensus matrix (M) to distance matrix (D) used for hierarchical
            clustering. D_{i,j} = (1 - M_{i,j})^factor. This number influences the gradient of distance with respect
            to consensus. A smaller number will lead to shaper increase of distance at consensus close to 1.
        Returns
        -------
        """
        if self.consensus_matrix is None:
            raise ValueError('Consensus matrix is not defined.')
        distance_matrix = (1 - self.consensus_matrix) ** consensus_conversion_power
        linkage_matrix = linkage(squareform(distance_matrix), method='complete')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        linkage_matrix_sorted = linkage_matrix[linkage_matrix[:, 2].argsort()[::-1]]
        max_d = linkage_matrix_sorted[n_clusters - 2, 2]
        self.threshold_r = max_d
        sorted_indices = np.argsort(labels)
        consensus_matrix_sorted = self.consensus_matrix[sorted_indices][:, sorted_indices]
        if eem_dataset.index is not None:
            eem_index_sorted = [eem_dataset.index[i] for i in sorted_indices]
            self.index_sorted = eem_index_sorted
        if eem_dataset.ref is not None:
            eem_ref_sorted = eem_dataset.ref.iloc[sorted_indices, :]
            self.ref_sorted = eem_ref_sorted
        sc = silhouette_score(X=distance_matrix, labels=labels, metric='precomputed')
        self.silhouette_score = sc
        self.distance_matrix = distance_matrix
        self.linkage_matrix = linkage_matrix
        self.consensus_matrix_sorted = consensus_matrix_sorted
        # ---------Get final clusters and cluster-specific models-------
        clusters = {}
        cluster_specific_models = {}
        for j in set(list(labels)):
            eem_stack_j = eem_dataset.eem_stack[np.where(labels == j)]
            if eem_dataset.ref is not None:
                ref_j = eem_dataset.ref.iloc[np.where(labels == j)]
            else:
                ref_j = None
            if eem_dataset.index is not None:
                index_j = [eem_dataset.index[k] for k, idx in enumerate(labels) if idx == j]
            else:
                index_j = None
            cluster_j = [j] * eem_stack_j.shape[0]
            clusters[j] = EEMDataset(eem_stack=eem_stack_j, ex_range=eem_dataset.ex_range,
                                     em_range=eem_dataset.em_range, ref=ref_j, index=index_j, cluster=cluster_j)
            model = copy.deepcopy(self.base_model)
            # model = PARAFAC(rank=self.rank, non_negativity=self.non_negativity, init=self.init,
            #                 tf_normalization=self.tf_normalization,
            #                 loadings_normalization=self.loadings_normalization, sort_em=self.sort_em)
            model.fit(clusters[j])
            cluster_specific_models[j] = model
        self.labels = labels
        self.eem_clusters = clusters
        self.cluster_specific_models = cluster_specific_models


    def predict(self, eem_dataset: EEMDataset):
        """
        Fit the cluster-specific models to a given EEM dataset. Each EEM in the EEM dataset is fitted to the model that
        produce the least RMSE.
        Parameters
        ----------
        eem_dataset: EEMDataset
            The EEM dataset to be predicted.
        Returns
        -------
        best_model_label: pd.DataFrame
            The best-fit model for every EEM.
        score_all: pd.DataFrame
            The score fitted with each cluster-specific model.
        fmax_all: pd.DataFrame
            The fmax fitted with each cluster-specific model.
        sample_error: pd.DataFrame
            The RMSE fitted with each cluster-specific model.
        """
        sample_error = []
        score_all = []
        fmax_all = []
        for label, m in self.cluster_specific_models.items():
            score_m, fmax_m, eem_stack_re_m = m.predict(eem_dataset)
            res = m.eem_stack_train - eem_stack_re_m
            n_pixels = m.eem_stack_train.shape[1] * m.eem_stack_train.shape[2]
            rmse = sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels)
            sample_error.append(rmse)
            score_all.append(score_m)
            fmax_all.append(fmax_m)
        score_all = pd.DataFrame(np.array(score_all), index=eem_dataset.index,
                                 columns=list(self.cluster_specific_models.keys()))
        fmax_all = pd.DataFrame(np.array(fmax_all), index=eem_dataset.index,
                                columns=list(self.cluster_specific_models.keys()))
        best_model_idx = np.argmin(sample_error, axis=0)
        # least_model_errors = np.min(sample_error, axis=0)
        # score_opt = np.array([score_all[i, j] for j, i in enumerate(best_model_idx)])
        # fmax_opt = np.array([fmax_all[i, j] for j, i in enumerate(best_model_idx)])
        best_model_label = np.array([list(self.cluster_specific_models.keys())[idx] for idx in best_model_idx])
        best_model_label = pd.DataFrame(best_model_label, index=eem_dataset.index, columns=['best-fit model'])
        sample_error = pd.DataFrame(np.array(sample_error), index=eem_dataset.index,
                                    columns=list(self.cluster_specific_models.keys()))
        return best_model_label, score_all, fmax_all, sample_error

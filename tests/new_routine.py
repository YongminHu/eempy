import pickle

import numpy as np
from scipy.stats import zscore
from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax, plot_eem_stack
from itertools import product
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import f_oneway, kruskal
from statsmodels import robust

np.random.seed(42)

# ---------------Read EEM dataset-----------------

# eem_dataset_path = \
#     "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset_path = \
    "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/sample_276_ex_274_em_310_mfem_5_gaussian_rsu_rs_interpolated.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset.median_filter(footprint=(5, 5), copy=False)
eem_dataset.cutting(ex_min=274, ex_max=500, em_min=312, em_max=500, copy=False)
# eem_dataset.raman_scattering_removal(width=15, interpolation_method='nan', copy=False)
# eem_dataset.eem_stack = np.nan_to_num(eem_dataset.eem_stack, copy=True, nan=0)
eem_dataset_july, _ = eem_dataset.filter_by_index(None, ['2024-07-'], copy=True)
eem_dataset_october, _ = eem_dataset.filter_by_index(None, ['2024-10-'], copy=True)
eem_dataset_original, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
eem_dataset_quenched, _ = eem_dataset.filter_by_index(['B1C2'], None, copy=True)
idx_top_oct = [i for i in range(len(eem_dataset_october.index)) if 'B1C1' in eem_dataset_october.index[i]]
idx_bot_oct = [i for i in range(len(eem_dataset_october.index)) if 'B1C2' in eem_dataset_october.index[i]]
idx_top_jul = [i for i in range(len(eem_dataset_july.index)) if 'B1C1' in eem_dataset_july.index[i]]
idx_bot_jul = [i for i in range(len(eem_dataset_july.index)) if 'B1C2' in eem_dataset_july.index[i]]

eem_dataset_bac_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
eem_dataset_bac = read_eem_dataset_from_json(eem_dataset_bac_path)
bacteria_eem = eem_dataset_bac.eem_stack[-5]
bacteria_eem = eem_interpolation(bacteria_eem, eem_dataset_bac.ex_range, eem_dataset_bac.em_range,
                                 eem_dataset.ex_range, eem_dataset.em_range, method='linear')
# bacteria_eem, _ = eem_raman_scattering_removal(bacteria_eem, eem_dataset.ex_range, eem_dataset.em_range,
#                                                width=10, interpolation_method='nan')
bacteria_eem = np.nan_to_num(bacteria_eem, nan=0)
prior_dict_ref = {0: bacteria_eem.reshape(-1)}

with open('C:/PhD/publication/2025_prior_knowledge/approx_components.pkl', 'rb') as file:
    approx_components = pickle.load(file)


def plot_all_f0f(model, eem_dataset, kw_top, kw_bot, target_analyte, eps=0.01, plot=True, zscore_threshold=3):
    target_train = eem_dataset.ref[target_analyte]
    valid_indices_train = target_train.index[~target_train.isna()]
    fmax = model.fmax
    fmax_original = fmax[fmax.index.str.contains(kw_top)]
    mask = fmax_original.index.isin(valid_indices_train)
    fmax_original = fmax_original[mask]
    fmax_quenched = fmax[fmax.index.str.contains(kw_bot)]
    fmax_quenched = fmax_quenched[mask]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()

    if plot:
        fig, axes = plt.subplots(ncols=2, nrows=(model.n_components + 1) // 2, figsize=(8, 10))
        for rank_target in range(model.n_components):
            fmax_ratio_target = fmax_ratio[:, rank_target]
            fmax_ratio_target_valid = fmax_ratio_target[(fmax_ratio_target >= 0) & (fmax_ratio_target <= 1e3)]
            fmax_ratio_z_scores = zscore(fmax_ratio_target_valid, nan_policy='omit')
            ratio_nan = 1 - fmax_ratio_target_valid.shape[0] / fmax_ratio_target.shape[0]
            fmax_ratio_target_filtered = fmax_ratio_target_valid[np.abs(fmax_ratio_z_scores) <= zscore_threshold]

            if (model.n_components + 1) // 2 > 1:
                counts, bins, patches = axes[rank_target // 2, rank_target % 2].hist(fmax_ratio_target_filtered,
                                                                                     bins=30,
                                                                                     density=True, alpha=0.5,
                                                                                     color='blue',
                                                                                     label='training', zorder=0,
                                                                                     edgecolor='black')
                axes[rank_target // 2, rank_target % 2].set_ylabel('Density', fontsize=18)
                axes[rank_target // 2, rank_target % 2].set_xlabel(f'C{rank_target + 1}' + ' $F_{0}/F$', fontsize=18)
                axes[rank_target // 2, rank_target % 2].tick_params(axis='both', labelsize=16)
                axes[rank_target // 2, rank_target % 2].text(
                    0.01, 0.99,
                    f'nan_ratio: {ratio_nan:2f}',
                    transform=axes[rank_target // 2, rank_target % 2].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left'
                )
            else:
                counts, bins, patches = axes[rank_target % 2].hist(fmax_ratio_target_filtered, bins=30,
                                                                   density=True, alpha=0.5, color='blue',
                                                                   label='training', zorder=0,
                                                                   edgecolor='black')
                axes[rank_target % 2].set_ylabel('Density', fontsize=18)
                axes[rank_target % 2].set_xlabel(f'C{rank_target + 1}' + ' $F_{0}/F$', fontsize=18)
                axes[rank_target % 2].tick_params(axis='both', labelsize=16)
                axes[rank_target % 2].text(
                    0.01, 0.99,
                    f'nan_ratio: {ratio_nan:2f}',
                    transform=axes[rank_target % 2].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left'
                )
        fig.tight_layout()
        fig.show()
    # clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None)
    clustering = DBSCAN(eps=eps, min_samples=int(fmax_ratio.shape[0] / 3))
    outlier_labels = clustering.fit_predict(fmax_ratio)
    qualifier_indices = list(fmax_original.index[outlier_labels != -1]) + list(
        fmax_quenched.index[outlier_labels != -1])
    outliers_indices = list(fmax_original.index[outlier_labels == -1]) + list(fmax_quenched.index[outlier_labels == -1])
    eem_dataset_cleaned, _ = eem_dataset.filter_by_index(mandatory_keywords=None, optional_keywords=qualifier_indices)
    return fmax_ratio, eem_dataset_cleaned, outliers_indices


def plot_fmax_vs_truth(fmax_train, fmax_test, truth_train, truth_test, n, info_dict, fit_intercept=True):
    info_dict_copy = copy.deepcopy(info_dict)
    fig, ax = plt.subplots()

    # Plot training data
    ax.plot(fmax_train.iloc[:, n], truth_train.to_numpy(), 'o', label='training')

    # Train model on training data
    lr_n = LinearRegression(fit_intercept=fit_intercept)
    mask_train_n = ~np.isnan(truth_train.to_numpy())
    lr_n.fit(fmax_train.iloc[mask_train_n, [n]].to_numpy(), truth_train.iloc[mask_train_n])
    y_pred_train = lr_n.predict(fmax_train.iloc[mask_train_n, [n]].to_numpy())
    rmse_train = np.sqrt(mean_squared_error(truth_train.iloc[mask_train_n], y_pred_train))
    r2_train = r2_score(truth_train.iloc[mask_train_n], y_pred_train)

    info_dict_copy['r2_train'] = np.round(r2_train, 3)
    info_dict_copy['rmse_train'] = np.round(rmse_train, 3)
    info_dict_copy['fit_intercept'] = True if lr_n.fit_intercept else False

    x_line = np.array([0, 10000])
    if lr_n.fit_intercept:
        ax.plot(x_line, x_line * lr_n.coef_[0] + lr_n.intercept_, '--')
    else:
        ax.plot(x_line, x_line * lr_n.coef_[0], '--')

    # If test data is provided, evaluate and plot it
    if fmax_test is not None and truth_test is not None:
        ax.plot(fmax_test.iloc[:, n], truth_test.to_numpy(), 'o', label='testing')
        y_pred_test = lr_n.predict(fmax_test.iloc[:, [n]].to_numpy())
        mask_test_n = ~np.isnan(truth_test.to_numpy())
        rmse_test = np.sqrt(mean_squared_error(truth_test.iloc[mask_test_n].to_numpy(), y_pred_test[mask_test_n]))
        r2_test = r2_score(truth_test.iloc[mask_test_n], y_pred_test[mask_test_n])
        info_dict_copy['r2_test'] = np.round(r2_test, 3)
        info_dict_copy['rmse_test'] = np.round(rmse_test, 3)
        x_max = max(fmax_train.iloc[:, n].max(), fmax_test.iloc[:, n].max()) + 100
        y_max = max(truth_train.max(), truth_test.max()) + 0.5
    else:
        x_max = fmax_train.iloc[:, n].max() + 100
        y_max = truth_train.max() + 0.5

    ax.text(
        0.01, 0.99,
        '\n'.join(f'{k}: {v}' for k, v in info_dict_copy.items()),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left'
    )
    ax.set_title(f'Fmax C{n + 1} vs. {truth_train.name}', fontsize=18)
    ax.set_xlabel(f'Fmax C{n + 1}', fontsize=18)
    ax.set_ylabel(f'{truth_train.name}', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])
    ax.legend(loc='best', bbox_to_anchor=(0.95, 0.25), fontsize=16)
    fig.tight_layout()
    fig.show()


def plot_all_components(eem_model):
    plot_eem_stack(eem_model.components, eem_model.ex_range, eem_model.em_range,
                   titles=[f'C{i + 1}' for i in range(eem_model.n_components)])



def get_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values_product = product(*(param_grid[key] for key in keys))
    return [dict(zip(keys, values)) for values in values_product]


def mean_pairwise_correlation(vectors):
    n = len(vectors)
    corrs = [
        abs(pearsonr(vectors[i], vectors[j])[0])
        for i, j in combinations(range(n), 2)
    ]
    return np.mean(corrs)


def all_split_half_combinations(lst):
    n = len(lst) // 2
    indices = range(len(lst))
    result = []

    for comb_indices in itertools.combinations(indices, n):
        group1 = [lst[i] for i in comb_indices]
        group2 = [lst[i] for i in indices if i not in comb_indices]
        result.append((group1, group2))

    return result


def rank_one_approximation_svd(A):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # Rank-one approximation: only keep the first singular value/vector
    A1 = S[0] * np.outer(U[:, 0], VT[0, :])
    # Score: ratio of first singular value squared to total energy
    total_energy = np.sum(S**2)
    rank_one_energy = S[0]**2
    score = rank_one_energy / total_energy if total_energy > 0 else 0.0
    return A1, score


def rank_one_approximation_nmf(A, n_components=2, max_iter=1000, random_state=0):
    """
    Perform NMF and extract the dominant rank-one component.

    Parameters:
    A (np.ndarray): Non-negative matrix to factor.
    n_components (int): Number of components for NMF.
    max_iter (int): Max iterations for NMF.
    random_state (int): Random seed for reproducibility.

    Returns:
    dominant_component (np.ndarray): Dominant rank-one approximation (W[:, i] @ H[i, :]).
    i_dominant (int): Index of the dominant component.
    contribution_ratio (float): Ratio of Frobenius norm of dominant component to total approximation.
    """
    if np.any(A < 0):
        raise ValueError("Matrix A must be non-negative.")

    model = NMF(n_components=n_components, init='nndsvda', max_iter=max_iter, random_state=random_state)
    W = model.fit_transform(A)
    H = model.components_

    # Compute rank-one approximations and their Frobenius norms
    components = [np.outer(W[:, i], H[i, :]) for i in range(n_components)]
    energies = [np.linalg.norm(comp, 'fro')**2 for comp in components]

    # Identify dominant component
    i_dominant = int(np.argmax(energies))
    dominant_component = components[i_dominant]
    total_energy = np.sum([np.linalg.norm(W @ H, 'fro')**2])
    contribution_ratio = energies[i_dominant] / total_energy if total_energy > 0 else 0.0

    return dominant_component, contribution_ratio


# # -------------------Step 1: Detection of Outlier Samples with High Reconstruction Error-------------------
# dataset_train = eem_dataset_october
# n_components = 4
# param = {
#     # "prior_dict_H": {k: model_nmf_components_r1_dict[k] for k in [0, 1, 2]},
#     "gamma_H": 0,
#     "lam": 0,
#     # "fit_rank_one": {0: True, 1: True, 2: True, 3: True}
# }
#
# model = EEMNMF(
#         n_components=n_components,
#         max_iter_nnls=100,
#         max_iter_als=500,
#         init='nndsvd',
#         random_state=42,
#         solver='hals',
#         normalization=None,
#         sort_components_by_em=False,
#         # prior_ref_components=model_nmf_components_r1_dict,
#         kw_top='B1C1',
#         kw_bot='B1C2',
#         tol=1e-5,
#         **param
#     )
#
# eem_dataset_october_clean, outlier_indices, outlier_indices_history = model.robust_fit(
#     dataset_train, zscore_threshold=3.5, max_iter_outlier_removal=1, n_splits=6)
#
# # model.gamma_H = 3
# # model.prior_dict_H = {r: model_nmf_components_r1_dict_clean[r] for r in [2]}
# # model.lam = 1e6
# # model.fit(dataset_train_clean)
# plt.close()
# plot_all_components(model)
# dataset_test = eem_dataset_july
# # fmax_train = model.fmax
# _, fmax_train, _ = model.predict(
#     eem_dataset=eem_dataset_october_clean,
#     fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
#     idx_top=[i for i in range(len(eem_dataset_october_clean.index)) if 'B1C1' in eem_dataset_october_clean.index[i]],
#     idx_bot=[i for i in range(len(eem_dataset_october_clean.index)) if 'B1C2' in eem_dataset_october_clean.index[i]],
# )
# _, fmax_test, _ = model.predict(
#     eem_dataset=dataset_test,
#     fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
#     idx_top=[i for i in range(len(dataset_test.index)) if 'B1C1' in dataset_test.index[i]],
#     idx_bot=[i for i in range(len(dataset_test.index)) if 'B1C2' in dataset_test.index[i]],
# )
# truth_train = eem_dataset_october_clean.ref['TCC (million #/mL)'][~eem_dataset_october_clean.ref.index.isin(outlier_indices)]
# truth_test = dataset_test.ref['TCC (million #/mL)']
# plot_fmax_vs_truth(fmax_train=fmax_train, fmax_test=fmax_test,
#                    truth_train=truth_train,
#                    truth_test=truth_test,
#                    n=-1, info_dict=param, fit_intercept=True,
#                    )
# fmax_ratio, _, _ = plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)', plot=True)

# eem_dataset_october_clean.ref = eem_dataset_october_clean.ref[['TCC (million #/mL)', 'DOC (mg/L)']]

# with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned.pkl",
#           'wb') as file:
#     pickle.dump(eem_dataset_october_clean, file)

with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned.pkl",
          'rb') as file:
    eem_dataset_october_clean = pickle.load(file)


# # -------------------Step 2: Generate rank-one approximations as priors-------------------
# dataset_train = eem_dataset_october_clean
# n_components = 4
#
# model_all = EEMNMF(
#     n_components=n_components,
#     fit_rank_one=False,
#     max_iter_nnls=100,
#     max_iter_als=500,
#     init='nndsvd',
#     random_state=42,
#     solver='hals',
#     normalization=None,
#     sort_components_by_em=True,
#     tol=1e-5,
#     kw_top='B1C1',
#     kw_bot='B1C2'
# )
# model_all.fit(eem_dataset=dataset_train)
# plot_all_components(model_all)
# components_r1 = np.array([rank_one_approximation_nmf(model_all.components[i])[0] for i in range(n_components)])
# plot_eem_stack(components_r1, dataset_train.ex_range, dataset_train.em_range, titles=[f'C{i + 1} R1' for i in range(n_components)])
# model_all_ref_components = {k: components_r1[k].reshape(-1) for k in range(n_components)}
#
# model_all.prior_ref_components = model_all_ref_components
# sv = SplitValidation(base_model=model_all, n_split=6)
# sv.fit(dataset_train)
# rank1_approx_priors = []
# for r in range(4):
#     cr = []
#     for sub_model in sv.subset_specific_models.values():
#         cr.append(sub_model.components[r])
#     component_set_r = EEMDataset(
#         eem_stack=np.array(cr),
#         ex_range=model_all.ex_range, em_range=model_all.em_range
#     )
#     cr_parafac = PARAFAC(n_components=2, solver='hals',sort_components_by_em=True)
#     cr_parafac.fit(component_set_r)
#     r_primary = np.argmax(np.mean(cr_parafac.fmax.to_numpy(), axis=0))
#     print(np.mean(cr_parafac.fmax.to_numpy(), axis=0))
#     rank1_approx_priors.append(cr_parafac.components[r_primary])
# plot_eem_stack(np.array(rank1_approx_priors), dataset_train.ex_range, dataset_train.em_range, titles=[f'C{i + 1} R1' for i in range(n_components)])
# rank1_approx_priors = {k: rank1_approx_priors[k].reshape(-1) for k in range(n_components)}

# with open("C:/PhD/publication/2025_prior_knowledge/rank1_approx_priors_Oct2024.pkl",
#           'wb') as file:
#     pickle.dump(rank1_approx_priors, file)
#
with open("C:/PhD/publication/2025_prior_knowledge/rank1_approx_priors_Oct2024.pkl",
          'rb') as file:
    rank1_approx_priors = pickle.load(file)

rank1_approx_priors_reshaped = np.array([c.reshape([eem_dataset_october_clean.eem_stack.shape[1], eem_dataset_october_clean.eem_stack.shape[2]]) for c in rank1_approx_priors.values()])
plot_eem_stack(rank1_approx_priors_reshaped, eem_dataset_october_clean.ex_range, eem_dataset_october_clean.em_range, [1,2,3,4])

# -------------------Step 2.5: Detection of Components Sensitive to Rank-One Constraints-------------------
# #
# n_components = 4
# dataset_train = eem_dataset_october_clean
#
# model_parafac = EEMNMF(
#     n_components=4,
#     fit_rank_one={r: True for r in range(n_components)},
#     solver='hals',
#     init='nndsvd',
#     random_state=42
# )
# model_parafac.fit(dataset_train)
# model_parafac_components_dict = {k: model_parafac.components[k].reshape(-1) for k in range(n_components)}
#
# correlation_sim_all=[{} for i in range(n_components)]
# for r in range(n_components):
#     # fit_rank_one = {r: True for r in [i for i in range(n_components) if i != r]}
#     # fit_rank_one = {r: True}
#     prior_dict_H = {k: model_parafac.components[k].reshape(-1) for k in range(n_components) if k != r}
#     model = EEMNMF(
#         n_components=n_components,
#         fit_rank_one=False,
#         max_iter_nnls=100,
#         max_iter_als=500,
#         init='nndsvd',
#         random_state=42,
#         solver='hals',
#         normalization=None,
#         sort_components_by_em=False,
#         prior_ref_components=model_parafac_components_dict,
#         prior_dict_H=prior_dict_H,
#         gamma_H=1e8,
#     )
#     model.fit(eem_dataset=dataset_train)
#     for k in range(n_components):
#         # cosine_sim = cosine_similarity(model.components[k].flatten().reshape(1, -1),
#         #                                model_standard.components[k].flatten().reshape(1, -1))[0, 0]
#         correlation_sim = np.corrcoef(model.components[k].flatten(),
#                                       model_parafac.components[k].flatten())[0, 1]
#         correlation_sim_all[r][k] = correlation_sim
#     plot_all_components(model)
# correlation_sim_all_df = pd.DataFrame(correlation_sim_all)



# ------------Step 3: cross-validation & hyperparameter optimization-------

dataset_train = eem_dataset_october_clean
n_components = 4

param_grid = {
    'n_components': [4],
    'init': ['nndsvd'],
    'gamma_H': [0, 25, 50, 75, 100],
    'prior_dict_H': [
        # None,
        {k: rank1_approx_priors[k] for k in [0, 1, 2]},
    ],
    'lam': [0],
    'max_iter_als': [300],
    'max_iter_nnls': [500],
    'fit_rank_one': [
        False,
        # {r: True for r in range(n_components)}
    ]
}

param_combinations = get_param_combinations(param_grid)
components_lists_all = []

for k, p in enumerate(param_combinations):
    print(f"param_combinations: {k + 1};")
    components_list = [[] for i in range(p['n_components'])]
    fmax_ratio_list = [[] for i in range(p['n_components'])]
    model = EEMNMF(
        solver='hals',
        random_state=42,
        sort_components_by_em=False,
        prior_ref_components=rank1_approx_priors,
        kw_top='B1C1',
        kw_bot='B1C2',
        normalization=None,
        tol=1e-8,
        **p
    )
    model.fit(dataset_train)
    components_lists_all.append(model.components)
    _, fmax_train, eem_re_train = model.predict(
        eem_dataset=dataset_train,
        fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
        idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
        idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
    )
    explained_variance = model.explained_variance()
    param_combinations[k]['explained_variance'] = explained_variance
    for ref_col in dataset_train.ref.columns:
        mask_ref = ~np.isnan(dataset_train.ref[ref_col].to_numpy())
        y_train = dataset_train.ref[ref_col].to_numpy()[mask_ref]
        for r in range(model.n_components):
            x_train_r = fmax_train.iloc[mask_ref, [r]].to_numpy()
            lr = LinearRegression(fit_intercept=True)
            lr.fit(x_train_r, y_train)
            coef_r = lr.coef_[0]
            intercept_r = lr.intercept_
            y_pred = lr.predict(x_train_r)
            r2_r = lr.score(x_train_r, y_train)
            rmse_r = np.sqrt(mean_squared_error(y_train, y_pred))
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-coef_fit_intercept=True'] = coef_r
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-intercept_fit_intercept=True'] = intercept_r
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-r2_with_fit_intercept=True'] = r2_r
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-rmse_with_fit_intercept=True'] = rmse_r
            lr = LinearRegression(fit_intercept=False)
            lr.fit(x_train_r, y_train)
            coef_r = lr.coef_[0]
            intercept_r = lr.intercept_
            y_pred = lr.predict(x_train_r)
            r2_r = lr.score(x_train_r, y_train)
            rmse_r = np.sqrt(mean_squared_error(y_train, y_pred))
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-coef_fit_intercept=False'] = coef_r
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-intercept_fit_intercept=False'] = intercept_r
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-r2_fit_intercept=False'] = r2_r
            param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-rmse_fit_intercept=False'] = rmse_r
    if model.idx_top is not None and model.idx_bot is not None:
        fmax_train_top = fmax_train[fmax_train.index.str.contains('B1C1')]
        fmax_train_bot = fmax_train[fmax_train.index.str.contains('B1C2')]
        fmax_ratio = fmax_train_top.to_numpy()/fmax_train_bot.to_numpy()
        fmax_ratio_means = np.mean(fmax_ratio, axis=0)
        fmax_ratio_std = np.std(fmax_ratio, axis=0)
        fmax_ratio_max = np.max(fmax_ratio, axis=0)
        fmax_ratio_min = np.min(fmax_ratio, axis=0)
        for r in range(n_components):
            param_combinations[k][f'C{r+1}-fmax_ratio_means'] = fmax_ratio_means[r]
            param_combinations[k][f'C{r+1}-fmax_ratio_std'] = fmax_ratio_std[r]
            param_combinations[k][f'C{r+1}-fmax_ratio_min'] = fmax_ratio_max[r]
            param_combinations[k][f'C{r+1}-fmax_ratio_max'] = fmax_ratio_min[r]
    sv = SplitValidation(base_model=model, n_split=4, random_state=42)
    sv.fit(dataset_train)
    sim_components = sv.compare_components()
    sim_component_sim = sim_components.mean()
    for c in range(model.n_components):
        param_combinations[k][sim_component_sim.index[c]+' similarities'] = sim_component_sim.iloc[c]
    plot_all_components(model)

param_combinations_df = pd.DataFrame(param_combinations)


# with open("C:/PhD/publication/2025_prior_knowledge/param_combinations.pkl",
#           'wb') as file:
#     pickle.dump(param_combinations_df, file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/components_lists_all.pkl",
#           'wb') as file:
#     pickle.dump(components_lists_all, file)



# with open("C:/PhD/publication/2025_prior_knowledge/param_combinations.pkl",
#           'rb') as file:
#     param_combinations_df = pickle.load(file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/components_lists_all.pkl",
#           'rb') as file:
#     components_lists_all = pickle.load(file)


# -------------------Step 3: Optimize hyperparameters with Split-half Cross-Validation-------------------



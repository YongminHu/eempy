import copy
import pickle

import numpy as np
import pandas as pd
import sns
from scipy.stats import zscore
from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax, plot_eem_stack
from itertools import product
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.cluster import DBSCAN

np.random.seed(42)

# ---------------Read EEM dataset-----------------

eem_dataset_path = \
    "tests/EawagGAC_2025_unquenched_unfiltered.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
# eem_dataset.median_filter(footprint=(5, 5), copy=False)
# eem_dataset.raman_scattering_removal(width=15, interpolation_method="nan", copy=False)
eem_dataset.cutting(ex_min=250, ex_max=450, em_min=300, em_max=600, inplace=True)
# eem_dataset, _ = eem_dataset.filter_by_index(["2024-10"], None, copy=True)

eem_dataset_feb = eem_dataset.filter_by_index(["2025-02-"], None, inplace=False)
eem_dataset_filtered = eem_dataset.filter_by_index(["-filtered"], None, inplace=False)
indices_non_feb = [i for i in eem_dataset.index if i not in eem_dataset_feb.index and i not in eem_dataset_filtered.index]
eem_dataset_non_feb = eem_dataset.filter_by_index(None, indices_non_feb, inplace=False)


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


def plot_fmax_vs_truth(fmax_train, truth_train, n, info_dict=None, fmax_test=None, truth_test=None, fit_intercept=True):
    info_dict_copy = copy.deepcopy(info_dict) if info_dict is not None else {}
    fig, ax = plt.subplots(figsize=(4,4))
    # Plot training data
    ax.plot(fmax_train.iloc[:, n], truth_train.to_numpy(), 'o', label='training')

    # Train model on training data
    lr_n = LinearRegression(fit_intercept=fit_intercept)
    mask_train_n = ~np.isnan(truth_train.to_numpy())
    lr_n.fit(fmax_train.iloc[mask_train_n, [n]].to_numpy(), truth_train.iloc[mask_train_n])
    y_pred_train = lr_n.predict(fmax_train.iloc[mask_train_n, [n]].to_numpy())
    rmse_train = np.sqrt(mean_squared_error(truth_train.iloc[mask_train_n], y_pred_train))
    r2_train = r2_score(truth_train.iloc[mask_train_n], y_pred_train)

    info_dict_copy['r2'] = np.round(r2_train, 3)
    info_dict_copy['rmse'] = np.round(rmse_train, 3)
    # info_dict_copy['fit_intercept'] = True if lr_n.fit_intercept else False

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
        y_max = max(truth_train.max(), truth_test.max()) + 0.3
    else:
        x_max = fmax_train.iloc[:, n].max() + 100
        y_max = truth_train.max() + 0.3

    ax.text(
        0.01, 0.99,
        '\n'.join(f'{k}: {v}' for k, v in info_dict_copy.items()),
        transform=ax.transAxes,
        fontsize=15,
        verticalalignment='top',
        horizontalalignment='left'
    )
    # ax.set_title(f'Fmax C{n + 1} vs. {truth_train.name}', fontsize=18)
    ax.set_xlabel(f'Fmax C{n + 1}', fontsize=18)
    ax.set_ylabel(f'{truth_train.name}', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])
    # ax.legend(loc='best', bbox_to_anchor=(0.95, 0.25), fontsize=16)
    fig.tight_layout()
    fig.show()


def plot_all_components(eem_model):
    plot_eem_stack(
        eem_model.components,
        eem_model.ex_range,
        eem_model.em_range,
        n_cols=5,
        cbar=None,
        plot_x_axis_label=False,
        plot_y_axis_label=False,
                   )



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


def random_split_columns(arr, columns, n_splits, random_state=42):
    """
    Splits specified columns of a NumPy array into n_splits random components,
    preserving the original column order. Each set of split values replaces the
    original column and sums row-wise to the original value.

    Parameters:
    arr (np.ndarray): 2D array of shape (n_samples, n_features).
    columns (list): List of column indices to split.
    n_splits (int): Number of splits per column.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    np.ndarray: Modified array with split columns in original positions.
    """
    rng = np.random.default_rng(random_state)
    arr = np.asarray(arr)
    n_rows, n_cols = arr.shape
    columns = sorted(columns)

    result_cols = []
    current_col = 0

    for col in columns:
        # Append untouched columns before this split column
        while current_col < col:
            result_cols.append(arr[:, current_col][:, np.newaxis])
            current_col += 1

        # Split the column
        col_vals = arr[:, col]
        rand_fracs = rng.random((n_rows, n_splits))
        rand_fracs /= rand_fracs.sum(axis=1, keepdims=True)
        split_vals = rand_fracs * col_vals[:, np.newaxis]

        # Append split columns
        for i in range(n_splits):
            result_cols.append(split_vals[:, i][:, np.newaxis])

        current_col += 1  # Skip the original column

    # Append remaining untouched columns
    while current_col < n_cols:
        result_cols.append(arr[:, current_col][:, np.newaxis])
        current_col += 1

    return np.hstack(result_cols)

def get_component_set(base_model, dataset_train, n_splits=6):
    model_standard = PARAFAC(
        n_components=base_model.n_components,
        solver="hals",
        max_iter_nnls=300,
        max_iter_als=300,
        sort_components_by_em=True,
    )
    model_standard.fit(dataset_train)
    prior_ref_components = {k: model_standard.components[k].reshape(-1) for k in range(base_model.n_components)}
    base_model.prior_ref_components = prior_ref_components
    sv = SplitValidation(base_model=base_model, n_splits=n_splits)
    sv.fit(dataset_train)

    c_all = []
    for sub_model in sv.subset_specific_models.values():
        c_all.append(sub_model.components)
    component_set = EEMDataset(
        eem_stack=np.concatenate(c_all, axis=0),
        ex_range=dataset_train.ex_range, em_range=dataset_train.em_range
    )
    return component_set


# # -------------------Step 1: Detection of Outlier Samples with High Reconstruction Error-------------------
dataset_train = eem_dataset
n_components = 5
param = {
    # "prior_dict_H": {k: model_nmf_components_r1_dict[k] for k in [0, 1, 2]},
    # "gamma_H": 0,
    # "lam": 0,
    # "fit_rank_one": {0: True, 1: True, 2: True, 3: True}
}

model = EEMNMF(
        n_components=n_components,
        fit_rank_one={k: True for k in range(n_components)},
        max_iter_nnls=1000,
        max_iter_als=1000,
        init='nndsvd',
        random_state=42,
        solver='hals',
        sort_components_by_em=False,
        # prior_ref_components=model_nmf_components_r1_dict,
        # kw_top='B1C1',
        # kw_bot='B1C2',
        tol=1e-6,
        **param
    )

eem_dataset_clean, outlier_indices, outlier_indices_history = model.robust_fit(
    dataset_train, zscore_threshold=3, max_iter_outlier_removal=1, n_splits=6)

# model.gamma_H = 3
# model.prior_dict_H = {r: model_nmf_components_r1_dict_clean[r] for r in [2]}
# model.lam = 1e6
# model.fit(dataset_train_clean)
plt.close()
plot_all_components(model)
# dataset_test = eem_dataset_july
# fmax_train = model.fmax
_, fmax_train, _ = model.predict(
    eem_dataset=eem_dataset_clean,
    fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
    idx_top=[i for i in range(len(eem_dataset_clean.index)) if 'B1C1' in eem_dataset_clean.index[i]],
    idx_bot=[i for i in range(len(eem_dataset_clean.index)) if 'B1C2' in eem_dataset_clean.index[i]],
)
# _, fmax_test, _ = model.predict(
#     eem_dataset=dataset_test,
#     fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
#     idx_top=[i for i in range(len(dataset_test.index)) if 'B1C1' in dataset_test.index[i]],
#     idx_bot=[i for i in range(len(dataset_test.index)) if 'B1C2' in dataset_test.index[i]],
# )
truth_train = eem_dataset_clean.ref['TCC (million #/mL)'][~eem_dataset_clean.ref.index.isin(outlier_indices)]
# truth_test = dataset_test.ref['TCC (million #/mL)']
plot_fmax_vs_truth(
    fmax_train=fmax_train,
    # fmax_test=fmax_test,
    truth_train=truth_train,
    # truth_test=truth_test,
    n=2, info_dict=param, fit_intercept=True
    )
# fmax_ratio, _, _ = plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)', plot=True)

# _ = eem_dataset_clean.to_json("tests/Waterhub_Oct2024_unquenched_unfiltered_cleaned_82.json")
eem_dataset_clean = read_eem_dataset_from_json("tests/Waterhub_Jan2022_unquenched_unfiltered_cleaned_164.json")

eem_dataset_clean.ref = eem_dataset_clean.ref[['TCC (million #/mL)', 'DOC (mg/L)']]

# with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned_2.pkl",
#           'wb') as file:
#     pickle.dump(eem_dataset_clean, file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned.pkl",
#           'rb') as file:
#     eem_dataset_clean = pickle.load(file)


# -------------------Step 2: Generate rank-one approximations as priors-------------------


#
# with open("/Users/User/Documents/PhD/publication/2025_prior_knowledge/rank1_approx_priors_Oct2024.pkl",
#           'wb') as file:
#     pickle.dump(rank1_approx_priors, file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/rank1_approx_priors_Oct2024.pkl",
#           'rb') as file:
#     rank1_approx_priors = pickle.load(file)

# rank1_approx_priors_reshaped = np.array([c.reshape([eem_dataset_october_clean.eem_stack.shape[1], eem_dataset_october_clean.eem_stack.shape[2]]) for c in rank1_approx_priors.values()])
# plot_eem_stack(rank1_approx_priors_reshaped, eem_dataset_october_clean.ex_range, eem_dataset_october_clean.em_range, [1,2,3,4])
#
# # -------------------Step 2.5: Detection of Components Sensitive to Rank-One Constraints-------------------
# #
# n_components = 4
# dataset_train = eem_dataset_clean
#
# model_parafac = PARAFAC(
#     n_components=4,
#     # fit_rank_one={r: True for r in range(n_components)},
#     solver='hals',
#     max_iter_als=300,
#     max_iter_nnls=300,
#     init='svd',
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
#         max_iter_nnls=300,
#         max_iter_als=300,
#         init='svd',
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
#
#
#
# # ------------Step 3: cross-validation & hyperparameter optimization-------
#
# dataset_train = eem_dataset_clean
# n_components = 4
#
# param_grid = {
#     'n_components': [4],
#     'init': ['nndsvda'],
#     'gamma_H': [2500],
#     'prior_dict_H': [
#         # None,
#         {k: rank1_approx_priors[k] for k in [1]},
#     ],
#     'lam': [0],
#     'max_iter_als': [300],
#     'max_iter_nnls': [300],
#     'fit_rank_one': [
#         False,
#         # {r: True for r in range(n_components)}
#     ]
# }
#
# param_combinations = get_param_combinations(param_grid)
# components_lists_all = []
#
# for k, p in enumerate(param_combinations):
#     print(f"param_combinations: {k + 1};")
#     components_list = [[] for i in range(p['n_components'])]
#     fmax_ratio_list = [[] for i in range(p['n_components'])]
#     model = EEMNMF(
#         solver='hals',
#         random_state=42,
#         sort_components_by_em=False,
#         prior_ref_components=rank1_approx_priors,
#         # kw_top='B1C1',
#         # kw_bot='B1C2',
#         normalization=None,
#         tol=1e-6,
#         **p
#     )
#     model.fit(dataset_train)
#     plot_all_components(model)
#     components_lists_all.append(model.components)
#     _, fmax_train, eem_re_train = model.predict(
#         eem_dataset=dataset_train,
#         fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
#         idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
#         idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
#     )
#     # explained_variance = model.variance_explained()
#     # param_combinations[k]['explained_variance'] = explained_variance
#     for ref_col in dataset_train.ref.columns:
#         mask_ref = ~np.isnan(dataset_train.ref[ref_col].to_numpy())
#         y_train = dataset_train.ref[ref_col].to_numpy()[mask_ref]
#         for r in range(model.n_components):
#             x_train_r = fmax_train.iloc[mask_ref, [r]].to_numpy()
#             lr = LinearRegression(fit_intercept=True)
#             lr.fit(x_train_r, y_train)
#             coef_r = lr.coef_[0]
#             intercept_r = lr.intercept_
#             y_pred = lr.predict(x_train_r)
#             r2_r = lr.score(x_train_r, y_train)
#             rmse_r = np.sqrt(mean_squared_error(y_train, y_pred))
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-coef_fit_intercept=True'] = coef_r
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-intercept_fit_intercept=True'] = intercept_r
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-r2_with_fit_intercept=True'] = r2_r
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-rmse_with_fit_intercept=True'] = rmse_r
#             lr = LinearRegression(fit_intercept=False)
#             lr.fit(x_train_r, y_train)
#             coef_r = lr.coef_[0]
#             intercept_r = lr.intercept_
#             y_pred = lr.predict(x_train_r)
#             r2_r = lr.score(x_train_r, y_train)
#             rmse_r = np.sqrt(mean_squared_error(y_train, y_pred))
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-coef_fit_intercept=False'] = coef_r
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-intercept_fit_intercept=False'] = intercept_r
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-r2_fit_intercept=False'] = r2_r
#             param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-rmse_fit_intercept=False'] = rmse_r
#     # if model.idx_top is not None and model.idx_bot is not None:
#     #     fmax_train_top = fmax_train[fmax_train.index.str.contains('B1C1')]
#     #     fmax_train_bot = fmax_train[fmax_train.index.str.contains('B1C2')]
#     #     fmax_ratio = fmax_train_top.to_numpy()/fmax_train_bot.to_numpy()
#     #     fmax_ratio_means = np.mean(fmax_ratio, axis=0)
#     #     fmax_ratio_std = np.std(fmax_ratio, axis=0)
#     #     fmax_ratio_max = np.max(fmax_ratio, axis=0)
#     #     fmax_ratio_min = np.min(fmax_ratio, axis=0)
#     #     for r in range(n_components):
#     #         param_combinations[k][f'C{r+1}-fmax_ratio_means'] = fmax_ratio_means[r]
#     #         param_combinations[k][f'C{r+1}-fmax_ratio_std'] = fmax_ratio_std[r]
#     #         param_combinations[k][f'C{r+1}-fmax_ratio_min'] = fmax_ratio_max[r]
#     #         param_combinations[k][f'C{r+1}-fmax_ratio_max'] = fmax_ratio_min[r]
#     #     _ = plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)')
#     # sv = SplitValidation(base_model=model, n_split=4, random_state=42)
#     # sv.fit(dataset_train)
#     # sim_components = sv.compare_components()
#     # sim_component_sim = sim_components.mean()
#     # for c in range(model.n_components):
#     #     param_combinations[k][sim_component_sim.index[c]+' similarities'] = sim_component_sim.iloc[c]
#
#
# param_combinations_df = pd.DataFrame(param_combinations)
#

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


# ------------------Normal PARAFAC--------------------

dataset_train = eem_dataset
dataset_train_unquenched = dataset_train.filter_by_index("B", None, inplace=False)
dataset_train_rs_removed = copy.deepcopy(dataset_train)
dataset_train_rs_removed.raman_scattering_removal(width=15, interpolation_method='zero', inplace=True)
dataset_train_rs_removed.rayleigh_scattering_removal(width_o1=20, width_o2=40, interpolation_method_o2='zero', inplace=True)
dataset_train_rs_removed_unquenched = dataset_train_rs_removed.filter_by_index("B", None, inplace=False)


n_components = 5
model_normal_parafac = PARAFAC(
    n_components=n_components,
    init='svd',
    solver='hals',
    max_iter_nnls=1000,
    max_iter_als=1000,
    sort_components_by_em=True,
    random_state=42,
)
model_normal_parafac.fit(dataset_train)
# plot_all_components(model_normal_parafac)
plot_eem_stack(
    model_normal_parafac.components,
    model_normal_parafac.ex_range,
    model_normal_parafac.em_range,
    n_cols=4,
    cbar=None,
    plot_x_axis_label=False,
    plot_y_axis_label=False,)
# plot_fmax_vs_truth(
#     fmax_train=model_normal_parafac.fmax,
#     truth_train=dataset_train.ref['TCC (million #/mL)'],
#     info_dict={}, n=0, fit_intercept=True
# )
prior_components = {k: model_normal_parafac.components[k].reshape(-1) for k in range(model_normal_parafac.n_components)}

# def deconvolute_fmax(model_test, target_rs, eps=10):
#     n_col_tot = model_test.n_components + len(target_rs)
#     a_deconvoluted = np.zeros([model_test.eem_stack.shape[0], n_col_tot])
#     b_deconvoluted = np.zeros([model_test.eem_stack.shape[1], n_col_tot])
#     c_deconvoluted = np.zeros([model_test.eem_stack.shape[2], n_col_tot])
#     a_deconvoluted = np.zeros([n_col_tot, model_test.eem_stack.shape[1]*model_test.eem_stack.shape[2]])
#
#     for target_r in target_rs:
#         model_test_fmax = model_test.fmax.to_numpy()[:, target_r]
#         model_test_fmax_quenched = model_test_fmax[model_test.fmax.index.str.contains('B1C2')]
#         model_test_fmax_unquenched = model_test_fmax[model_test.fmax.index.str.contains('B1C1')]
#         model_test_fmax_ratio = model_test_fmax_unquenched / model_test_fmax_quenched
#         beta_guess1 = np.percentile(model_test_fmax_ratio, 5)
#         beta_guess2 = np.percentile(model_test_fmax_ratio, 95)
#         M = np.array([[1,1], [beta_guess1, beta_guess2]])
#         fmax_deconvoluted = []
#         for i in range(model_test_fmax_ratio.shape[0]):
#             fmax_divided_i = np.linalg.solve(M, np.array([model_test_fmax_quenched[i], model_test_fmax_unquenched[i]]))
#             fmax_deconvoluted.append(fmax_divided_i)
#         fmax_deconvoluted_quenched = np.clip(np.array(fmax_deconvoluted), eps, None)
#         fmax_deconvoluted_unquenched = fmax_deconvoluted_quenched * M[1, :]
#         fmax_deconvoluted = np.concatenate([fmax_deconvoluted_unquenched, fmax_deconvoluted_quenched], axis=0)
#         a_deconvoluted = np.concatenate([fmax_deconvoluted, model_standard.fmax.to_numpy()[:, 1:]], axis=1)
#         b_deconvoluted = np.flipud(model_standard.ex_loadings.to_numpy()[:, [0,0,1,2,3,4]])
#         c_deconvoluted = model_standard.em_loadings.to_numpy()[:, [0,0,1,2,3,4]]
#         h_deconvoluted = model_standard.components.reshape((model_standard.components.shape[0],-1))[[0,0,1,2,3,4], :]

# -----------first layer of robustness check-------

base_model_layer1 = EEMNMF(
    n_components=5,
    solver="hals",
    max_iter_nnls=300,
    max_iter_als=300,
    prior_ref_components=prior_components,
    init='svd',
)
base_model_layer2 = EEMNMF(
    n_components=5,
    solver="hals",
    max_iter_nnls=300,
    max_iter_als=300,
    init='svd',
)
Rparafac = RobustPARAFAC(base_model_layer1, base_model_layer2, n_splits_layer1=6, random_state=42)
Rparafac.fit(dataset_train_unquenched)
plot_all_components(Rparafac.base_model_layer2)


base_model_layer2_test = EEMNMF(
    n_components=7,
    solver="hals",
    max_iter_nnls=300,
    max_iter_als=300,
    sort_components_by_em=True,
    init='svd',
    random_state=42
)
base_model_layer2_test.fit(Rparafac.component_set_layer1)
plot_all_components(base_model_layer2_test)
sim = component_similarity(model_normal_parafac.components, base_model_layer2_test.components)

def best_average_similarity(similarity_matrix):
    cost_matrix = -similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_similarity = similarity_matrix[row_ind, col_ind].sum()
    average_similarity = total_similarity / len(similarity_matrix)
    return average_similarity
print(best_average_similarity(sim.to_numpy()))

# with open("tests/Rparafac_EawagGAC_2025_6_components.pkl",
#           'rb') as file:
#     Rparafac = pickle.load(file)


_, fmax_robust, _ = base_model_layer2_test.predict(dataset_train_unquenched)
robust_init_a = fmax_robust.to_numpy()
robust_init_b = np.flipud(base_model_layer2_test.ex_loadings.to_numpy())
robust_init_c = base_model_layer2_test.em_loadings.to_numpy()
robust_model = PARAFAC(
    n_components=6,
    solver="hals",
    max_iter_nnls=500,
    max_iter_als=500,
    init="custom",
    custom_init=[robust_init_a, robust_init_b, robust_init_c],
)
robust_model.fit(dataset_train_unquenched)
plot_all_components(robust_model)


# ---------High-resolution EEM decomposition-----
model_standard = model_normal_parafac

component_set = EEMDataset(model_standard.components, model_standard.ex_range, model_standard.em_range)
component_set.raman_scattering_removal(width=15, interpolation_method='zero', inplace=True)
component_set.rayleigh_scattering_removal(width_o1=20, width_o2=40, interpolation_method_o2="zero", inplace=True)
# component_set.cutting(ex_min=250, ex_max=325, em_min=300, em_max=370,copy=False)
# cutting_mask = ~(component_set.eem_stack<1e-2).all(axis=(1,2))
# component_set.eem_stack = component_set.eem_stack[cutting_mask]

# prior_components = {k: model_standard.components[k].reshape(-1) for k in range(model_standard.n_components)}
# prior_ex = {k: np.flipud(model_standard.ex_loadings.to_numpy()[:, k]) for k in range(model_standard.n_components)}
# prior_em = {k: model_standard.em_loadings.to_numpy()[:, k] for k in range(model_standard.n_components)}

prior_components_rs_removed = {k: component_set.eem_stack[k].reshape(-1) for k in range(component_set.eem_stack.shape[0])}

n_components_original = 5
r_to_deconvolute = 0
n_deconvoluted = 2
new_order = [0 for _ in range(n_components_original+n_deconvoluted-1)]
for i in range(n_components_original):
    if i < r_to_deconvolute:
        new_order[i] = i
    elif i == r_to_deconvolute:
        for j in range(n_deconvoluted):
            new_order[i+j] = i
    elif i > r_to_deconvolute:
        new_order[i+n_deconvoluted-1] = i
fmax_standard = model_standard.fmax.to_numpy()
fmax_init = random_split_columns(fmax_standard,[r_to_deconvolute],n_deconvoluted)
# ex_init = np.flipud(model_standard.ex_loadings.to_numpy()[:, new_order])
# em_init = model_standard.em_loadings.to_numpy()[:, new_order]
components_init = model_standard.components.reshape((model_standard.components.shape[0],-1))[new_order, :]
components_init_rs_removed = component_set.eem_stack.reshape((n_components_original,-1))[new_order, :]
# components_init_test = model_hr_layer1.components.reshape((model_standard.components.shape[0],-1))[new_order, :]
# dataset_train_rs_removed_unquenched.cutting(ex_min=250, ex_max=325, em_min=300, em_max=370,copy=False)

model_hr_layer1 = EEMNMF(
    n_components=n_components_original + n_deconvoluted - 1,
    max_iter_nnls=500,
    max_iter_als=500,
    random_state=42,
    solver='hals',
    sort_components_by_em=False,
    tol=1e-8,
    init='custom',
    custom_init=[fmax_init, components_init_rs_removed],
    fixed_components=[i for i in range(n_components_original + n_deconvoluted - 1) if new_order[i] != r_to_deconvolute],
    # fixed_components=[i for i in range(n_components_original + n_deconvoluted - 1) if new_order[i] not in [0,3]],
    prior_dict_W = {new_order.index(r_to_deconvolute): dataset_train_rs_removed_unquenched.ref['TCC (million #/mL)'].to_numpy()},
    prior_dict_H={i: prior_components_rs_removed[k] for i, k in enumerate(new_order)},
    gamma_H={k: 0 for k in range(n_components_original + n_deconvoluted - 1) if new_order[k] == r_to_deconvolute},
    # gamma_H={k: 1e6 for k in range(n_components_original + n_deconvoluted - 1) if new_order[k] in [0,3]},
    # gamma_H = {0: 1e6, 1: 1e6, 4: 0}
    gamma_W={new_order.index(r_to_deconvolute): 1e7},
    # lam=3e3,
    # idx_top="B1C1",
    # idx_bot="B1C2",
)
model_hr_layer1.fit(dataset_train_rs_removed_unquenched)
plot_all_components(model_hr_layer1)
component_hr_layer1 = EEMDataset(model_hr_layer1.components, model_hr_layer1.ex_range, model_hr_layer1.em_range)
component_hr_layer1.raman_scattering_removal(width=10, interpolation_method='nan', inplace=True)
component_hr_layer1.rayleigh_scattering_removal(width_o1=20, width_o2=40, interpolation_method_o2="nan", inplace=True)
plot_eem_stack(component_hr_layer1.eem_stack, model_hr_layer1.ex_range, model_hr_layer1.em_range, n_cols=5,
        cbar=None,
        plot_x_axis_label=False,
        plot_y_axis_label=False,)

target_matrix = component_hr_layer1.eem_stack[r_to_deconvolute]
target_matrix = eem_gaussian_filter(target_matrix, sigma=2)

results = analyze_matrix_maxima(target_matrix, model_hr_layer1.em_range,
                      np.flip(model_hr_layer1.ex_range), threshold_ratio=0.3)
print(results)
# model_hr_layer2 = PARAFAC(
#     n_components=6,
#     max_iter_nnls=300,
#     max_iter_als=300,
#     solver='hals',
# )
# Rparafac_hr = RobustPARAFAC(model_hr_layer1, model_hr_layer2, n_splits_layer1=6, random_state=42)
# Rparafac_hr.fit(dataset_train_unquenched)
# plot_all_components(Rparafac_hr.base_model_layer2)

# model_hr_layer2_test = EEMNMF(
#     n_components=8,
#     solver="hals",
#     max_iter_nnls=300,
#     max_iter_als=300,
# )
# model_hr_layer2_test.fit(Rparafac_hr.component_set_layer1)
# plot_all_components(model_hr_layer2_test)

# fmax = model_hr_layer1.fmax
# for i in range(model_hr_layer1.n_components):
#     if new_order[i]==r_to_deconvolute:
#         r1_approx, _ = rank_one_approximation_nmf(model_hr_layer1.components[i], 2, random_state=42)
#         model_hr_layer1.components[i] = r1_approx
_, fmax, _ = model_hr_layer1.predict(
    eem_dataset=dataset_train_rs_removed_unquenched,
    fit_beta=True if model_hr_layer1.lam is not None and model_hr_layer1.idx_bot is not None and model_hr_layer1.idx_top is not None else False,
    idx_top=[i for i in range(len(dataset_train_rs_removed_unquenched.index)) if 'B1C1' in dataset_train_rs_removed_unquenched.index[i]],
    idx_bot=[i for i in range(len(dataset_train_rs_removed_unquenched.index)) if 'B1C2' in dataset_train_rs_removed_unquenched.index[i]],
)
# gamma_display = list(param["gamma_W"].values())[0] if isinstance(model_hr_layer1, PARAFAC) else list(param["gamma_W"].values())[0]
plot_fmax_vs_truth(
    fmax_train=fmax,
    truth_train=dataset_train_unquenched.ref['TCC (million #/mL)'],
    # info_dict={'gamma': f'{gamma_display:.1E}'},
    n=1, fit_intercept=True
)

# plot_all_f0f(model_hr_layer1, dataset_train_rs_removed, "B1C1", "B1C2", "TCC (million #/mL)")


# ---------------------


base_model2 = EEMNMF(
    n_components=n_components_original+n_deconvoluted-1,
    # fit_rank_one={k: True for k in range(n_components)},
    max_iter_nnls=500,
    max_iter_als=500,
    init='custom',
    custom_init=[fmax_init, components_init_rs_removed],
    fixed_components=[i for i in range(n_components_original+n_deconvoluted-1) if new_order[i]!=r_to_deconvolute],
    random_state=42,
    solver='hals',
    sort_components_by_em=False,
    tol=1e-8,
    # kw_top='B1C1',
    # kw_bot='B1C2',
    # prior_dict_ex={i: prior_ex[k] for i, k in enumerate(new_order)},
    # prior_dict_em={i: prior_em[k] for i, k in enumerate(new_order)},
    # prior_dict_W={new_order.index(r_to_deconvolute): dataset_train_rs_removed.ref['TCC (million #/mL)'].to_numpy()},
    # gamma_sample=1e6,
    # prior_dict_H={i: prior_components_rs_removed[k] for i, k in enumerate(new_order)},
    # **param
)

component_set_layer2 = get_component_set(base_model2, dataset_train_rs_removed_unquenched)


component_set_model = EEMNMF(
    n_components=n_components_original+n_deconvoluted-2,
    solver="hals",
    max_iter_nnls=300,
    max_iter_als=300,
    init='svd',
    # prior_dict_H={i: prior_components_rs_removed[k] for i, k in enumerate(new_order)},
    # gamma_H=1e9,
    # prior_ref_components=prior_components,
)
component_set_model.fit(component_set_layer2)
plot_all_components(component_set_model)

_, fmax_pred, _ = component_set_model.predict(dataset_train_unquenched)
plot_fmax_vs_truth(
    fmax_train=fmax_pred,
    truth_train=dataset_train_unquenched.ref['TCC (million #/mL)'],
    info_dict={}, n=1, fit_intercept=True
)

#-----------------------

dataset_train_grid = dataset_train_rs_removed_unquenched

param_grid = {
    # 'gamma_H': [
    #     {k: g for k in range(n_components_original+n_deconvoluted-1) if new_order[k] == r_to_deconvolute} for g in [
    #         1e5,
    #         # 1e4,
    #         # 2.5e4,
    #         # 5e4,
    #         # 7.5e4,
    #         # 2.5e5,
    #         # 5e5,
    #         # 7.5e5,
    #         # 1e6
    #     ]
    # ],
    # 'gamma_W': [
    #     {new_order.index(r_to_deconvolute): g} for g in [
    #         # 0,
    #         # 1,
    #         # 10,
    #         # 25,
    #         # 50,
    #         # 100,
    #         # 250,
    #         # 500,
    #         # 1000,
    #         # 2500,
    #         # 5000,
    #         # 10000,
    #         1e8
    #     ]
    # ],
    'n_components': [
        # n_components_original + n_deconvoluted - 1,
        # 3,
        # 4,
        # 5,
        6,
        # 7,
    ],
}

children_grid = {
    "r_to_deconvolute": [
        0
    ],
    "n_deconvoluted": [
        1,2,3,4
    ]
}

param_combinations = get_param_combinations(children_grid)
# param_combinations = get_param_combinations(children_grid)
components_lists_all = []
n_components_original=5

for k, p in enumerate(param_combinations):

    n_deconvoluted = p["n_deconvoluted"]
    r_to_deconvolute = p["r_to_deconvolute"]
    new_order = [0 for _ in range(n_components_original + n_deconvoluted - 1)]
    for i in range(n_components_original):
        if i < r_to_deconvolute:
            new_order[i] = i
        elif i == r_to_deconvolute:
            for j in range(n_deconvoluted):
                new_order[i + j] = i
        elif i > r_to_deconvolute:
            new_order[i + n_deconvoluted - 1] = i
    fmax_init = random_split_columns(model_standard.fmax.to_numpy(), [r_to_deconvolute], n_deconvoluted)
    ex_init = np.flipud(model_standard.ex_loadings.to_numpy()[:, new_order])
    em_init = model_standard.em_loadings.to_numpy()[:, new_order]
    components_init = model_standard.components.reshape((model_standard.components.shape[0], -1))[new_order, :]
    components_init_rs_removed = component_set.eem_stack.reshape((model_standard.components.shape[0], -1))[new_order, :]

    # p['gamma_em'] = p['gamma_ex']
    print(f"param_combinations: {k + 1};")
    model = EEMNMF(
        n_components=n_components_original+n_deconvoluted-1,
        max_iter_nnls=500,
        max_iter_als=500,
        random_state=42,
        solver='hals',
        sort_components_by_em=True,
        tol=1e-8,
        init='custom',
        custom_init=[fmax_init, components_init_rs_removed],
        fixed_components=[i for i in range(n_components_original + n_deconvoluted - 1) if new_order[i] != r_to_deconvolute],
        # prior_dict_W = {new_order.index(r_to_deconvolute): dataset_train_grid.ref['TCC (million #/mL)'].to_numpy()},
        prior_dict_H={i: prior_components_rs_removed[k] for i, k in enumerate(new_order)},
        gamma_H={k: 1e6 for k in range(n_components_original+n_deconvoluted-1) if new_order[k] == r_to_deconvolute},
        # lam=3e3,
        # idx_top="B1C1",
        # idx_bot="B1C2",
        # **p
    )
    # model = PARAFAC(
    #     init='svd',
    #     solver='hals',
    #     max_iter_nnls=300,
    #     max_iter_als=300,
    #     sort_components_by_em=True,
    #     random_state=42,
    #     **p
    # )
    model.fit(dataset_train_grid)
    plot_all_components(model)
    components_list = [[] for i in range(model.n_components)]
    fmax_ratio_list = [[] for i in range(model.n_components)]
    components_lists_all.append(model.components)
    fmax_train = model.fmax
    explained_variance = model.variance_explained()
    # gamma_display = list(p["gamma_ex"].values())[0] if isinstance(model, PARAFAC) else list(p["gamma_H"].values())[0]
    # param_combinations[k]["gamma_display"] = gamma_display
    param_combinations[k]['explained_variance'] = explained_variance
    sv = SplitValidation(base_model=copy.deepcopy(model), n_splits=4, random_state=42)
    sv.fit(dataset_train_grid)
    sim = sv.compare_components()
    sim_mean = sim.mean()
    for c in range(model.n_components):
        param_combinations[k][sim_mean.index[c] + ' similarities'] = sim_mean.iloc[c]
    param_combinations[k]['overall similarities'] = np.mean(sim_mean)
    children_r = [k for k in range(model.n_components) if new_order[k] == r_to_deconvolute]
    param_combinations[k]["split-half stability"] = np.mean(sim_mean.iloc[children_r])
    param_combinations[k]["children similarities"] = mean_pairwise_correlation([model.components[i].reshape(-1) for i in children_r])
    # cvdf = sv.correlation_cv("TCC (million #/mL)")
    # cvdf_mean = cvdf.mean()
    # param_combinations[k].update(cvdf_mean.to_dict())

    # for ref_col in ['TCC (million #/mL)']:
    #     mask_ref = ~np.isnan(dataset_train_grid.ref[ref_col].to_numpy())
    #     y_train = dataset_train_grid.ref[ref_col].to_numpy()[mask_ref]
    #     for r in range(model.n_components):
    #         x_train_r = fmax_train.iloc[mask_ref, [r]].to_numpy()
    #         lr = LinearRegression(fit_intercept=True)
    #         lr.fit(x_train_r, y_train)
    #         coef_r = lr.coef_[0]
    #         intercept_r = lr.intercept_
    #         y_pred = lr.predict(x_train_r)
    #         r2_r = lr.score(x_train_r, y_train)
    #         rmse_r = np.sqrt(mean_squared_error(y_train, y_pred))
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-coef_fit_intercept=True'] = coef_r
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-intercept_fit_intercept=True'] = intercept_r
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-r2_with_fit_intercept=True'] = r2_r
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-rmse_with_fit_intercept=True'] = rmse_r
    #         lr = LinearRegression(fit_intercept=False)
    #         lr.fit(x_train_r, y_train)
    #         coef_r = lr.coef_[0]
    #         intercept_r = lr.intercept_
    #         y_pred = lr.predict(x_train_r)
    #         r2_r = lr.score(x_train_r, y_train)
    #         rmse_r = np.sqrt(mean_squared_error(y_train, y_pred))
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-coef_fit_intercept=False'] = coef_r
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-intercept_fit_intercept=False'] = intercept_r
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-r2_fit_intercept=False'] = r2_r
    #         param_combinations[k][ref_col + '-' + f'C{r + 1} Fmax' + '-rmse_fit_intercept=False'] = rmse_r

    # plot_fmax_vs_truth(
    #     fmax_train=fmax_train,
    #     truth_train=dataset_train_grid.ref['TCC (million #/mL)'],
    #     info_dict={'gamma': f'{gamma_display:.1E}'}, n=0, fit_intercept=True
    # )
    # plot_fmax_vs_truth(
    #     fmax_train=fmax_train,
    #     truth_train=dataset_train.ref['TCC (million #/mL)'],
    #     info_dict={'gamma': f'{gamma_display:.1E}'}, n=2, fit_intercept=True
    # )
param_combinations_df = pd.DataFrame(param_combinations)


# with open("tests/Waterhub_Oct2024_param_combination_df.pkl",
#           'wb') as file:
#     pickle.dump(param_combinations_df, file)

#----------------For Eawag_GAC--------------
colors = [
    '#332288', '#88CCEE', '#44AA99', '#117733',
    '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499'
]
fmax_final = model_hr_layer1.fmax
fmax_final = model_hr_layer1.fmax
fmax_final = fmax_final[~fmax_final.index.str.contains("-filtered")]
fmax_final = fmax_final[~fmax_final.index.str.contains("2025-02")]
fmax_final = fmax_final[~fmax_final.index.str.contains("2025-03-04")]

original_r = r_to_deconvolute
children_r = [k for k in range(n_components_original+n_deconvoluted-1) if new_order[k] == r_to_deconvolute]
fmax_children_sum = fmax_final[fmax_final.columns[children_r]].sum(axis=1)
fmax_ratio_children = np.zeros((len(children_r), 5, int(fmax_children_sum.shape[0]/5)))
for r in children_r:
    fmax_child = fmax_final.iloc[:, r]
    fmax_ratio_child = fmax_child / fmax_children_sum
    for i, kw in enumerate(["influent", "SP1", "SP2", "SP3", "Eff"]):
        fmax_ratio_child_loc = fmax_ratio_child[fmax_ratio_child.index.str.contains(kw)]
        fmax_ratio_children[r, i, :] = fmax_ratio_child_loc.to_numpy()

means = fmax_ratio_children.mean(axis=2)             # shape: (n_series, n_samples)
errors = fmax_ratio_children.std(axis=2, ddof=1)     # standard deviation
x = np.arange(5)

# Plot
fig, axes = plt.subplots(1, len(children_r), figsize=(12, 4), sharey=False)
series_names = ['C1.1 Fmax', 'C1.2 Fmax', 'C1 Fmax']

for i in range(len(children_r)):
    ax = axes[i]
    box = ax.boxplot(fmax_ratio_children[i].T, patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor(colors[i])
    # ax.set_title(sensor_names[i])
    ax.set_xlabel('Location', fontsize=18)
    ax.set_ylabel(series_names[i] + r" / $\sum_{i=1}^{2} \text{C1.i} \, \text{Fmax}$", fontsize=16)
    ax.set_xticks(range(1, 5 + 1))
    ax.set_ylim((0,1))
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticklabels(["inf.", "top", "mid", "bot", "eff."], fontsize=16)
    ax.grid(True, which='major', axis='y')

# plt.suptitle('Box Plots per Sensor (One Subplot Each)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


fmax_final_normal_parafac = model_normal_parafac.fmax.iloc[:, original_r]
fmax_final_normal_parafac = fmax_final_normal_parafac[~fmax_final_normal_parafac.index.str.contains("-filtered")]
fmax_final_normal_parafac = fmax_final_normal_parafac[~fmax_final_normal_parafac.index.str.contains("2025-02")]
fmax_final_normal_parafac = fmax_final_normal_parafac[~fmax_final_normal_parafac.index.str.contains("2025-03-04")]

fmax_final_normal_parafac_influent = fmax_final_normal_parafac[fmax_final_normal_parafac.index.str.contains("influent")]
fmax_normalized_children = np.zeros((len(children_r), 5, int(fmax_children_sum.shape[0]/5)))
fmax_normalized_mother = np.zeros((5, int(fmax_children_sum.shape[0]/5)))
for r in children_r:
    fmax_child = fmax_final.iloc[:, r]
    for i, kw in enumerate(["influent", "SP1", "SP2", "SP3", "Eff"]):
        fmax_loc = fmax_child[fmax_child.index.str.contains(kw)]
        fmax_loc_normalized = fmax_loc.to_numpy() / fmax_final_normal_parafac_influent.to_numpy()
        fmax_normalized_children[r, i, :] = fmax_loc_normalized
        fmax_loc_mother = fmax_final_normal_parafac[fmax_final_normal_parafac.index.str.contains(kw)]
        fmax_normalized_mother[i, :] = fmax_loc_mother.to_numpy() / fmax_final_normal_parafac_influent.to_numpy()


series_means = fmax_normalized_children.mean(axis=2)   # shape: (n_series, n_samples)
ref_means = fmax_normalized_mother.mean(axis=1)         # shape: (n_samples,)

# Plotting
x = np.arange(5)  # x positions per sample
width = 0.35              # bar width

fig, ax = plt.subplots(figsize=(12, 4))

# Plot stacked bars (offset left)
bottom = np.zeros(5)
for i in range(len(children_r)):
    ax.bar(x + width/2, series_means[i], width, bottom=bottom, color=colors[i], label=series_names[i])
    bottom += series_means[i]

# Plot reference bars (offset right)
ax.bar(x - width/2, ref_means, width, color='gray', label=series_names[-1])

# Customize
ax.set_xticks(x)
ax.set_xticklabels(["inf.", "top", "mid", "bot", "eff."], fontsize=16)
ax.set_xlabel('Location', fontsize=18)
ax.set_ylabel('Average Fmax normalized by \nC1 Fmax of influent', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=13)
ax.grid(True, axis='y')
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

#
# #-----------------For Waterhub quenching--------------
#
# eem_dataset_path_test = \
# "tests/Waterhub_2024_sample_276_ex_274_em_310_mfem_5_gaussian_rsu_rs_interpolated.json"
# eem_dataset_test = read_eem_dataset_from_json(eem_dataset_path_test)
# eem_dataset_test, _ = eem_dataset_test.filter_by_index(["2024-10-"], None, copy=True)
# eem_dataset_test.raman_scattering_removal(width=15, interpolation_method='zero', copy=False)
# eem_dataset_test.rayleigh_scattering_removal(width_o1=20, width_o2=40, interpolation_method_o2='zero', copy=False)
#
#
# model_trained = model_hr_layer1
# dataset_trained = eem_dataset
# components_cut = model_trained.components
# components_cut, _, = process_eem_stack(
#     components_cut,
#     eem_cutting,
#     ex_range_old = dataset_trained.ex_range,
#     em_range_old = dataset_trained.em_range,
#     ex_min_new = min(eem_dataset_test.ex_range),
#     ex_max_new = max(eem_dataset_test.ex_range),
#     em_min_new = min(eem_dataset_test.em_range),
#     em_max_new = max(eem_dataset_test.em_range),
# )
# model_cut = copy.deepcopy(model_trained)
# model_cut.components = components_cut
# _, fmax_test, _ = model_cut.predict(eem_dataset_test)
# model_cut.fmax = fmax_test
# plot_all_f0f(model_cut, eem_dataset_test, kw_top="B1C1", kw_bot="B1C2", target_analyte="TCC (million #/mL)")
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
def plot_matrix_with_maxima_fit(matrix, every_n=1, row_indices=None, x_axis=None, label_offset=5, series_labels=None, mask=None):
    """
    Plots selected rows of the matrix as curves, marks and labels the maximum in each selected row
    using arrows and points with alternating directions, and fits a regression or connecting line
    through the maxima.

    Parameters:
    - matrix (ndarray): 2D array (rows x columns).
    - every_n (int): Plot every nth row (ignored if row_indices is provided).
    - row_indices (list): Specific row indices to plot.
    - x_axis (array-like): Custom x-axis values (must match number of columns).
    - label_offset (float): Horizontal distance between peak and label text (in x-axis units).
    - series_labels (list): Optional labels (e.g., excitation wavelengths) for legend.

    Returns:
    - dict with maxima info and regression result (if available).
    """
    matrix = np.asarray(matrix)
    num_rows, num_cols = matrix.shape

    # Validate x_axis
    if x_axis is None:
        x_axis = np.arange(num_cols)
    else:
        x_axis = np.asarray(x_axis)
        if len(x_axis) != num_cols:
            raise ValueError("Length of x_axis must match number of columns in matrix.")

    # Select rows to plot
    if row_indices is not None:
        row_indices = np.asarray(row_indices)
    else:
        row_indices = np.arange(0, num_rows, every_n)

    max_positions = []
    max_values = []
    max_x_coords = []

    # Collect maxima data
    peaks_info = []
    for i in row_indices:
        row = matrix[i]
        max_idx = np.argmax(row)
        max_val = row[max_idx]
        max_x = x_axis[max_idx]
        peaks_info.append((i, max_idx, max_val, max_x))
        max_positions.append(max_idx)
        max_values.append(max_val)
        max_x_coords.append(max_x)

    # Sort by y-value to alternate label directions for vertically close peaks
    peaks_info_sorted = sorted(peaks_info, key=lambda x: x[2])  # sort by max_val (y)
    direction_map = {}
    last_side = 0  # Start with right
    for row_i, *_ in peaks_info_sorted:
        direction_map[row_i] = -label_offset if last_side == 1 else label_offset
        last_side = 1 - last_side

    # Start plotting
    plt.figure(figsize=(7.6, 6))

    for i, max_idx, max_val, max_x in peaks_info:
        row = matrix[i]
        label = f'{i}th Row' if series_labels is None else f'ex={series_labels[i]} nm'
        if mask is not None:
            row_mask = mask[i]
            is_valid = row_mask.astype(bool)
            color = plt.gca()._get_lines.get_next_color()  # Get consistent color for this row
            start=0
            while start < len(row):
                current = is_valid[start]
                end = start + 1
                while end < len(row) and is_valid[end] == current:
                    end += 1
                style = '-' if current else '--'
                if start == 0:
                    plt.plot([], [], "-", color=color, label=label)
                plt.plot(x_axis[start:end], row[start:end], style, color=color)
                start = end
        else:
            plt.plot(x_axis, row, label=label)

        # Plot peak marker
        plt.plot(max_x, max_val, 'ko')

        # Arrow label
        offset = direction_map[i]
        label_x = max_x + offset
        ha = 'left' if offset > 0 else 'right'
        plt.annotate(f'{max_x:.2f}',
                     xy=(max_x, max_val),
                     xytext=(label_x, max_val),
                     textcoords='data',
                     arrowprops=dict(arrowstyle='->', color='black'),
                     ha=ha, va='center', fontsize=14)

    max_positions = np.array(max_positions)
    max_values = np.array(max_values)
    max_x_coords = np.array(max_x_coords)

    # Optional: regression line (commented out in your version)
    if not np.all(max_positions == max_positions[0]):
        slope, intercept, r_value, p_value, std_err = linregress(max_x_coords, max_values)
        x_fit = np.array([min(max_x_coords), max(max_x_coords)])
        y_fit = slope * x_fit + intercept
        # Uncomment if needed:
        # plt.plot(x_fit, y_fit, 'k--', label='Linear fit to maxima')
        regression_info = {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        }
    else:
        regression_info = None

    plt.xlabel('Emission wavelength (nm)', fontsize=20)
    plt.xlim((300, 400))
    plt.ylabel('Intensity (A.U.)', fontsize=20)
    plt.ylim((0, 1))
    plt.legend(fontsize=14, loc='upper right')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "row_indices": row_indices,
        "max_indices": max_positions,
        "max_values": max_values,
        "max_x_coords": max_x_coords,
        "regression": regression_info
    }




target_ex = [
    250,
    260,
    270,
    280,
    290,
    294,
    298
    # 310,
    # 320,
    # 350,360,370,380,385,390,394,400
    # 420, 430, 440, 450
]
target_ex_idx = [
    -dichotomy_search(model_hr_layer1.ex_range, ex)-1 for ex in target_ex
]
target_matrix = model_hr_layer1.components[1]
# target_matrix = model_normal_parafac.components[4]
# target_matrix = base_model_layer2_test.components[4]
target_matrix, mask_raman = eem_raman_scattering_removal(target_matrix, model_hr_layer1.ex_range, model_hr_layer1.em_range, width=15)
target_matrix, mask_rayleigh_o1, mask_rayleigh_o2 = eem_rayleigh_scattering_removal(target_matrix,model_hr_layer1.ex_range, model_hr_layer1.em_range,
                                                   width_o1=20, width_o2=40,interpolation_method_o2="linear")
mask_total = mask_raman.astype(bool) & mask_rayleigh_o1.astype(bool) & mask_rayleigh_o2.astype(bool)
target_matrix = eem_gaussian_filter(target_matrix, sigma=1)
target_matrix = eem_median_filter(target_matrix, (1,5))
plot_matrix_with_maxima_fit(target_matrix, row_indices=target_ex_idx,
                            x_axis=model_hr_layer1.em_range, label_offset=15,
                            series_labels=np.flip(model_hr_layer1.ex_range),
                            mask=mask_total,
                            )


#-------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks

def analyze_matrix_maxima(matrix, x_axis, y_axis, threshold_ratio=0.3):
    """
    Analyzes the maximum values in a matrix and produces two plots:
    1. Regression of row-wise maxima x-positions vs. y-axis
    2. Max intensity per row vs. y-axis with local peak annotations

    Parameters:
    - matrix: 2D ndarray (rows x columns)
    - x_axis: 1D array of column positions (length = num columns)
    - y_axis: 1D array of row positions (length = num rows)
    - threshold_ratio: threshold for filtering rows based on max intensity

    Returns:
    - Dictionary with maxima and regression info
    """
    matrix = np.asarray(matrix)
    x_axis = np.asarray(x_axis)
    y_axis = np.asarray(y_axis)

    if matrix.shape != (len(y_axis), len(x_axis)):
        raise ValueError("Shape mismatch: matrix must be (len(y_axis), len(x_axis))")

    # 1. Global maximum
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    overall_y_idx, overall_x_idx = max_idx
    overall_max_coords = (x_axis[overall_x_idx], y_axis[overall_y_idx])
    overall_max_value = matrix[overall_y_idx, overall_x_idx]

    # 2. Row-wise maxima
    rowwise_max_x = []
    rowwise_max_vals = []
    for row in matrix:
        max_col_idx = np.argmax(row)
        rowwise_max_x.append(x_axis[max_col_idx])
        rowwise_max_vals.append(np.max(row))
    rowwise_max_x = np.array(rowwise_max_x)
    rowwise_max_vals = np.array(rowwise_max_vals)

    # 3. Apply threshold filter
    mask = rowwise_max_vals >= threshold_ratio * overall_max_value
    filtered_x = rowwise_max_x[mask]
    filtered_y = y_axis[mask]
    filtered_indices = np.where(mask)[0]

    # 4. Regression Plot
    if len(filtered_x) >= 2:
        regression = linregress(filtered_y, filtered_x)
        regression_slope = regression.slope

        plt.figure(figsize=(6, 5))
        plt.scatter(filtered_y, filtered_x, color='blue', label='Filtered maxima')

        # Regression line
        x_fit = np.array([min(filtered_y), max(filtered_y)])
        y_fit = regression.intercept + regression.slope * x_fit
        plt.plot(x_fit, y_fit, 'k--', label=f'Regression (slope = {regression.slope:.3f})')

        # Annotate overall maximum
        overall_x_at_y = rowwise_max_x[overall_y_idx]
        overall_y = y_axis[overall_y_idx]
        plt.annotate(f'{overall_x_at_y:.2f}',
                     xy=(overall_y, overall_x_at_y),
                     xytext=(overall_y + 5, overall_x_at_y),
                     arrowprops=dict(arrowstyle='->'), fontsize=12, color='red')

        if len(filtered_y) > 0:
            max_y_idx = np.argmax(filtered_y)  # Find index with highest Y-axis value
            last_y = filtered_y[max_y_idx]
            last_x = filtered_x[max_y_idx]
            plt.annotate(f'{last_x:.2f}',
                         xy=(last_y, last_x),
                         xytext=(last_y + 5, last_x),
                         arrowprops=dict(arrowstyle='->'), fontsize=12, color='green')

        plt.xlabel('Y-axis coordinate')
        plt.ylabel('X position of row maxima')
        plt.title('Regression of Filtered Row-wise Maxima')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        regression = None
        regression_slope = None

    # 5. Row-wise max values vs y-axis (intensity)
    plt.figure(figsize=(6, 5))
    plt.plot(y_axis, rowwise_max_vals, 'o-', color='green', label='Row-wise maxima')

    # Local peaks in 1D signal
    peaks, _ = find_peaks(rowwise_max_vals)
    for i in peaks:
        y = y_axis[i]
        val = rowwise_max_vals[i]
        plt.annotate(f'{y:.2f}', xy=(y, val), xytext=(y, val + 0.03),
                     textcoords='data', ha='center', fontsize=11,
                     arrowprops=dict(arrowstyle='->'))

    plt.xlabel('Y-axis coordinate')
    plt.ylabel('Row-wise max intensity')
    plt.title('Row-wise Max Intensity vs. Y-axis')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "overall_max_coords": overall_max_coords,
        # "rowwise_max_x": rowwise_max_x,
        # "rowwise_max_vals": rowwise_max_vals,
        # "filtered_indices": filtered_indices,
        # "regression_slope": regression_slope,
        # "regression_result": regression
    }




result = analyze_matrix_maxima(component_hr_layer1.eem_stack[0], model_hr_layer1.em_range, np.flip(model_hr_layer1.ex_range))

print(result)
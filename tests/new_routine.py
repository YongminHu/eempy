import copy
import pickle

import numpy as np
import pandas as pd
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
from scipy.optimize import nnls
from statsmodels import robust

np.random.seed(42)

# ---------------Read EEM dataset-----------------

# eem_dataset_path = \
#     "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset_path = \
    "C:/PhD/publication/2025_prior_knowledge/Waterhub_Oct2024_unquenched_unfiltered.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset.median_filter(footprint=(5, 5), copy=False)
eem_dataset.cutting(ex_min=260, ex_max=500, em_min=312, em_max=600, copy=False)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['2024-10-'], copy=True)
# eem_dataset.raman_scattering_removal(width=15, interpolation_method='nan', copy=False)
# eem_dataset.eem_stack = np.nan_to_num(eem_dataset.eem_stack, copy=True, nan=0)
# eem_dataset_july, _ = eem_dataset.filter_by_index(None, ['2024-07-'], copy=True)

# eem_dataset_original, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
# eem_dataset_quenched, _ = eem_dataset.filter_by_index(['B1C2'], None, copy=True)
# idx_top_oct = [i for i in range(len(eem_dataset_october.index)) if 'B1C1' in eem_dataset_october.index[i]]
# idx_bot_oct = [i for i in range(len(eem_dataset_october.index)) if 'B1C2' in eem_dataset_october.index[i]]
# idx_top_jul = [i for i in range(len(eem_dataset_july.index)) if 'B1C1' in eem_dataset_july.index[i]]
# idx_bot_jul = [i for i in range(len(eem_dataset_july.index)) if 'B1C2' in eem_dataset_july.index[i]]

# eem_dataset_bac_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
# eem_dataset_bac = read_eem_dataset_from_json(eem_dataset_bac_path)
# bacteria_eem = eem_dataset_bac.eem_stack[-5]
# bacteria_eem = eem_interpolation(bacteria_eem, eem_dataset_bac.ex_range, eem_dataset_bac.em_range,
#                                  eem_dataset.ex_range, eem_dataset.em_range, method='linear')
# # bacteria_eem, _ = eem_raman_scattering_removal(bacteria_eem, eem_dataset.ex_range, eem_dataset.em_range,
# #                                                width=10, interpolation_method='nan')
# bacteria_eem = np.nan_to_num(bacteria_eem, nan=0)
# prior_dict_ref = {0: bacteria_eem.reshape(-1)}
#
# with open('C:/PhD/publication/2025_prior_knowledge/approx_components.pkl', 'rb') as file:
#     approx_components = pickle.load(file)


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


def plot_fmax_vs_truth(fmax_train, truth_train, n, info_dict, fmax_test=None, truth_test=None, fit_intercept=True):
    info_dict_copy = copy.deepcopy(info_dict)
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
        y_max = max(truth_train.max(), truth_test.max()) + 0.5
    else:
        x_max = fmax_train.iloc[:, n].max() + 100
        y_max = truth_train.max() + 0.5

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


# # -------------------Step 1: Detection of Outlier Samples with High Reconstruction Error-------------------
dataset_train = eem_dataset
n_components = 4
param = {
    # "prior_dict_H": {k: model_nmf_components_r1_dict[k] for k in [0, 1, 2]},
    "gamma_H": 0,
    "lam": 0,
    # "fit_rank_one": {0: True, 1: True, 2: True, 3: True}
}

model = EEMNMF(
        n_components=n_components,
        max_iter_nnls=300,
        max_iter_als=300,
        init='custom',

        random_state=42,
        solver='hals',
        normalization=None,
        sort_components_by_em=False,
        # prior_ref_components=model_nmf_components_r1_dict,
        # kw_top='B1C1',
        # kw_bot='B1C2',
        tol=1e-5,
        **param
    )

eem_dataset_clean, outlier_indices, outlier_indices_history = model.robust_fit(
    dataset_train, zscore_threshold=3.5, max_iter_outlier_removal=1, n_splits=4)

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
    n=3, info_dict=param, fit_intercept=True
    )
# fmax_ratio, _, _ = plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)', plot=True)

eem_dataset_clean.ref = eem_dataset_clean.ref[['TCC (million #/mL)', 'DOC (mg/L)']]

# with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned_2.pkl",
#           'wb') as file:
#     pickle.dump(eem_dataset_clean, file)

with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned.pkl",
          'rb') as file:
    eem_dataset_clean = pickle.load(file)


# -------------------Step 2: Generate rank-one approximations as priors-------------------
dataset_train = eem_dataset_clean
n_components = 5

model_all = PARAFAC(
    n_components=n_components,
    # fit_rank_one=False,
    max_iter_nnls=300,
    max_iter_als=300,
    init='svd',
    random_state=42,
    solver='hals',
    # normalization=None,
    sort_components_by_em=True,
    tol=1e-6,
    # kw_top='B1C1',
    # kw_bot='B1C2'
)
model_all.fit(eem_dataset=dataset_train)
plot_all_components(model_all)

prior_ex = {k: np.flip(model_all.ex_loadings.to_numpy()[:, k]) for k in range(n_components)}
prior_em = {k: model_all.em_loadings.to_numpy()[:, k] for k in range(n_components)}

with open("C:/PhD/publication/2025_prior_knowledge/prior_ex_Oct2024.pkl",
          'wb') as file:
    pickle.dump(prior_ex, file)
with open("C:/PhD/publication/2025_prior_knowledge/prior_em_Oct2024.pkl",
          'wb') as file:
    pickle.dump(prior_em, file)


components_r1 = np.array([rank_one_approximation_nmf(model_all.components[i])[0] for i in range(n_components)])
plot_eem_stack(components_r1, dataset_train.ex_range, dataset_train.em_range, titles=[f'C{i + 1} R1' for i in range(n_components)])
model_all_ref_components = {k: components_r1[k].reshape(-1) for k in range(n_components)}

model_all.prior_ref_components = model_all_ref_components
sv = SplitValidation(base_model=model_all, n_split=6)
sv.fit(dataset_train)
rank1_approx_priors = []
for r in range(n_components):
    cr = []
    for sub_model in sv.subset_specific_models.values():
        cr.append(sub_model.components[r])
    component_set_r = EEMDataset(
        eem_stack=np.array(cr),
        ex_range=model_all.ex_range, em_range=model_all.em_range
    )
    cr_parafac = PARAFAC(n_components=2, solver='hals',sort_components_by_em=True)
    cr_parafac.fit(component_set_r)
    ve = cr_parafac.variance_explained
    print(ve)
    r_primary = np.argmax(np.mean(cr_parafac.fmax.to_numpy(), axis=0))
    print(np.mean(cr_parafac.fmax.to_numpy(), axis=0))
    rank1_approx_priors.append(cr_parafac.components[r_primary])
plot_eem_stack(np.array(rank1_approx_priors), dataset_train.ex_range, dataset_train.em_range, titles=[f'C{i + 1} R1' for i in range(n_components)])
rank1_approx_priors = {k: rank1_approx_priors[k].reshape(-1) for k in range(n_components)}

with open("C:/PhD/publication/2025_prior_knowledge/rank1_approx_priors_Jan2022.pkl",
          'wb') as file:
    pickle.dump(rank1_approx_priors, file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/rank1_approx_priors_Oct2024.pkl",
#           'rb') as file:
#     rank1_approx_priors = pickle.load(file)

# rank1_approx_priors_reshaped = np.array([c.reshape([eem_dataset_october_clean.eem_stack.shape[1], eem_dataset_october_clean.eem_stack.shape[2]]) for c in rank1_approx_priors.values()])
# plot_eem_stack(rank1_approx_priors_reshaped, eem_dataset_october_clean.ex_range, eem_dataset_october_clean.em_range, [1,2,3,4])

# -------------------Step 2.5: Detection of Components Sensitive to Rank-One Constraints-------------------
#
n_components = 4
dataset_train = eem_dataset_clean

model_parafac = PARAFAC(
    n_components=4,
    # fit_rank_one={r: True for r in range(n_components)},
    solver='hals',
    max_iter_als=300,
    max_iter_nnls=300,
    init='svd',
    random_state=42
)
model_parafac.fit(dataset_train)
model_parafac_components_dict = {k: model_parafac.components[k].reshape(-1) for k in range(n_components)}

correlation_sim_all=[{} for i in range(n_components)]
for r in range(n_components):
    # fit_rank_one = {r: True for r in [i for i in range(n_components) if i != r]}
    # fit_rank_one = {r: True}
    prior_dict_H = {k: model_parafac.components[k].reshape(-1) for k in range(n_components) if k != r}
    model = EEMNMF(
        n_components=n_components,
        fit_rank_one=False,
        max_iter_nnls=300,
        max_iter_als=300,
        init='svd',
        random_state=42,
        solver='hals',
        normalization=None,
        sort_components_by_em=False,
        prior_ref_components=model_parafac_components_dict,
        prior_dict_H=prior_dict_H,
        gamma_H=1e8,
    )
    model.fit(eem_dataset=dataset_train)
    for k in range(n_components):
        # cosine_sim = cosine_similarity(model.components[k].flatten().reshape(1, -1),
        #                                model_standard.components[k].flatten().reshape(1, -1))[0, 0]
        correlation_sim = np.corrcoef(model.components[k].flatten(),
                                      model_parafac.components[k].flatten())[0, 1]
        correlation_sim_all[r][k] = correlation_sim
    plot_all_components(model)
correlation_sim_all_df = pd.DataFrame(correlation_sim_all)



# ------------Step 3: cross-validation & hyperparameter optimization-------

dataset_train = eem_dataset_clean
n_components = 4

param_grid = {
    'n_components': [4],
    'init': ['nndsvda'],
    'gamma_H': [2500],
    'prior_dict_H': [
        # None,
        {k: rank1_approx_priors[k] for k in [1]},
    ],
    'lam': [0],
    'max_iter_als': [300],
    'max_iter_nnls': [300],
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
        # kw_top='B1C1',
        # kw_bot='B1C2',
        normalization=None,
        tol=1e-6,
        **p
    )
    model.fit(dataset_train)
    plot_all_components(model)
    components_lists_all.append(model.components)
    _, fmax_train, eem_re_train = model.predict(
        eem_dataset=dataset_train,
        fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
        idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
        idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
    )
    # explained_variance = model.variance_explained()
    # param_combinations[k]['explained_variance'] = explained_variance
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
    # if model.idx_top is not None and model.idx_bot is not None:
    #     fmax_train_top = fmax_train[fmax_train.index.str.contains('B1C1')]
    #     fmax_train_bot = fmax_train[fmax_train.index.str.contains('B1C2')]
    #     fmax_ratio = fmax_train_top.to_numpy()/fmax_train_bot.to_numpy()
    #     fmax_ratio_means = np.mean(fmax_ratio, axis=0)
    #     fmax_ratio_std = np.std(fmax_ratio, axis=0)
    #     fmax_ratio_max = np.max(fmax_ratio, axis=0)
    #     fmax_ratio_min = np.min(fmax_ratio, axis=0)
    #     for r in range(n_components):
    #         param_combinations[k][f'C{r+1}-fmax_ratio_means'] = fmax_ratio_means[r]
    #         param_combinations[k][f'C{r+1}-fmax_ratio_std'] = fmax_ratio_std[r]
    #         param_combinations[k][f'C{r+1}-fmax_ratio_min'] = fmax_ratio_max[r]
    #         param_combinations[k][f'C{r+1}-fmax_ratio_max'] = fmax_ratio_min[r]
    #     _ = plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)')
    # sv = SplitValidation(base_model=model, n_split=4, random_state=42)
    # sv.fit(dataset_train)
    # sim_components = sv.compare_components()
    # sim_component_sim = sim_components.mean()
    # for c in range(model.n_components):
    #     param_combinations[k][sim_component_sim.index[c]+' similarities'] = sim_component_sim.iloc[c]


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


# --------------------------------------

# with open("C:/PhD/publication/2025_prior_knowledge/prior_ex_Oct2024.pkl",
#           'rb') as file:
#     prior_ex = pickle.load(file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/prior_em_Oct2024.pkl",
#           'rb') as file:
#     prior_em = pickle.load(file)

# with open("C:/PhD/publication/2025_prior_knowledge/eem_dataset_Oct2024_cleaned.pkl",
#           'rb') as file:
#     eem_dataset_clean = pickle.load(file)

dataset_train = eem_dataset
dataset_train_unquenched, _ = dataset_train.filter_by_index('B1C1', None, copy=True)
dataset_train_rs_removed = copy.deepcopy(dataset_train)
dataset_train_rs_removed.raman_scattering_removal(width=15, interpolation_method='zero', copy=False)
dataset_train_rs_removed_unquenched, _ = dataset_train_rs_removed.filter_by_index('B1C1', None, copy=True)


n_components = 5
model_standard = PARAFAC(
    n_components=n_components,
    init='svd',
    solver='hals',
    max_iter_nnls=300,
    max_iter_als=300
)
model_standard.fit(dataset_train_unquenched)
plot_all_components(model_standard)
plot_fmax_vs_truth(
    fmax_train=model_standard.fmax,
    truth_train=dataset_train_unquenched.ref['TCC (million #/mL)'],
    info_dict={}, n=0, fit_intercept=True
)


component_set = EEMDataset(model_standard.components, model_standard.ex_range, model_standard.em_range)
component_set.raman_scattering_removal(width=15, interpolation_method='zero', copy=False)


prior_ex = {k: np.flipud(model_standard.ex_loadings.to_numpy()[:, k]) for k in range(model_standard.n_components)}
prior_em = {k: model_standard.em_loadings.to_numpy()[:, k] for k in range(model_standard.n_components)}
prior_components = {k: model_standard.components[k].reshape(-1) for k in range(model_standard.n_components)}
prior_components_rs_removed = {k: component_set.eem_stack[k].reshape(-1) for k in range(model_standard.n_components)}

fmax_init = random_split_columns(model_standard.fmax.to_numpy(),[0,3],2)
ex_init = np.flipud(model_standard.ex_loadings.to_numpy()[:, [0,0,1,2,3,3,4]])
em_init = model_standard.em_loadings.to_numpy()[:, [0,0,1,2,3,3,4]]
components_init = model_standard.components.reshape((model_standard.components.shape[0],-1))[[0,0,1,2,3,3,4], :]
components_init_rs_removed = component_set.eem_stack.reshape((model_standard.components.shape[0],-1))[[0,0,1,2,3,3,4], :]

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


n_components = 7
param = {
    # "gamma_ex": {0: 1e6, 1: 1e6, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
    # "gamma_em": {0: 1e6, 1: 1e6, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
    "gamma_H": {0: 3e5, 1: 3e5, 2: 1e8, 3: 1e8, 4: 3e5, 5: 3e5, 6: 1e8},
    # "lam": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
}

model = EEMNMF(
    n_components=n_components,
    max_iter_nnls=300,
    max_iter_als=300,
    init='custom',
    custom_init=[fmax_init, components_init_rs_removed],
    random_state=42,
    solver='hals',
    sort_components_by_em=False,
    tol=1e-6,
    # kw_top='B1C1',
    # kw_bot='B1C2',
    # prior_dict_ex={i: prior_ex[k] for i, k in enumerate([0,0,1,2,3,4])},
    # prior_dict_em={i: prior_em[k] for i, k in enumerate([0,0,1,2,3,4])},
    prior_dict_H={i: prior_components_rs_removed[k] for i, k in enumerate([0,0,1,2,3,3,4])},
    **param
)
model.fit(eem_dataset=dataset_train_rs_removed_unquenched)
plot_all_components(model)
# _ = plot_all_f0f(model, dataset_train_rs_removed_unquenched, 'B1C1', 'B1C2', 'TCC (million #/mL)', zscore_threshold=6)
fmax = model.fmax
# _, fmax, _ = model.predict(
#     eem_dataset=dataset_train,
#     fit_beta=True if model.lam is not None and model.idx_bot is not None and model.idx_top is not None else False,
#     idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
#     idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
# )
gamma_display = param["gamma_H"][0] if isinstance(model, EEMNMF) else param["gamma_ex"][0]
plot_fmax_vs_truth(
    fmax_train=fmax,
    truth_train=dataset_train_rs_removed_unquenched.ref['TCC (million #/mL)'],
    info_dict={'gamma': f'{gamma_display:.1E}'}, n=0, fit_intercept=True
)


#-----------------------

dataset_train = dataset_train_unquenched

param_grid = {
    'n_components': [6],
    'gamma_ex': [
        # {0: 0, 1: 0, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
        # {0: 1e2, 1: 1e2, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
        # {0: 1e4, 1: 1e4, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
        # {0: 1e5, 1: 1e5, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
        # {0: 1e6, 1: 1e6, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
        # {0: 1e7, 1: 1e7, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
        {0: 1e8, 1: 1e8, 2: 1e8, 3: 1e8, 4: 1e8, 5: 1e8},
    ],
}

param_combinations = get_param_combinations(param_grid)
components_lists_all = []

for k, p in enumerate(param_combinations):
    p['gamma_em'] = p['gamma_ex']
    print(f"param_combinations: {k + 1};")
    components_list = [[] for i in range(p['n_components'])]
    fmax_ratio_list = [[] for i in range(p['n_components'])]
    model = PARAFAC(
        max_iter_nnls=300,
        max_iter_als=300,
        init='nndsvd',
        random_state=42,
        solver='hals',
        sort_components_by_em=False,
        tol=1e-6,
        prior_dict_ex={i: prior_ex[k] for i, k in enumerate([0,0,1,2,3,4])},
        prior_dict_em={i: prior_em[k] for i, k in enumerate([0,0,1,2,3,4])},
        # prior_dict_H={i: prior_components[k] for i, k in enumerate([0,0,1,2,3,4])},
        **p
    )
    model.fit(dataset_train)
    plot_all_components(model)
    components_lists_all.append(model.components)
    fmax_train = model.fmax
    explained_variance = model.variance_explained()
    param_combinations[k]['explained_variance'] = explained_variance
    for ref_col in ['TCC (million #/mL)']:
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
    sim_component_sim = component_similarity(model_standard.components[[0]], model.components)
    param_combinations[k]['C1 similarity'] = sim_component_sim.iloc[0, 0]
    param_combinations[k]['C2 similarity'] = sim_component_sim.iloc[0, 1]
    gamma_display = p["gamma_ex"][0]
    plot_fmax_vs_truth(
        fmax_train=fmax_train,
        truth_train=dataset_train.ref['TCC (million #/mL)'],
        info_dict={'gamma': f'{gamma_display:.1E}'}, n=0, fit_intercept=True
    )
    plot_fmax_vs_truth(
        fmax_train=fmax_train,
        truth_train=dataset_train.ref['TCC (million #/mL)'],
        info_dict={'gamma': f'{gamma_display:.1E}'}, n=1, fit_intercept=True
    )
param_combinations_df = pd.DataFrame(param_combinations)

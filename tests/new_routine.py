import copy
import pickle

import numpy as np
import pandas as pd
from scipy.stats import zscore
from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax, plot_eem_stack
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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


def plot_fmax_vs_truth(fmax_train, fmax_test, truth_train, truth_test, n, info_dict):
    info_dict_copy = copy.deepcopy(info_dict)
    fig, ax = plt.subplots()
    ax.plot(fmax_train.iloc[:, n], truth_train.to_numpy(), 'o', label='training')
    ax.plot(fmax_test.iloc[:, n], truth_test.to_numpy(), 'o', label='testing')
    lr_n = LinearRegression(fit_intercept=True)
    mask_train_n = ~np.isnan(truth_train.to_numpy())
    lr_n.fit(fmax_train.iloc[mask_train_n, [n]].to_numpy(), truth_train.iloc[mask_train_n])
    y_pred_train = lr_n.predict(fmax_train.iloc[mask_train_n, [n]].to_numpy())
    rmse_train = np.sqrt(mean_squared_error(truth_train.iloc[mask_train_n], y_pred_train))
    r2_train = r2_score(truth_train.iloc[mask_train_n], y_pred_train)
    y_pred_test = lr_n.predict(fmax_test.iloc[:, [n]].to_numpy())
    mask_test_n = ~np.isnan(truth_test.to_numpy())
    rmse_test = np.sqrt(mean_squared_error(truth_test.iloc[mask_test_n].to_numpy(), y_pred_test[mask_test_n]))
    r2_test = r2_score(truth_test.iloc[mask_test_n], y_pred_test[mask_test_n])
    info_dict_copy['r2_train'] = np.round(r2_train, decimals=3)
    info_dict_copy['r2_test'] = np.round(r2_test, decimals=3)
    info_dict_copy['rmse_train'] = np.round(rmse_train, decimals=3)
    info_dict_copy['rmse_test'] = np.round(rmse_test, decimals=3)
    info_dict_copy['fit_intercept'] = True if lr_n.fit_intercept else False
    if lr_n.fit_intercept:
        ax.plot([0, 10000], np.array([0, 10000]) * lr_n.coef_[0] + lr_n.intercept_, '--')
    else:
        ax.plot([0, 10000], np.array([0, 10000]) * lr_n.coef_[0], '--')
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
    ax.set_xlim([0, max(
        max(fmax_train.iloc[:, n].to_numpy()),
        max(fmax_test.iloc[:, n].to_numpy())
    ) + 100
                 ])
    ax.set_ylim([0, max(
        max(truth_train.to_numpy()),
        max(truth_test.to_numpy())
    ) + 0.5
                 ])
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


def rank_one_approximation(A):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # Rank-one approximation: only keep the first singular value/vector
    A1 = S[0] * np.outer(U[:, 0], VT[0, :])
    # Score: ratio of first singular value squared to total energy
    total_energy = np.sum(S**2)
    rank_one_energy = S[0]**2
    score = rank_one_energy / total_energy if total_energy > 0 else 0.0
    return A1, score

# -------------------Step 1: Detection of Components Sensitive to Rank-One Constraints-------------------
dataset_train = eem_dataset_october
n_components = 4
model_parafac = EEMNMF(
    n_components=n_components,
    fit_rank_one={r: True for r in range(n_components)},
    max_iter_nnls=200,
    max_iter_als=800,
    init='ordinary_cp',
    random_state=42,
    solver='hals',
    normalization=None,
    sort_components_by_em=False,
    prior_ref_components=approx_components
)
model_parafac.fit(eem_dataset=dataset_train)
model_parafac_components_dict = {r: model_parafac.components[r].reshape(-1) for r in range(n_components)}
correlation_sim_all = [{} for i in range(n_components)]
plot_all_components(model_parafac)

model_nmf = EEMNMF(
    n_components=n_components,
    fit_rank_one=False,
    max_iter_nnls=200,
    max_iter_als=500,
    init='nndsvd',
    random_state=42,
    solver='hals',
    normalization=None,
    prior_ref_components=model_parafac_components_dict
)
model_nmf.fit(eem_dataset=dataset_train)
plot_all_components(model_nmf)
components_r1 = np.array([rank_one_approximation(model_nmf.components[i])[0] for i in range(n_components)])
plot_eem_stack(components_r1, eem_dataset.ex_range, eem_dataset.em_range, titles=[f'C{i + 1} R1' for i in range(n_components)])
model_nmf_components_r1_dict = {k: components_r1[k].reshape(-1) for k in range(n_components)}

for r in range(n_components):
    # fit_rank_one = {r: True for r in [i for i in range(n_components) if i != r]}
    # fit_rank_one = {r: True}
    prior_dict_H = {k: model_parafac.components[k].reshape(-1) for k in range(n_components) if k != r}
    model = EEMNMF(
        n_components=n_components,
        fit_rank_one=False,
        max_iter_nnls=200,
        max_iter_als=800,
        init='ordinary_nmf',
        random_state=42,
        solver='hals',
        normalization=None,
        sort_components_by_em=False,
        prior_ref_components=model_parafac_components_dict,
        prior_dict_H=prior_dict_H,
        gamma_H=1e5,
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

# -------------------Step 2: Detection of Outlier Samples with High Reconstruction Error-------------------

model = EEMNMF(
        n_components=n_components,
        fit_rank_one=False,
        max_iter_nnls=200,
        max_iter_als=500,
        init='nndsvd',
        random_state=42,
        solver='hals',
        normalization=None,
        sort_components_by_em=False,
        prior_ref_components=model_parafac_components_dict,
        prior_dict_H={k: model_nmf_components_r1_dict[k] for k in [0, 1, 2]},
        gamma_H=1e5,
    )

def fitting_outlier_detection(model, eem_dataset, zscore_threshold=3):
    model_work = copy.deepcopy(model)
    model_work.fit(eem_dataset=eem_dataset)
    _, fmax, eem_re = model.predict(
            eem_dataset,
            fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
            idx_top=[i for i in range(len(eem_dataset.index)) if 'B1C1' in eem_dataset.index[i]],
            idx_bot=[i for i in range(len(eem_dataset.index)) if 'B1C2' in eem_dataset.index[i]],
        )
    res = eem_dataset.eem_stack - eem_re
    n_pixels = res.shape[1] * res.shape[2]
    rmse = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)
    median = np.median(rmse)
    mad = robust.mad(rmse)
    modified_z_scores = 0.6745 * (rmse - median) / mad
    outlier_indices = np.where(modified_z_scores > zscore_threshold)[0]
    outlier_indices = [eem_dataset.index[i] for i in outlier_indices]
    return outlier_indices

outlier_indices_train = [0]
if outlier_indices_train:
    outlier_indices_train = fitting_outlier_detection(model=model, eem_dataset=eem_dataset_october, zscore_threshold=3)

    outlier_unquenched = [i for i, idx in enumerate(dataset_train_unquenched.index) if idx in outlier_indices_train]
    outlier_quenched = [i for i, idx in enumerate(dataset_train_quenched.index) if idx in samples_to_remove]
    number_outliers = list(set(outlier_quenched + outlier_unquenched))
    qualified_indices = [idx for i, idx in enumerate(dataset_train_unquenched.index) if i not in number_outliers] + \
                         [idx for i, idx in enumerate(dataset_train_quenched.index) if i not in number_outliers]
    eem_dataset_october_cleaned, _ = dataset_train.filter_by_index(None, qualified_indices, copy=True)


# ----------cross-validation & hyperparameter optimization-------
dataset_train = eem_dataset_october

param_grid = {
    'n_components': [4],
    'init': ['nndsvd'],
    'gamma_W': [0],
    'gamma_A': [0],
    'gamma_H': [1e5],
    'prior_dict_H': [
        # None,
        {k: model_nmf_components_r1_dict[k] for k in [0, 1, 2]},
    ],
    'lam': [0],
    'max_iter_als': [100],
    'max_iter_nnls': [500],
    'fit_rank_one': [
        False,
        # {0: True,},
        # {1: True,},
        # {2: True,},
        # {3: True,},
        # {0: True, 1: True,},
        # {0: True, 2: True,},
        # {0: True, 3: True,},
        # {1: True, 2: True,},
        # {1: True, 3: True,},
        # {2: True, 3: True,},
        # {0: True, 1: True, 2: True,},
        # {0: True, 1: True, 3: True,},
        # {0: True, 2: True, 3: True,},
        # {1: True, 2: True, 3: True,},
        # {0: True, 1: True, 2: True, 3: True,},
    ]
}

param_combinations = get_param_combinations(param_grid)
dataset_train_splits = []
dataset_train_unquenched, _ = dataset_train.filter_by_index('B1C1', None, copy=True)
initial_sub_eem_datasets_unquenched = dataset_train_unquenched.splitting(n_split=4, random_state=42)
dataset_train_quenched, _ = dataset_train.filter_by_index('B1C2', None, copy=True)
for subset in initial_sub_eem_datasets_unquenched:
    pos = [dataset_train_unquenched.index.index(idx) for idx in subset.index]
    quenched_index = [dataset_train_quenched.index[idx] for idx in pos]
    sub_eem_dataset_quenched, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
    subset.sort_by_index()
    sub_eem_dataset_quenched.sort_by_index()
    dataset_train_splits.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))

splits = all_split_half_combinations(dataset_train_splits)
components_lists_all = []

for k, p in enumerate(param_combinations):
    print(f"param_combinations: {k + 1};")
    r2_train, r2_test, rmse_train, rmse_test = 0, 0, 0, 0
    components_list = [[] for i in range(p['n_components'])]
    fmax_ratio_list = [[] for i in range(p['n_components'])]
    for j, (a, b) in enumerate(splits):
        print(f"split: {j + 1}; ")
        d_train = combine_eem_datasets(a)
        d_test = combine_eem_datasets(b)
        sample_prior = {0: d_train.ref['TCC (million #/mL)']}
        model = EEMNMF(
            solver='hals',
            random_state=42,
            # prior_dict_W=sample_prior,
            # prior_dict_H=approx_components,
            sort_components_by_em=False,
            prior_ref_components=model_nmf_components_r1_dict,
            idx_top=[i for i in range(len(d_train.index)) if 'B1C1' in d_train.index[i]],
            idx_bot=[i for i in range(len(d_train.index)) if 'B1C2' in d_train.index[i]],
            normalization=None,
            **p
        )
        model.fit(d_train)
        if j == 0:
            components_ref = {model.fmax.columns[i]: model.components[i] for i in range(model.n_components)}
        model_dict = align_components_by_components({0: model}, components_ref)
        model = model_dict[0]
        fmax_train = model.fmax
        components = model.components
        # plot_outlier_plots(
        #     model=model, estimator_rank=0, indicator='TCC (million #/mL)',
        #     dataset_test=d_test, dataset_train=d_train
        # )
        _, fmax_test, eem_re_test = model.predict(
            d_test,
            fit_beta=True if model.lam > 0 and model.idx_bot is not None and model.idx_top is not None else False,
            idx_top=[i for i in range(len(d_test.index)) if 'B1C1' in d_test.index[i]],
            idx_bot=[i for i in range(len(d_test.index)) if 'B1C2' in d_test.index[i]],
        )
        lr = LinearRegression(fit_intercept=True)
        mask_train = ~np.isnan(d_train.ref['TCC (million #/mL)'].to_numpy())
        X_train = fmax_train.iloc[mask_train, [0]].to_numpy()
        y_train = d_train.ref['TCC (million #/mL)'].to_numpy()[mask_train]
        lr.fit(X_train, y_train)
        mask_test = ~np.isnan(d_test.ref['TCC (million #/mL)'].to_numpy())
        X_test = fmax_test.iloc[mask_test, [0]].to_numpy()
        y_test = d_test.ref['TCC (million #/mL)'].to_numpy()[mask_test]
        r2_train += lr.score(X_train, y_train) / len(splits)
        r2_test += lr.score(X_test, y_test) / len(splits)
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_test)
        rmse_train += np.sqrt(mean_squared_error(y_train, y_pred_train)) / len(splits)
        rmse_test += np.sqrt(mean_squared_error(y_test, y_pred_test)) / len(splits)
        # plt.close()
        # plot_all_components(model)
        # plot_fmax_vs_truth(fmax_train=fmax_train, fmax_test=fmax_test,
        #                    truth_train=d_train.ref[indicator], truth_test=d_test.ref[indicator],
        #                    n=0, info_dict=p)
        fmax_ratio, _, _ = plot_all_f0f(model, d_train, 'B1C1', 'B1C2', 'TCC (million #/mL)', plot=False)
        for rr in range(len(components_list)):
            components_list[rr].append(model.components[rr].reshape(-1))
            fmax_ratio_list[rr].append(fmax_ratio[:, rr])
    param_combinations[k]['r2_train'] = r2_train
    param_combinations[k]['rmse_train'] = rmse_train
    param_combinations[k]['r2_test'] = r2_test
    param_combinations[k]['rmse_test'] = rmse_test
    for z in range(len(components_list)):
        param_combinations[k][f'C{z + 1} similarities'] = mean_pairwise_correlation(components_list[z])
    for z in range(len(components_list)):
        param_combinations[k][f'C{z + 1} fmax ratios ANOVA F'] = f_oneway(*fmax_ratio_list[z])[0]
    for z in range(len(components_list)):
        param_combinations[k][f'C{z + 1} fmax ratios ANOVA p'] = f_oneway(*fmax_ratio_list[z])[1]
    for z in range(len(components_list)):
        param_combinations[k][f'C{z + 1} fmax ratios H-test H'] = kruskal(*fmax_ratio_list[z])[0]
    for z in range(len(components_list)):
        param_combinations[k][f'C{z + 1} fmax ratios H-test p'] = kruskal(*fmax_ratio_list[z])[1]
    for z in range(len(components_list)):
        param_combinations[k][f'C{z + 1} fmax ratios mean variance'] = np.mean(
            [np.var(group, ddof=1) for group in fmax_ratio_list[z]])
    components_lists_all.append(components_list)


param_combinations_df = pd.DataFrame(param_combinations)

with open("C:/PhD/publication/2025_prior_knowledge/param_combinations.pkl",
          'wb') as file:
    pickle.dump(param_combinations_df, file)

with open("C:/PhD/publication/2025_prior_knowledge/components_lists_all.pkl",
          'wb') as file:
    pickle.dump(components_lists_all, file)


# with open("C:/PhD/publication/2025_prior_knowledge/param_combinations.pkl",
#           'rb') as file:
#     param_combinations_df = pickle.load(file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/components_lists_all.pkl",
#           'rb') as file:
#     components_lists_all = pickle.load(file)

n_is_outlier_train_all = []
n_is_outlier_test_all = []
for k, p in enumerate(param_combinations):
    n_is_outlier_train = pd.Series(np.zeros(len(dataset_train.index)), index=dataset_train.index, dtype=int)
    n_is_outlier_test = pd.Series(np.zeros(len(dataset_train.index)), index=dataset_train.index, dtype=int)
    for j, (a, b) in enumerate(splits):
        d_train = combine_eem_datasets(a)
        d_test = combine_eem_datasets(b)
        components = np.array([components_lists_all[k][r][j].reshape(d_train.eem_stack.shape[1:]) for r in range(p['n_components'])])
        # plot_eem_stack(components, d_train.ex_range, d_train.em_range, titles=[f'C{i + 1}' for i in range(p['n_components'])])
        _, _, eem_re_train = eems_fit_components(d_train.eem_stack,
                                                 components,
                                                 fit_intercept=False)
        _, _, eem_re_test = eems_fit_components(d_test.eem_stack,
                                                components,
                                                fit_intercept=False)

        res_train = d_train.eem_stack - eem_re_train
        n_pixels = res_train.shape[1] * res_train.shape[2]
        error_train = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)
        outlier_indices_train = np.where(zscore(error_train) > 2)[0]
        outlier_indices_train = np.array([d_train.index[i] for i in outlier_indices_train])
        n_is_outlier_train.loc[outlier_indices_train] += 1

        res_test = d_test.eem_stack - eem_re_test
        n_pixels = res_test.shape[1] * res_test.shape[2]
        error_test = np.sqrt(np.sum(res_test ** 2, axis=(1, 2)) / n_pixels)
        outlier_indices_test = np.where(zscore(error_test) > 2)[0]
        outlier_indices_test = np.array([d_test.index[i] for i in outlier_indices_test])
        n_is_outlier_test.loc[outlier_indices_test] += 1

    n_is_outlier_train_all.append(n_is_outlier_train)
    n_is_outlier_test_all.append(n_is_outlier_test)

n_is_outlier_train_df = pd.concat(n_is_outlier_train_all, axis=1)
n_is_outlier_test_df = pd.concat(n_is_outlier_test_all, axis=1)
samples_to_remove = n_is_outlier_train_df.index[n_is_outlier_train_df.sum(axis=1) >= 72].to_list()
outlier_unquenched = [i for i, idx in enumerate(dataset_train_unquenched.index) if idx in samples_to_remove]
outlier_quenched = [i for i, idx in enumerate(dataset_train_quenched.index) if idx in samples_to_remove]
number_outliers = list(set(outlier_quenched + outlier_unquenched))
qualified_indices = [idx for i, idx in enumerate(dataset_train_unquenched.index) if i not in number_outliers] + \
                     [idx for i, idx in enumerate(dataset_train_quenched.index) if i not in number_outliers]
eem_dataset_october_cleaned, _ = dataset_train.filter_by_index(None, qualified_indices, copy=True)


# -------------------Step 3: Optimize hyperparameters with Split-half Cross-Validation-------------------



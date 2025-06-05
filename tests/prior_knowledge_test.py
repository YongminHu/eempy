import copy
import pickle

import numpy as np
import pandas as pd
from scipy.stats import zscore
from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from itertools import product
import seaborn as sns

np.random.seed(42)

# ---------------Read EEM dataset-----------------

# eem_dataset_path = \
#     "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset_path = \
    "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/sample_276_ex_274_em_310_mfem_5_gaussian_rsu.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
# eem_dataset.raman_scattering_removal(width=15, interpolation_method='nan',copy=False)
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
# bacteria_eem = eem_interpolation(bacteria_eem, eem_dataset_bac.ex_range, eem_dataset_bac.em_range,
#                                  eem_dataset.ex_range, eem_dataset.em_range, method='linear')
# bacteria_eem, _ = eem_raman_scattering_removal(bacteria_eem, eem_dataset.ex_range, eem_dataset.em_range,
#                                             width=10, interpolation_method='nan')
# bacteria_eem = np.nan_to_num(bacteria_eem, nan=0)
prior_dict_ref = {0: bacteria_eem.reshape(-1)}

def plot_all_f0f(model, eem_dataset, kw_top, kw_bot, target_analyte):
    target_train = eem_dataset.ref[target_analyte]
    valid_indices_train = target_train.index[~target_train.isna()]
    fmax = model.fmax
    fmax_original = fmax[fmax.index.str.contains(kw_top)]
    mask = fmax_original.index.isin(valid_indices_train)
    fmax_original = fmax_original[mask]
    fmax_quenched = fmax[fmax.index.str.contains(kw_bot)]
    fmax_quenched = fmax_quenched[mask]
    fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()

    fig, axes = plt.subplots(ncols=2, nrows=(params['n_components']+1)//2, figsize=(8, 10))
    for rank_target in range(params['n_components']):
        fmax_ratio_target = fmax_ratio[:, rank_target]
        fmax_ratio_target_valid = fmax_ratio_target[(fmax_ratio_target >= 0) & (fmax_ratio_target <= 1e3)]
        fmax_ratio_z_scores = zscore(fmax_ratio_target_valid, nan_policy='omit')
        ratio_nan = 1 - fmax_ratio_target_valid.shape[0]/fmax_ratio_target.shape[0]
        fmax_ratio_target_filtered = fmax_ratio_target_valid[np.abs(fmax_ratio_z_scores) <= 3]

        if (params['n_components']+1)//2 > 1:
            counts, bins, patches = axes[rank_target // 2, rank_target % 2].hist(fmax_ratio_target_filtered, bins=30,
                                                                                 density=True, alpha=0.5, color='blue',
                                                                                 label='training', zorder=0,
                                                                                 edgecolor='black')
            axes[rank_target//2, rank_target % 2].set_ylabel('Density', fontsize=18)
            axes[rank_target//2, rank_target % 2].set_xlabel(f'C{rank_target+1}'+' $F_{0}/F$', fontsize=18)
            axes[rank_target//2, rank_target % 2].tick_params(axis='both', labelsize=16)
            axes[rank_target//2, rank_target % 2].text(
                0.01, 0.99,
                f'nan_ratio: {ratio_nan:2f}',
                transform=axes[rank_target//2, rank_target % 2].transAxes,
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
            axes[rank_target % 2].set_xlabel(f'C{rank_target+1}'+' $F_{0}/F$', fontsize=18)
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

def plot_all_components(eem_dataset):
    fig, ax = plt.subplots(
        nrows=(params['n_components'] + 1) // 2, ncols=2,
    )
    plt.subplots_adjust(
        left=0,  # distance from left of figure (0 = 0%, 1 = 100%)
        right=1,  # distance from right
        bottom=0,
        top=1,
        wspace=0,  # width between subplots
        hspace=0  # height between subplots
    )
    for i in range(params['n_components']):
        if i < params['n_components']:
            f, a, im = plot_eem(
                eem_dataset.components[i],
                ex_range=eem_dataset.ex_range,
                em_range=eem_dataset.em_range,
                display=False,
                title=f'Component {i + 1}'
            )
            canvas = FigureCanvas(f)
            canvas.draw()
            # Get the RGBA image as a NumPy array
            img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(canvas.get_width_height()[::-1] + (4,))
            if (params['n_components'] + 1) // 2 > 1:
                ax[i // 2, i % 2].imshow(img_array)
                ax[i // 2, i % 2].axis('off')  # Hides ticks, spines, etc.
            else:
                ax[i % 2].imshow(img_array)
                ax[i % 2].axis('off')  # Hides ticks, spines, etc.
    fig.show()

model = PARAFAC(n_components=5, tf_normalization=False)
model.fit(eem_dataset_october)
#
# # -------------prior decomposition function test---------
#
# A, B, C, beta = cp_hals_prior_ratio(
#     tensor=eem_dataset_october.eem_stack,
#     rank=5,
#     prior_dict_A={0: eem_dataset_october.ref['TCC (million #/mL)'].to_numpy()},
#     gamma_A=3e8,
#     prior_ref_components=prior_dict_ref,
#     lam=0,
#     idx_top=[i for i in range(len(eem_dataset_october.index)) if 'B1C1' in eem_dataset_october.index[i]],
#     idx_bot=[i for i in range(len(eem_dataset_october.index)) if 'B1C2' in eem_dataset_october.index[i]],
#     tol=1e-9,
#     init='ordinary_cp',
#     random_state=42
# )
# plt.plot(A[:, 0], eem_dataset_october.ref['TCC (million #/mL)'], 'o')
# plt.show()
# for r in range(A.shape[0]):
#     plot_eem(np.outer(B[:, r], C[:, r]),
#              ex_range=eem_dataset_original.ex_range,
#              em_range=eem_dataset_original.em_range,
#              display=True
#              )
#
# A, B, beta = nmf_hals_prior_ratio(
#     X=eem_dataset_july.eem_stack.reshape([eem_dataset_july.eem_stack.shape[0], -1]),
#     idx_top=idx_top_jul,
#     idx_bot=idx_bot_jul,
#     lam=0, #2e3
#     rank=3,
#     prior_dict_W={2: eem_dataset_july.ref['TCC (million #/mL)'].to_numpy()},
#     gamma_W=0,
#     init='nndsvda',
#     alpha_H=0,
#     l1_ratio=0,
# )
# A_test = solve_W(
#     X1=eem_stack_to_2d(eem_dataset_october.eem_stack)[idx_bot_oct],
#     X2=eem_stack_to_2d(eem_dataset_october.eem_stack)[idx_top_oct],
#     H=B,
#     beta=None
# )
# A_test = A_test*beta
# r = 0
# plt.plot(A[:, r], eem_dataset_july.ref['TCC (million #/mL)'], 'o')
# plt.plot(A_test[:, r], eem_dataset_october.ref['TCC (million #/mL)'].dropna(), 'o')
# plt.show()
# plot_eem(B[r, :].reshape(eem_dataset_october.eem_stack.shape[1:]),
#          ex_range=eem_dataset_october.ex_range,
#          em_range=eem_dataset_october.em_range,
#          display=True
#          )
# fmax_ratio = A[idx_top_oct] / A[idx_bot_oct]
# fmax_ratio_z_scores = zscore(fmax_ratio[:, r])
# fmax_ratio_filtered = fmax_ratio[np.abs(fmax_ratio_z_scores) <= 3]
# plt.hist(fmax_ratio_filtered[:, r], density=True)
# plt.show()


# -----------model training-------------
# dataset_train, dataset_test = eem_dataset_october.splitting(2)
# dataset_train_splits = []
# dataset_train_unquenched, _ = eem_dataset_october.filter_by_index('B1C1', None, copy=True)
# initial_sub_eem_datasets_unquenched = dataset_train_unquenched.splitting(n_split=2, random_state=42)
# dataset_train_quenched, _ = eem_dataset_october.filter_by_index('B1C2', None, copy=True)
# for subset in initial_sub_eem_datasets_unquenched:
#     pos = [dataset_train_unquenched.index.index(idx) for idx in subset.index]
#     quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
#     sub_eem_dataset_quenched, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
#     dataset_train_splits.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))
# dataset_train = dataset_train_splits[0]
# dataset_test = dataset_train_splits[1]
dataset_train = eem_dataset_october
dataset_test = eem_dataset_july
indicator = 'TCC (million #/mL)'
sample_prior = {0: dataset_train.ref[indicator]}
params = {
    'n_components': 7,
    'init': 'ordinary_nmf',
    'gamma_sample': 0,
    'alpha_component': 0,
    'alpha_sample': 0,
    'l1_ratio': 0,
    'max_iter_als': 100,
    'max_iter_nnls': 800,
    'lam': 3e6, # 1e6
    'random_state': 42
}
# model = EEMNMF(
#     solver='hals',
#     prior_dict_sample=sample_prior,
#     normalization=None,
#     sort_em=False,
#     prior_ref_components=prior_dict_ref,
#     idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
#     idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
#     **params
# )
# model.fit(dataset_train)
# fmax_train = model.nmf_fmax
# components = model.components
# lr = LinearRegression(fit_intercept=False)
# mask_train = ~np.isnan(dataset_train.ref[indicator].to_numpy())
# X_train = fmax_train.iloc[mask_train, [list(sample_prior.keys())[0]]].to_numpy()
# y_train = dataset_train.ref[indicator].to_numpy()[mask_train]
# lr.fit(X_train, y_train)
# y_pred_train = lr.predict(X_train)
# rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
# r2_train = lr.score(X_train, y_train)
params = {
    'n_components': 5,
    'init': 'svd',
    'gamma_sample': 0,
    'max_iter_als': 100,
    'max_iter_nnls': 800,
    'lam': 0, # 1e8
    'random_state': 42
}
model = PARAFAC(
        solver='mu',
        # prior_dict_sample=sample_prior,
        tf_normalization=False,
        # sort_em=False,
        # prior_ref_components=prior_dict_ref,
        # idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
        # idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
        **params
)
model.fit(dataset_train)
fmax_train = model.fmax
components = model.components
lr = LinearRegression(fit_intercept=False)
mask_train = ~np.isnan(dataset_train.ref[indicator].to_numpy())
X_train = fmax_train.iloc[mask_train, [list(sample_prior.keys())[0]]].to_numpy()
y_train = dataset_train.ref[indicator].to_numpy()[mask_train]
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = lr.score(X_train, y_train)


# -----------model testing-------------
_, fmax_test, eem_re_test = model.predict(
    dataset_test,
    fit_beta=True,
    idx_top=[i for i in range(len(dataset_test.index)) if 'B1C1' in dataset_test.index[i]],
    idx_bot=[i for i in range(len(dataset_test.index)) if 'B1C2' in dataset_test.index[i]],
                                          )
sample_test_truth = {0: dataset_test.ref[indicator]}
mask_test = ~np.isnan(dataset_test.ref[indicator].to_numpy())
X_test = fmax_test.iloc[mask_test, [list(sample_prior.keys())[0]]].to_numpy()
y_test = dataset_test.ref[indicator].to_numpy()[mask_test]
y_pred_test = lr.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = lr.score(X_test, y_test)


# -----------plot components----------

plot_all_components(model)

# -----------plot Fmax vs. prior variables----------
info_dict = params.copy()
# info_dict['r2_train'] = np.round(r2_train, decimals=3)
# info_dict['r2_test'] = np.round(r2_test, decimals=3)
# info_dict['rmse_train'] = np.round(rmse_train, decimals=3)
# info_dict['rmse_test'] = np.round(rmse_test, decimals=3)
fig, ax = plt.subplots(nrows=1, ncols=len(sample_prior))
n_override = 0
for i, ((n, p), (n2, t)) in enumerate(zip(sample_prior.items(), sample_test_truth.items())):
    if n_override is not None:
        n = n_override
        n2 = n_override
    if len(sample_prior) == 1:
        ax.plot(fmax_train.iloc[:, n], p.to_numpy(), 'o', label='training')
        ax.plot(fmax_test.iloc[:, n], t.to_numpy(), 'o', label='testing')
        lr_n = LinearRegression(fit_intercept=False)
        mask_train_n = ~np.isnan(dataset_train.ref[indicator].to_numpy())
        lr_n.fit(fmax_train.iloc[mask_train_n, [n]].to_numpy(), p.iloc[mask_train_n])
        y_pred_train = lr.predict(fmax_train.iloc[mask_train_n, [n]].to_numpy())
        rmse_train = np.sqrt(mean_squared_error(p.iloc[mask_train_n], y_pred_train))
        r2_train = lr.score(fmax_train.iloc[mask_train_n, [n]], p.iloc[mask_train_n])
        y_pred_test = lr.predict(fmax_test.iloc[:, [n]].to_numpy())
        mask_test_n = ~np.isnan(dataset_test.ref[indicator].to_numpy())
        rmse_test = np.sqrt(mean_squared_error(t.iloc[mask_test_n].to_numpy(), y_pred_test[mask_test_n]))
        r2_test = lr.score(fmax_test.iloc[mask_test_n, [n]].to_numpy(), t.iloc[mask_test_n].to_numpy())
        info_dict['r2_train'] = np.round(r2_train, decimals=3)
        info_dict['r2_test'] = np.round(r2_test, decimals=3)
        info_dict['rmse_train'] = np.round(rmse_train, decimals=3)
        info_dict['rmse_test'] = np.round(rmse_test, decimals=3)
        info_dict['fit_intercept'] = True if lr_n.fit_intercept else False
        if lr_n.fit_intercept:
            ax.plot([0, 10000], np.array([0, 10000])*lr_n.coef_[0]+lr_n.intercept_, '--')
        else:
            ax.plot([0, 10000], np.array([0, 10000]) * lr_n.coef_[0], '--')
        ax.text(
            0.01, 0.99,
            '\n'.join(f'{k}: {v}' for k, v in info_dict.items()),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left'
        )
        ax.set_title(f'Fmax C{n + 1} vs. {p.name}', fontsize=18)
        ax.set_xlabel(f'Fmax C{n + 1}', fontsize=18)
        ax.set_ylabel(f'{p.name}', fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_xlim([0, max(
            max(fmax_train.iloc[:, n].to_numpy()),
            max(fmax_test.iloc[:, n].to_numpy())
        )+100
                     ])
        ax.set_ylim([0, max(
            max(p.to_numpy()),
            max(t.to_numpy())
        )+0.5
                     ])
        ax.legend(loc='best', bbox_to_anchor=(0.95, 0.25), fontsize=16)
    else:
        ax[i].plot(fmax_train.iloc[:, n], p.to_numpy(), 'o')
        ax[i].plot(fmax_test.iloc[:, n], t.to_numpy(), 'o')
        ax[i].set_title(f'Fmax C{n + 1} vs. {p.name}')
        ax[i].set_xlabel(f'Fmax C{n + 1}')
        ax[i].set_ylabel(f'{p.name}')
fig.tight_layout()
fig.show()

# ------------apparent F0/F distributions-------------


plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)')


# ----------cross-validation & hyperparameter optimization-------
dataset_train = eem_dataset_october
dataset_test = eem_dataset_july
indicator = 'TCC (million #/mL)'
param_grid = {
    'n_components': [4, 5, 6],
    'init': ['ordinary_nmf'],
    'gamma_sample': [0],
    'alpha_component': [0],
    'l1_ratio': [0],
    'lam': [0]
}

def get_param_combinations(param_grid):
    """
    Generates all combinations of parameters from a grid.

    Parameters:
        param_grid (dict): Dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        List of dictionaries, each representing one combination of parameters.
    """
    keys = list(param_grid.keys())
    values_product = product(*(param_grid[key] for key in keys))
    return [dict(zip(keys, values)) for values in values_product]

param_combinations = get_param_combinations(param_grid)
dataset_train_splits = []
dataset_train_unquenched, _ = dataset_train.filter_by_index('B1C1', None, copy=True)
initial_sub_eem_datasets_unquenched = dataset_train_unquenched.splitting(n_split=5, random_state=42)
dataset_train_quenched, _ = dataset_train.filter_by_index('B1C2', None, copy=True)
for subset in initial_sub_eem_datasets_unquenched:
    pos = [dataset_train_unquenched.index.index(idx) for idx in subset.index]
    quenched_index = [eem_dataset_quenched.index[idx] for idx in pos]
    sub_eem_dataset_quenched, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
    dataset_train_splits.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))

for k, p in enumerate(param_combinations):
    r2 = 0
    rmse = 0
    for i in range(len(dataset_train_splits)):
        d_train = combine_eem_datasets(dataset_train_splits[:i] + dataset_train_splits[i + 1:])
        d_test = dataset_train_splits[i]
        sample_prior = {1: d_train.ref['TCC (million #/mL)']}
        # model = EEMNMF(
        #     solver='hals',
        #     prior_dict_sample=sample_prior,
        #     normalization=None,
        #     sort_em=False,
        #     prior_ref_components=prior_dict_ref,
        #     idx_top=[i for i in range(len(d_train.index)) if 'B1C1' in d_train.index[i]],
        #     idx_bot=[i for i in range(len(d_train.index)) if 'B1C2' in d_train.index[i]],
        #     **p
        # )
        model = PARAFAC(
            n_components=p['n_components'],
        )
        model.fit(d_train)
        fmax_train = model.fmax
        components = model.components
        _, fmax_test, eem_re_test = model.predict(
            d_test,
            idx_top=[i for i in range(len(d_test.index)) if 'B1C1' in d_test.index[i]],
            idx_bot=[i for i in range(len(d_test.index)) if 'B1C2' in d_test.index[i]],
        )
        lr = LinearRegression(fit_intercept=False)
        mask_train = ~np.isnan(d_train.ref['TCC (million #/mL)'].to_numpy())
        X_train = fmax_train.iloc[mask_train, [1]].to_numpy()
        y_train = d_train.ref['TCC (million #/mL)'].to_numpy()[mask_train]
        lr.fit(X_train, y_train)
        mask_test = ~np.isnan(d_test.ref['TCC (million #/mL)'].to_numpy())
        X_test = fmax_test.iloc[mask_test, [1]].to_numpy()
        y_test = d_test.ref['TCC (million #/mL)'].to_numpy()[mask_test]
        r2 += lr.score(X_test, y_test) / len(dataset_train_splits)
        y_pred_test = lr.predict(X_test)
        rmse += np.sqrt(mean_squared_error(y_test, y_pred_test))/len(dataset_train_splits)
    param_combinations[k]['r2'] = r2
    param_combinations[k]['rmse'] = rmse

# with open("C:/PhD/publication/2025_prior_knowledge/param_combinations.pkl",
#           'wb') as file:
#     pickle.dump(param_combinations, file)
#
# with open("C:/PhD/publication/2025_prior_knowledge/param_combinations.pkl",
#           'rb') as file:
#     param_combinations = pickle.load(file)
#
# r = 5
# gamma_sample_list, lam_list, rmse_list = [], [], []
# for combo in param_combinations:
#     if combo['n_components'] == r:
#         gamma_sample_list.append(combo['gamma_sample'])
#         lam_list.append(combo['lam'])
#         rmse_list.append(combo['rmse'])
#
# df = pd.DataFrame({
#     'gamma': gamma_sample_list,
#     'lambda': lam_list,
#     'rmse': rmse_list
# })
#
# # Pivot to create a 2D grid
# heatmap_data = df.pivot(index='lambda', columns='gamma', values='rmse')
# heatmap_array = heatmap_data.values
#
# # Plot using subplots
# fig, ax = plt.subplots(figsize=(6, 5))
# im = ax.imshow(heatmap_array, cmap='viridis', origin='lower', aspect='auto')
#
# # Set tick labels
# ax.set_xticks(np.arange(len(heatmap_data.columns)))
# ax.set_yticks(np.arange(len(heatmap_data.index)))
# ax.set_xticklabels(heatmap_data.columns)
# ax.set_yticklabels(heatmap_data.index)
#
# # Labels and title
# ax.set_xlabel('Gamma')
# ax.set_ylabel('Lambda')
# ax.set_title(f'rank={r}')
#
# # Colorbar
# cbar = fig.colorbar(im, ax=ax, label='RMSE')
#
# fig.tight_layout()
# fig.show()
#
# ------------Kmethod + regulated NMF--------------

dataset_train = eem_dataset_october
dataset_test = eem_dataset_july
indicator = 'TCC (million #/mL)'
sample_prior = {0: dataset_train.ref[indicator]}
params = {
    'n_components': 4,
    'init': 'ordinary_nmf',
    'gamma_sample': 0,
    'alpha_component': 0,
    'alpha_sample': 0,
    'l1_ratio': 0,
    'max_iter_als': 100,
    'max_iter_nnls': 800,
    'lam': 1e6, # 1e6
    'random_state': 42
}
model = EEMNMF(
    solver='hals',
    prior_dict_sample=None,
    normalization=None,
    sort_em=False,
    prior_ref_components=prior_dict_ref,
    kw_top='B1C1',
    kw_bot='B1C2',
    **params
)

kmodel = KMethod(
    base_model=model,
    n_initial_splits=4,
    distance_metric="reconstruction_error_with_beta",
    max_iter=10,
    kw_top='B1C1',
    kw_bot='B1C2',
)
cluster_labels, label_history, error_history = kmodel.base_clustering(eem_dataset=dataset_train)
df_cluster = pd.DataFrame(cluster_labels, index=dataset_train.index)
dataset_train.cluster = cluster_labels
cluster_specific_models = {}
for label in list(set(cluster_labels)):
    cluster, _ = dataset_train.filter_by_cluster(cluster_names=label)
    model_cluster = copy.deepcopy(model)
    model_cluster.fit(cluster)
    cluster_specific_models[label] = model_cluster
    plot_all_components(model_cluster)
    plot_all_f0f(model_cluster, cluster, 'B1C1', 'B1C2', 'TCC (million #/mL)')

with open("C:/PhD/publication/2025_prior_knowledge/cluster_specific_models_october.pkl",
          'wb') as file:
    pickle.dump(cluster_specific_models, file)

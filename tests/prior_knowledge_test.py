import copy
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import zscore
from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from itertools import product
from itertools import combinations
from sklearn.metrics import r2_score

np.random.seed(42)

# ---------------Read EEM dataset-----------------

# eem_dataset_path = \
#     "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset_path = \
    "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/sample_276_ex_274_em_310_mfem_5_gaussian_rsu_rs_interpolated.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset.median_filter(footprint=(5, 5), copy=False)
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

with open('C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/approx_components.pkl', 'rb') as file:
    approx_components = pickle.load(file)


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

    fig, axes = plt.subplots(ncols=2, nrows=(model.n_components + 1) // 2, figsize=(8, 10))
    for rank_target in range(model.n_components):
        fmax_ratio_target = fmax_ratio[:, rank_target]
        fmax_ratio_target_valid = fmax_ratio_target[(fmax_ratio_target >= 0) & (fmax_ratio_target <= 1e3)]
        fmax_ratio_z_scores = zscore(fmax_ratio_target_valid, nan_policy='omit')
        ratio_nan = 1 - fmax_ratio_target_valid.shape[0] / fmax_ratio_target.shape[0]
        fmax_ratio_target_filtered = fmax_ratio_target_valid[np.abs(fmax_ratio_z_scores) <= 3]

        if (model.n_components + 1) // 2 > 1:
            counts, bins, patches = axes[rank_target // 2, rank_target % 2].hist(fmax_ratio_target_filtered, bins=30,
                                                                                 density=True, alpha=0.5, color='blue',
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


def plot_fmax_vs_truth(fmax_train, fmax_test, truth_train, truth_test, n, info_dict):
    info_dict_copy = copy.deepcopy(info_dict)
    fig, ax = plt.subplots(nrows=1, ncols=len(sample_prior))
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
    fig, ax = plt.subplots(
        nrows=(eem_model.n_components + 1) // 2, ncols=2,
    )
    plt.subplots_adjust(
        left=0,  # distance from left of figure (0 = 0%, 1 = 100%)
        right=1,  # distance from right
        bottom=0,
        top=1,
        wspace=0,  # width between subplots
        hspace=0  # height between subplots
    )
    for i in range(eem_model.n_components):
        if i < eem_model.n_components:
            f, a, im = plot_eem(
                eem_model.components[i],
                ex_range=eem_model.ex_range,
                em_range=eem_model.em_range,
                display=False,
                title=f'Component {i + 1}'
            )
            canvas = FigureCanvas(f)
            canvas.draw()
            # Get the RGBA image as a NumPy array
            img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(canvas.get_width_height()[::-1] + (4,))
            if (eem_model.n_components + 1) // 2 > 1:
                ax[i // 2, i % 2].imshow(img_array)
                ax[i // 2, i % 2].axis('off')  # Hides ticks, spines, etc.
            else:
                ax[i % 2].imshow(img_array)
                ax[i % 2].axis('off')  # Hides ticks, spines, etc.
    fig.show()


def plot_outlier_plots(model, indicator, estimator_rank, dataset_train, dataset_test, criteria='reconstruction_error'):
    # fmax_train = model.fmax
    _, fmax_train, eem_re_train = model.predict(
        dataset_train,
        fit_beta=True,
        idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
        idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
    )
    _, fmax_test, eem_re_test = model.predict(
        dataset_test,
        fit_beta=True,
        idx_top=[i for i in range(len(dataset_test.index)) if 'B1C1' in dataset_test.index[i]],
        idx_bot=[i for i in range(len(dataset_test.index)) if 'B1C2' in dataset_test.index[i]],
    )
    if criteria == 'reconstruction_error':
        # error_train = model.sample_rmse().to_numpy().reshape(-1)
        res_train = dataset_train.eem_stack - eem_re_train
        n_pixels = res_train.shape[1] * res_train.shape[2]
        error_train = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)

        res_test = dataset_test.eem_stack - eem_re_test
        n_pixels = res_test.shape[1] * res_test.shape[2]
        error_test = np.sqrt(np.sum(res_test ** 2, axis=(1, 2)) / n_pixels)

    binwidth = np.max(np.concatenate([error_train, error_test])) - np.min(np.concatenate([error_train, error_test]))
    binwidth = binwidth / 50
    binrange = (np.min(np.concatenate([error_train, error_test]) - binwidth, axis=0),
                np.max(np.concatenate([error_train, error_test]) + binwidth, axis=0)
                )
    error_train_z_scores = zscore(error_train)
    filtered_error_train = error_train[np.abs(error_train_z_scores) <= 3]
    threshold_upper = np.quantile(filtered_error_train, 1)
    threshold_lower = np.quantile(filtered_error_train, 0)
    fig_hist, ax_hist = plt.subplots()
    counts, bins, patches = ax_hist.hist(error_train,
                                         bins=np.arange(binrange[0], binrange[1] + binwidth, binwidth),
                                         density=True, alpha=0.5, color='blue', label='training', zorder=0,
                                         edgecolor='black')
    # Histogram for outliers (just to show label with hatching)
    counts, bins, patches = ax_hist.hist([-100], bins=np.arange(binrange[0], binrange[1] + binwidth, binwidth),
                                         density=True, alpha=0.5, color='orange', label='test (qualified)',
                                         edgecolor='red', zorder=2)
    # Histogram for test (qualified) data
    counts, bins, patches = ax_hist.hist(error_test,
                                         bins=np.arange(binrange[0], binrange[1] + binwidth, binwidth),
                                         density=True, alpha=0.5, color='orange', label='test (outliers)', zorder=1,
                                         edgecolor='black')

    for bar in ax_hist.patches:
        # Calculate the midpoint of the bin
        bin_left = bar.get_x()
        bin_width = bar.get_width()
        bin_mid = bin_left + bin_width / 2

        # Check if the midpoint is above the threshold
        if (threshold_upper <= bin_mid or threshold_lower >= bin_mid) and bar.zorder == 1:
            # Add hatch pattern and change color
            bar.set_hatch("////")  # Hatch pattern (e.g., "////", "xxx", "..")
            bar.set_edgecolor('red')
    plt.xlim(binrange)
    plt.xlabel("RMSE", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    fig_hist.show()

    truth_train = dataset_train.ref[indicator]
    truth_test = dataset_test.ref[indicator]
    valid_indices_train = truth_train.index[~truth_train.isna()]
    valid_indices_test = truth_test.index[~truth_test.isna()]
    truth_train_valid = truth_train.loc[valid_indices_train]
    truth_test_valid = truth_test.loc[valid_indices_test]
    fmax_train_valid = fmax_train.loc[valid_indices_train]
    fmax_test_valid = fmax_test.loc[valid_indices_test]
    error_train_valid = error_train[fmax_train.index.isin(valid_indices_train)]
    error_test_valid = error_test[fmax_test.index.isin(valid_indices_test)]

    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(truth_train_valid.to_numpy().reshape(-1, 1), fmax_train_valid.iloc[:, estimator_rank])
    a = reg.coef_
    b = reg.intercept_

    fig_plot, ax_plot = plt.subplots()
    ax_plot.plot(
        [-1, 10],
        a * np.array([-1, 10]) + b,
        '--',
        color='blue',
        label='reg. training'
    )
    ax_plot.scatter(fmax_train_valid.iloc[:, estimator_rank], truth_train_valid, label='training', color='blue',
                    alpha=0.6)
    ax_plot.scatter(
        fmax_test_valid.iloc[
            (error_test_valid < threshold_upper) & (error_test_valid > threshold_lower), estimator_rank],
        truth_test_valid[(error_test_valid < threshold_upper) & (error_test_valid > threshold_lower)],
        label='test (qualified)', color='orange', alpha=0.6)
    ax_plot.scatter(
        fmax_test_valid.iloc[
            (error_test_valid >= threshold_upper) | (error_test_valid <= threshold_lower), estimator_rank],
        truth_test_valid[(error_test_valid >= threshold_upper) | (error_test_valid <= threshold_lower)],
        label='test (outliers)', color='red', alpha=0.6)
    ax_plot.set_ylabel(indicator, fontsize=20)
    ax_plot.set_xlabel(f'C{estimator_rank + 1} Fmax', fontsize=20)
    ax_plot.set_xlim([0, 2500])
    ax_plot.set_ylim([0, 2.5])
    fig_plot.legend(
        bbox_to_anchor=[1.02, 0.37],
        # bbox_to_anchor=[0.58, 0.63],
        fontsize=16
    )
    ax_plot.tick_params(labelsize=16)
    fig_plot.tight_layout()
    fig_plot.show()


# -----------model training-------------
# dataset_train, dataset_test = eem_dataset_october.splitting(2)
dataset_train_splits = []
dataset_train_unquenched, _ = eem_dataset_october.filter_by_index('B1C1', None, copy=True)
initial_sub_eem_datasets_unquenched = dataset_train_unquenched.splitting(n_split=3, random_state=42)
dataset_train_quenched, _ = eem_dataset_october.filter_by_index('B1C2', None, copy=True)
for subset in initial_sub_eem_datasets_unquenched:
    pos = [dataset_train_unquenched.index.index(idx) for idx in subset.index]
    quenched_index = [dataset_train_quenched.index[idx] for idx in pos]
    sub_eem_dataset_quenched, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
    subset.sort_by_index()
    sub_eem_dataset_quenched.sort_by_index()
    dataset_train_splits.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))
dataset_train = dataset_train_splits[0]
dataset_test = dataset_train_splits[2]
# dataset_train = eem_dataset_october
# dataset_test = eem_dataset_july
indicator = 'TCC (million #/mL)'
sample_prior = {0: dataset_train.ref[indicator]}
params = {
    'n_components': 5,
    'init': 'svd',
    # 'gamma_W': 0,
    # 'gamma_H': 0,
    # 'gamma_A': 0,
    # 'alpha_component': 0,
    # 'alpha_sample': 0,
    # 'l1_ratio': 0,
    'max_iter_als': 100,
    'max_iter_nnls': 800,
    'lam': 3e4,  # 1e6
    # 'random_state': 42,
    # 'fit_rank_one': {
    #     0: True,
    #     1: True,
    #     2: True,
    #     3: True,
    #     4: True,
    #     5: True,
    # },
}

model = PARAFAC(
    solver='hals',
    # prior_dict_W=sample_prior,
    # prior_dict_H=approx_components,
    # normalization=None,
    sort_em=False,
    prior_ref_components=prior_dict_ref,
    idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
    idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
    **params
)
model.fit(dataset_train)

fmax_train = model.fmax
components = model.components
lr = LinearRegression(fit_intercept=True)
mask_train = ~np.isnan(dataset_train.ref[indicator].to_numpy())
X_train = fmax_train.iloc[mask_train, [list(sample_prior.keys())[0]]].to_numpy()
y_train = dataset_train.ref[indicator].to_numpy()[mask_train]
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = lr.score(X_train, y_train)
slope_train = lr.coef_
intercept_train = lr.intercept_
#
# params = {
#     'n_components': 5,
#     'init': 'nndsvda',
#     'gamma_sample': 0,
#     'max_iter_als': 100,
#     'max_iter_nnls': 800,
#     'lam': 1e6, # 1e8
#     'random_state': 42
# }
# model = PARAFAC(
#         solver='hals',
#         # prior_dict_sample=sample_prior,
#         # tf_normalization=False,
#         # sort_em=False,
#         prior_ref_components=prior_dict_ref,
#         idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
#         idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
#         **params
# )
# model.fit(dataset_train)
# fmax_train = model.fmax
# components = model.components
# lr = LinearRegression(fit_intercept=False)
# mask_train = ~np.isnan(dataset_train.ref[indicator].to_numpy())
# X_train = fmax_train.iloc[mask_train, [list(sample_prior.keys())[0]]].to_numpy()
# y_train = dataset_train.ref[indicator].to_numpy()[mask_train]
# lr.fit(X_train, y_train)
# y_pred_train = lr.predict(X_train)
# rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
# r2_train = lr.score(X_train, y_train)


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

plot_fmax_vs_truth(fmax_train=fmax_train, fmax_test=fmax_test,
                   truth_train=dataset_train.ref[indicator], truth_test=dataset_test.ref[indicator],
                   n=0, info_dict=params)

# ------------apparent F0/F distributions-------------


plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)')

# -----------outliers----------

plot_outlier_plots(
    model=model, estimator_rank=0, indicator='TCC (million #/mL)',
    dataset_test=dataset_test, dataset_train=dataset_train
)

# ----------cross-validation & hyperparameter optimization-------
dataset_train = eem_dataset_october
dataset_test = eem_dataset_july
indicator = 'TCC (million #/mL)'

model_ref = PARAFAC(
    n_components=5,
    solver='hals',
    sort_em=False,
    lam=0,
    prior_ref_components=prior_dict_ref,
    # prior_dict_H=approx_components,
    idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
    idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
)
model_ref.fit(dataset_train)
components_ref = {model_ref.fmax.columns[i]: model_ref.components[i] for i in range(model_ref.n_components)}

param_grid = {
    'n_components': [5],
    'init': ['svd'],
    # 'gamma_W': [0],
    # 'gamma_H': [0],
    # 'l1_ratio': [0],
    'lam': [0],
}

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

for k, p in enumerate(param_combinations):
    r2_train, r2_test, rmse_train, rmse_test = 0, 0, 0, 0
    components_list = [[] for i in range(p['n_components'])]
    for a, b in splits:
        d_train = combine_eem_datasets(a)
        d_test = combine_eem_datasets(b)
        sample_prior = {0: d_train.ref['TCC (million #/mL)']}
        model = PARAFAC(
            solver='hals',
            # prior_dict_W=sample_prior,
            # prior_dict_H=approx_components,
            sort_em=False,
            prior_ref_components=prior_dict_ref,
            idx_top=[i for i in range(len(d_train.index)) if 'B1C1' in d_train.index[i]],
            idx_bot=[i for i in range(len(d_train.index)) if 'B1C2' in d_train.index[i]],
            # fit_rank_one={
            #                     0: True,
            #                     1: True,
            #                     2: True,
            #                     3: True,
            #                     4: True,
            #                     5: True,
            #              },
            **p
        )
        model.fit(d_train)
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
        model_dict = align_components_by_components({0: model}, components_ref, model_type='parafac')
        model = model_dict[0]
        plt.close()
        plot_all_components(model)
        plot_fmax_vs_truth(fmax_train=fmax_train, fmax_test=fmax_test,
                           truth_train=d_train.ref[indicator], truth_test=d_test.ref[indicator],
                           n=0, info_dict=p)
        plot_all_f0f(model, d_train, 'B1C1', 'B1C2', 'TCC (million #/mL)')
        for j in range(len(components_list)):
            components_list[j].append(model.components[j].reshape(-1))
    param_combinations[k]['r2_train'] = r2_train
    param_combinations[k]['rmse_train'] = rmse_train
    param_combinations[k]['r2_test'] = r2_test
    param_combinations[k]['rmse_test'] = rmse_test
    param_combinations[k]['component_similarities'] = [mean_pairwise_correlation(c) for c in components_list]

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
    'lam': 1e6,  # 1e6
    'random_state': 42
}
model = EEMNMF(
    solver='hals',
    prior_dict_W=None,
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
clusters = {}
for label in list(set(cluster_labels)):
    cluster, _ = dataset_train.filter_by_cluster(cluster_names=label)
    clusters[label] = cluster

# with open("C:/PhD/publication/2025_prior_knowledge/clusters_all.pkl",
#           'wb') as file:
#     pickle.dump(clusters, file)

with open("C:/PhD/publication/2025_prior_knowledge/clusters_all.pkl",
          'rb') as file:
    clusters = pickle.load(file)

cluster_specific_models = {}
for label, cluster in clusters.items():
    model_cluster = copy.deepcopy(model)
    model_cluster.fit(cluster)
    cluster_specific_models[label] = model_cluster
    plot_all_components(model_cluster)
    plot_all_f0f(model_cluster, cluster, 'B1C1', 'B1C2', 'TCC (million #/mL)')

with open("C:/PhD/publication/2025_prior_knowledge/cluster_specific_models_all.pkl",
          'wb') as file:
    pickle.dump(cluster_specific_models, file)

# ------------

dataset_train_splits = []
dataset_train_unquenched, _ = eem_dataset_october.filter_by_index('B1C1', None, copy=True)
initial_sub_eem_datasets_unquenched = dataset_train_unquenched.splitting(n_split=2, random_state=42)
dataset_train_quenched, _ = eem_dataset_october.filter_by_index('B1C2', None, copy=True)
for subset in initial_sub_eem_datasets_unquenched:
    pos = [dataset_train_unquenched.index.index(idx) for idx in subset.index]
    quenched_index = [dataset_train_quenched.index[idx] for idx in pos]
    sub_eem_dataset_quenched, _ = eem_dataset.filter_by_index(None, quenched_index, copy=True)
    subset.sort_by_index()
    sub_eem_dataset_quenched.sort_by_index()
    dataset_train_splits.append(combine_eem_datasets([subset, sub_eem_dataset_quenched]))
dataset_train = dataset_train_splits[0]
dataset_test = dataset_train_splits[1]
# dataset_train = eem_dataset_october
# dataset_test = eem_dataset_july
indicator = 'TCC (million #/mL)'
sample_prior = {0: dataset_train.ref[indicator]}
params = {
    'n_components': 4,
    'init': 'ordinary_nmf',
    'gamma_H': 0,
    'alpha_component': 0,
    'alpha_sample': 0,
    'l1_ratio': 0,
    'max_iter_als': 100,
    'max_iter_nnls': 800,
    'lam': 2e5,  # 1e6
    'random_state': 42,
}


def auto_nmf(model: EEMNMF, dataset_train, dataset_test, gamma_H=1e5):
    approx_components = {}
    for r in range(model.n_components):
        model.fit_rank_one = {r: True}
        model.fit(dataset_train)
        if r != 1:
            approx_components[r] = model.components[r].reshape(-1)
    model.fit_rank_one = False
    model.gamma_H = gamma_H
    model.prior_dict_H = approx_components
    model.fit(dataset_train)
    plot_all_components(model)
    fmax_train = model.fmax
    _, fmax_test, _ = model.predict(
        dataset_test,
        fit_beta=True if model.lam > 0 else False,
        idx_top=[i for i in range(len(dataset_test.index)) if 'B1C1' in dataset_test.index[i]],
        idx_bot=[i for i in range(len(dataset_test.index)) if 'B1C2' in dataset_test.index[i]],
    )
    plot_fmax_vs_truth(fmax_train=fmax_train, fmax_test=fmax_test,
                       truth_train=dataset_train.ref['TCC (million #/mL)'],
                       truth_test=dataset_test.ref['TCC (million #/mL)'],
                       n=0, info_dict={})
    plot_all_f0f(model, dataset_train, 'B1C1', 'B1C2', 'TCC (million #/mL)')
    return model, approx_components


model = EEMNMF(
    solver='hals',
    prior_dict_W=None,
    normalization=None,
    sort_em=False,
    prior_ref_components=prior_dict_ref,
    kw_top='B1C1',
    kw_bot='B1C2',
    **params
)
model_opt, approx_components = auto_nmf(
    model=model,
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    gamma_H=1e5
)

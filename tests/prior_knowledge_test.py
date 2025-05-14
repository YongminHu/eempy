import matplotlib.pyplot as plt
from scipy.stats import zscore

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from itertools import product
import seaborn as sns

# ---------------Read EEM dataset-----------------

# eem_dataset_path = \
#     "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset_path = \
    "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset.eem_stack = np.nan_to_num(eem_dataset.eem_stack, copy=True, nan=0)
eem_dataset_july, _ = eem_dataset.filter_by_index(None, ['2024-07-'], copy=True)
eem_dataset_october, _ = eem_dataset.filter_by_index(None, ['2024-10-'], copy=True)
eem_dataset_original, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
eem_dataset_quenched, _ = eem_dataset.filter_by_index(['B1C2'], None, copy=True)
idx_top_oct = [i for i in range(len(eem_dataset_october.index)) if 'B1C1' in eem_dataset_october.index[i]]
idx_bot_oct = [i for i in range(len(eem_dataset_october.index)) if 'B1C2' in eem_dataset_october.index[i]]
idx_top_jul = [i for i in range(len(eem_dataset_july.index)) if 'B1C1' in eem_dataset_july.index[i]]
idx_bot_jul = [i for i in range(len(eem_dataset_july.index)) if 'B1C2' in eem_dataset_july.index[i]]

#
# # -------------prior decomposition function test---------
#
# A, B, C = cp_hals_prior(
#     tensor=eem_dataset_original.eem_stack,
#     rank=6,
#     prior_dict_A={0: eem_dataset_original.ref['TCC (million #/mL)'].to_numpy()},
#     gamma_A=7.51 * 1e3,
#     tol=1e-9,
#     init='nndsvda'
# )
# plt.plot(A[:, 0], eem_dataset_original.ref['TCC (million #/mL)'], 'o')
# plt.show()
# plot_eem(np.outer(B[:, 0], C[:, 0]),
#          ex_range=eem_dataset_original.ex_range,
#          em_range=eem_dataset_original.em_range,
#          display=True
#          )
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
dataset_train = eem_dataset_october
dataset_test = eem_dataset_july
sample_prior = {2: dataset_train.ref['TCC (million #/mL)']}
params = {
    'n_components': 4,
    'init': 'nndsvda',
    'gamma_sample': 5e6,
    'alpha_component': 0,
    'l1_ratio': 0,
    'max_iter_als': 100,
    'max_iter_nnls': 500,
    'lam': 1e9
}
model = EEMNMF(
    solver='hals',
    prior_dict_sample=sample_prior,
    normalization=None,
    sort_em=False,
    # idx_top=[i for i in range(len(dataset_train.index)) if 'B1C1' in dataset_train.index[i]],
    # idx_bot=[i for i in range(len(dataset_train.index)) if 'B1C2' in dataset_train.index[i]],
    **params
)
model.fit(dataset_train)
fmax_train = model.nmf_fmax
components = model.components
lr = LinearRegression(fit_intercept=True)
mask_train = ~np.isnan(dataset_train.ref['TCC (million #/mL)'].to_numpy())
X_train = fmax_train.iloc[mask_train, [list(sample_prior.keys())[0]]].to_numpy()
y_train = dataset_train.ref['TCC (million #/mL)'].to_numpy()[mask_train]
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = lr.score(X_train, y_train)

# -----------model testing-------------

_, fmax_test, eem_re_test = model.predict(
    dataset_test,
    idx_top=[i for i in range(len(dataset_test.index)) if 'B1C1' in dataset_test.index[i]],
    idx_bot=[i for i in range(len(dataset_test.index)) if 'B1C2' in dataset_test.index[i]],
                                          )
sample_test_truth = {2: dataset_test.ref['TCC (million #/mL)']}
mask_test = ~np.isnan(dataset_test.ref['TCC (million #/mL)'].to_numpy())
X_test = fmax_test.iloc[mask_test, [list(sample_prior.keys())[0]]].to_numpy()
y_test = dataset_test.ref['TCC (million #/mL)'].to_numpy()[mask_test]
y_pred_test = lr.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = lr.score(X_test, y_test)


# -----------plot components----------
plt.close()
fig, ax = plt.subplots(
    nrows=(params['n_components'] - 1) // 2 + 1, ncols=2,
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
            components[i],
            ex_range=eem_dataset_original.ex_range,
            em_range=eem_dataset_original.em_range,
            display=False,
            title=f'Component {i + 1}'
        )
        canvas = FigureCanvas(f)
        canvas.draw()

        # Get the RGBA image as a NumPy array
        img_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(canvas.get_width_height()[::-1] + (4,))
        ax[i // 2, i % 2].imshow(img_array)
    ax[i // 2, i % 2].axis('off')  # Hides ticks, spines, etc.
fig.show()

# -----------plot Fmax vs. prior variables----------
info_dict = params.copy()
info_dict['r2_train'] = np.round(r2_train, decimals=3)
info_dict['r2_test'] = np.round(r2_test, decimals=3)
info_dict['rmse_train'] = np.round(rmse_train, decimals=3)
info_dict['rmse_test'] = np.round(rmse_test, decimals=3)
fig, ax = plt.subplots(nrows=1, ncols=len(sample_prior))
n_override = 0
for i, ((n, p), (n2, t)) in enumerate(zip(sample_prior.items(), sample_test_truth.items())):
    if n_override is not None:
        n = n_override
        n2 = n_override
    if len(sample_prior) == 1:
        ax.plot(fmax_train.iloc[:, n], p.to_numpy(), 'o')
        ax.plot(fmax_test.iloc[:, n], t.to_numpy(), 'o')
        ax.text(
            0.01, 0.99,
            '\n'.join(f'{k}: {v}' for k, v in info_dict.items()),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left'
        )
        ax.set_title(f'Fmax C{n + 1} vs. {p.name}')
        ax.set_xlabel(f'Fmax C{n + 1}')
        ax.set_ylabel(f'{p.name}')
    else:
        ax[i].plot(fmax_train.iloc[:, n], p.to_numpy(), 'o')
        ax[i].plot(fmax_test.iloc[:, n], t.to_numpy(), 'o')
        ax[i].set_title(f'Fmax C{n + 1} vs. {p.name}')
        ax[i].set_xlabel(f'Fmax C{n + 1}')
        ax[i].set_ylabel(f'{p.name}')
fig.show()

# ------------apparent F0/F distributions-------------
rank_target = 2
target_analyte = 'TCC (million #/mL)'
target_train = dataset_train.ref[target_analyte]
valid_indices_train = target_train.index[~target_train.isna()]
target_train = target_train.dropna().to_numpy()
fmax_original_train = fmax_train[fmax_train.index.str.contains('B1C1')]
mask_train = fmax_original_train.index.isin(valid_indices_train)
fmax_original_train = fmax_original_train[mask_train]
fmax_quenched_train = fmax_train[fmax_train.index.str.contains('B1C2')]
fmax_quenched_train = fmax_quenched_train[mask_train]
fmax_ratio_train = fmax_original_train.to_numpy() / fmax_quenched_train.to_numpy()
fmax_ratio_target_train = fmax_ratio_train[:, rank_target]
fmax_ratio_train_z_scores = zscore(fmax_ratio_target_train)
fmax_ratio_target_train_filtered = fmax_ratio_target_train[np.abs(fmax_ratio_train_z_scores) <= 3]

plt.close()
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(fmax_ratio_target_train_filtered, bins=30,
                                density=True, alpha=0.5, color='blue', label='training', zorder=0, edgecolor='black')
fig.tight_layout()
fig.show()

#
# # ----------cross-validation & hyperparameter optimization-------
#
# param_grid = {
#     'n_components': [4],
#     'init': ['nndsvda'],
#     'gamma_sample': [2.15e7, 2.2e7, 2.25e7, 2.3e7, 2.35e7],
#     'alpha_component': [0],
#     'l1_ratio': [0]
# }
#
# def get_param_combinations(param_grid):
#     """
#     Generates all combinations of parameters from a grid.
#
#     Parameters:
#         param_grid (dict): Dictionary where keys are parameter names and values are lists of possible values.
#
#     Returns:
#         List of dictionaries, each representing one combination of parameters.
#     """
#     keys = list(param_grid.keys())
#     values_product = product(*(param_grid[key] for key in keys))
#     return [dict(zip(keys, values)) for values in values_product]
#
# param_combinations = get_param_combinations(param_grid)
# dataset_splits = dataset_train.splitting(n_split=5)
#
# for k, p in enumerate(param_combinations):
#     r2 = 0
#     rmse = 0
#     for i in range(len(dataset_splits)):
#         d_train = combine_eem_datasets(dataset_splits[:i]+dataset_splits[i+1:])
#         d_test = dataset_splits[i]
#         sample_prior = {0: d_train.ref['TCC (million #/mL)']}
#         model = EEMNMF(
#             solver='hals',
#             prior_dict_sample=sample_prior,
#             normalization=None,
#             sort_em=False,
#             **p
#         )
#         model.fit(d_train)
#         fmax_train = model.nmf_fmax
#         components = model.components
#         _, fmax_test, eem_re_test = model.predict(d_test)
#         lr = LinearRegression(fit_intercept=False)
#         mask_train = ~np.isnan(d_train.ref['TCC (million #/mL)'].to_numpy())
#         X_train = fmax_train.iloc[mask_train, [0]].to_numpy()
#         y_train = d_train.ref['TCC (million #/mL)'].to_numpy()[mask_train]
#         lr.fit(X_train, y_train)
#         mask_test = ~np.isnan(d_test.ref['TCC (million #/mL)'].to_numpy())
#         X_test = fmax_test.iloc[mask_test, [0]].to_numpy()
#         y_test = d_test.ref['TCC (million #/mL)'].to_numpy()[mask_test]
#         r2 += lr.score(X_test, y_test) / len(dataset_splits)
#         y_pred_test = lr.predict(X_test)
#         rmse += np.sqrt(mean_squared_error(y_test, y_pred_test))/len(dataset_splits)
#     param_combinations[k]['r2'] = r2
#     param_combinations[k]['rmse'] = rmse


# eem_stack = eem_dataset_original.eem_stack
# eem_stack_2d = eem_stack.reshape([eem_stack.shape[0], -1])
# eem_stack_2d = np.nan_to_num(eem_stack_2d, nan=0)
# A, B = nmf_hals_prior(X=eem_stack_2d, rank=rank, prior_dict_W=sample_prior, gamma_W=1e7, init='nndsvda')
# components = B.reshape([B.shape[0], eem_stack.shape[1], eem_stack.shape[2]])
# plt.plot(A[:, 0], eem_dataset_original.ref['TCC (million #/mL)'].to_numpy(), 'o')
# plt.show()

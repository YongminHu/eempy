from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import *
from scipy.stats import pearsonr
from matplotlib.colors import TABLEAU_COLORS
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter, HourLocator
from scipy.stats import zscore
from matplotlib.colors import BoundaryNorm, ListedColormap
colors = list(TABLEAU_COLORS.values())

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_260_ex_274_em_310_mfem_3.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)
eem_dataset_original, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
abs_stack, ex_range_abs, _ = read_abs_dataset('C:/PhD/Fluo-detect/_data/_greywater/2024_quenching', ['ABS', 'B1C1'])
# eem_dataset.gaussian_filter(sigma=1, truncate=3, copy=False)

# ------------Define conditions--------------

kw_dict = {
    'nj': [['M3'], ['2024-07-12', '2024-07-13', '2024-07-15', '2024-07-16', '2024-07-17'], 4],
    'sj': [['M3'], ['2024-07-18', '2024-07-19'], 4],
    'no': [['M3'], ['2024-10-16', '2024-10-22'], 4],
    'so': [['M3'], ['2024-10-18'], 3],
    'hf': [['M3'], ['2024-10-17'], 3],
    'sc1': [['G3'], None, 4],
    'sc2': [['G2'], None, 4],
    'sc3': [['G1'], None, 4],
    'cc': [['M3'], ['2024-10-21'], 4],
    # 'all': [['2024'], None, 4]
}

eem_dataset_normal_oct, _ = eem_dataset_original.filter_by_index(None, ['2024-10-15', '2024-10-16', '2024-10-22'], copy=True)
eem_dataset_normal_oct_col, _ = eem_dataset_normal_oct.filter_by_index(None, ['G1', 'G2', 'G3'], copy=True)
eem_dataset_normal_oct_eff, _ = eem_dataset_normal_oct.filter_by_index(None, ['M3'], copy=True)
eem_dataset_lowflow_oct_col, _ = eem_dataset_original.filter_by_index(['2024-10-18'], ['G1', 'G2', 'G3'], copy=True)
eem_dataset_lowflow_oct_eff, _ = eem_dataset_original.filter_by_index(['2024-10-18', 'M3'], None, copy=True)
eem_dataset_highflow_oct_col, _ = eem_dataset_original.filter_by_index(['2024-10-17'], ['G1', 'G2', 'G3'], copy=True)
eem_dataset_highflow_oct_eff, _ = eem_dataset_original.filter_by_index(['2024-10-17', 'M3'], None, copy=True)
eem_dataset_crossconnection_oct_eff, _ = eem_dataset_original.filter_by_index(['2024-10-21'], None,  copy=True)

dataset_divisions = {
    'N-col.': eem_dataset_normal_oct_col,
    'N-eff.': eem_dataset_normal_oct_eff,
    'LF-col.': eem_dataset_lowflow_oct_col,
    'LF-eff.': eem_dataset_lowflow_oct_eff,
    'HF-col.': eem_dataset_highflow_oct_col,
    'HF-eff.': eem_dataset_highflow_oct_eff,
    'CC-eff.': eem_dataset_crossconnection_oct_eff
}



#
# # # --------N_components vs. r_tcc and mean(F0/F)--------
# # # --------Fig 1: Histplots of model on testing dataset: effect of N components-------
# dataset_train, _ = eem_dataset.filter_by_index(None,
#                                                [
#                                                    '2024-'
#                                                ]
#                                                )
#
# r_list = [4, 5, 6]
# fmax_col = 0
# target_name = 'TCC (millioin #/mL)'
# hist_fmax_ratio = plt.figure()
# for i, r_i in enumerate(r_list):
#     model = PARAFAC(n_components=r_i, init='svd', non_negativity=True,
#                     tf_normalization=True, sort_em=True, loadings_normalization='maximum')
#     model.fit(dataset_train)
#     target_test_re = dataset_train.ref[target_name]
#     valid_indices_test_re = target_test_re.index[~target_test_re.isna()]
#     target_test_re = target_test_re.dropna().to_numpy()
#     fmax_test_re = model.fmax
#     fmax_original_test_re = fmax_test_re[fmax_test_re.index.str.contains('B1C1')]
#     mask_test_re = fmax_original_test_re.index.isin(valid_indices_test_re)
#     fmax_original_test_re = fmax_original_test_re[mask_test_re]
#     fmax_quenched_test_re = fmax_test_re[fmax_test_re.index.str.contains('B1C2')]
#     fmax_quenched_test_re = fmax_quenched_test_re[mask_test_re]
#     fmax_ratio_test_re = fmax_original_test_re.to_numpy() / fmax_quenched_test_re.to_numpy()
#     fmax_ratio_target_test_re = fmax_ratio_test_re[:, fmax_col]
#     r_target_test_re, p_target_test_re = pearsonr(target_test_re, fmax_original_test_re.iloc[:, fmax_col])
#     sns.histplot(fmax_ratio_target_test_re, binwidth=0.01, binrange=(0.85, 1.32), kde=False, stat='density',
#                  alpha=0.3, label=f'n_components={r_i},' '$r_{TCC}=$' + f'{r_target_test_re:.2f}', color=colors[i])
#     plt.vlines(np.mean(fmax_ratio_target_test_re), ymin=0, ymax=100, color=colors[i], linestyles='dashed')
# plt.vlines(0.98, ymin=0, ymax=100, color='black', linestyles='dashed')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
#
# # --------Table 1: n_components determination with F0/F and numerical methods---------
#
# def calculate_split_half_score(dataset, n_components, n_try):
#     score = 0
#     for i in range(n_try):
#         split_set = dataset.splitting(n_split=2, rule='random')
#         model1 = PARAFAC(n_components=n_components)
#         model1.fit(split_set[0])
#         model2 = PARAFAC(n_components=n_components)
#         model2.fit(split_set[1])
#         model2 = align_components_by_loadings({'model2': model2}, model1.ex_loadings, model1.em_loadings)['model2']
#         exl1 = model1.ex_loadings
#         eml1 = model1.em_loadings
#         exl2 = model2.ex_loadings
#         eml2 = model2.em_loadings
#         m_sim_ex = loadings_similarity(exl1, exl2)
#         m_sim_em = loadings_similarity(eml1, eml2)
#         m_sim = (m_sim_ex + m_sim_em) / 2
#         score += np.mean(np.diag(m_sim.to_numpy()))
#     return score / n_try
#
#
# kw_dict_tbl1 = {
#     'july+october': [None, '2024'],
#     'july': ['2024-07', None],
#     'october': ['2024-10', None],
# }
# target_name = 'TCC (million #/mL)'
# fmax_col = 0
# scores_results = {}
# for name, kw in kw_dict_tbl1.items():
#     dataset, _ = eem_dataset.filter_by_index(kw[0], kw[1])
#     results_all_r = {}
#     for r in [3, 4, 5, 6]:
#         model = PARAFAC(n_components=r)
#         model.fit(dataset)
#         cc = model.core_consistency()
#         ve = model.variance_explained()
#         # shs = calculate_split_half_score(dataset, r, 30)
#         target = dataset.ref[target_name]
#         valid_indices = target.index[~target.isna()]
#         target = target.dropna().to_numpy()
#         fmax = model.fmax
#         fmax_original = fmax[fmax.index.str.contains('B1C1')]
#         mask = fmax_original.index.isin(valid_indices)
#         fmax_original = fmax_original[mask]
#         fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
#         fmax_quenched = fmax_quenched[mask]
#         fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#         fmax_ratio_target = fmax_ratio[:, fmax_col]
#         r_target, p_target = pearsonr(target, fmax_original.iloc[:, fmax_col])
#         f0f_mean = np.mean(fmax_ratio_target)
#         f0f_std = np.std(fmax_ratio_target)
#         results_r = {
#             'r_target': r_target,
#             'p_target': p_target,
#             'cc': cc,
#             've': ve,
#             # 'shs': shs,
#             'f0f_mean': f0f_mean,
#             'f0f_std': f0f_std,
#         }
#         results_all_r[r] = results_r
#     scores_results[name] = results_all_r
#
# --------Fig 2: applying established model to quantify new samples: outlier detection-------

dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   # '2024-07-13',
                                                   # '2024-07-15',
                                                   # '2024-07-16',
                                                   # '2024-07-17',
                                                   # '2024-07-18',
                                                   # '2024-07-19',
                                                   '2024'
                                               ]
                                               )
dataset_train_original, _ = dataset_train.filter_by_index(['B1C1'], None)

dataset_test, _ = eem_dataset.filter_by_index(None,
                                              [
                                                  '2024-10-'
                                              ]
                                              )
dataset_test_original, _ = dataset_test.filter_by_index(['B1C1'], None)

indices_test_in_scenarios = {}
for name, kw in kw_dict.items():
    dataset_test_filtered, _ = dataset_test.filter_by_index(kw[0], kw[1], copy=True)
    indices_test_in_scenarios[name] = dataset_test_filtered.index

r = 4
n_outliers = 40
fmax_col = 2
target_name = 'DOC (mg/L)'
model = PARAFAC(n_components=r, init='svd', non_negativity=True,
                tf_normalization=True, sort_components_by_em=True, loadings_normalization='maximum')
model.fit(dataset_train)
plot_loadings({0:model},)

# # # export model
# info_dict = {
#     'name': 'October_5_component',
#     'creator': 'Yongmin Hu',
#     'date': '2025-03',
#     'email': 'yongmin.hu@eawag.ch',
# }
# model.export('C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/model_October_5_component.txt', info_dict)

target_train = dataset_train.ref[target_name]
valid_indices_train = target_train.index[~target_train.isna()]
target_train = target_train.dropna().to_numpy()
fmax_train = model.nnls_fmax

# #----
# # peak-picking
# dataset_train.gaussian_filter(sigma=1, truncate=3, copy=False)
# pp1, _, _ = dataset_train.peak_picking(ex=274, em=323)
# pp2, _, _ = dataset_train.peak_picking(ex=300, em=383)
# pp3, _, _ = dataset_train.peak_picking(ex=336, em=407)
# fmax_train = pd.concat([pp1, pp2, pp3], axis=1)
# #---

fmax_original_train = fmax_train[fmax_train.index.str.contains('B1C1')]
mask_train = fmax_original_train.index.isin(valid_indices_train)
fmax_original_train = fmax_original_train[mask_train]
fmax_quenched_train = fmax_train[fmax_train.index.str.contains('B1C2')]
fmax_quenched_train = fmax_quenched_train[mask_train]
fmax_ratio_train = fmax_original_train.to_numpy() / fmax_quenched_train.to_numpy()
fmax_ratio_target_train = fmax_ratio_train[:, fmax_col]
r_target_train, p_target_train = pearsonr(target_train, fmax_original_train.iloc[:, fmax_col])
reg = LinearRegression(positive=True, fit_intercept=False)
reg.fit(target_train.reshape(-1, 1), fmax_original_train.iloc[:, fmax_col])
slope_train = reg.coef_
intercept_train = reg.intercept_
r2_train = reg.score(target_train.reshape(-1, 1), fmax_original_train.iloc[:, fmax_col])
target_train_pred = (fmax_original_train.iloc[:, fmax_col] - intercept_train) / slope_train
residual_train = np.abs(target_train - target_train_pred)
relative_error_train = abs(target_train - target_train_pred) / target_train * 100

target_test_true = dataset_test.ref[target_name]
valid_indices_test = target_test_true.index[~target_test_true.isna()]
target_test_true = target_test_true.dropna().to_numpy()
_, fmax_test, _ = model.predict(eem_dataset=dataset_test)

# # ----
# dataset_test.gaussian_filter(sigma=1, truncate=3, copy=False)
# pp1, _, _ = dataset_test.peak_picking(ex=274, em=323)
# pp2, _, _ = dataset_test.peak_picking(ex=300, em=383)
# pp3, _, _ = dataset_test.peak_picking(ex=336, em=407)
# fmax_test = pd.concat([pp1, pp2, pp3], axis=1)
# # ----

fmax_original_test = fmax_test[fmax_test.index.str.contains('B1C1')]
mask_test = fmax_original_test.index.isin(valid_indices_test)
fmax_original_test = fmax_original_test[mask_test]
fmax_quenched_test = fmax_test[fmax_test.index.str.contains('B1C2')]
fmax_quenched_test = fmax_quenched_test[mask_test]
fmax_ratio_test = fmax_original_test.to_numpy() / fmax_quenched_test.to_numpy()
fmax_ratio_target_test = fmax_ratio_test[:, fmax_col]
fmax_ratio_target_test_df = pd.DataFrame(fmax_ratio_target_test, index=fmax_original_test.index)
r_target_test, p_target_test = pearsonr(target_test_true, fmax_original_test.iloc[:, fmax_col])
target_test_pred = (fmax_original_test.iloc[:, fmax_col] - intercept_train) / slope_train

residual_test = np.abs(target_test_true - target_test_pred)
relative_error_test = abs(target_test_true - target_test_pred) / target_test_true * 100


# ---------Histplot of F0/F for training and testing, with outliers labeled---------

def round_2d(num, direction):
    if direction == 'up':
        return math.ceil(num * 100) / 100
    elif direction == 'down':
        return math.floor(num * 100) / 100

# ------------fluorescence indices-----------
# eem_dataset_path_bulk = \
#     "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_130_ex_250_em_280_mfem_5_gaussian_1.json"
# eem_dataset_bulk = read_eem_dataset_from_json(eem_dataset_path_bulk)
# eem_dataset_bulk, _ = eem_dataset_bulk.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)
# abs_stack_bulk, ex_range_abs_bulk, _ = read_abs_dataset('C:/PhD/Fluo-detect/_data/_greywater/2024_quenching', ['ABS', 'B1C1'])
# dataset_train_bulk, _ = eem_dataset_bulk.filter_by_index(None,
#                                                [
#                                                    '2024-07-13',
#                                                    '2024-07-15',
#                                                    '2024-07-16',
#                                                    '2024-07-17',
#                                                    '2024-07-18',
#                                                    '2024-07-19',
#                                                ]
#                                                )
# dataset_test_bulk, _ = eem_dataset_bulk.filter_by_index(None,
#                                               [
#                                                   '2024-10-'
#                                               ]
#                                               )
# fmax_ratio_target_train = dataset_train_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 320).to_numpy() / 1e5
# fmax_ratio_target_test = dataset_test_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 320).to_numpy() / 1e5
# fmax_ratio_target_train = dataset_train_bulk.hix()
# fmax_ratio_target_test = dataset_test_bulk.hix()
# fmax_ratio_target_test_df = pd.DataFrame(fmax_ratio_target_test, index=dataset_test_original.index)


# # #------------numerical indicators-------
# _, fmax_train, recon_eem_stack_train = model.predict(dataset_train_original)
# res_train = dataset_train_original.eem_stack - recon_eem_stack_train
# n_pixels = res_train.shape[1] * res_train.shape[2]
# rmse_train = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)
# rmse_train_df = pd.DataFrame(rmse_train, index=fmax_train.index)
# relative_rmse_train = rmse_train / np.average(
#     dataset_train_original.eem_stack,
#     axis=(1, 2)
# )
# _, fmax_test, recon_eem_stack_test = model.predict(dataset_test_original)
# res_test = dataset_test_original.eem_stack - recon_eem_stack_test
# n_pixels = res_test.shape[1] * res_test.shape[2]
# rmse_test = np.sqrt(np.sum(res_test ** 2, axis=(1, 2)) / n_pixels)
# rmse_test_df = pd.DataFrame(rmse_test, index=fmax_test.index)
# relative_rmse_test = rmse_test / np.average(
#     dataset_test_original.eem_stack,
#     axis=(1, 2))
# relative_rmse_test_df = pd.DataFrame(relative_rmse_test, index=fmax_test.index)
# fmax_ratio_target_train = relative_rmse_train
# fmax_ratio_target_test = relative_rmse_test
# fmax_ratio_target_test_df = pd.DataFrame(rmse_test, index=dataset_test_original.index)


binwidth = 0.005
# threshold = round_2d(np.max(fmax_ratio_target_train), 'up')
binrange = (round_2d(np.min(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) - binwidth, axis=0), 'down'),
            round_2d(np.max(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) + binwidth, axis=0), 'up')
            )
# threshold = np.max(fmax_ratio_target_train)
fmax_ratio_train_z_scores = zscore(fmax_ratio_target_train)
filtered_fmax_ratio_target_train = fmax_ratio_target_train[np.abs(fmax_ratio_train_z_scores) <= 2.5]
threshold_upper = np.quantile(filtered_fmax_ratio_target_train, 1)
threshold_lower = np.quantile(filtered_fmax_ratio_target_train, 0)
# threshold = 1.148
# binrange = (np.min(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) - 0.02, axis=0),
#             np.max(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) + 0.02, axis=0)
#             )
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(fmax_ratio_target_train, bins=np.arange(binrange[0], binrange[1] + binwidth, binwidth),
                                density=True, alpha=0.5, color='blue', label='training', zorder=0, edgecolor='black')
# Histogram for outliers (just to show label with hatching)
counts, bins, patches = ax.hist([0], bins=np.arange(binrange[0], binrange[1] + binwidth, binwidth),
                                density=True, alpha=0.5, color='orange', label='test (qualified)',
                                edgecolor='red', zorder=2)
# Histogram for test (qualified) data
counts, bins, patches = ax.hist(fmax_ratio_target_test, bins=np.arange(binrange[0], binrange[1] + binwidth, binwidth),
                                density=True, alpha=0.5, color='orange', label='test (outliers)', zorder=1, edgecolor='black')


for bar in ax.patches:
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
plt.xlabel("C{i} apparent ".format(i=fmax_col + 1) + "$F_{0}/F$", fontsize=20)
# plt.xlabel("$AQY_{320}$", fontsize=20)
plt.xlabel("HIX")
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=16)
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()


# ---------Fmax vs. TCC or DOC in training and testing------------

plt.figure()

reg = LinearRegression(positive=True, fit_intercept=False)
reg.fit(target_train.reshape(-1, 1), fmax_original_train.iloc[:, fmax_col])
a = reg.coef_
b = reg.intercept_

plt.plot(
    [-1, 10],
    a * np.array([-1, 10]) + b,
    '--',
    color='blue',
    label='reg. training'
)
plt.scatter(target_train, fmax_original_train.iloc[:, fmax_col], label='training', color='blue', alpha=0.6)
plt.scatter(target_test_true[(fmax_ratio_target_test < threshold_upper) & (fmax_ratio_target_test > threshold_lower)],
            fmax_original_test.iloc[(fmax_ratio_target_test < threshold_upper) & (fmax_ratio_target_test > threshold_lower), fmax_col],
            label='test (qualified)', color='orange', alpha=0.6)
plt.scatter(target_test_true[(fmax_ratio_target_test >= threshold_upper) | (fmax_ratio_target_test <= threshold_lower)],
            fmax_original_test.iloc[(fmax_ratio_target_test >= threshold_upper) | (fmax_ratio_target_test <= threshold_lower), fmax_col],
            label='test (outliers)', color='red', alpha=0.6)
plt.xlabel(target_name, fontsize=20)
plt.ylabel(f'C{fmax_col + 1} Fmax', fontsize=20)
# plt.ylabel(f'Intensity \n (ex = 336 nm, em = 407 nm)', fontsize=18)
plt.legend(
    bbox_to_anchor=[1.02, 0.37],
    # bbox_to_anchor=[0.58, 0.63],
    fontsize=16
)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.xlim([0, 2.5])
plt.ylim([0, 2500])
plt.show()

# ---------Boxplots of RMSE for training, testing (qualified and outliers)---------

# plt.figure(figsize=(2.2, 4))
plt.figure(figsize=(5, 3.5))
bplot = plt.boxplot(
    [
        relative_error_train,
        relative_error_test[(fmax_ratio_target_test < threshold_upper) & (fmax_ratio_target_test > threshold_lower)],
        relative_error_test[(fmax_ratio_target_test >= threshold_upper) | (fmax_ratio_target_test <= threshold_lower)]
    ],
    labels=('training', 'test (qualified)', 'test (outliers)'),
    patch_artist=True,
    widths=0.75
)
for patch, color in zip(bplot['boxes'], ['blue', 'orange', 'red']):
    patch.set_facecolor(color)
plt.ylabel('relative error (%)', fontsize=16)
plt.tick_params(labelsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



# ---------Table of outlier rates of different indicators----------

metric_dict = {}
metric_dict['apparent $F_{0}/F^{*}$'] = [fmax_ratio_train[:, 2], fmax_ratio_test[:, 2]]
# metric_dict['C2 apparent $F_{0}/F$'] = [fmax_ratio_train[:, 1], fmax_ratio_test[:, 1]]
# metric_dict['C3 apparent $F_{0}/F$'] = [fmax_ratio_train[:, 2], fmax_ratio_test[:, 2]]

#------------fluorescence indices-----------
eem_dataset_path_bulk = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_130_ex_250_em_280_mfem_5_gaussian_1.json"
eem_dataset_bulk = read_eem_dataset_from_json(eem_dataset_path_bulk)
abs_stack_bulk, ex_range_abs_bulk, _ = read_abs_dataset('C:/PhD/Fluo-detect/_data/_greywater/2024_quenching', ['ABS', 'B1C1'])
dataset_train_bulk, _ = eem_dataset_bulk.filter_by_index(None,
                                               [
                                                   '2024-07-13',
                                                   '2024-07-15',
                                                   '2024-07-16',
                                                   '2024-07-17',
                                                   '2024-07-18',
                                                   '2024-07-19',
                                               ]
                                               )
dataset_test_bulk, _ = eem_dataset_bulk.filter_by_index(None,
                                              [
                                                  '2024-10-'
                                              ]
                                              )
aqy254_train = dataset_train_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 254).to_numpy() / 1e5
aqy254_test = dataset_test_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 254).to_numpy() / 1e5

aqy280_train = dataset_train_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 280).to_numpy() / 1e5
aqy280_test = dataset_test_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 280).to_numpy() / 1e5

aqy320_train = dataset_train_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 320).to_numpy() / 1e5
aqy320_test = dataset_test_bulk.aqy(abs_stack_bulk, ex_range_abs_bulk, 320).to_numpy() / 1e5
metric_dict['$AQY_{254}$'] = [aqy254_train, aqy254_test]
metric_dict['$AQY_{280}$'] = [aqy280_train, aqy280_test]
metric_dict['$AQY_{320}$'] = [aqy320_train, aqy320_test]

metric_dict['BIX'] = [dataset_train_bulk.bix().to_numpy(), dataset_test_bulk.bix().to_numpy()]
metric_dict['HIX'] = [dataset_train_bulk.hix().to_numpy(), dataset_test_bulk.hix().to_numpy()]

#------------numerical indicators-------

_, fmax_train, recon_eem_stack_train = model.predict(dataset_train_original)
res_train = dataset_train_original.eem_stack - recon_eem_stack_train
n_pixels = res_train.shape[1] * res_train.shape[2]
rmse_train = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)
relative_rmse_train = rmse_train / np.average(
    dataset_train_original.eem_stack,
    axis=(1, 2)
)
_, fmax_test, recon_eem_stack_test = model.predict(dataset_test_original)
res_test = dataset_test_original.eem_stack - recon_eem_stack_test
n_pixels = res_test.shape[1] * res_test.shape[2]
rmse_test = np.sqrt(np.sum(res_test ** 2, axis=(1, 2)) / n_pixels)
relative_rmse_test = rmse_test / np.average(
    dataset_test_original.eem_stack,
    axis=(1, 2))

metric_dict['RE'] = [rmse_train, rmse_test]
metric_dict['Relative RE'] = [relative_rmse_train, relative_rmse_test]

outlier_rates_by_conditions = []
outlier_rates_by_error_quantiles = []
outlier_rates_by_error_values = []

relative_error_quantile_labels = pd.qcut(relative_error_test, q=10, labels=False)
quantile_groups = [[] for _ in range(10)]
for idx, label in zip(dataset_test_original.index, relative_error_quantile_labels):
    quantile_groups[label].append(idx)
quantile_groups = {f'{q*10}-{q*10+10} quantile': idx for q, idx in enumerate(quantile_groups)}

value_groups = {'0%-25%':[], '25%-50%':[], '50%-100%':[], '>100%':[]}
for idx, e in zip(dataset_test_original.index, relative_error_test):
    # if e <= 25:
    #     value_groups['0%-25%'].append(idx)
    if 25<e<=50:
        value_groups['25%-50%'].append(idx)
    elif 50<e<=100:
        value_groups['50%-100%'].append(idx)
    elif e>100:
        value_groups['>100%'].append(idx)

# value_groups = { '0.25-0.5':[], '0.5-1':[], '>1':[]}
# for idx, e in zip(dataset_test_original.index, residual_test):
#     # if e <= 0.25:
#     #     value_groups['0-0.25'].append(idx)
#     if 0.25<e<=0.5:
#         value_groups['0.25-0.5'].append(idx)
#     elif 0.5<e<=1:
#         value_groups['0.5-1'].append(idx)
#     elif e>1:
#         value_groups['>1'].append(idx)



for name, metric in metric_dict.items():
    metric_train, metric_test = metric
    metric_train_z_scores = zscore(metric_train)
    filtered_metric_train = metric_train[np.abs(metric_train_z_scores) <= 2.5]
    threshold_upper = np.quantile(filtered_metric_train, 1)
    threshold_lower = np.quantile(filtered_metric_train, 0)
    print(threshold_upper)
    outlier_boolean = list((metric_test > threshold_upper) | (metric_test < threshold_lower))
    outlier_indices = [di for di, oi in zip(dataset_test_original.index, outlier_boolean) if oi]
    outlier_rates_by_conditions_i = []
    outlier_rates_by_errors_quantiles_i = []
    outlier_rates_by_error_values_i = []
    for sub_dataset in dataset_divisions.values():
        num_outliers = sum([idx in outlier_indices for idx in sub_dataset.index])
        num_total = len(sub_dataset.index)
        outlier_rates_by_conditions_i.append(num_outliers/num_total*100)
    for q in quantile_groups.values():
        num_outliers = sum([idx in outlier_indices for idx in q])
        num_total = len(q)
        outlier_rates_by_errors_quantiles_i.append(num_outliers / num_total * 100)
    for e in value_groups.values():
        num_outliers = sum([idx in outlier_indices for idx in e])
        num_total = len(e)
        outlier_rates_by_error_values_i.append(num_outliers / num_total * 100)
    outlier_rates_by_conditions.append(outlier_rates_by_conditions_i)
    outlier_rates_by_error_quantiles.append(outlier_rates_by_errors_quantiles_i)
    outlier_rates_by_error_values.append(outlier_rates_by_error_values_i)

outlier_rates_by_conditions = pd.DataFrame(outlier_rates_by_conditions, index=list(metric_dict.keys()), columns=list(dataset_divisions.keys()))
outlier_rates_by_error_quantiles = pd.DataFrame(outlier_rates_by_error_quantiles, index=list(metric_dict.keys()), columns=list(quantile_groups.keys()))
outlier_rates_by_error_values = pd.DataFrame(outlier_rates_by_error_values, index=list(metric_dict.keys()), columns=list(value_groups.keys()))



data = outlier_rates_by_error_values
viridis = plt.cm.get_cmap('viridis', 10)  # 10 discrete colors
colors = viridis(np.arange(10))
cmap = ListedColormap(colors)
bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap=cmap, norm=norm)

# Tick at every pixel without labels
ax.xaxis.set_ticks_position('top')
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_yticklabels(data.index)
ax.set_xticklabels(data.columns, rotation=90)

# Grid at pixel borders
ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
ax.tick_params(axis='both', which='major', length=8, labelsize=14)
ax.tick_params(axis='both', which='minor', length=0)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
ax.grid(which='major', visible=False)
# Add discrete colorbar
# cbar = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=np.arange(len(colors)))
# cbar.ax.set_yticklabels([str(i) for i in range(len(colors))])  # Optional: custom labels
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax)
cb.set_label('Outlier rate (%)', fontsize=18)

# Set ticks and labels at midpoints of each interval
tick_positions = [bounds[i] for i in range(len(bounds))]
tick_labels = [f'{bounds[i]}' for i in range(len(bounds))]
cb.set_ticks(tick_positions)
cb.ax.tick_params(labelsize=14)
cb.set_ticklabels(tick_labels)
plt.tight_layout()
plt.show()




#------------
model = PARAFAC(n_components=4)
model.fit(dataset_train)

for fmax_col, target_name in zip([0, 1, 2], ['TCC (million #/mL)', 'DOC (mg/L)', 'DOC (mg/L)']):
    fmax_original_train = fmax_train[fmax_train.index.str.contains('B1C1')]
    mask_train = fmax_original_train.index.isin(valid_indices_train)
    fmax_original_train = fmax_original_train[mask_train]
    fmax_quenched_train = fmax_train[fmax_train.index.str.contains('B1C2')]
    fmax_quenched_train = fmax_quenched_train[mask_train]
    fmax_ratio_train = fmax_original_train.to_numpy() / fmax_quenched_train.to_numpy()
    fmax_ratio_target_train = fmax_ratio_train[:, fmax_col]
    r_target_train, p_target_train = pearsonr(target_train, fmax_original_train.iloc[:, fmax_col])
    slope_train, intercept_train = np.polyfit(target_train, fmax_original_train.iloc[:, fmax_col], deg=1)
    target_train_pred = (fmax_original_train.iloc[:, fmax_col] - intercept_train) / slope_train
    residual_train = np.abs(target_train - target_train_pred)
    relative_error_train = abs(target_train - target_train_pred) / target_train * 100
    target_test_true = dataset_test.ref[target_name]
    valid_indices_test = target_test_true.index[~target_test_true.isna()]
    target_test_true = target_test_true.dropna().to_numpy()
    _, fmax_test, recon_eem_stack_test = model.predict(eem_dataset=dataset_test)
    fmax_original_test = fmax_test[fmax_test.index.str.contains('B1C1')]
    mask_test = fmax_original_test.index.isin(valid_indices_test)
    fmax_original_test = fmax_original_test[mask_test]
    fmax_quenched_test = fmax_test[fmax_test.index.str.contains('B1C2')]
    fmax_quenched_test = fmax_quenched_test[mask_test]
    fmax_ratio_test = fmax_original_test.to_numpy() / fmax_quenched_test.to_numpy()
    fmax_ratio_target_test = fmax_ratio_test[:, fmax_col]
    fmax_ratio_target_test_df = pd.DataFrame(fmax_ratio_target_test, index=fmax_original_test.index)
    r_target_test, p_target_test = pearsonr(target_test_true, fmax_original_test.iloc[:, fmax_col])
    slope_test, intercept_test = np.polyfit(target_test_true, fmax_original_test.iloc[:, fmax_col], deg=1)
    target_test_pred = (fmax_original_test.iloc[:, fmax_col] - intercept_train) / slope_train
    residual_test = np.abs(target_test_true - target_test_pred)
    relative_error_test = abs(target_test_true - target_test_pred) / target_test_true * 100
    fmax_ratio_train_z_scores = zscore(fmax_ratio_target_train)
    filtered_fmax_ratio_target_train = fmax_ratio_target_train[np.abs(fmax_ratio_train_z_scores) <= 2.5]
    threshold_upper = np.quantile(filtered_fmax_ratio_target_train, 1)
    threshold_lower = np.quantile(filtered_fmax_ratio_target_train, 0)
    recon_eem_stack_train = model.eem_stack_reconstructed
    res_train = dataset_train.eem_stack - recon_eem_stack_train
    n_pixels = recon_eem_stack_train.shape[1] * recon_eem_stack_train.shape[2]
    if fmax_col == 0:
        res_train, _ = process_eem_stack(res_train, eem_cutting,
                                        ex_range_old=dataset_test.ex_range,
                                        em_range_old=dataset_test.em_range,
                                        ex_min_new=274, # 274   274   300
                                        ex_max_new=300, # 300   340   370
                                        em_min_new=310, # 310   330   365
                                        em_max_new=370, # 370   450   470
                                        )
    elif fmax_col == 1:
        res_train, _ = process_eem_stack(res_train, eem_cutting,
                                        ex_range_old=dataset_test.ex_range,
                                        em_range_old=dataset_test.em_range,
                                        ex_min_new=274, # 274   274   300
                                        ex_max_new=340, # 300   340   370
                                        em_min_new=330, # 310   330   365
                                        em_max_new=470, # 370   450   470
                                        )
    elif fmax_col == 2:
        res_train, _ = process_eem_stack(res_train, eem_cutting,
                                        ex_range_old=dataset_test.ex_range,
                                        em_range_old=dataset_test.em_range,
                                        ex_min_new=300, # 274   274   300
                                        ex_max_new=370, # 300   340   370
                                        em_min_new=365, # 310   330   365
                                        em_max_new=470, # 370   450   470
                                        )
    rmse_train = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)
    rmse_train_df = pd.DataFrame(rmse_train, index=fmax_train.index)
    relative_rmse_train = rmse_train / np.average(dataset_train.eem_stack, axis=(1, 2))
    res_test = dataset_test.eem_stack - recon_eem_stack_test
    if fmax_col == 0:
        res_train, _ = process_eem_stack(res_train, eem_cutting,
                                         ex_range_old=dataset_test.ex_range,
                                         em_range_old=dataset_test.em_range,
                                         ex_min_new=274,  # 274   274   300
                                         ex_max_new=300,  # 300   340   370
                                         em_min_new=310,  # 310   330   365
                                         em_max_new=370,  # 370   450   470
                                         )
    n_pixels = recon_eem_stack_test.shape[1] * recon_eem_stack_test.shape[2]
    rmse_test = np.sqrt(np.sum(res_test ** 2, axis=(1, 2)) / n_pixels)
    rmse_test_df = pd.DataFrame(rmse_test, index=fmax_test.index)
    relative_rmse_test = rmse_test / np.average(dataset_test.eem_stack, axis=(1, 2))



# plt.figure(figsize=(2.3, 4))
# bplot = plt.boxplot(
#     [
#         residual_train,
#         residual_test[fmax_ratio_target_test <= threshold],
#         residual_test[fmax_ratio_target_test > threshold]
#     ],
#     labels=('training', 'test (qualified)', 'test (outliers)'),
#     patch_artist=True,
#     widths=0.75
# )
# for patch, color in zip(bplot['boxes'], ['blue', 'orange', 'red']):
#     patch.set_facecolor(color)
# plt.ylabel('absolute residual\n (million #/mL)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
#
# # ---------Histplot of outlier rates in each scenario---------
# # outlier_indices = fmax_original_test.index[fmax_ratio_target_test > threshold]
# # outlier_rates = {}
# # for name, indices in indices_test_in_scenarios.items():
# #     n_outliers = 0
# #     for idx in indices:
# #         if idx in outlier_indices:
# #             n_outliers += 1
# #     outlier_rates[name] = (n_outliers / len(indices) * 100 * 2) if indices else 0
# # fig, ax = plt.subplots(figsize=(8, 5))
# # ax.bar(list(outlier_rates.keys())[2:], list(outlier_rates.values())[2:], color='red')
# # ax.set_ylim([0, 100])
# # ax.set_ylabel('Outlier rate (%)', fontsize=18)
# # ax.tick_params(labelsize=20)
# # fig.tight_layout()
# # fig.show()
#
# # ---------Fig 3: timeseries of anomalies detected-----------

time = [datetime.strptime(t[0:16], '%Y-%m-%d-%H-%M') for t in fmax_original_train.index]
sampling_point_labels_dict = {
    'GAC top': 'G1',
    'GAC middle': 'G2',
    'GAC bottom': 'G3',
    'GAC effluent': 'M3',
}
sampling_point_labels = []
for idx in fmax_original_train.index:
    for label, kw in sampling_point_labels_dict.items():
        if kw in idx:
            sampling_point_labels.append(label)
            break

markers = []
for i in sampling_point_labels:
    if i == 'GAC top':
        markers.append('^')
    elif i == 'GAC middle':
        markers.append('s')
    elif i == 'GAC bottom':
        markers.append('v')
    elif i == 'GAC effluent':
        markers.append('o')

df = pd.DataFrame(
    {
        target_name: target_train,
        f'C{fmax_col + 1} Fmax': fmax_original_train.iloc[:, fmax_col].to_numpy(),
        f'C{fmax_col + 1} ' + '$F_{0}/F$': fmax_ratio_target_train,
        'sampling point': sampling_point_labels,
        'markers': markers,
        'is_outlier': fmax_ratio_target_train > threshold

        # target_name: target_test_true,
        # f'C{fmax_col + 1} Fmax': fmax_original_test.iloc[:, fmax_col].to_numpy(),
        # f'C{fmax_col + 1} ' + '$F_{0}/F$': fmax_ratio_target_test,
        # 'sampling point': sampling_point_labels,
        # 'markers': markers,
        # 'is_outlier': fmax_ratio_target_test > threshold
    },
    index=time,
)

# df.loc[datetime(2024, 7, 13, 11, 20)] = [-1000, -1000, -1000, 'GAC effluent', 'o', False]
# df.loc[datetime(2024, 7, 13, 14, 54)] = [-1000, -1000, -1000, 'GAC effluent', 'o', False]

# Automatically detect unique days
unique_days = df.index.normalize().unique().sort_values()
n_days = len(unique_days)

# Create subplots with adjusted spacing
fig, axes = plt.subplots(1, n_days, figsize=(max(3.5 * n_days, 8), 4),
                         gridspec_kw={'wspace': 0, 'right': 0.8})
if n_days == 1:
    axes = [axes]

# Configure plot styles
plot_config = {
    f'C{fmax_col + 1} ' + '$F_{0}/F$': {'color': '#1f77b4', 'marker': 'o'},
    target_name: {'color': '#2ca02c', 'marker': '^'},
    f'C{fmax_col + 1} Fmax': {'color': '#ff7f0e', 'marker': 's'},
}

# Create shared axis system
main_ax = axes[0]
main_ax.yaxis.set_label_position('left')
twin_right1 = main_ax.twinx()
twin_right2 = main_ax.twinx()
twin_right2.spines.right.set_position(("axes", 1.3))
twin_right2.spines.right.set_color((0, 0, 0, 0))

# Configure main axis colors
main_ax.yaxis.label.set_color(plot_config[f'C{fmax_col + 1} ' + '$F_{0}/F$']['color'])
twin_right1.yaxis.label.set_color((0, 0, 0, 0))
twin_right2.yaxis.label.set_color((0, 0, 0, 0))
main_ax.tick_params(axis='y', colors=plot_config[f'C{fmax_col + 1} ' + '$F_{0}/F$']['color'], labelsize=14)
twin_right1.tick_params(axis='y', colors=(0, 0, 0, 0))
twin_right2.tick_params(axis='y', colors=(0, 0, 0, 0))

# Plot data
for i, day in enumerate(unique_days):
    ax = axes[i]
    daily_data = df[df.index.normalize() == day]

    ax.sharey(main_ax)

    # Create shared right twins
    tr1 = ax.twinx()
    tr2 = ax.twinx()
    tr1.sharey(twin_right1)
    tr2.sharey(twin_right2)
    tr2.spines.right.set_position(("axes", 1.35))
    ax.set_xlabel(day.strftime('%Y-%m-%d'), fontsize=12)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # Hide right axes for non-last plots
    if i != 0:
        # ax.yaxis.set_visible(False)
        ax.yaxis.label.set_color((0, 0, 0, 0))
        ax.tick_params(axis='y', colors=(0, 0, 0, 0))
    if i != len(axes) - 1:
        tr1.set_ylabel('')
        tr2.set_ylabel('')
        tr1.yaxis.label.set_color((0, 0, 0, 0))
        tr2.yaxis.label.set_color((0, 0, 0, 0))
        tr1.tick_params(axis='y', colors=(0, 0, 0, 0))
        tr2.tick_params(axis='y', colors=(0, 0, 0, 0))
        tr2.spines.right.set_color((0, 0, 0, 0))
    else:
        tr1.set_ylabel(target_name, fontsize=18)
        tr2.set_ylabel(f'C{fmax_col + 1} Fmax', fontsize=18)
        tr1.yaxis.label.set_color(plot_config[target_name]['color'])
        tr2.yaxis.label.set_color(plot_config[f'C{fmax_col + 1} Fmax']['color'])
        tr1.tick_params(axis='y', colors=plot_config[target_name]['color'], labelsize=14)
        tr2.tick_params(axis='y', colors=plot_config[f'C{fmax_col + 1} Fmax']['color'], labelsize=14)

    # Plot variables
    p1, = ax.plot(daily_data.index, daily_data[f'C{fmax_col + 1} ' + '$F_{0}/F$'],
                  color=plot_config[f'C{fmax_col + 1} ' + '$F_{0}/F$']['color'],
                  linestyle='-', label=f'C{fmax_col + 1} ' + '$F_{0}/F$')

    p2, = tr1.plot(daily_data.index, daily_data[target_name],
                   color=plot_config[target_name]['color'],
                   linestyle='-', label=target_name)

    p3, = tr2.plot(daily_data.index, daily_data[f'C{fmax_col + 1} Fmax'],
                   color=plot_config[f'C{fmax_col + 1} Fmax']['color'],
                   linestyle='-', label=f'C{fmax_col + 1} Fmax')
    for j, m in enumerate(daily_data['markers'].to_list()):
        ax.plot(daily_data.index[j], daily_data[f'C{fmax_col + 1} ' + '$F_{0}/F$'].iloc[j], markersize=8,
                markeredgecolor='black',
                marker=m, linestyle='', color=plot_config[f'C{fmax_col + 1} ' + '$F_{0}/F$']['color'])
        tr1.plot(daily_data.index[j], daily_data[target_name].iloc[j], markersize=8, markeredgecolor='black',
                 marker=m, linestyle='', color=plot_config[target_name]['color'])
        tr2.plot(daily_data.index[j], daily_data[f'C{fmax_col + 1} Fmax'].iloc[j], markersize=8,
                 markeredgecolor='black',
                 marker=m, linestyle='', color=plot_config[f'C{fmax_col + 1} Fmax']['color'])
        # if daily_data['is_outlier'].iloc[j]:
        #     gap_left = (daily_data.index[j] - daily_data.index[j - 1]) / 2 if j != 0 else pd.Timedelta(hours=0.5)
        #     gap_right = (daily_data.index[j + 1] - daily_data.index[j]) / 2 if j != len(
        #         daily_data['markers'].to_list()) - 1 else pd.Timedelta(hours=0.5)
        #     ax.axvspan(daily_data.index[j] - gap_left, daily_data.index[j] + gap_right, color='red', alpha=0.3)

    # Format subplot
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.grid(alpha=0.3, axis='x')

    buffer = pd.Timedelta(hours=1.25)
    # Set dynamic xlim with buffer
    if not daily_data.empty:
        ax.set_xlim(daily_data.index.min() - buffer,
                    daily_data.index.max() + buffer)
        ax.xaxis.set_major_locator(HourLocator(interval=2))
        l1 = ax.hlines(xmin=daily_data.index.min() - buffer, xmax=daily_data.index.max() + buffer,
                       y=threshold, color=plot_config[f'C{fmax_col + 1} ' + '$F_{0}/F$']['color'],
                       linestyles='--')
        # if i == 0:
        #     t1 = ax.text(x=daily_data.index.min() - buffer + pd.Timedelta(hours=0.5),
        #                  y=threshold + 0.02, s='$F_{0}/F=$' + f'{threshold:.2f}',
        #                  c=plot_config[f'C{fmax_col + 1} ' + '$F_{0}/F$']['color'], fontsize=14)
        outlier_rate = np.sum(daily_data['is_outlier']) / daily_data.shape[0] * 100
        outlier_rate_eff = np.sum(daily_data['is_outlier'][daily_data['sampling point'] == 'GAC effluent']) / np.sum(
            daily_data['sampling point'] == 'GAC effluent') * 100
        outlier_rate_col = np.sum(daily_data['is_outlier'][daily_data['sampling point'] != 'GAC effluent']) / np.sum(
            daily_data['sampling point'] != 'GAC effluent') * 100
        # if not np.isnan(outlier_rate_col):
        #     t2 = ax.text(1, 0.99, '$OR_{col}$=' + f'{outlier_rate_col:.2f}%', transform=ax.transAxes,
        #                  fontsize=14, color='black', verticalalignment='top', horizontalalignment='right')
        # if not np.isnan(outlier_rate_eff):
        #     t3 = ax.text(1, 0.91, '$OR_{eff}$=' + f'{outlier_rate_eff:.2f}%', transform=ax.transAxes,
        #                  fontsize=14, color='black', verticalalignment='top', horizontalalignment='right')
    tr1.set_ylim([0, 2.5])
    tr2.set_ylim([0, 2500])

# Configure axis labels
main_ax.set_ylabel(f'C{fmax_col + 1} ' + '$F_{0}/F$', fontsize=18)
main_ax.set_ylim([0.95, 1.35])

# Final adjustments
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.show()
#
# # ----------create legends--------
# plt.figure()
# plt.plot([0, 0], [1, 1], label='$F_{0}/F$', color='#1f77b4')
# plt.plot([0, 0], [1, 1], label='TCC or DOC', color='#2ca02c')
# plt.plot([0, 0], [1, 1], label='Fmax', color='#ff7f0e')
# plt.legend(ncol=3, title='variables', fontsize=12, title_fontsize=12, frameon=True)
# plt.show()
#
# plt.figure()
# plt.plot([0, 0], [1, 1], marker='^', markersize=6, markeredgecolor='black', color='white', label='BAC top')
# plt.plot([0, 0], [1, 1], marker='s', markersize=6, markeredgecolor='black', color='white', label='BAC middle')
# plt.plot([0, 0], [1, 1], marker='v', markersize=6, markeredgecolor='black', color='white', label='BAC bottom')
# plt.plot([0, 0], [1, 1], marker='o', markersize=6, markeredgecolor='black', color='white', label='BAC effluent')
# plt.legend(ncol=4, title='sampling point', fontsize=10, title_fontsize=10, frameon=True, bbox_to_anchor=(1, -0.2))
# plt.tight_layout()
# plt.show()

# ----------Other numerical outlier detection methods: rmse and leverage--------

dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   '2024-07-13',
                                                   '2024-07-15',
                                                   '2024-07-16',
                                                   '2024-07-17',
                                                   '2024-07-18',
                                                   '2024-07-19',
                                               ]
                                               )

dataset_test, _ = eem_dataset.filter_by_index(['B1C1'],
                                              [
                                                  '2024-10-'
                                              ]
                                              )

model = PARAFAC(n_components=4, loadings_normalization='maximum', tf_normalization=True)
model.fit(dataset_train)
fmax_train = model.nnls_fmax
leverage_train = model.leverage()
recon_eem_stack_train = model.eem_stack_reconstructed
res_train = dataset_train.eem_stack - recon_eem_stack_train
# plot_eem(recon_eem_stack_train[0], ex_range=dataset_train.ex_range, em_range=dataset_train.em_range, auto_intensity_range=False, vmin=0, vmax=800)
# res_train, _ = process_eem_stack(res_train, eem_rayleigh_scattering_removal, ex_range=dataset_train.ex_range, em_range=dataset_train.em_range)
# res_train = process_eem_stack(res_train, eem_gaussian_filter, sigma=1, truncate=3)
res_train, _ = process_eem_stack(res_train, eem_cutting,
                                ex_range_old=dataset_train.ex_range,
                                em_range_old=dataset_train.em_range,
                                ex_min_new=274, # 274   274   300
                                ex_max_new=294, # 294   332   366
                                em_min_new=310, # 310   338   367
                                em_max_new=469, # 369   457   466
                                )
dataset_train_cut, _ = process_eem_stack(dataset_train.eem_stack, eem_cutting,
                                ex_range_old=dataset_train.ex_range,
                                em_range_old=dataset_train.em_range,
                                ex_min_new=274, # 274   274   300
                                ex_max_new=294, # 294   332   366
                                em_min_new=310, # 310   338   367
                                em_max_new=469, # 369   457   466
                                )
n_pixels = res_train.shape[1] * res_train.shape[2]
rmse_train = np.sqrt(np.sum(res_train ** 2, axis=(1, 2)) / n_pixels)
rmse_train_df = pd.DataFrame(rmse_train, index=fmax_train.index)
relative_rmse_train = rmse_train / np.average(
    dataset_train_cut,
    # dataset_train.eem_stack,
    axis=(1, 2)
)


# rmse_train = model.sample_rmse().to_numpy().reshape(-1)
# relative_rmse_train = model.sample_relative_rmse().to_numpy().reshape(-1)

# ------------rmse and relative rmse----------
_, fmax_test, recon_eem_stack_test = model.predict(dataset_test)
res_test = dataset_test.eem_stack - recon_eem_stack_test
# res_test, _ = process_eem_stack(res_test, eem_rayleigh_scattering_removal, ex_range=dataset_test.ex_range, em_range=dataset_test.em_range)
# res_test = process_eem_stack(res_test, eem_gaussian_filter, sigma=1, truncate=3)
res_test, _ = process_eem_stack(res_test, eem_cutting,
                                ex_range_old=dataset_test.ex_range,
                                em_range_old=dataset_test.em_range,
                                ex_min_new=274, # 274   274   300
                                ex_max_new=294, # 294   332   366
                                em_min_new=310, # 310   338   367
                                em_max_new=469, # 369   457   466
                                )
dataset_test_cut, _ = process_eem_stack(dataset_test.eem_stack, eem_cutting,
                                ex_range_old=dataset_test.ex_range,
                                em_range_old=dataset_test.em_range,
                                ex_min_new=274, # 274   274   300
                                ex_max_new=294, # 294   332   366
                                em_min_new=310, # 310   338   367
                                em_max_new=469, # 369   457   466
                                )
n_pixels = res_test.shape[1] * res_test.shape[2]
rmse_test = np.sqrt(np.sum(res_test ** 2, axis=(1, 2)) / n_pixels)
rmse_test_df = pd.DataFrame(rmse_test, index=fmax_test.index)
relative_rmse_test = rmse_test / np.average(
    dataset_test_cut,
    # dataset_test.eem_stack,
    axis=(1, 2))
relative_rmse_test_df = pd.DataFrame(relative_rmse_test, index=fmax_test.index)

# target_name = 'TCC (million #/mL)'
target_name = 'TCC (million #/mL)'
fmax_col = 0
target_train = dataset_train.ref[target_name]
valid_indices_train = target_train.index[~target_train.isna()]
mask_train = fmax_train.index.isin(valid_indices_train)
target_train = target_train.dropna().to_numpy()
fmax_train = fmax_train.iloc[mask_train, :]
target_test_true = dataset_test.ref[target_name]
slope_train, intercept_train = np.polyfit(target_train, fmax_train.iloc[:, fmax_col], deg=1)
target_train_pred = (fmax_train.iloc[:, fmax_col] - intercept_train) / slope_train
residual_train = np.abs(target_train - target_train_pred)
relative_error_train = abs(target_train - target_train_pred) / target_train * 100
target_test_pred = (fmax_test.iloc[:, fmax_col] - intercept_train) / slope_train
residual_test = np.abs(target_test_true - target_test_pred)
relative_error_test = abs(target_test_true - target_test_pred) / target_test_true * 100

# plot_eem(dataset_train.eem_stack[1], ex_range=dataset_train.ex_range, em_range=dataset_train.em_range)
# plot_eem(model.eem_stack_reconstructed[1], ex_range=dataset_train.ex_range, em_range=dataset_train.em_range)

# # ------------leverage---------
# indices_test = dataset_test.index
# leverage_test = {}
# for idx in indices_test:
#     one_sample_dataset, _ = dataset_test.filter_by_index([idx], None)
#     new_dataset = combine_eem_datasets([dataset_train, one_sample_dataset])
#     model = PARAFAC(n_components=4)
#     model.fit(new_dataset)
#     leverage = model.leverage()
#     leverage_test[one_sample_dataset.index[0]] = leverage


# -----------boxplots of training and testing---------
def round_2d(num, direction):
    if direction == 'up':
        return math.ceil(num * 100) / 100
    elif direction == 'down':
        return math.floor(num * 100) / 100


def round_0d(num, direction):
    if direction == 'up':
        return math.ceil(num)
    elif direction == 'down':
        return math.floor(num)


# -----------rmse---------
indicator_train = rmse_train
indicator_test = rmse_test
indicator_train_z_scores = zscore(indicator_train)
filtered_indicator_train = indicator_train[np.abs(indicator_train_z_scores) <= 2.5]
threshold = np.quantile(filtered_indicator_train, 1)
binwidth = 5
binrange = (threshold - binwidth * np.ceil((threshold - np.min(indicator_train))/binwidth),
            threshold + binwidth * np.ceil((np.max(indicator_test) - threshold)/binwidth)
            )
plt.figure()
ax = sns.histplot(indicator_train, binwidth=binwidth, binrange=binrange, kde=False, stat='density', color="blue",
                  alpha=0.5, label='training', zorder=0)
sns.histplot(indicator_test, binwidth=binwidth, binrange=binrange, kde=False, stat='density', color="orange",
             alpha=0.5, label='test (qualified)', zorder=1)
sns.histplot([-100], binwidth=binwidth, binrange=binrange, kde=True, stat='density', color="orange",
             alpha=0.5, label='test (outliers)', hatch='////', edgecolor='red')
for bar in ax.patches:
    # Calculate the midpoint of the bin
    bin_left = bar.get_x()
    bin_width = bar.get_width()
    bin_mid = bin_left + bin_width / 2
    print(bin_left)

    # Check if the midpoint is above the threshold
    if threshold-0.0001 <= bin_left and bar.zorder == 1:
        # Add hatch pattern and change color
        bar.set_hatch("////")  # Hatch pattern (e.g., "////", "xxx", "..")
        bar.set_edgecolor('red')
plt.xlim(binrange)
plt.xlabel("Relative reconstruction error", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=16, loc='upper right')
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()


plt.figure()
a, b = np.polyfit(target_train, fmax_train.iloc[:, fmax_col], deg=1)
plt.plot(
    [-1, 10],
    a * np.array([-1, 10]) + b,
    '--',
    color='blue',
    label='reg. training'
)
plt.scatter(target_train, fmax_train.iloc[:, fmax_col], label='training', color='blue', alpha=0.6)
plt.scatter(target_test_true[indicator_test <= threshold],
            fmax_test.iloc[indicator_test <= threshold, fmax_col],
            label='test (qualified)', color='orange', alpha=0.6)
plt.scatter(target_test_true[indicator_test > threshold],
            fmax_test.iloc[indicator_test > threshold, fmax_col],
            label='test (outliers)', color='red', alpha=0.6)
plt.xlabel(target_name, fontsize=20)
plt.ylabel(f'C{fmax_col + 1} Fmax', fontsize=20)
plt.legend(
    # bbox_to_anchor=[1.02, 0.37],
    # bbox_to_anchor=[0.58, 0.63],
    fontsize=16
)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.xlim([0, 2.5])
plt.ylim([0, 2500])
plt.show()


# plt.figure(figsize=(2.3, 4))
# bplot = plt.boxplot(
#     [
#         relative_error_train,
#         relative_error_test[indicator_test <= threshold],
#         relative_error_test[indicator_test > threshold]
#     ],
#     labels=('training', 'test (qualified)', 'test (outliers)'),
#     patch_artist=True,
#     widths=0.75
# )
# for patch, color in zip(bplot['boxes'], ['blue', 'orange', 'red']):
#     patch.set_facecolor(color)
# plt.ylabel('relative error (%)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(2.3, 4))
# bplot = plt.boxplot(
#     [
#         residual_train,
#         residual_test[indicator_test <= threshold],
#         residual_test[indicator_test > threshold]
#     ],
#     labels=('training', 'test (qualified)', 'test (outliers)'),
#     patch_artist=True,
#     widths=0.75
# )
# for patch, color in zip(bplot['boxes'], ['blue', 'orange', 'red']):
#     patch.set_facecolor(color)
# plt.ylabel('absolute residual\n (million #/mL)', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()

# ----------relative rmse----------
indicator_train = relative_rmse_train
indicator_test = relative_rmse_test
# threshold = round_2d(np.max(indicator_train), 'up')
binrange = (round_2d(np.min(np.concatenate([indicator_train, indicator_test]) - 0.05, axis=0), 'down'),
            round_2d(np.max(np.concatenate([indicator_train, indicator_test]) + 0.05, axis=0), 'up')
            )
threshold = np.max(indicator_train)
# threshold = np.quantile(indicator_train, 0.95)
# binrange = (np.min(np.concatenate([indicator_train, indicator_test]) - 0.02, axis=0),
#             np.max(np.concatenate([indicator_train, indicator_test]) + 0.02, axis=0)
#             )
plt.figure()
ax = sns.histplot(indicator_train, binwidth=0.025, binrange=binrange, kde=False, stat='density', color="blue",
                  alpha=0.5, label='training', zorder=0)
sns.histplot(indicator_test, binwidth=0.025, binrange=binrange, kde=False, stat='density', color="orange",
             alpha=0.5, label='test (qualified)', zorder=1)
sns.histplot([-100], binwidth=0.025, binrange=binrange, kde=True, stat='density', color="orange",
             alpha=0.5, label='test (outliers)', hatch='////', edgecolor='red')
for bar in ax.patches:
    # Calculate the midpoint of the bin
    bin_left = bar.get_x()
    bin_width = bar.get_width()
    bin_mid = bin_left + bin_width / 2

    # Check if the midpoint is above the threshold
    if threshold <= bin_mid - bin_width and bar.zorder == 1:
        # Add hatch pattern and change color
        bar.set_hatch("////")  # Hatch pattern (e.g., "////", "xxx", "..")
        bar.set_edgecolor('red')
plt.xlim(binrange)
plt.xlabel("Relative EEM-RMSE", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=16, loc='upper right')
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()
#
# #----------leverage---------
# indicator_train = leverage_train.to_numpy().reshape(-1)
# indicator_test = np.array(leverage_test)

# plt.figure()
# for sample_idx, lvg in leverage_test.items():
#     lvg_train_samples = lvg.iloc[:-1]
#     plt.plot(np.max(lvg_train_samples), lvg.iloc[-1], 'o', color='black')
# plt.plot([0, 1], [0, 1], color='black', linestyle='--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel('Maximum leverage among training samples', fontsize=16)
# plt.ylabel('Leverage of the test sample', fontsize=16)
# plt.tick_params(labelsize=12)
# plt.tight_layout()
# plt.show()

#--------multi-variate regression for DOC--------

dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   '2024-07-13',
                                                   '2024-07-15',
                                                   '2024-07-16',
                                                   '2024-07-17',
                                                   '2024-07-18',
                                                   '2024-07-19',
                                               ]
                                               )

dataset_test, _ = eem_dataset.filter_by_index(None,
                                              [
                                                  '2024-10-'
                                              ]
                                              )

indices_test_in_scenarios = {}

r = 4
n_outliers = 40
fmax_col = [0, 1, 2, 3]
target_name = 'DOC (mg/L)'
model = PARAFAC(n_components=r, init='svd', non_negativity=True,
                tf_normalization=True, sort_components_by_em=True, loadings_normalization='maximum')
model.fit(dataset_train)

target_train = dataset_train.ref[target_name]
valid_indices_train = target_train.index[~target_train.isna()]
target_train = target_train.dropna().to_numpy()
fmax_train = model.nnls_fmax
fmax_original_train = fmax_train[fmax_train.index.str.contains('B1C1')]
mask_train = fmax_original_train.index.isin(valid_indices_train)
fmax_original_train = fmax_original_train[mask_train]
fmax_quenched_train = fmax_train[fmax_train.index.str.contains('B1C2')]
fmax_quenched_train = fmax_quenched_train[mask_train]
# fmax_ratio_train = fmax_original_train.to_numpy() / fmax_quenched_train.to_numpy()
# fmax_ratio_target_train = fmax_ratio_train[:, fmax_col]
# r_target_train, p_target_train = pearsonr(target_train, fmax_original_train.iloc[:, fmax_col])
reg = LinearRegression(positive=True)
reg.fit(fmax_original_train.iloc[:, fmax_col], target_train)
weights = reg.coef_
intercept = reg.intercept_
target_train_pred = reg.predict(fmax_original_train.iloc[:, fmax_col])
r_target_train, p_target_train = pearsonr(target_train, target_train_pred)
residual_train = np.abs(target_train - target_train_pred)
relative_error_train = abs(target_train - target_train_pred) / target_train * 100

target_test_true = dataset_test.ref[target_name]
valid_indices_test = target_test_true.index[~target_test_true.isna()]
target_test_true = target_test_true.dropna().to_numpy()
_, fmax_test, _ = model.predict(eem_dataset=dataset_test)
fmax_original_test = fmax_test[fmax_test.index.str.contains('B1C1')]
mask_test = fmax_original_test.index.isin(valid_indices_test)
fmax_original_test = fmax_original_test[mask_test]
fmax_quenched_test = fmax_test[fmax_test.index.str.contains('B1C2')]
fmax_quenched_test = fmax_quenched_test[mask_test]
target_test_pred = reg.predict(fmax_original_test.iloc[:, fmax_col])
r_target_test, p_target_test = pearsonr(target_test_true, target_test_pred)

residual_test = np.abs(target_test_true - target_test_pred)
relative_error_test = abs(target_test_true - target_test_pred) / target_test_true * 100

plt.figure()
a, b = np.polyfit(target_train, target_train_pred, deg=1)
plt.plot(
    [-1, 10],
    a * np.array([-1, 10]) + b,
    '--',
    color='blue',
    label='reg. training'
)
plt.scatter(target_train, target_train_pred, label='training', color='blue', alpha=0.6)
plt.scatter(target_test_true,
            target_test_pred, label='test', color='red', alpha=0.6)
plt.xlabel('True ' + target_name, fontsize=20)
plt.ylabel(f'Predicted DOC (mg/L)', fontsize=20)
plt.legend(
    # bbox_to_anchor=[1.02, 0.37],
    # bbox_to_anchor=[0.58, 0.63],
    fontsize=16
)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.xlim([0, 2.5])
# plt.ylim([0, 2500])
plt.show()
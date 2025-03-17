import pandas as pd

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import *
from scipy.stats import pearsonr
from matplotlib.colors import TABLEAU_COLORS
from datetime import datetime
import seaborn as sns
from matplotlib.dates import DateFormatter, HourLocator

colors = list(TABLEAU_COLORS.values())

# ------------Read EEM dataset-------------
eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_286_ex_274_em_310_mfem_7_gaussian.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['M3', 'G1', 'G2', 'G3'], copy=True)

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

# # --------N_components vs. r_tcc and mean(F0/F)--------
# # --------Fig 1: Histplots of model on testing dataset: effect of N components-------
dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   '2024-'
                                               ]
                                               )

r_list = [4, 5, 6]
fmax_col = 0
target_name = 'TCC (millioin #/mL)'
hist_fmax_ratio = plt.figure()
for i, r_i in enumerate(r_list):
    model = PARAFAC(n_components=r_i, init='svd', non_negativity=True,
                    tf_normalization=True, sort_em=True, loadings_normalization='maximum')
    model.fit(dataset_train)
    target_test_re = dataset_train.ref[target_name]
    valid_indices_test_re = target_test_re.index[~target_test_re.isna()]
    target_test_re = target_test_re.dropna().to_numpy()
    fmax_test_re = model.fmax
    fmax_original_test_re = fmax_test_re[fmax_test_re.index.str.contains('B1C1')]
    mask_test_re = fmax_original_test_re.index.isin(valid_indices_test_re)
    fmax_original_test_re = fmax_original_test_re[mask_test_re]
    fmax_quenched_test_re = fmax_test_re[fmax_test_re.index.str.contains('B1C2')]
    fmax_quenched_test_re = fmax_quenched_test_re[mask_test_re]
    fmax_ratio_test_re = fmax_original_test_re.to_numpy() / fmax_quenched_test_re.to_numpy()
    fmax_ratio_target_test_re = fmax_ratio_test_re[:, fmax_col]
    r_target_test_re, p_target_test_re = pearsonr(target_test_re, fmax_original_test_re.iloc[:, fmax_col])
    sns.histplot(fmax_ratio_target_test_re, binwidth=0.01, binrange=(0.85, 1.32), kde=False, stat='density',
                 alpha=0.3, label=f'n_components={r_i},' '$r_{TCC}=$' + f'{r_target_test_re:.2f}', color=colors[i])
    plt.vlines(np.mean(fmax_ratio_target_test_re), ymin=0, ymax=100, color=colors[i], linestyles='dashed')
plt.vlines(0.98, ymin=0, ymax=100, color='black', linestyles='dashed')
plt.legend()
plt.tight_layout()
plt.show()


# --------Table 1: n_components determination with F0/F and numerical methods---------

def calculate_split_half_score(dataset, n_components, n_try):
    score = 0
    for i in range(n_try):
        split_set = dataset.splitting(n_split=2, rule='random')
        model1 = PARAFAC(n_components=n_components)
        model1.fit(split_set[0])
        model2 = PARAFAC(n_components=n_components)
        model2.fit(split_set[1])
        model2 = align_components_by_loadings({'model2': model2}, model1.ex_loadings, model1.em_loadings)['model2']
        exl1 = model1.ex_loadings
        eml1 = model1.em_loadings
        exl2 = model2.ex_loadings
        eml2 = model2.em_loadings
        m_sim_ex = loadings_similarity(exl1, exl2)
        m_sim_em = loadings_similarity(eml1, eml2)
        m_sim = (m_sim_ex + m_sim_em) / 2
        score += np.mean(np.diag(m_sim.to_numpy()))
    return score / n_try


kw_dict_tbl1 = {
    'july+october': [None, '2024'],
    'july': ['2024-07', None],
    'october': ['2024-10', None],
}
target_name = 'TCC (million #/mL)'
fmax_col = 0
scores_results = {}
for name, kw in kw_dict_tbl1.items():
    dataset, _ = eem_dataset.filter_by_index(kw[0], kw[1])
    results_all_r = {}
    for r in [3, 4, 5, 6]:
        model = PARAFAC(n_components=r)
        model.fit(dataset)
        cc = model.core_consistency()
        ve = model.variance_explained()
        # shs = calculate_split_half_score(dataset, r, 30)
        target = dataset.ref[target_name]
        valid_indices = target.index[~target.isna()]
        target = target.dropna().to_numpy()
        fmax = model.fmax
        fmax_original = fmax[fmax.index.str.contains('B1C1')]
        mask = fmax_original.index.isin(valid_indices)
        fmax_original = fmax_original[mask]
        fmax_quenched = fmax[fmax.index.str.contains('B1C2')]
        fmax_quenched = fmax_quenched[mask]
        fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
        fmax_ratio_target = fmax_ratio[:, fmax_col]
        r_target, p_target = pearsonr(target, fmax_original.iloc[:, fmax_col])
        f0f_mean = np.mean(fmax_ratio_target)
        f0f_std = np.std(fmax_ratio_target)
        results_r = {
            'r_target': r_target,
            'p_target': p_target,
            'cc': cc,
            've': ve,
            # 'shs': shs,
            'f0f_mean': f0f_mean,
            'f0f_std': f0f_std,
        }
        results_all_r[r] = results_r
    scores_results[name] = results_all_r

# --------Fig 2: applying established model to quantify new samples: outlier detection-------

dataset_train, _ = eem_dataset.filter_by_index(None,
                                               [
                                                   '2024-07-13',
                                                   '2024-07-15',
                                                   '2024-07-16',
                                                   '2024-07-17',
                                                   '2024-07-18',
                                                   '2024-07-19',
                                                   # '2024-10-17',
                                                   # '2024-10-21',
                                                   # 'G1',
                                                   # 'G2',
                                                   # 'G3',
                                                   # '2024-'
                                               ]
                                               )

dataset_test, _ = eem_dataset.filter_by_index(None,
                                              [
                                                  # '2024-07-15',
                                                  # '2024-07-16',
                                                  # '2024-07-17',
                                                  # '2024-07-18',
                                                  # '2024-07-19',
                                                  '2024-10-'
                                                  # '2024-10-17',
                                                  # '2024-10-21',
                                                  # 'G1',
                                                  # 'G2',
                                                  # 'G3',
                                              ]
                                              )

indices_test_in_scenarios = {}
for name, kw in kw_dict.items():
    dataset_test_filtered, _ = dataset_test.filter_by_index(kw[0], kw[1], copy=True)
    indices_test_in_scenarios[name] = dataset_test_filtered.index

r = 4
n_outliers = 40
fmax_col = 0
target_name = 'TCC (million #/mL)'
model = PARAFAC(n_components=r, init='svd', non_negativity=True,
                tf_normalization=True, sort_em=True, loadings_normalization='maximum')
model.fit(dataset_train)

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
fmax_train = model.fmax
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
_, fmax_test, _ = model.predict(eem_dataset=dataset_test)
fmax_original_test = fmax_test[fmax_test.index.str.contains('B1C1')]
mask_test = fmax_original_test.index.isin(valid_indices_test)
fmax_original_test = fmax_original_test[mask_test]
fmax_quenched_test = fmax_test[fmax_test.index.str.contains('B1C2')]
fmax_quenched_test = fmax_quenched_test[mask_test]
fmax_ratio_test = fmax_original_test.to_numpy() / fmax_quenched_test.to_numpy()
fmax_ratio_target_test = fmax_ratio_test[:, fmax_col]
r_target_test, p_target_test = pearsonr(target_test_true, fmax_original_test.iloc[:, fmax_col])
slope_test, intercept_test = np.polyfit(target_test_true, fmax_original_test.iloc[:, fmax_col], deg=1)
target_test_pred = (fmax_original_test.iloc[:, fmax_col] - intercept_train) / slope_train

residual_test = np.abs(target_test_true - target_test_pred)
relative_error_test = abs(target_test_true - target_test_pred) / target_test_true * 100


# ---------Histplot of F0/F for training and testing, with outliers labeled---------

def round_2d(num, direction):
    if direction == 'up':
        return math.ceil(num * 100) / 100
    elif direction == 'down':
        return math.floor(num * 100) / 100


# threshold = round_2d(np.max(fmax_ratio_target_train), 'up')
binrange = (round_2d(np.min(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) - 0.02, axis=0), 'down'),
            round_2d(np.max(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) + 0.02, axis=0), 'up')
            )
# threshold = np.max(fmax_ratio_target_train)
threshold = np.quantile(fmax_ratio_target_train, 0.95)
# binrange = (np.min(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) - 0.02, axis=0),
#             np.max(np.concatenate([fmax_ratio_target_train, fmax_ratio_target_test]) + 0.02, axis=0)
#             )
plt.figure()
ax = sns.histplot(fmax_ratio_target_train, binwidth=0.01, binrange=binrange, kde=False, stat='density', color="blue",
                  alpha=0.5, label='training')
sns.histplot(fmax_ratio_target_test, binwidth=0.01, binrange=binrange, kde=False, stat='density', color="orange",
             alpha=0.5, label='test (qualified)')
sns.histplot([0], binwidth=0.01, binrange=binrange, kde=True, stat='density', color="orange",
             alpha=0.5, label='test (outliers)', hatch='////', edgecolor='red')
for bar in ax.patches:
    # Calculate the midpoint of the bin
    bin_left = bar.get_x()
    bin_width = bar.get_width()
    bin_mid = bin_left + bin_width / 2

    # Check if the midpoint is above the threshold
    if threshold <= bin_mid:
        # Add hatch pattern and change color
        bar.set_hatch("////")  # Hatch pattern (e.g., "////", "xxx", "..")
        bar.set_edgecolor('red')
plt.xlim(binrange)
plt.xlabel("C{i} ".format(i=fmax_col + 1) + "$F_{0}/F$", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=16, bbox_to_anchor=(0.5, 0.7))
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()

# ---------Fmax vs. TCC or DOC in training and testing------------

plt.figure()
a, b = np.polyfit(target_train, fmax_original_train.iloc[:, fmax_col], deg=1)
plt.plot(
    [-1, 10],
    a * np.array([-1, 10]) + b,
    '--',
    color='blue',
    label='reg. training'
)
plt.scatter(target_train, fmax_original_train.iloc[:, fmax_col], label='training', color='blue', alpha=0.6)
plt.scatter(target_test_true[fmax_ratio_target_test <= threshold],
            fmax_original_test.iloc[fmax_ratio_target_test <= threshold, fmax_col],
            label='test (qualified)', color='orange', alpha=0.6)
plt.scatter(target_test_true[fmax_ratio_target_test > threshold],
            fmax_original_test.iloc[fmax_ratio_target_test > threshold, fmax_col],
            label='test (outliers)', color='red', alpha=0.6)
plt.xlabel(target_name, fontsize=20)
plt.ylabel(f'C{fmax_col + 1} Fmax', fontsize=20)
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
        relative_error_test[fmax_ratio_target_test <= threshold],
        relative_error_test[fmax_ratio_target_test > threshold]
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

# ---------Histplot of outlier rates in each scenario---------
# outlier_indices = fmax_original_test.index[fmax_ratio_target_test > threshold]
# outlier_rates = {}
# for name, indices in indices_test_in_scenarios.items():
#     n_outliers = 0
#     for idx in indices:
#         if idx in outlier_indices:
#             n_outliers += 1
#     outlier_rates[name] = (n_outliers / len(indices) * 100 * 2) if indices else 0
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.bar(list(outlier_rates.keys())[2:], list(outlier_rates.values())[2:], color='red')
# ax.set_ylim([0, 100])
# ax.set_ylabel('Outlier rate (%)', fontsize=18)
# ax.tick_params(labelsize=20)
# fig.tight_layout()
# fig.show()

# ---------Fig 3: timeseries of anomalies detected-----------

time = [datetime.strptime(t[0:16], '%Y-%m-%d-%H-%M') for t in fmax_original_test.index]
sampling_point_labels_dict = {
    'GAC top': 'G1',
    'GAC middle': 'G2',
    'GAC bottom': 'G3',
    'GAC effluent': 'M3',
}
sampling_point_labels = []
for idx in fmax_original_test.index:
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
        # target_name: target_train,
        # f'C{fmax_col + 1} Fmax': fmax_original_train.iloc[:, fmax_col].to_numpy(),
        # f'C{fmax_col + 1} ' + '$F_{0}/F$': fmax_ratio_target_train,
        # 'sampling point': sampling_point_labels,
        # 'markers': markers,
        # 'is_outlier': fmax_ratio_target_train > threshold

        target_name: target_test_true,
        f'C{fmax_col + 1} Fmax': fmax_original_test.iloc[:, fmax_col].to_numpy(),
        f'C{fmax_col + 1} ' + '$F_{0}/F$': fmax_ratio_target_test,
        'sampling point': sampling_point_labels,
        'markers': markers,
        'is_outlier': fmax_ratio_target_test > threshold
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
        if not np.isnan(outlier_rate_col):
            t2 = ax.text(1, 0.99, '$OR_{col}$=' + f'{outlier_rate_col:.2f}%', transform=ax.transAxes,
                         fontsize=14, color='black', verticalalignment='top', horizontalalignment='right')
        if not np.isnan(outlier_rate_eff):
            t3 = ax.text(1, 0.91, '$OR_{eff}$=' + f'{outlier_rate_eff:.2f}%', transform=ax.transAxes,
                         fontsize=14, color='black', verticalalignment='top', horizontalalignment='right')
    tr1.set_ylim([0, 2.5])
    tr2.set_ylim([0, 2500])

# Configure axis labels
main_ax.set_ylabel(f'C{fmax_col + 1} ' + '$F_{0}/F$', fontsize=18)
main_ax.set_ylim([0.95, 1.35])

# Final adjustments
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.show()

# ----------create legends--------
plt.figure()
plt.plot([0, 0], [1, 1], label='$F_{0}/F$', color='#1f77b4')
plt.plot([0, 0], [1, 1], label='TCC or DOC', color='#2ca02c')
plt.plot([0, 0], [1, 1], label='Fmax', color='#ff7f0e')
plt.legend(ncol=3, title='variables', fontsize=12, title_fontsize=12, frameon=True)
plt.show()

plt.figure()
plt.plot([0, 0], [1, 1], marker='^', markersize=6, markeredgecolor='black', color='white', label='BAC top')
plt.plot([0, 0], [1, 1], marker='s', markersize=6, markeredgecolor='black', color='white', label='BAC middle')
plt.plot([0, 0], [1, 1], marker='v', markersize=6, markeredgecolor='black', color='white', label='BAC bottom')
plt.plot([0, 0], [1, 1], marker='o', markersize=6, markeredgecolor='black', color='white', label='BAC effluent')
plt.legend(ncol=4, title='sampling point', fontsize=10, title_fontsize=10, frameon=True, bbox_to_anchor=(1, -0.2))
plt.tight_layout()
plt.show()

# ----------Other numerical outlier detection methods: rmse and leverage--------

dataset_train, _ = eem_dataset.filter_by_index(['B1C1'],
                                               [
                                                   '2024-07-13',
                                                   '2024-07-15',
                                                   '2024-07-16',
                                                   '2024-07-17',
                                                   '2024-07-18',
                                                   '2024-07-19',
                                                   # '2024-10-17',
                                                   # '2024-10-21',
                                                   # 'G1',
                                                   # 'G2',
                                                   # 'G3',
                                                   # '2024-'
                                               ]
                                               )

dataset_test, _ = eem_dataset.filter_by_index(['B1C1'],
                                              [
                                                  # '2024-07-15',
                                                  # '2024-07-16',
                                                  # '2024-07-17',
                                                  # '2024-07-18',
                                                  # '2024-07-19',
                                                  '2024-10-'
                                                  # '2024-10-17',
                                                  # '2024-10-21',
                                                  # 'G1',
                                                  # 'G2',
                                                  # 'G3',
                                              ]
                                              )

model = PARAFAC(n_components=4)
model.fit(dataset_train)
fmax_train = model.fmax
leverage_train = model.leverage()
rmse_train = model.sample_rmse().to_numpy().reshape(-1)
relative_rmse_train = model.sample_relative_rmse().to_numpy().reshape(-1)

# ------------rmse and relative rmse----------
_, fmax_test, recon_eem_stack = model.predict(dataset_test)
res = dataset_test.eem_stack - recon_eem_stack
n_pixels = recon_eem_stack.shape[1] * recon_eem_stack.shape[2]
rmse_test = np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels)
rmse_test_df = pd.DataFrame(rmse_test, index=fmax_test.index)
relative_rmse_test = rmse_test / np.average(dataset_test.eem_stack, axis=(1, 2))

# plot_eem(dataset_train.eem_stack[1], ex_range=dataset_train.ex_range, em_range=dataset_train.em_range)
# plot_eem(model.eem_stack_reconstructed[1], ex_range=dataset_train.ex_range, em_range=dataset_train.em_range)

# ------------leverage---------
indices_test = dataset_test.index
leverage_test = []
for idx in indices_test:
    one_sample_dataset, _ = dataset_test.filter_by_index([idx], None)
    new_dataset = combine_eem_datasets([dataset_train, one_sample_dataset])
    model = PARAFAC(n_components=4)
    model.fit(new_dataset)
    leverage = model.leverage()
    leverage_test.append(leverage.loc[one_sample_dataset.index[0]].to_numpy()[0])


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
# threshold = round_2d(np.max(indicator_train), 'up')
binrange = (round_2d(np.min(np.concatenate([indicator_train, indicator_test]) - 20, axis=0), 'down'),
            round_2d(np.max(np.concatenate([indicator_train, indicator_test]) + 20, axis=0), 'up')
            )
# threshold = np.max(indicator_train)
threshold = np.quantile(indicator_train, 0.95)
# binrange = (np.min(np.concatenate([indicator_train, indicator_test]) - 0.02, axis=0),
#             np.max(np.concatenate([indicator_train, indicator_test]) + 0.02, axis=0)
#             )
plt.figure()
ax = sns.histplot(indicator_train, binwidth=10, binrange=binrange, kde=False, stat='density', color="blue",
                  alpha=0.5, label='training', zorder=0)
sns.histplot(indicator_test, binwidth=10, binrange=binrange, kde=False, stat='density', color="orange",
             alpha=0.5, label='test (qualified)', zorder=1)
sns.histplot([-100], binwidth=10, binrange=binrange, kde=True, stat='density', color="orange",
             alpha=0.5, label='test (outliers)', hatch='////', edgecolor='red')
for bar in ax.patches:
    # Calculate the midpoint of the bin
    bin_left = bar.get_x()
    bin_width = bar.get_width()
    bin_mid = bin_left + bin_width / 2

    # Check if the midpoint is above the threshold
    if threshold <= bin_mid and bar.zorder == 1:
        # Add hatch pattern and change color
        bar.set_hatch("////")  # Hatch pattern (e.g., "////", "xxx", "..")
        bar.set_edgecolor('red')
plt.xlim(binrange)
plt.xlabel("EEM-RMSE", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=16, loc='upper right')
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()

target_name = 'DOC'
target_train = dataset_train.ref[target_name]
target_test_true = dataset_test.ref[target_name]
fmax_col = 1
plt.figure()
# a, b = np.polyfit(target_train, fmax_train.iloc[:, fmax_col], deg=1)
# plt.plot(
#     [-1, 10],
#     a * np.array([-1, 10]) + b,
#     '--',
#     color='blue',
#     label='reg. training'
# )
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
    bbox_to_anchor=[1.02, 0.37],
    # bbox_to_anchor=[0.58, 0.63],
    fontsize=16
)
plt.tick_params(labelsize=16)
plt.tight_layout()
# plt.xlim([0, 2.5])
# plt.ylim([0, 2500])
plt.show()

# ----------relative rmse----------
indicator_train = relative_rmse_train
indicator_test = relative_rmse_test
# threshold = round_2d(np.max(indicator_train), 'up')
binrange = (round_2d(np.min(np.concatenate([indicator_train, indicator_test]) - 0.05, axis=0), 'down'),
            round_2d(np.max(np.concatenate([indicator_train, indicator_test]) + 0.05, axis=0), 'up')
            )
# threshold = np.max(indicator_train)
threshold = np.quantile(indicator_train, 0.95)
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
    if threshold <= bin_mid and bar.zorder == 1:
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

#----------leverage---------
indicator_train = leverage_train.to_numpy().reshape(-1)
indicator_test = np.array(leverage_test)
# threshold = round_2d(np.max(indicator_train), 'up')
binrange = (round_2d(np.min(np.concatenate([indicator_train, indicator_test]) - 0.05, axis=0), 'down'),
            round_2d(np.max(np.concatenate([indicator_train, indicator_test]) + 0.05, axis=0), 'up')
            )
# threshold = np.max(indicator_train)
threshold = np.quantile(indicator_train, 0.95)
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
    if threshold <= bin_mid and bar.zorder == 1:
        # Add hatch pattern and change color
        bar.set_hatch("////")  # Hatch pattern (e.g., "////", "xxx", "..")
        bar.set_edgecolor('red')
plt.xlim(binrange)
plt.xlabel("Sample leverage", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=16, loc='upper right')
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.show()

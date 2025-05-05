import matplotlib.pyplot as plt
from scipy.stats import zscore

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

# ---------------Read EEM dataset-----------------

# eem_dataset_path = \
#     "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/nan_sample_260_ex_274_em_310_mfem_5.json"
eem_dataset_path = \
    "C:/PhD\Fluo-detect/_data/_greywater/2024_quenching/sample_260_ex_274_em_310_mfem_3.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['2024-10-'], copy=True)
eem_dataset_original, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
eem_dataset_quenched, _ = eem_dataset.filter_by_index(['B1C2'], None, copy=True)

# -------------prior decomposition function test---------

A, B, C = cp_hals_prior(
    tensor=eem_dataset_original.eem_stack,
    rank=6,
    prior_dict_A={0: eem_dataset_original.ref['TCC (million #/mL)'].to_numpy()},
    gamma_A=7.51 * 1e3,
    tol=1e-9,
    init='nndsvda'
)
plt.plot(A[:, 0], eem_dataset_original.ref['TCC (million #/mL)'], 'o')
plt.show()
plot_eem(np.outer(B[:, 0], C[:, 0]),
         ex_range=eem_dataset_original.ex_range,
         em_range=eem_dataset_original.em_range,
         display=True
         )


A, B = nmf_hals_prior(
    X=eem_dataset_original.eem_stack.reshape([eem_dataset_original.eem_stack.shape[0], -1]),
    rank=4,
    prior_dict_W={0: eem_dataset_original.ref['TCC (million #/mL)'].to_numpy()},
    gamma_W=1.5e7,
    init='nndsvda',
    alpha_H=1,
    l1_ratio=0,
)
plt.plot(A[:, 0], eem_dataset_original.ref['TCC (million #/mL)'], 'o')
plt.show()
plot_eem(B[3, :].reshape(eem_dataset_original.eem_stack.shape[1:]),
         ex_range=eem_dataset_original.ex_range,
         em_range=eem_dataset_original.em_range,
         display=True
         )



# -----------model training-------------
dataset_train = eem_dataset
sample_prior = {0: dataset_train.ref['TCC (million #/mL)']}
rank = 3
model = EEMNMF(
    n_components=rank,
    solver='hals',
    prior_dict_sample=sample_prior,
    gamma_sample=1e9,
    normalization=None,
    sort_em=False
)
model.fit(dataset_train)
fmax_train = model.nmf_fmax
components = model.components

# -----------plot components----------
fig, ax = plt.subplots(
    nrows=rank // 2 + 1, ncols=2,
)
plt.subplots_adjust(
    left=0,  # distance from left of figure (0 = 0%, 1 = 100%)
    right=1,  # distance from right
    bottom=0,
    top=1,
    wspace=0,  # width between subplots
    hspace=0  # height between subplots
)
for i in range(4):
    if i < rank:
        f, a, im = plot_eem(
            components[i],
            ex_range=eem_dataset_original.ex_range,
            em_range=eem_dataset_original.em_range,
            display=False,
            title=f'Component {i+1}'
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
fig, ax = plt.subplots(nrows=1, ncols=len(sample_prior))
for i, (r, p) in enumerate(sample_prior.items()):
    if len(sample_prior) == 1:
        ax.plot(fmax_train.iloc[:, r], p.to_numpy(), 'o')
        ax.set_title(f'Fmax C{r+1} vs. {p.name}')
        ax.set_xlabel(f'Fmax C{r+1}')
        ax.set_ylabel(f'{p.name}')
    else:
        ax[i].plot(fmax_train.iloc[:, r], p, 'o')
        ax[i].set_title(f'Fmax C{r+1} vs. {p.name}')
        ax[i].set_xlabel(f'Fmax C{r+1}')
        ax[i].set_ylabel(f'{p.name}')
fig.show()


# ------------apparent F0/F distributions-------------
rank_target = 0
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

fig, ax = plt.subplots()
counts, bins, patches = ax.hist(fmax_ratio_target_train_filtered, bins=30,
                                density=True, alpha=0.5, color='blue', label='training', zorder=0, edgecolor='black')
plt.tight_layout()
plt.show()




# eem_stack = eem_dataset_original.eem_stack
# eem_stack_2d = eem_stack.reshape([eem_stack.shape[0], -1])
# eem_stack_2d = np.nan_to_num(eem_stack_2d, nan=0)
# A, B = nmf_hals_prior(X=eem_stack_2d, rank=rank, prior_dict_W=sample_prior, gamma_W=1e7, init='nndsvda')
# components = B.reshape([B.shape[0], eem_stack.shape[1], eem_stack.shape[2]])
# plt.plot(A[:, 0], eem_dataset_original.ref['TCC (million #/mL)'].to_numpy(), 'o')
# plt.show()

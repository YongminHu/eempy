from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from sklearn.metrics import mean_squared_error


# ---------------Effect of initialization-----------------
eem_dataset_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
true_components = np.array([eem_dataset.eem_stack[-5], eem_dataset.eem_stack[0]])
_, fmax_measured, _ = eems_fit_components(eem_dataset.eem_stack, true_components)
fmax_measured = pd.DataFrame(fmax_measured, index=eem_dataset.index)
eem_dataset_o, _ = eem_dataset.filter_by_index(['0gL'], None, copy=True)
eem_dataset_q, _ = eem_dataset.filter_by_index(['2_5gL'], None, copy=True)

X1 = eem_dataset_o.eem_stack
X2 = eem_dataset_q.eem_stack
X = np.concatenate((X1, X2), axis=0)


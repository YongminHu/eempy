import numpy as np

from eempy.read_data import read_eem_dataset, read_abs_dataset, read_eem, read_eem_dataset_from_json
from eempy.eem_processing import *
from eempy.plot import plot_eem, plot_loadings, plot_fmax
from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------Read EEM dataset-----------------

eem_dataset_path = \
    "C:/PhD/Fluo-detect/_data/_greywater/2024_quenching/sample_260_ex_274_em_310_mfem_3.json"
eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
eem_dataset, _ = eem_dataset.filter_by_index(None, ['2024-10-'], copy=True)
eem_dataset_original, _ = eem_dataset.filter_by_index(['B1C1'], None, copy=True)
eem_dataset_quenched, _ = eem_dataset.filter_by_index(['B1C2'], None, copy=True)


# eem_dataset_path = 'C:/PhD/Fluo-detect/_data/20240313_BSA_Ecoli/synthetic_samples.json'
# eem_dataset = read_eem_dataset_from_json(eem_dataset_path)
# eem_up = eem_dataset.eem_stack[eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKI-reS')]
# eem_bot = eem_dataset.eem_stack[eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKISYM.')]
# eem_new = np.concatenate([eem_up[:-8], eem_bot[-8:]], axis=0)
# # plot_eem(eem_up, eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=1500)
# # plot_eem(eem_new, eem_dataset.ex_range, eem_dataset.em_range, vmin=0, vmax=1500)
# eem_stack_new = np.delete(eem_dataset.eem_stack,
#                           eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKI-reS'),
#                           axis=0)
# eem_dataset.index.remove('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKI-reS')
# replaced_idx = eem_dataset.index.index('B1S12024-03-11-Ecoli_BSA_3to1+3_75gLKISYM.')
# eem_stack_new[replaced_idx] = eem_new
# eem_dataset.eem_stack = eem_stack_new


#------------nmf_prior--------------
X1 = eem_dataset_original.eem_stack
# X2 = eem_dataset_q.eem_stack
# X = np.concatenate((X1, X2), axis=0)
X = X1.reshape([X1.shape[0], -1])
z_dict = {0: eem_dataset_original.ref["TCC"].to_numpy().reshape([eem_dataset_original.ref["TCC"].shape[0], 1])}
rank = 4
max_iter = 500

U, V = initialization_2d(X, rank=rank)
factors = [U, V.T]

# for c in range(factors[1].shape[1]):
#     plot_eem(factors[1][:, c].reshape([X1.shape[1], X1.shape[2]]), ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range)

# factors, replaced_rank = replace_factor_with_prior(factors, eem_dataset_o.ref["TCC"].to_numpy(), 0, 3, X=X,
#                                                    show_replaced_rank=True)

# for c in range(factors[1].shape[1]):
#     plot_eem(factors[1][:, c].reshape([X1.shape[1], X1.shape[2]]), ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range)

for i in range(max_iter):
    for mode_to_update in range(X.ndim):
        mode_fixed = [i for i in range(X.ndim) if i != mode_to_update][0]
        new_order = [mode_fixed] + [mode_to_update]
        X_reshaped = X.transpose(new_order)
        U = factors[mode_fixed]
        V = factors[mode_to_update].T
        UtU = U.T.dot(U)
        UtM = U.T.dot(X_reshaped)
        if mode_to_update == 0:
            V = hals_prior_nnls(UtM, UtU, regularization_dict={1: eem_dataset_original.ref["TCC"].to_numpy()}, V=V,
                                l=0.75, n_iter_max=200, epsilon=1e-8)
        else:
            V = hals_nnls_normal(UtM, UtU, V)
        factors[mode_to_update] = V.T
        # V = hals_nnls_normal(UtM, UtU, V, epsilon=1e-8)

components = [factors[1][:, i].reshape([eem_dataset.eem_stack.shape[1], -1]) for i in range(rank)]
for c in components:
    plot_eem(c, ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# ax1.plot(eem_dataset_o.ref['TCC'], loadings[0][:int(X.shape[0]/2), 1], 'o', color='red')

for i in range(rank):
    r, _ = pearsonr(eem_dataset_original.ref['TCC'], factors[0][:, i])
    ax1.plot(eem_dataset_original.ref['TCC'], factors[0][:, i], 'o', label=f"C{i}, r={r:2g}")
ax1.legend()
fig1.show()
# ax2.plot([1.66, 0.83, 1.245, 0.415, 0], loadings[0][:int(X.shape[0]/2), 1], 'o', color='red')
# fig2.show()

#------------nmf_ic--------------

# X1 = eem_dataset_o.eem_stack
# # X2 = eem_dataset_q.eem_stack
# # X = np.concatenate((X1, X2), axis=0)
# X = X1.reshape([X1.shape[0], -1])
# z_dict = {0: eem_dataset_o.ref["TCC"].to_numpy().reshape([eem_dataset_o.ref["TCC"].shape[0], 1])}
# rank = 4
# max_iter = 300
#
# U, V = initialization_2d(X, rank=rank)
# loadings = [U, V.T]
# for i in range(max_iter):
#     for mode_to_update in range(X.ndim):
#         mode_fixed = [i for i in range(X.ndim) if i != mode_to_update][0]
#         new_order = [mode_fixed] + [mode_to_update]
#         X_reshaped = X.transpose(new_order)
#         U = loadings[mode_fixed]
#         V = loadings[mode_to_update].T
#         UtU = U.T.dot(U)
#         UtM = U.T.dot(X_reshaped)
#         if mode_to_update == 0:
#             V = hals_pr_nnls2(UtM, UtU, z=z_dict[0], V=V, p_coefficient=1e7, n_iter_max_inner=100, n_iter_max_outer=200)
#         else:
#             V = hals_nnls_normal(UtM, UtU, V)
#         loadings[mode_to_update] = V.T
#         # V = hals_nnls_normal(UtM, UtU, V, epsilon=1e-8)
#
# components = [loadings[1][:, i].reshape([eem_dataset.eem_stack.shape[1], -1]) for i in range(rank)]
# for c in components:
#     plot_eem(c, ex_range=eem_dataset.ex_range, em_range=eem_dataset.em_range)
#
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# # ax1.plot(eem_dataset_o.ref['TCC'], loadings[0][:int(X.shape[0]/2), 1], 'o', color='red')
# for i in range(rank):
#     ax1.plot(eem_dataset_o.ref['TCC'], loadings[0][:, i], 'o')
# fig1.show()
# # ax2.plot([1.66, 0.83, 1.245, 0.415, 0], loadings[0][:int(X.shape[0]/2), 1], 'o', color='red')
# # fig2.show()

# #------------parafac_ic--------------
# rank = 4
# ic_coeffcients = [500, 1000, 2000, 2500, 7500]
# kw_o = "B1C1"
# kw_q = "B1C2"
#
# n_cols = 2
# n_rows = rank // n_cols + 1
# fig_fmax_ratio_box = make_subplots(rows=n_rows,
#                                    cols=n_cols,
#                                    subplot_titles=[f"component{i+1}" for i in range(rank)])
# fig_fmax_ratio_std = make_subplots(rows=n_rows,
#                                    cols=n_cols,
#                                    subplot_titles=[f"component{i+1}" for i in range(rank)])
# fig_cor = make_subplots(rows=n_rows,
#                         cols=n_cols,
#                         subplot_titles=[f"component{i+1}" for i in range(rank)])
# fig_p = make_subplots(rows=n_rows,
#                       cols=n_cols,
#                       subplot_titles=[f"component{i+1}" for i in range(rank)])
#
# fmax_ratio_stds = {f"component {i+1}": [] for i in range(rank)}
# cors = {f"component {i+1}": [] for i in range(rank)}
# ps = {f"component {i+1}": [] for i in range(rank)}
# for icc in ic_coeffcients:
#     model = PRPARAFAC(n_components=rank, n_iter_max=500, p_coefficient=icc,
#                       z_dict={0: eem_dataset_o.ref["TCC"].to_numpy().reshape([eem_dataset_o.ref["TCC"].shape[0], 1])})
#     model.fit(eem_dataset_o)
#     fmax_tot = model.fmax
#     fmax_original = fmax_tot[fmax_tot.index.str.contains(kw_o)]
#     # fmax_quenched = fmax_tot[fmax_tot.index.str.contains(kw_q)]
#     # fmax_ratio = fmax_original.to_numpy() / fmax_quenched.to_numpy()
#     # fmax_ratio_std = np.std(fmax_ratio, axis=0)
#     for i in range(rank):
#         # fmax_ratio_stds[f"component {i+1}"].append(fmax_ratio_std[i])
#         # fig_fmax_ratio_box.add_trace(
#         #     go.Box(
#         #         y=fmax_ratio[:, i],
#         #         name=str(icc),
#         #         showlegend=False,
#         #     ),
#         #     row=i // n_cols + 1, col=i % n_cols + 1
#         # )
#         cor, p = pearsonr(fmax_original.iloc[:, i], eem_dataset_o.ref["TCC"])
#         cors[f"component {i+1}"].append(cor)
#         ps[f"component {i+1}"].append(p)
#
# for i in range(rank):
#     # fig_fmax_ratio_std.add_trace(
#     #     go.Scatter(
#     #         x=[str(c) for c in ic_coeffcients],  # Position the text at the x location of the box
#     #         y=fmax_ratio_stds[f'component {i+1}'],  # Position the text near the mean of the box values
#     #         # text=[f'std={ratio_std:.2g}<br>cor_tcc={cor_tcc:.2g}<br>cor_doc={cor_doc:.2g}'],  # Custom text
#     #         mode='markers',  # Only display text
#     #         textposition="top right",  # Position text relative to the point
#     #         showlegend=False,  # Hide legend for this trace
#     #         marker=dict(
#     #             size=10,  # Marker size
#     #             symbol='circle',  # Marker symbol
#     #             opacity=0.8  # Transparency
#     #         ),
#     #     ),
#     #     row=i // n_cols + 1, col=i % n_cols + 1
#     # )
#     fig_cor.add_trace(
#         go.Scatter(
#             x=[str(c) for c in ic_coeffcients],  # Position the text at the x location of the box
#             y=cors[f'component {i+1}'],  # Position the text near the mean of the box values
#             # text=[f'std={ratio_std:.2g}<br>cor_tcc={cor_tcc:.2g}<br>cor_doc={cor_doc:.2g}'],  # Custom text
#             mode='markers',  # Only display text
#             textposition="top right",  # Position text relative to the point
#             showlegend=False,  # Hide legend for this trace
#             marker=dict(
#                 size=10,  # Marker size
#                 symbol='circle',  # Marker symbol
#                 opacity=0.8  # Transparency
#             ),
#         ),
#         row=i // n_cols + 1, col=i % n_cols + 1
#     )
#     fig_p.add_trace(
#         go.Scatter(
#             x=[str(c) for c in ic_coeffcients],  # Position the text at the x location of the box
#             y=ps[f'component {i+1}'],  # Position the text near the mean of the box values
#             # text=[f'std={ratio_std:.2g}<br>cor_tcc={cor_tcc:.2g}<br>cor_doc={cor_doc:.2g}'],  # Custom text
#             mode='markers',  # Only display text
#             textposition="top right",  # Position text relative to the point
#             showlegend=False,  # Hide legend for this trace
#             marker=dict(
#                 size=10,  # Marker size
#                 symbol='circle',  # Marker symbol
#                 opacity=0.8  # Transparency
#             ),
#         ),
#         row=i // n_cols + 1, col=i % n_cols + 1
#     )
#
# # fig_fmax_ratio_box.show()
# # fig_fmax_ratio_std.show()
# fig_cor.show()
# fig_p.show()



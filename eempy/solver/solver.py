"""Solver routines for excitation–emission matrix (EEM) decomposition.

This module provides numerical solvers used to decompose an EEM stack into a
small number of non-negative components using matrix (NMF) and tensor (CP/PARAFAC)
factorization models.

Implemented features include:
- HALS/ALS-style NNLS updates for NMF and CP factors,
- quadratic (Tikhonov) priors with NaN-masked entries,
- elastic-net regularization (L1/L2 mix),
- optional paired-sample ratio constraints on sample-mode scores,
- elementwise masking to ignore missing or invalid tensor entries.

"""

import numpy as np
import tensorly as tl
from eempy.eem_processing import process_eem_stack, eem_nan_imputing
from sklearn.decomposition import NMF
from scipy.linalg import khatri_rao
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment, nnls
from scipy.spatial.distance import cdist
from tensorly.decomposition import non_negative_parafac_hals
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import unfolding_dot_khatri_rao
from tensorly.base import unfold


def masked_unfolding_dot_khatri_rao(tensor, factors, mode, mask):
    """
    Compute a masked unfolded tensor times the Khatri–Rao product (MTTKRP).

    This is a helper for masked CP/PARAFAC updates. It first applies an elementwise
    mask to the tensor (typically 0/1), unfolds the masked tensor along `mode`, and
    right-multiplies by the Khatri–Rao product of the other factor matrices.

    Parameters
    ----------
    tensor : array-like, shape (I, J, K)
        Input 3-way tensor (e.g., an EEM stack) to be decomposed.
    factors : tuple or list
        CP factors in the TensorLy format `(weights, [A, B, C])` or simply the list
        `[A, B, C]`, where each factor has shape `(dimension, rank)`.
    mode : int
        Unfolding mode (0, 1, or 2).
    mask : array-like, shape (I, J, K)
        Elementwise mask (1 = keep / valid entry, 0 = ignore entry).

    Returns
    -------
    mttkrp : tensorly.tensor or np.ndarray, shape (tensor.shape[mode], rank)
        The masked MTTKRP matrix used in HALS/ALS updates.
    """
    masked_tensor = tl.tensor(tensor * mask, dtype=float)
    return tl.dot(unfold(masked_tensor, mode), tl.tenalg.khatri_rao(factors[1], skip_matrix=mode))


def masked_tensor_norm_error(tensor, reconstruction, mask):
    """
    Compute the reconstruction error on observed (masked) entries.

    Parameters
    ----------
    tensor : array-like
        Original tensor.
    reconstruction : array-like
        Reconstructed tensor of the same shape as `tensor`.
    mask : array-like
        Elementwise mask (1 = keep / valid entry, 0 = ignore entry).

    Returns
    -------
    error : float
        Frobenius norm of `(tensor - reconstruction) * mask`.
    """
    return tl.norm((tensor - reconstruction) * mask)


def unfolded_eem_stack_initialization(M, rank, method='nndsvd'):
    """
    Initialize non-negative matrix factors from an unfolded EEM stack.

    Given a non-negative matrix `M` (typically an unfolding of a 3D EEM stack),
    this routine produces an initialization `(W, H)` suitable for NMF / HALS updates.

    Supported initializations include:
    - `ordinary_nmf` : scikit-learn NMF with NNDSVD init.
    - `svd` : absolute-value SVD-based factors (non-negative by clipping).
    - `nndsvd`, `nndsvda`, `nndsvdar` : NNDSVD variants (Boutsidis & Gallopoulos),
      where zeros are kept (`nndsvd`), filled with the mean (`nndsvda`), or filled
      with small random values (`nndsvdar`).

    Parameters
    ----------
    M : np.ndarray, shape (m, n)
        Non-negative matrix to factorize (e.g., unfolded tensor).
    rank : int
        Number of components (latent factors).
    method : str, {"ordinary_nmf", "svd", "nndsvd", "nndsvda", "nndsvdar"}, optional
        Initialization method. Default is "nndsvd".

    Returns
    -------
    W : np.ndarray, shape (m, rank)
        Left factor initialization.
    H : np.ndarray, shape (rank, n)
        Right factor initialization.
    """
    if method == 'ordinary_nmf':
        nmf_model = NMF(n_components=rank, init='nndsvd')
        W = nmf_model.fit_transform(M)
        H = nmf_model.components_
        return W, H
    # Step 1: Compute SVD of V
    U, S, VT = np.linalg.svd(M, full_matrices=False)  # SVD decomposition

    # Step 2: Keep the top-r components
    U_r = U[:, :rank]
    S_r = S[:rank]
    VT_r = VT[:rank, :]

    if method == 'svd':
        W = np.abs(U_r) * np.sqrt(S_r)[None, :]
        H = np.sqrt(S_r)[:, None] * np.abs(VT_r)
        W = np.clip(W, a_min=1e-6, a_max=None)
        H = np.clip(H, a_min=1e-6, a_max=None)

    else:
        # Step 3: Initialize W and H
        W = np.zeros((M.shape[0], rank))
        H = np.zeros((rank, M.shape[1]))

        for k in range(rank):
            u_k = U_r[:, k]
            v_k = VT_r[k, :]

            # Positive and negative parts
            u_k_pos = np.maximum(u_k, 0)
            u_k_neg = np.maximum(-u_k, 0)
            v_k_pos = np.maximum(v_k, 0)
            v_k_neg = np.maximum(-v_k, 0)

            # Normalize
            u_norm_pos = np.linalg.norm(u_k_pos)
            v_norm_pos = np.linalg.norm(v_k_pos)

            # Assign components
            if u_norm_pos * v_norm_pos > 0:
                W[:, k] = np.sqrt(S_r[k]) * (u_k_pos / u_norm_pos)
                H[k, :] = np.sqrt(S_r[k]) * (v_k_pos / v_norm_pos)
            else:
                W[:, k] = np.sqrt(S_r[k]) * (u_k_neg / np.linalg.norm(u_k_neg))
                H[k, :] = np.sqrt(S_r[k]) * (v_k_neg / np.linalg.norm(v_k_neg))

        # Step 4: Handle zero entries
        if method == 'nndsvd':
            pass
        if method == 'nndsvda':
            W[W == 0] = np.mean(M)
            H[H == 0] = np.mean(M)
        if method == 'nndsvdar':
            W[W == 0] = np.random.uniform(0, np.mean(M) / 100, W[W == 0].shape)
            H[H == 0] = np.random.uniform(0, np.mean(M) / 100, H[H == 0].shape)

    return W, H


def hals_prior_nnls(
        UtM,
        UtU,
        prior_dict=None,
        gamma=None,
        alpha=0,
        l1_ratio=0,
        V=None,
        max_iter=500,
        tol=1e-8,
        eps=1e-8,
        fixed_components=None,
):
    """
    HALS-style non-negative least squares update with optional priors and elastic-net.

    This routine updates a factor matrix `V` (shape `(rank, n_features)`) in a
    least-squares subproblem of the form used in NMF and CP-ALS/HALS:

        min_{V >= 0}  1/2 || UtM - UtU @ V ||_F^2
                     + sum_k (gamma_k/2) ||mask_k * (V_k - p_k)||_2^2
                     + alpha * [ l1_ratio * ||V||_1 + (1 - l1_ratio)/2 * ||V||_F^2 ]

    where:
    - `UtM` corresponds to UᵀM (or a mode-wise MTTKRP in CP),
    - `UtU` corresponds to UᵀU (Gram matrix),
    - `prior_dict[k] = p_k` is an optional per-component prior vector for row `k`,
      with NaNs allowed to indicate "no prior" at specific entries,
    - `gamma` can be a scalar applied to all prior components or a dict `{k: gamma_k}`.

    The update is performed row-by-row (FastHALS residual update), and can keep
    selected components fixed (useful when seeding or freezing known spectra).

    Parameters
    ----------
    UtM : np.ndarray, shape (rank, n)
        Cross-product term (e.g., UᵀM or MTTKRPᵀ), with one row per component.
    UtU : np.ndarray, shape (rank, rank)
        Gram matrix term (e.g., UᵀU).
    prior_dict : dict[int, array-like], optional
        Mapping from component index `k` to a prior vector of length `n`. NaNs are
        treated as "no prior" for those entries.
    gamma : float or dict[int, float], optional
        Prior strength. If a float is provided, it is applied to all keys in
        `prior_dict`. If a dict is provided, strengths are component-specific.
    alpha : float, optional
        Overall elastic-net strength (0 disables elastic-net).
    l1_ratio : float, optional
        Elastic-net mixing parameter in [0, 1]. 0 = pure L2, 1 = pure L1.
    V : np.ndarray, shape (rank, n), optional
        Initial value for `V`. If None, a least-squares solution is computed and
        clipped to be positive.
    max_iter : int, optional
        Maximum number of HALS iterations.
    tol : float, optional
        Relative convergence tolerance based on HALS update magnitude.
    eps : float, optional
        Small positive constant used for numerical stability and positivity floor.
    fixed_components : list[int], optional
        Indices of components (rows of `V`) that should not be updated.

    Returns
    -------
    V : np.ndarray, shape (rank, n)
        Updated non-negative factor matrix.
    """
    fixed_components = [] if fixed_components is None else fixed_components
    r, n = UtM.shape
    prior_dict = {} if prior_dict is None else prior_dict

    if gamma is None:
        gamma = {}
    elif not isinstance(gamma, dict):
        gamma = {k: float(gamma) for k in prior_dict.keys()}

    # Elastic-net constants
    l2_pen = alpha * (1 - l1_ratio)
    l1_pen = alpha * l1_ratio

    # Initialize V if not provided
    if V is None:
        V_np = np.linalg.solve(UtU + eps * np.eye(r), UtM)
        V = np.clip(V_np, a_min=eps, a_max=None)
        VVt = V @ V.T
        scale = (UtM * V).sum() / ((UtU * VVt).sum() + eps)
        V = V * scale

    # Initialize residual: R = UtM - UtU @ V
    R = UtM - UtU @ V

    for it in range(max_iter):
        delta = 0.0

        for k in range(r):
            if k in fixed_components:
                continue
            ukk = UtU[k, k]
            if ukk < eps:
                continue

            # Local residual with V[k]'s contribution restored
            Rk = R[k] + ukk * V[k]

            # Numerator and denominator
            num = Rk.copy()
            denom = ukk + l2_pen

            # Per-component prior
            gamma_k = gamma.get(k, 0.0)
            if gamma_k > 0 and k in prior_dict:
                p_arr = np.asarray(prior_dict[k], dtype=float)
                mask = np.isfinite(p_arr).astype(float)
                p_clean = np.nan_to_num(p_arr, nan=0.0)
                num += gamma_k * mask * p_clean
                denom += gamma_k

            # L1 penalty
            if l1_pen:
                num -= l1_pen

            # Update rule and clipping
            v_new = np.clip(num / (denom + eps), a_min=eps, a_max=None)

            # Track and apply update
            diff = v_new - V[k]
            delta += np.dot(diff, diff)

            V[k] = v_new
            R -= np.outer(UtU[:, k], diff)  # FastHALS residual update

        if it > 0 and delta / (prev_delta + eps) < tol:
            break
        prev_delta = delta

    return V


def cp_hals_prior(
        tensor,
        rank,
        prior_dict_A=None,
        prior_dict_B=None,
        prior_dict_C=None,
        gamma_A=0,
        gamma_B=0,
        gamma_C=0,
        prior_ref_components=None,
        alpha_A=0,
        alpha_B=0,
        alpha_C=0,
        l1_ratio=0,
        lam=0,
        idx_top=None,
        idx_bot=None,
        max_iter_als=200,
        max_iter_nnls=500,
        tol=1e-9,
        eps=1e-8,
        init='svd',
        custom_init=None,
        random_state=None,
        fixed_components=None,
        mask=None,
):
    """
    Non-negative CP/PARAFAC decomposition for EEM stacks using HALS with priors and masking.

    This solver factorizes a 3-way tensor `X` (typically an excitation–emission matrix
    stack with shape `(n_samples, n_ex, n_em)`) into non-negative CP factors:

        X ≈ [[A, B, C]]

    where:
    - `A` (n_samples × rank) contains sample scores / concentrations,
    - `B` (n_ex × rank) contains excitation loadings,
    - `C` (n_em × rank) contains emission loadings.

    Optional features supported by this implementation:
    - Elementwise masking (for ignoring NaNs or predefined regions).
    - Quadratic priors on any factor (A/B/C), with NaNs allowed to skip entries.
    - Elastic-net regularization on any factor (L1/L2 mix).
    - A ratio constraint on paired rows of A: A[idx_top] ≈ beta * A[idx_bot],
      controlled by `lam` and estimated per component (`beta`).
    - Component alignment to reference components using the Hungarian algorithm
      (`prior_ref_components`).

    Parameters
    ----------
    tensor : array-like, shape (I, J, K)
        Input tensor (EEM stack). Non-negative values are expected.
    rank : int
        Number of CP components.
    prior_dict_A : dict[int, array-like], optional
        Priors for columns of `A` (sample-mode factor). Each prior vector must have
        length `I`. NaNs indicate entries without a prior.
    prior_dict_B : dict[int, array-like], optional
        Priors for columns of `B` (excitation factor), length `J`.
    prior_dict_C : dict[int, array-like], optional
        Priors for columns of `C` (emission factor), length `K`.
    gamma_A : float or dict[int, float], optional
        Prior weight(s) for `A`.
    gamma_B : float or dict[int, float], optional
        Prior weight(s) for `B`.
    gamma_C : float or dict[int, float], optional
        Prior weight(s) for `C`.
    prior_ref_components : dict[int, array-like], optional
        Reference components used to permute factors for consistent component order.
        Each value should be a flattened (J*K,) reference spectrum (e.g., outer(B,C)).
    alpha_A : float, optional
        Elastic-net strength for `A`.
    alpha_B : float, optional
        Elastic-net strength for `B`.
    alpha_C : float, optional
        Elastic-net strength for `C`.
    l1_ratio : float, optional
        Elastic-net mixing parameter in [0, 1].
    lam : float or dict[int, float], optional
        Ratio-penalty weight(s) for the sample-mode factor `A`. If provided together
        with `idx_top` and `idx_bot`, `beta` is estimated per component.
    idx_top : sequence[int], optional
        Row indices in `A` corresponding to the "top/original" samples in paired data.
    idx_bot : sequence[int], optional
        Row indices in `A` corresponding to the "bottom/perturbed" samples in paired data.
    max_iter_als : int, optional
        Maximum number of outer ALS iterations.
    max_iter_nnls : int, optional
        Maximum number of inner HALS iterations for each NNLS subproblem.
    tol : float, optional
        Relative convergence tolerance on masked reconstruction error.
    eps : float, optional
        Small positive constant for numerical stability and positivity floor.
    init : str, {"random", "svd", "nndsvd", "nndsvda", "nndsvdar", "ordinary_cp", "custom"}, optional
        Initialization method for factors.
    custom_init : tuple(A, B, C), optional
        Custom initialization, only used when `init="custom"`.
    random_state : int or None, optional
        Random seed for reproducible initialization.
    fixed_components : list[int], optional
        Component indices to keep fixed during updates (useful when seeding known spectra).
    mask : array-like, shape (I, J, K), optional
        Elementwise mask (1 = keep / valid entry, 0 = ignore entry). If None, finite
        entries of `tensor` are treated as valid.

    Returns
    -------
    A : np.ndarray, shape (I, rank)
        Sample-mode factor (scores / concentrations).
    B : np.ndarray, shape (J, rank)
        Excitation factor loadings.
    C : np.ndarray, shape (K, rank)
        Emission factor loadings.
    beta : np.ndarray or None, shape (rank,)
        Estimated per-component ratio between paired sample rows when the ratio
        penalty is enabled; otherwise None.
    """
    # Ensure tensor
    X = tl.tensor(tensor, dtype=float)
    if mask is None:
        mask = np.isfinite(X)  # fallback if user just passed in nan-masked data
    mask = tl.tensor(mask.astype(float))
    I, J, K = X.shape
    rng = np.random.RandomState(random_state)
    if np.isfinite(X).any():
        X = process_eem_stack(X, eem_nan_imputing, ex_range=np.arange(X.shape[1]), em_range=np.arange(X.shape[2]))

    # Initialize factors A, B, C

    if init == 'random':
        A = tl.clip(rng.rand(I, rank), a_min=eps)
        B = tl.clip(rng.rand(J, rank), a_min=eps)
        C = tl.clip(rng.rand(K, rank), a_min=eps)
    elif init in ('svd', 'nndsvd', 'nndsvda', 'nndsvdar'):
        # Use 2D initialization on each mode unfolding
        # Mode-0 init for A
        X1 = tl.unfold(X, mode=0)
        W1, _ = unfolded_eem_stack_initialization(tl.to_numpy(X1), rank, method=init)
        A = tl.tensor(np.clip(W1, a_min=eps, a_max=None), dtype=float)
        # Mode-1 init for B
        X2 = tl.unfold(X, mode=1)
        W2, _ = unfolded_eem_stack_initialization(tl.to_numpy(X2), rank, method=init)
        B = tl.tensor(np.clip(W2, a_min=eps, a_max=None), dtype=float)
        # Mode-2 init for C
        X3 = tl.unfold(X, mode=2)
        W3, _ = unfolded_eem_stack_initialization(tl.to_numpy(X3), rank, method=init)
        C = tl.tensor(np.clip(W3, a_min=eps, a_max=None), dtype=float)
    elif init == 'ordinary_cp':
        A, B, C = non_negative_parafac_hals(X, rank=rank, random_state=random_state)[1]
    elif init == 'custom':
        A, B, C = custom_init
    else:
        raise ValueError(f"Unknown init mode: {init}")

    # Default empty priors
    if prior_dict_B is None:
        prior_dict_B = {}
    if prior_dict_C is None:
        prior_dict_C = {}
    if prior_dict_A is None:
        prior_dict_A = {}
    elif prior_ref_components is not None:
        H = np.zeros([rank, B.shape[0] * C.shape[0]])
        for r in range(rank):
            component = np.array([B[:, r]]).T.dot(np.array([C[:, r]]))
            H[r, :] = component.reshape(-1)
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, h_idx = linear_sum_assignment(cost_mat)
        query_idx = [prior_keys[i] for i in query_idx]
        A_new, B_new, C_new = np.zeros(A.shape), np.zeros(B.shape), np.zeros(C.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, h_idx):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
            r_list_query.remove(qi)
            r_list_ref.remove(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
        A, B, C = A_new, B_new, C_new

    if np.isnan(A).any():
        print(f"A contains NaN in init")
    if np.isnan(B).any():
        print(f"B contains NaN in init")
    if np.isnan(C).any():
        print(f"C contains NaN in init")

    if isinstance(lam, dict):
        lam = {k: lam.get(k, 0.0) for k in range(rank)}
    elif lam > 0:
        lam = {k: lam for k in range(rank)}

    if lam is not None and idx_top is not None and idx_bot is not None:
        beta = np.ones(rank, dtype=float)
    else:
        beta = None
    prev_error = masked_tensor_norm_error(X, cp_to_tensor((None, [A, B, C])), mask)
    for iteration in range(max_iter_als):
        # Update B:
        UtM = masked_unfolding_dot_khatri_rao(X, (None, [A, B, C]), 1, mask).T
        if np.isnan(UtM).any():
            print(f"UtM contains NaN in {iteration}")
        UtU = (C.T @ C) * (A.T @ A)
        if np.isnan(UtU).any():
            print(f"UtU contains NaN in {iteration}")
        B = hals_prior_nnls(
            UtM=UtM,
            UtU=UtU,
            prior_dict=prior_dict_B,
            V=B.T,
            gamma=gamma_B,
            alpha=alpha_B,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls,
            fixed_components=fixed_components,
        )
        B = B.T
        if np.isnan(B).any():
            print(f"B contains NaN in {iteration}")

        # Update C:
        UtM = masked_unfolding_dot_khatri_rao(X, (None, [A, B, C]), 2, mask).T
        if np.isnan(UtM).any():
            print(f"UtM contains NaN in {iteration}")
        UtU = (B.T @ B) * (A.T @ A)  # shape (rank, rank)
        if np.isnan(UtU).any():
            print(f"UtU contains NaN in {iteration}")
        C = hals_prior_nnls(
            UtM=UtM,
            UtU=UtU,
            prior_dict=prior_dict_C,
            V=C.T,
            gamma=gamma_C,
            alpha=alpha_C,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls,
            # fixed_components=fixed_components,
        )
        C = C.T
        if np.isnan(C).any():
            print(f"C contains NaN in {iteration}")

        if lam is not None and idx_top is not None and idx_bot is not None:
            # --- Update W via ratio-aware HALS columns ---
            UtM = masked_unfolding_dot_khatri_rao(X, (None, [A, B, C]), 0, mask).T
            UtU = (C.T @ C) * (B.T @ B)
            for k in range(rank):
                Rk = UtM[k].copy()
                for j in range(rank):
                    if j != k:
                        Rk -= UtU[k, j] * A[:, j]
                d = UtU[k, k]
                A[:, k] = hals_column_with_ratio(
                    Rk=Rk,
                    hk_norm2=d,
                    beta_k=beta[k],
                    lam=lam[k],
                    k=k,
                    prior_dict=prior_dict_A,
                    gamma=gamma_A,
                    alpha=alpha_A,
                    l1_ratio=l1_ratio,
                    idx_top=idx_top,
                    idx_bot=idx_bot,
                    eps=eps
                )

            # --- Beta‐step (closed form) ---
            beta = update_beta(A, idx_top=idx_top, idx_bot=idx_bot, eps=eps)

        else:
            # Update A:
            UtM = masked_unfolding_dot_khatri_rao(X, (None, [A, B, C]), 0, mask).T
            if np.isnan(UtM).any():
                print(f"UtM contains NaN in {iteration}")
            UtU = (C.T @ C) * (B.T @ B)
            if np.isnan(UtU).any():
                print(f"UtU contains NaN in {iteration}")
            A = hals_prior_nnls(
                UtM=UtM,
                UtU=UtU,
                prior_dict=prior_dict_A,
                V=A.T,
                gamma=gamma_A,
                alpha=alpha_A,
                l1_ratio=l1_ratio,
                tol=tol,
                eps=eps,
                max_iter=max_iter_nnls
            )
            A = A.T
            if np.isnan(A).any():
                print(f"A contains NaN in {iteration}")

        # Check convergence
        reconstructed = cp_to_tensor((None, [A, B, C]))
        err = masked_tensor_norm_error(X, reconstructed, mask)
        if abs(prev_error - err) / (prev_error + eps) < tol:
            break
        prev_error = err

    if prior_ref_components is not None:
        H = np.zeros([rank, B.shape[0] * C.shape[0]])
        for r in range(rank):
            component = np.array([B[:, r]]).T.dot(np.array([C[:, r]]))
            H[r, :] = component.reshape(-1)
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, h_idx = linear_sum_assignment(cost_mat)
        query_idx = [prior_keys[i] for i in query_idx]
        A_new, B_new, C_new = np.zeros(A.shape), np.zeros(B.shape), np.zeros(C.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, h_idx):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
            r_list_query.remove(qi)
            r_list_ref.remove(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            A_new[:, qi] = A[:, ri]
            B_new[:, qi] = B[:, ri]
            C_new[:, qi] = C[:, ri]
        A, B, C = A_new, B_new, C_new

    return A, B, C, beta


def replace_factor_with_prior(factors, prior, replaced_mode, replaced_rank="best-fit", frozen_rank=None,
                              project_prior=True, X=None, show_replaced_rank=False):
    """
    Replace a single component in a two-factor decomposition with a prior vector.

    This helper is primarily used for matrix factorization models (e.g., NMF on an
    unfolded EEM or a component matrix) where `factors` contains two factor matrices.
    It can (i) select which component to replace by maximizing correlation with the
    prior ("best-fit"), (ii) directly replace a specified component, and (iii)
    optionally project the prior to the other factor to keep the reconstruction
    consistent.

    Notes
    -----
    - Replacement is performed in-place on `factors`.
    - When `project_prior=True`, the method computes a residual excluding the
      replaced component and projects the prior onto the other mode using a
      non-negative least-squares-style clipping.

    Parameters
    ----------
    factors : list[np.ndarray]
        Factor matrices for a rank-`r` matrix model. Expected shapes are
        `[F0, F1]` where `F0.shape = (m, r)` and `F1.shape = (n, r)`.
    prior : array-like, shape (m,) or (n,)
        Prior vector to insert into the selected component of `factors[replaced_mode]`.
    replaced_mode : int
        Which factor to modify: 0 or 1.
    replaced_rank : int or {"best-fit"}, optional
        Component index to replace. If "best-fit", the component with the highest
        Pearson correlation to `prior` is replaced (excluding `frozen_rank`).
    frozen_rank : int or None, optional
        Component index that must not be replaced (useful when one component is fixed).
    project_prior : bool, optional
        If True, also update the complementary factor column by projecting the prior
        onto the residual (requires `X`).
    X : np.ndarray, optional
        Data matrix being factorized, shape (m, n). Required if `project_prior=True`.
    show_replaced_rank : bool, optional
        If True, also return the selected `replaced_rank`.

    Returns
    -------
    factors : list[np.ndarray]
        Updated factor matrices.
    replaced_rank : int, optional
        Only returned when `show_replaced_rank=True`.
    """
    rank = factors[replaced_mode].shape[1]
    if replaced_rank == "best-fit":
        sim = -1
        replaced_rank = 0
        for j in [i for i in range(rank) if i != frozen_rank]:
            factor_init = factors[replaced_mode][:, j]
            r, _ = pearsonr(factor_init, prior)
            if r > sim:
                sim = r
                replaced_rank = j
        factors[replaced_mode][:, replaced_rank] = prior
    else:
        factors[replaced_mode][:, replaced_rank] = prior

    if project_prior and X is not None:
        residual = X
        for j in [i for i in range(rank) if i != replaced_rank]:
            residual -= np.outer(factors[0][:, j], factors[1][:, j])
        if replaced_mode == 0:
            E = residual.T
        elif replaced_mode == 1:
            E = residual
        projection = E @ prior / np.inner(prior, prior)
        projection[projection < 0] = 0
        factors[int(1 - replaced_mode)][:, replaced_rank] = projection

    if show_replaced_rank:
        return factors, replaced_rank
    else:
        return factors


def hals_column_with_ratio(
        Rk,
        hk_norm2,
        beta_k,
        lam,
        k,
        prior_dict=None,
        gamma=0.0,
        alpha=0.0,
        l1_ratio=0.0,
        idx_top=None,
        idx_bot=None,
        eps=1e-8
):
    """
    HALS-style update for one column with a paired-row ratio penalty.

    This update is designed for sample-mode factors (e.g., `W` in NMF or `A` in CP)
    when the dataset contains paired samples (top/bottom) that are expected to follow
    a component-wise ratio:

        w[top_i] ≈ beta_k * w[bot_i]

    The subproblem solved for a single component column `w_k` is:

        1/2 || Rk - d * w_k ||_2^2
      + (lam/2) * sum_i (w[top_i] - beta_k * w[bot_i])^2
      + (gamma/2) * ||mask * (w_k - p_k)||_2^2
      + alpha * [ l1_ratio * ||w_k||_1 + (1 - l1_ratio)/2 * ||w_k||_2^2 ]

    where NaNs in the prior are ignored (masked out).

    Parameters
    ----------
    Rk : np.ndarray, shape (m,)
        Current HALS residual for component `k` (with the k-th contribution restored).
    hk_norm2 : float
        Squared norm of the corresponding component in the other factor(s)
        (i.e., the diagonal element `d` of the Gram matrix).
    beta_k : float
        Current ratio estimate for component `k`.
    lam : float
        Ratio penalty weight for component `k`.
    k : int
        Component index (used to look up `prior_dict[k]`).
    prior_dict : dict[int, array-like], optional
        Mapping from component index to a prior vector of length `m` (NaNs allowed).
    gamma : float, optional
        Prior weight.
    alpha : float, optional
        Elastic-net strength.
    l1_ratio : float, optional
        Elastic-net mixing parameter in [0, 1].
    idx_top : sequence[int]
        Indices of "top/original" rows in the paired sample design.
    idx_bot : sequence[int]
        Indices of "bottom/perturbed" rows paired with `idx_top`.
    eps : float, optional
        Small constant used as a positivity floor and for numerical stability.

    Returns
    -------
    w_new : np.ndarray, shape (m,)
        Updated non-negative column for component `k`.
    """
    m = Rk.shape[0]
    d = hk_norm2
    g = Rk.copy()

    # Elastic‐net
    l2_pen = alpha * (1 - l1_ratio)
    l1_pen = alpha * l1_ratio
    if l1_pen:
        g -= l1_pen

    # Prepare prior vector and mask
    if prior_dict is None:
        prior_dict = {}
    has_prior = (gamma > 0 and k in prior_dict)
    if has_prior:
        p_k = np.asarray(prior_dict[k], dtype=float)
        mask = np.isfinite(p_k)
        p_clean = np.nan_to_num(p_k, nan=0.0)
        # Add linear term once (will split per-entry below)
        g += gamma * p_clean

    w_new = np.empty_like(g)

    # Off-diagonal for ratio block
    off = -lam * beta_k

    # Solve each paired (t,b)
    for t, b in zip(idx_top, idx_bot):
        R1, R2 = g[t], g[b]
        # per-entry prior diag
        prior_t = gamma if (has_prior and mask[t]) else 0.0
        prior_b = gamma if (has_prior and mask[b]) else 0.0
        # build local diagonals
        a11 = d + lam + l2_pen + prior_t
        a22 = d + lam * beta_k ** 2 + l2_pen + prior_b
        det = a11 * a22 - off * off + eps

        # compute scalar updates
        w1 = (a22 * R1 - off * R2) / det
        w2 = (-off * R1 + a11 * R2) / det

        w_new[t] = float(max(eps, w1))
        w_new[b] = float(max(eps, w2))

    return w_new


def update_beta(
        W: np.ndarray,
        idx_top,
        idx_bot,
        eps: float = 0,
        boundaries: tuple = (0.95, 2)
) -> np.ndarray:
    """
    Estimate per-component ratios between paired sample rows.

    Given a concentration / score matrix `W` and paired row indices (`idx_top`, `idx_bot`),
    this function estimates a ratio `beta[j]` for each component `j` such that:

        W[idx_top, j] ≈ beta[j] * W[idx_bot, j]

    The least-squares solution is computed independently per component and then
    clamped to a user-provided interval.

    Parameters
    ----------
    W : np.ndarray, shape (m, r)
        Score / concentration matrix with `m` samples and `r` components.
    idx_top : sequence[int]
        Row indices corresponding to the "top/original" samples.
    idx_bot : sequence[int]
        Row indices corresponding to the "bottom/perturbed" samples.
        Must have the same length as `idx_top`.
    eps : float, optional
        Small constant added to the denominator to avoid division by zero.
    boundaries : tuple[float, float], optional
        (min_beta, max_beta) bounds used to clamp each estimated ratio.

    Returns
    -------
    beta : np.ndarray, shape (r,)
        Estimated ratio for each component, clamped to `[min_beta, max_beta]`.
    """
    W = np.asarray(W, dtype=float)
    idx_top = np.asarray(idx_top, dtype=int)
    idx_bot = np.asarray(idx_bot, dtype=int)
    if idx_top.shape != idx_bot.shape:
        raise ValueError("`idx_top` and `idx_bot` must have the same length")

    # Extract the paired rows
    W_top = W[idx_top, :]  # shape (p, r)
    W_bot = W[idx_bot, :]  # shape (p, r)

    # Compute numerator and denominator for each component j:
    #   numerator_j   = sum_i W_top[i,j] * W_bot[i,j]
    #   denominator_j = sum_i W_bot[i,j]^2
    num = np.sum(W_top * W_bot, axis=0)
    den = np.sum(W_bot * W_bot, axis=0) + eps

    beta = num / den

    # Clamp into the desired interval
    beta_min, beta_max = boundaries
    return np.clip(beta, beta_min, beta_max)


def nmf_hals_prior(
        X,
        rank,
        prior_dict_H=None,
        prior_dict_W=None,
        prior_dict_A=None,
        prior_dict_B=None,
        prior_dict_C=None,
        prior_ref_components=None,
        gamma_W=0,
        gamma_H=0,
        gamma_A=0,
        gamma_B=0,
        gamma_C=0,
        alpha_W=0,
        alpha_H=0,
        alpha_A=0,
        alpha_B=0,
        alpha_C=0,
        l1_ratio=0,
        idx_top=None,
        idx_bot=None,
        lam=0,
        fit_rank_one=False,
        component_shape=None,
        max_iter_als=500,
        max_iter_nnls=500,
        tol=1e-6,
        eps=1e-8,
        init='random',
        custom_init=None,
        fixed_components=None,
        random_state=None
):
    """
    Non-negative matrix factorization (NMF) with HALS updates, priors, and optional EEM rank-1 components.

    This solver decomposes a non-negative matrix `X` into:

        X ≈ W @ H

    where `W` (m × rank) contains sample scores / concentrations and `H` (rank × n)
    contains component spectra/features. It is designed for EEM decomposition where
    `X` can be an unfolded EEM stack and where components may be constrained by
    priors or by a paired-sample ratio constraint.

    Supported options include:
    - Quadratic priors on `W` and/or `H` (NaNs skip entries).
    - Elastic-net regularization on `W` and/or `H`.
    - A paired-row ratio penalty on `W`: W[idx_top] ≈ beta * W[idx_bot] with weight `lam`.
    - A hybrid model where selected components are forced to be rank-1 outer products
      in EEM space (CP-like): `H[k]` reshaped to `(n_ex, n_em)` is approximated by
      `outer(B[:,k], C[:,k])`. This is controlled via `fit_rank_one` and requires
      `component_shape`.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        Non-negative data matrix (e.g., unfolded EEM stack).
    rank : int
        Number of components.
    prior_dict_H : dict[int, array-like], optional
        Priors for rows of `H` (length `n`). NaNs skip entries.
    prior_dict_W : dict[int, array-like], optional
        Priors for columns of `W` (length `m`). NaNs skip entries.
    prior_dict_A : dict[int, array-like], optional
        Priors for CP sample-mode factor `A` when using rank-1 components.
    prior_dict_B : dict[int, array-like], optional
        Priors for CP excitation factor `B` when using rank-1 components.
    prior_dict_C : dict[int, array-like], optional
        Priors for CP emission factor `C` when using rank-1 components.
    prior_ref_components : dict[int, array-like], optional
        Reference components used to permute the solution for consistent ordering.
    gamma_W : float or dict[int, float], optional
        Prior weight(s) for `W`.
    gamma_H : float or dict[int, float], optional
        Prior weight(s) for `H`.
    gamma_A, gamma_B, gamma_C : float or dict[int, float], optional
        Prior weight(s) for CP factors in the rank-1 part.
    alpha_W : float, optional
        Elastic-net strength for `W`.
    alpha_H : float, optional
        Elastic-net strength for `H`.
    alpha_A, alpha_B, alpha_C : float, optional
        Elastic-net strengths for CP factors (rank-1 part).
    l1_ratio : float, optional
        Elastic-net mixing parameter in [0, 1].
    idx_top : sequence[int], optional
        Row indices in `W` for the paired "top/original" samples.
    idx_bot : sequence[int], optional
        Row indices in `W` for the paired "bottom/perturbed" samples.
    lam : float or dict[int, float], optional
        Ratio-penalty weight(s). Enable ratio constraint when provided together
        with `idx_top` and `idx_bot`.
    fit_rank_one : dict[int, bool] or False, optional
        If a dict, `fit_rank_one[k]=True` marks component `k` to be modeled as a
        rank-1 outer product in EEM space (CP-like). If False, standard NMF is used.
    component_shape : tuple[int, int], optional
        `(n_ex, n_em)` used to reshape rows of `H` when using rank-1 components.
    max_iter_als : int, optional
        Maximum number of outer ALS iterations.
    max_iter_nnls : int, optional
        Maximum number of HALS iterations per NNLS subproblem.
    tol : float, optional
        Relative convergence tolerance on the reconstruction error.
    eps : float, optional
        Small positive constant used as a positivity floor.
    init : str, {"random", "svd", "nndsvd", "nndsvda", "nndsvdar", "ordinary_nmf", "ordinary_cp", "custom"}, optional
        Initialization method.
    custom_init : tuple(W, H), optional
        Custom initialization, only used when `init="custom"`.
    fixed_components : list[int], optional
        Indices of components to keep fixed during updates.
    random_state : int or None, optional
        Random seed.

    Returns
    -------
    W : np.ndarray, shape (m, rank)
        Non-negative sample scores / concentrations.
    H : np.ndarray, shape (rank, n)
        Non-negative component matrix (flattened EEM components if applicable).
    beta : np.ndarray or None, shape (rank,)
        Per-component ratio estimates when the ratio penalty is enabled; otherwise None.
    """

    m, n = X.shape
    rng = np.random.RandomState(random_state)
    fixed_components = [] if fixed_components is None else fixed_components

    # 1) Initialize W, H
    if init == 'random':
        W = np.clip(rng.rand(m, rank), eps, None)
        H = np.clip(rng.rand(rank, n), eps, None)
    elif init in ('svd', 'nndsvd', 'nndsvda', 'nndsvdar'):
        W, H = unfolded_eem_stack_initialization(X, rank, init)
    elif init == 'ordinary_nmf':
        model = NMF(n_components=rank, init='nndsvda', random_state=random_state, max_iter=1000)
        W = model.fit_transform(X)
        H = model.components_
    elif init == 'ordinary_cp':
        _, factors_init = non_negative_parafac_hals(
            X.reshape([m, component_shape[0], component_shape[1]]),
            rank=rank,
            random_state=random_state
        )
        W = factors_init[0]
        H = khatri_rao(factors_init[1], factors_init[2]).T
        # H = np.array([np.outer(factors_init[1][:, r], factors_init[2][:, r]).flatten() for r in range(rank)])
    elif init == 'custom':
        W, H = custom_init
    else:
        raise ValueError(f"Unknown init {init}")

    # Default empty priors
    if prior_ref_components is not None:
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, h_idx = linear_sum_assignment(cost_mat)
        query_idx = [prior_keys[i] for i in query_idx]
        H_new, W_new = np.zeros(H.shape), np.zeros(W.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, h_idx):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
            r_list_query.remove(qi)
            r_list_ref.remove(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
        W, H = W_new, H_new

    if isinstance(lam, dict):
        lam = {k: lam.get(k, 0.0) for k in range(rank)}
    elif lam >= 0:
        lam = {k: lam for k in range(rank)}
    else:
        lam = None

    if lam is not None and idx_bot is not None and idx_top is not None:
        beta = np.ones(rank, dtype=float)
    else:
        beta = None
    prev_err = np.inf

    r_nmf, r_cp = [], []
    fixed_components_nmf, fixed_components_cp = [], []
    if fit_rank_one:
        for k in range(rank):
            if fit_rank_one.get(k, False):
                r_cp.append(k)
            else:
                r_nmf.append(k)
        for fc in fixed_components:
            if fc in r_nmf:
                fixed_components_nmf.append(r_nmf.index(fc))
            elif fc in r_cp:
                fixed_components_cp.append(r_cp.index(fc))

    prior_dict_W = {} if prior_dict_W is None else prior_dict_W
    prior_dict_H = {} if prior_dict_H is None else prior_dict_H
    prior_dict_A = {} if prior_dict_A is None else prior_dict_A
    prior_dict_B = {} if prior_dict_B is None else prior_dict_B
    prior_dict_C = {} if prior_dict_C is None else prior_dict_C

    if r_cp:
        prior_dict_W_nmf, prior_dict_H_nmf, prior_dict_A_cp, prior_dict_B_cp, prior_dict_C_cp = {}, {}, {}, {}, {}
        A = np.zeros((m, len(r_cp)))
        B = np.zeros((component_shape[0], len(r_cp)))
        C = np.zeros((component_shape[1], len(r_cp)))
        for i, r in enumerate(r_cp):
            U_r, S_r, VT_r = np.linalg.svd(H[r, :].reshape(component_shape), full_matrices=False)
            sigma1 = S_r[0]
            u1 = U_r[:, 0] * np.sqrt(sigma1)
            v1 = VT_r[0, :] * np.sqrt(sigma1)
            # Flip sign so that dominant peaks of singular vectors are positive
            if u1[np.argmax(np.abs(u1))] < 0:
                u1 = -u1
                v1 = -v1
            u1 = np.clip(u1, 0, None)  # Ensure non-negativity
            v1 = np.clip(v1, 0, None)  # Ensure non-negativity
            A[:, i] = W[:, r]
            B[:, i] = u1
            C[:, i] = v1
            if prior_dict_A is not None and r in prior_dict_A:
                prior_dict_A_cp[i] = prior_dict_A[r]
            if prior_dict_B is not None and r in prior_dict_B:
                prior_dict_B_cp[i] = prior_dict_B[r]
            if prior_dict_C is not None and r in prior_dict_C:
                prior_dict_C_cp[i] = prior_dict_C[r]

        if r_nmf:
            for j, r in enumerate(r_nmf):
                if prior_dict_W is not None and r in prior_dict_W:
                    prior_dict_W_nmf[j] = prior_dict_W[r]
                if prior_dict_H is not None and r in prior_dict_H:
                    prior_dict_H_nmf[j] = prior_dict_H[r]
            beta_cp = np.ones(len(r_cp), dtype=float) if beta is not None else None
            beta_nmf = np.ones(len(r_nmf), dtype=float) if beta is not None else None
            W = W[:, r_nmf]
            H = H[r_nmf, :]

    def one_step_nmf(X0, W0, H0, beta0, prior_dict_W0, prior_dict_H0, fixed_components_specific):
        UtM_H = tl.dot(tl.transpose(W0), X0)
        UtU_H = tl.dot(tl.transpose(W0), W0)
        H0 = hals_prior_nnls(
            UtM=UtM_H,
            UtU=UtU_H,
            prior_dict=prior_dict_H0,
            V=H0,
            gamma=gamma_H,
            alpha=alpha_H,
            l1_ratio=l1_ratio,
            max_iter=max_iter_nnls,
            tol=tol,
            eps=eps,
            fixed_components=fixed_components_specific,
        )

        if lam is not None and idx_bot is not None and idx_top is not None:
            # --- Update W via ratio-aware HALS columns ---
            UtM_W = tl.to_numpy(tl.dot(H0, tl.transpose(X0)))  # (r, m)
            UtU_W = tl.to_numpy(tl.dot(H0, tl.transpose(H0)))  # (r, r)
            for k in range(W0.shape[1]):
                Rk = UtM_W[k].copy()
                for j in range(W0.shape[1]):
                    if j != k:
                        Rk -= UtU_W[k, j] * W0[:, j]
                d = UtU_W[k, k]
                W0[:, k] = hals_column_with_ratio(
                    Rk=Rk,
                    hk_norm2=d,
                    beta_k=beta0[k],
                    lam=lam[k],
                    k=k,
                    prior_dict=prior_dict_W0,
                    gamma=gamma_W,
                    alpha=alpha_W,
                    l1_ratio=l1_ratio,
                    idx_top=idx_top,
                    idx_bot=idx_bot,
                    eps=eps
                )
            # --- Beta‐step (closed form) ---
            beta0 = update_beta(W0, idx_top=idx_top, idx_bot=idx_bot, eps=eps)
        else:
            # Update W (columns) via HALS on W^T
            UtM_W = tl.dot(H0, tl.transpose(X0))
            UtU_W = tl.dot(H0, tl.transpose(H0))
            W0t = hals_prior_nnls(
                UtM=UtM_W,
                UtU=UtU_W,
                prior_dict=prior_dict_W0,
                V=W0.T,
                gamma=gamma_W,
                alpha=alpha_W,
                l1_ratio=l1_ratio,
                tol=tol,
                eps=eps,
                max_iter=max_iter_nnls,
            )
            W0 = tl.transpose(W0t)

        return W0, H0, beta0 if lam is not None and idx_top is not None and idx_bot is not None else None

    def one_step_cp(X0, A0, B0, C0, beta0, prior_dict_A0, prior_dict_B0, prior_dict_C0, fixed_components_specific):

        M = X0.reshape((m, component_shape[0], component_shape[1]))
        # Update B:
        UtM = unfolding_dot_khatri_rao(M, (None, [A0, B0, C0]), 1).T
        UtU = (C0.T @ C0) * (A0.T @ A0)
        B0 = hals_prior_nnls(
            UtM=UtM,
            UtU=UtU,
            prior_dict=prior_dict_B0,
            V=B0.T,
            gamma=gamma_B,
            alpha=alpha_B,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls,
            fixed_components=fixed_components_specific,
        )
        B0 = B0.T

        # Update C:
        UtM = unfolding_dot_khatri_rao(M, (None, [A0, B0, C0]), 2).T
        UtU = (B0.T @ B0) * (A0.T @ A0)  # shape (rank, rank)
        C0 = hals_prior_nnls(
            UtM=UtM,
            UtU=UtU,
            prior_dict=prior_dict_C0,
            V=C0.T,
            gamma=gamma_C,
            alpha=alpha_C,
            l1_ratio=l1_ratio,
            tol=tol,
            eps=eps,
            max_iter=max_iter_nnls,
            fixed_components=fixed_components_specific,
        )
        C0 = C0.T

        if lam is not None and idx_top is not None and idx_bot is not None:
            # --- Update W via ratio-aware HALS columns ---
            UtM = unfolding_dot_khatri_rao(M, (None, [A0, B0, C0]), 0).T
            UtU = (C0.T @ C0) * (B0.T @ B0)
            for k in range(A0.shape[1]):
                Rk = UtM[k].copy()
                for j in range(A0.shape[1]):
                    if j != k:
                        Rk -= UtU[k, j] * A0[:, j]
                d = UtU[k, k]
                A0[:, k] = hals_column_with_ratio(
                    Rk=Rk,
                    hk_norm2=d,
                    beta_k=beta0[k],
                    lam=lam[k],
                    k=k,
                    prior_dict=prior_dict_A0,
                    gamma=gamma_A,
                    alpha=alpha_A,
                    l1_ratio=l1_ratio,
                    idx_top=idx_top,
                    idx_bot=idx_bot,
                    eps=eps
                )

            # --- Beta‐step (closed form) ---
            beta0 = update_beta(A0, idx_top=idx_top, idx_bot=idx_bot, eps=eps)

        else:
            # Update A:
            UtM = unfolding_dot_khatri_rao(M, (None, [A0, B0, C0]), 0).T
            UtU = (C0.T @ C0) * (B0.T @ B0)
            A0 = hals_prior_nnls(
                UtM=UtM,
                UtU=UtU,
                prior_dict=prior_dict_A0,
                V=A0.T,
                gamma=gamma_A,
                alpha=alpha_A,
                l1_ratio=l1_ratio,
                tol=tol,
                eps=eps,
                max_iter=max_iter_nnls
            )
            A0 = A0.T

        return A0, B0, C0, beta0 if lam is not None and idx_top is not None and idx_bot is not None else None

    for n_iter in range(max_iter_als):
        if r_cp and r_nmf:
            X_nmf = X - cp_to_tensor((None, [A, B, C])).reshape((m, -1))
            W, H, beta_nmf = one_step_nmf(X_nmf, W, H, beta_nmf, prior_dict_W_nmf, prior_dict_H_nmf,
                                          fixed_components_nmf)
            X_cp = X - W @ H
            A, B, C, beta_cp = one_step_cp(X_cp, A, B, C, beta_cp, prior_dict_A_cp, prior_dict_B_cp, prior_dict_C_cp,
                                           fixed_components_cp)
            err = tl.norm(X - W @ H - cp_to_tensor((None, [A, B, C])).reshape((m, -1)))
        elif r_cp and not r_nmf:
            A, B, C, beta = one_step_cp(X, A, B, C, beta, prior_dict_A_cp, prior_dict_B_cp, prior_dict_C_cp,
                                        fixed_components_cp)
            err = tl.norm(X - cp_to_tensor((None, [A, B, C])).reshape((m, -1)))
        else:
            W, H, beta = one_step_nmf(X, W, H, beta, prior_dict_W, prior_dict_H, fixed_components)
            err = tl.norm(X - W @ H)
            if np.isnan(W).any():
                print("n_iter: ", n_iter)
                raise ValueError("W contains NaN values")
            if np.isnan(H).any():
                print("n_iter: ", n_iter)
                raise ValueError("H contains NaN values")
        # --- Convergence check ---
        if n_iter > 0 and abs(prev_err - err) / (prev_err + eps) < tol:
            break
        prev_err = err

    if r_cp:
        W_final, H_final = np.zeros((m, rank), dtype=float), np.zeros((rank, n), dtype=float)
        for i, r in enumerate(r_cp):
            component_r = np.outer(B[:, i], C[:, i])
            H_final[r, :] = component_r.reshape(-1)
        for i, r in enumerate(r_nmf):
            H_final[r, :] = H[i, :]
        W_final[:, r_cp] = A
        if r_nmf:
            W_final[:, r_nmf] = W
            beta = np.zeros(rank, dtype=float)
            beta[r_cp] = beta_cp
            beta[r_nmf] = beta_nmf
        H = H_final
        W = W_final

    if prior_ref_components is not None:
        prior_keys = list(prior_ref_components.keys())
        queries = np.array([prior_ref_components[k] for k in prior_keys])
        cost_mat = cdist(queries, H, metric='correlation')
        # run Hungarian algorithm
        query_idx, h_idx = linear_sum_assignment(cost_mat)
        query_idx = [prior_keys[i] for i in query_idx]
        H_new, W_new = np.zeros(H.shape), np.zeros(W.shape)
        r_list_query, r_list_ref = [i for i in range(rank)], [i for i in range(rank)]
        for qi, ri in zip(query_idx, h_idx):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
            r_list_query.remove(qi)
            r_list_ref.remove(ri)
        for qi, ri in zip(r_list_query, r_list_ref):
            W_new[:, qi] = W[:, ri]
            H_new[qi, :] = H[ri, :]
        W, H = W_new, H_new

    return W, H, beta


def solve_W(X1, H, X2=None, beta=None, reg=0.0, non_negativity=True):
    """
    Solve for the sample-score matrix W in a (possibly constrained) linear regression subproblem.

    This helper solves for `W` in one of the following least-squares problems:

    1) Standard regression (single dataset):
           minimize_W || X1 - W @ H ||_F^2

    2) Paired / ratio-augmented regression (two datasets):
           minimize_W || X1 - W @ H ||_F^2
                     + || X2 - W @ diag(beta) @ H ||_F^2

    Optionally, non-negativity can be enforced via row-wise NNLS.

    Parameters
    ----------
    X1 : np.ndarray, shape (m, n)
        Primary data matrix.
    H : np.ndarray, shape (r, n)
        Component matrix (rows are components).
    X2 : np.ndarray, shape (m, n), optional
        Secondary data matrix used when `beta` is provided.
    beta : np.ndarray, shape (r,), optional
        Per-component scaling applied to the secondary term. If None, the second term
        is omitted.
    reg : float, optional
        Ridge (L2) regularization strength added to the normal equations (0 disables).
    non_negativity : bool, optional
        If True, enforce `W >= 0` using NNLS. If False, solve the unconstrained
        normal equations.

    Returns
    -------
    W : np.ndarray, shape (m, r)
        Estimated sample-score / concentration matrix.
    """
    # Validate shapes
    m, n = X1.shape
    r, n_H = H.shape
    assert n_H == n, "H must be of shape (r, n)"
    if beta is not None:
        assert X2 is not None and X2.shape == (m, n), "D must match shape of C when b is provided"
        assert beta.shape[0] == r, "b must have length r"

    # Prepare design matrix and target for NNLS if needed
    if non_negativity:
        # Build block design and targets
        # A: (2n x r) or (n x r) if b is None
        A1 = H.T  # n x r
        Y_blocks = [X1]
        A_blocks = [A1]
        if beta is not None:
            A2 = (np.diag(beta) @ H).T  # n x r
            A_blocks.append(A2)
            Y_blocks.append(X2)
        A = np.vstack(A_blocks)
        # Solve row-wise
        W = np.zeros((m, r))
        for i in range(m):
            y = np.hstack([Y_blocks[j][i] for j in range(len(Y_blocks))])
            W[i], _ = nnls(A, y)
        return W

    # Numerator and Denominator
    if beta is None:
        # Standard regression: only C and H
        numerator = X1 @ H.T
        denominator = H @ H.T
    else:
        B = np.diag(beta)
        numerator = X1 @ H.T + X2 @ H.T @ B
        HHT = H @ H.T
        denominator = HHT + B @ HHT @ B

    # Add ridge regularization if requested
    if reg > 0:
        denominator = denominator + reg * np.eye(r)

    # Solve for W (avoid explicit inverse)
    # Solve (denominator.T) @ X = numerator.T  => X = W.T
    W = np.linalg.solve(denominator.T, numerator.T).T
    return W


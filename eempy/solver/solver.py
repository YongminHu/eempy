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
    masked_tensor = tl.tensor(tensor * mask, dtype=float)
    return tl.dot(unfold(masked_tensor, mode), tl.tenalg.khatri_rao(factors[1], skip_matrix=mode))


def masked_tensor_norm_error(tensor, reconstruction, mask):
    return tl.norm((tensor - reconstruction) * mask)



def unfolded_eem_stack_initialization(M, rank, method='nndsvd'):
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
    FastHALS-style non‐negative least‐squares update for V in an NMF step,
    with per‐component quadratic priors, elastic‐net penalties, and fixed components.

     Solves for each row k of V (length n):
         min_{v>=0}  ½‖R_k - ukk·v‖²
                    + (γ_k/2)‖v - p_k‖²
                    + α[ℓ1‖v‖₁ + (1-ℓ1)/2‖v‖²]
     where
       - R_k = UtM[k] - ∑_{j≠k} UtU[k,j]·V[j] + ukk·V[k]
       - ukk = UtU[k,k]
       - γ_k = gamma_dict.get(k, 0) is the prior weight for component k
       - p_k = prior_dict.get(k, None) is its prior vector (NaNs skipped)
       - α = alpha is the overall elastic‐net weight
       - ℓ1 = l1_ratio mixes L1 vs L2 (0⇒pure L2, 1⇒pure L1)

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
    Perform non-negative PARAFAC/CP decomposition of a 3-way tensor using HALS with optional priors
    and elastic-net penalties on factor matrices A, B, C.

    Decomposes `tensor` of shape (I, J, K) into factors A (I x rank), B (J x rank), C (K x rank) such that:
        tensor ≈ [[A, B, C]]

    Parameters
    ----------
    tensor : array-like, shape (I, J, K)
        Input non-negative tensor.
    rank : int
        Number of components.
    prior_dict_A : dict {r: v_r}, optional
        Priors for columns of A: column r of A is penalized toward vector v_r.
    prior_dict_B : dict {r: v_r}, optional
        Priors for columns of B.
    prior_dict_C : dict {r: v_r}, optional
        Priors for columns of C.
    gamma_A, gamma_B, gamma_C : float, optional
        Quadratic prior weights for A, B, C.
    alpha_A, alpha_B, alpha_C : float, optional
        Elastic-net weights for A, B, C.
    l1_ratio : float in [0,1], optional
        Mix between L1 and L2 for elastic-net.
    max_iter_als : int, optional
        Maximum number of outer ALS iterations.
    max_iter_nnls : int, optional
        Maximum number of inner NNLS interations.
    tol : float, optional
        Convergence tolerance on reconstruction error.
    eps : float, optional
        Small constant to avoid zero division and ensure positivity.
    init : {'random', 'svd', 'nndsvd', 'nndsvda', 'nndsvdar'}, default 'random'
        Initialization scheme for factor matrices.
    random_state : int or None
        Random seed.
    fixed_components: list, optional
        List of indices of components that are fixed as initialization.
    mask: array-like, shape (I, J, K), optional
        Binary mask indicating positions not considered in optimization. 0: positions not considered; 1: valid positions.

    Returns
    -------
    A : ndarray, shape (I, rank)
    B : ndarray, shape (J, rank)
    C : ndarray, shape (K, rank)
    beta: ndarray, shape (rank,)
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
    HALS‐style update for one W‐column w_k ∈ ℝ^m solving

        ½||Rk - d·w_k||²
      + (lam/2)*Σ_i (w[t_i] - β·w[b_i])²
      + (γ/2)*Σ_i (w_i - p_k[i])²
      + α[ℓ1||w_k||₁ + ((1-ℓ1)/2)||w_k||²]

    where:
      - d = ||h_k||²,
      - Rk = UᵀM - Σ_{j≠k}(UᵀU)_{kj} w_j,
      - lam = λ is the ratio penalty,
      - γ = gamma is the prior penalty,
      - α,ℓ1_mix = elastic-net weights,
      - prior_dict[k] = length-m vector p_k (with NaNs where no prior),
      - idx_top/idx_bot pair row-indices covering all rows.
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
    Fit beta per component so that W[idx_top, j] ≈ beta[j] * W[idx_bot, j].

    Solves, for each component j,
        min_{beta_j} ∑_i (W_top[i,j] - beta_j * W_bot[i,j])^2
    which has the closed‐form
        beta_j = sum_i W_top[i,j] * W_bot[i,j]  /  (sum_i W_bot[i,j]^2).

    Parameters
    ----------
    W : np.ndarray, shape (m, r)
        Concentration matrix with m samples and r components.
    idx_top : sequence of ints
        Row indices in W corresponding to the “original” samples.
    idx_bot : sequence of ints
        Row indices in W corresponding to the “perturbed” samples.
        Must be the same length as idx_top.
    eps : float, optional
        Small constant to avoid division by zero when W_bot is nearly zero.
    boundaries : (min_beta, max_beta), optional
        Lower and upper bounds to clamp each estimated beta.

    Returns
    -------
    beta : np.ndarray, shape (r,)
        Estimated ratio for each of the r components, clamped to [min_beta, max_beta].
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
    ALS‐NMF with three penalties:
      - Elastic-net on H and W (alpha_H, alpha_W).
      - Quadratic priors on H and W via prior_dict_H/prior_dict_W (gamma_H, gamma_W).
      - Ratio penalty on W: W[idx_top] ≈ beta * W[idx_bot] (lam=gamma_W).

    Parameters
    ----------
    X : array-like (m, n)
        Non-negative data.
    rank : int
        Number of components.
    prior_dict_W : dict {k: p_k}, optional
        Priors for W columns (length m, NaN values are allowed).
    prior_dict_H : dict {k: p_k}, optional
        Priors for H rows (length n, NaN values are allowed).
    prior_dict_A : dict {k: p_k}, optional
        Priors for A columns (length m, NaN values are allowed).
    prior_dict_B : dict {k: p_k}, optional
        Priors for B columns (length n, NaN values are skip).
    prior_dict_C : dict {k: p_k}, optional
        Priors for C columns (length n, NaN values are skip).
    gamma_W, gamma_H, gamma_A, gamma_B, gamma_C : float
        Quadratic prior weights.
    alpha_W, alpha_H, alpha_A, alpha_B, alpha_C : float
        Elastic-net weights for W and H.
    l1_ratio : float [0,1]
        Mix parameter for elastic-net.
    idx_top, idx_bot : lists of int, length m/2
        Row‐index pairs covering all samples for the **ratio** penalty.
    lam : float
        Ratio penalty weight.
    fit_rank_one : dict {k: bool}, optional,
        If True, fit the k-th component as rank-1 matrix.
    component_shape : tuple (n_ex, n_em), optional,
        Shape of the components. Mandatory if fit_rank_one is not False or init is 'ordinary_cp'.
    max_iter_als : int
        Outer ALS iterations.
    max_iter_nnls : int
        Inner HALS iterations for H.
    tol : float
        Convergence tolerance.
    eps : float
        Small positive floor.
    init : {'random','svd',...}, custom_init, random_state : as before.

    Returns
    -------
    W : ndarray (m, rank)
    H : ndarray (rank, n)
    beta : ndarray (rank,)
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
            W, H, beta_nmf = one_step_nmf(X_nmf, W, H, beta_nmf, prior_dict_W_nmf, prior_dict_H_nmf, fixed_components_nmf)
            X_cp = X - W @ H
            A, B, C, beta_cp = one_step_cp(X_cp, A, B, C, beta_cp, prior_dict_A_cp, prior_dict_B_cp, prior_dict_C_cp, fixed_components_cp)
            err = tl.norm(X - W @ H - cp_to_tensor((None, [A, B, C])).reshape((m, -1)))
        elif r_cp and not r_nmf:
            A, B, C, beta = one_step_cp(X, A, B, C, beta, prior_dict_A_cp, prior_dict_B_cp, prior_dict_C_cp, fixed_components_cp)
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
    Solve for W in the regression problem:
        loss = ||X1 - W @ H||_F^2
             + ||X2 - W @ diag(beta) @ H||_F^2  (optional if b is provided)

    If b is None, reduces to standard regression: minimize ||C - W @ H||_F^2.  In that case D and b are ignored.

    Arguments:
        X1 (ndarray): m x n matrix.
        X2 (ndarray, optional): m x n matrix.  Required if b is not None.
        H (ndarray): r x n matrix.
        beta (ndarray, optional): vector of length r.  If None, drop the second term.
        reg (float): optional regularization (ridge) parameter.
        non_negativity (bool): whether to apply non-negativity.

    Returns:
        W (ndarray): m x r solution matrix.
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


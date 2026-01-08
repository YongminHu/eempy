import numpy as np
import warnings


def multi_matrices_khatri_rao(matrices, weights=None, skip_matrix=None, mask=None):
    """
    Khatri-Rao product of a list of matrices.
    This can be seen as a column-wise kronecker product. (see [1]_ for more details).
    If one matrix only is given, that matrix is directly returned.
    This function is adpated from tensorly.tenalg.khatri_rao

    Parameters
    ----------
    matrices : list of np.ndarray
        List of matrices with the same number of columns

    weights : np.ndarray, shape (m,), optional
        Array of weights for each rank, of length m, the number of columns of the factors
        (i.e. m == matrices[i].shape[1] for any factor).

    skip_matrix : int or None, optional
        If not None, index of a matrix to skip.

    mask : np.ndarray or None, optional
        Mask applied elementwise to the output after computing the Khatri-Rao product.
        It will be flattened and broadcast over columns.

    Returns
    -------
    khatri_rao_product : np.ndarray, shape (prod(n_i), m)
        Where ``prod(n_i) = prod([m.shape[0] for m in matrices])``.

    References
    ----------
    [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
        SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    if len(matrices) == 0:
        raise ValueError("matrices must contain at least one matrix.")

    # Convert inputs to numpy arrays (without copying when possible)
    matrices = [np.asarray(m) for m in matrices]
    if weights is not None:
        weights = np.asarray(weights)
    if mask is not None:
        mask = np.asarray(mask)

    # Khatri-rao of only one matrix: just return that matrix
    if len(matrices) == 1:
        res = matrices[0]
        m = mask.reshape(-1, 1) if mask is not None else 1
        return res * m

    if matrices[0].ndim == 2:
        n_columns = matrices[0].shape[1]
    else:
        n_columns = 1
        matrices = [m.reshape(-1, 1) for m in matrices]
        warnings.warn(
            "Khatri-rao of a series of vectors instead of matrices. "
            "Considering each as a matrix with 1 column."
        )

    for i, e in enumerate(matrices[1:]):
        if not i:
            if weights is None:
                res = matrices[0]
            else:
                res = matrices[0] * weights.reshape(1, -1)

        s1, s2 = res.shape
        s3, s4 = e.shape

        a = res.reshape(s1, 1, s2)
        b = e.reshape(1, s3, s4)
        res = (a * b).reshape(-1, n_columns)

    m = mask.reshape(-1, 1) if mask is not None else 1
    return res * m


def unfold_by_mode(tensor, mode):
    """
    Mode-n unfolding of a dense tensor (NumPy implementation).

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor.
    mode : int
        Mode along which to unfold.

    Returns
    -------
    unfolded : np.ndarray
        Unfolded matrix with shape ``(tensor.shape[mode], -1)``.
    """
    tensor = np.asarray(tensor)
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least 1 dimension.")
    if not (0 <= mode < tensor.ndim):
        raise ValueError(f"mode must be in [0, {tensor.ndim - 1}], got {mode}.")
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def calculate_mttkrp(tensor, cp_tensor, mode, mask=None):
    """
    Calculate Matricized Tensor Times Khatri–Rao Product (MTTKRP).

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to unfold.
    cp_tensor : tuple
        CP tensor represented as ``(weights, factors)`` where:
        - ``weights`` is a 1D array of length ``rank`` (or ``None``),
        - ``factors`` is a list of factor matrices, each with shape ``(I_k, rank)``.
    mode : int
        Mode on which to unfold ``tensor``. The factor matrix in this mode is skipped
        in the Khatri-Rao product.
    mask : np.ndarray
        Elementwise mask (1 = keep / valid entry, 0 = ignore entry).

    Returns
    -------
    mttkrp : np.ndarray
        ``dot(unfold(tensor, mode), conj(khatri_rao(factors, weights=weights, skip_matrix=mode)))``.

    Notes
    ------
    This function is adpated from tensorly.tenalg.unfolding_dot_khatri_rao.
    """
    weights, factors = cp_tensor
    if mask is not None:
        tensor = np.asarray(tensor * mask, dtype=float)
    kr_factors = multi_matrices_khatri_rao(factors, weights=weights, skip_matrix=mode)
    mttkrp = unfold_by_mode(tensor, mode) @ np.conj(kr_factors)
    return mttkrp


def masked_tensor_norm_error(tensor, reconstruction, mask):
    """
    Compute the reconstruction error on observed (masked) entries.

    Parameters
    ----------
    tensor : array-like
        Original tensor.
    reconstruction : array-like
        Reconstructed tensor with the same shape as ``tensor``.
    mask : array-like
        Elementwise mask (1 = keep / observed entry, 0 = ignore entry). Must broadcast to ``tensor``.

    Returns
    -------
    error : float
        Frobenius norm of ``(tensor - reconstruction) * mask``.
    """
    tensor = np.asarray(tensor)
    reconstruction = np.asarray(reconstruction)
    mask = np.asarray(mask)

    if tensor.shape != reconstruction.shape:
        raise ValueError(
            f"`tensor` and `reconstruction` must have the same shape, got {tensor.shape} and {reconstruction.shape}."
        )

    diff_masked = (tensor - reconstruction) * mask
    # Frobenius norm over all entries
    return float(np.linalg.norm(diff_masked.ravel(), ord=2))


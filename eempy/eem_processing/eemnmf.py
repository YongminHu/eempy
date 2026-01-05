import numpy as np
import pandas as pd

from math import sqrt

from scipy import stats
from sklearn.decomposition import NMF
from .basic import eems_fit_components, eem_stack_to_2d
from .eem_dataset import EEMDataset
from eempy.solver import nmf_with_prior_hals, solve_W

class EEMNMF:
    """
    Non-negative matrix factorization (NMF) model for an excitation–emission matrix (EEM) dataset.

    This class fits a low-rank NMF decomposition to a 3D EEM stack by unfolding it into a 2D non-negative matrix
    with shape (n_samples, n_pixels) and factorizing it into:
        - A non-negative sample score matrix W with shape (n_samples, n_components).
        - A non-negative component matrix H with shape (n_components, n_pixels), where n_pixels = n_ex * n_em`in the
        unfolded representation.

    The fitted NMF components are reshaped back to EEM form with shape (n_components, n_ex, n_em). Component
    amplitudes are reported as Fmax-like values using:
        - ``fmax`` : scores from the NMF factorization, rescaled to account for component normalization.
        - ``nnls_fmax`` : scores refit by non-negative least squares (NNLS) against the extracted components,
        which can differ slightly from ``fmax`` due to the non-exact NMF reconstruction and/or constraints.

    Optional regularization / constraints (solver-dependent) include:
        - Non-negativity (always enforced by this model).
        - Elastic-net regularization on ``W`` and/or ``H`` (L1/L2 mix).
        - Quadratic priors on ``W`` and/or ``H`` (controlled by ``prior_dict_W``, ``prior_dict_H`` and
          ``gamma_W``, ``gamma_H``), with NaNs allowed to skip entries. This is useful when fitted scores
          or spectral components are desired to be close (but not necessarily identical) to prior knowledge.
          For example, if a component’s concentration is known for some samples, a prior vector of length
          n_samples can be passed with real values for known samples and NaN for unknown samples.
        - A ratio constraint on paired rows of ``W``: ``W[idx_top] ≈ beta * W[idx_bot]``. This is useful when
          the ratios of component amplitudes between two sets of samples are desired to be constant. For example,
          if each sample is measured both unquenched and quenched using a fixed quencher dosage, then for a given
          chemically consistent component the ratio between unquenched and quenched amplitudes may be approximately
          constant across samples (Hu et al., ES&T, 2025). In this case, passing the unquenched and quenched sample
          indices to ``idx_top`` and ``idx_bot`` encourages a constant ratio. ``lam`` controls the strength of this
          regularization.

    Parameters
    ----------
    n_components : int
        Number of NMF components (rank of the factorization).
    solver : {'cd', 'mu', 'hals'}, default 'cd'
        Optimization algorithm used to fit NMF.
        - ``'cd'``: Coordinate Descent solver (scikit-learn ``decomposition.NMF``).
        - ``'mu'``: Multiplicative Updates solver (scikit-learn ``decomposition.NMF``).
        - ``'hals'``: Hierarchical Alternating Least Squares solver with optional priors/regularization
          (``eempy.solver.nmf_with_prior_hals``).
    init : str, default 'nndsvda'
        Initialization strategy passed to the selected solver.
        Common options include ``'random'``, ``'nndsvd'``, ``'nndsvda'``, ``'nndsvdar'`` (solver-dependent).
        For HALS, a custom initialization can be provided via ``custom_init`` when supported.
    custom_init : optional
        Custom initialization passed to the HALS solver (when supported by the backend implementation).
    fixed_components : optional
        Component(s) to keep fixed during fitting (backend-specific behavior).
    beta_loss : {'frobenius', 'kullback-leibler', 'itakura-saito'}, default 'frobenius'
        Beta divergence used by the ``'mu'`` solver. Ignored by ``'cd'`` and ``'hals'``.
    alpha_sample : float, default 0
        Regularization strength applied to the sample-mode factor matrix ``W`` (backend-specific).
        For scikit-learn, this maps to ``alpha_W``.
    alpha_component : float, default 0
        Regularization strength applied to the component matrix ``H`` (backend-specific).
        For scikit-learn, this maps to ``alpha_H``.
    l1_ratio : float, default 1
        Elastic-net mixing parameter used by the backend (``1`` corresponds to L1 only; ``0`` to L2 only).
    prior_dict_W : dict, optional
        Prior information for the sample-mode factor matrix ``W`` (HALS solver only).
        Keys are component indices (int); values are 1D arrays of length ``n_samples``.
        Use NaNs to indicate unknown entries that should not contribute to the penalty.
    prior_dict_H : dict, optional
        Prior information for the component matrix ``H`` (HALS solver only).
        Keys are component indices (int); values are 1D arrays of length ``n_pixels``.
        Use NaNs to indicate unknown entries that should not contribute to the penalty.
    prior_dict_A : dict, optional
        Additional prior mapping used by the HALS backend (backend-specific).
    prior_dict_B : dict, optional
        Additional prior mapping used by the HALS backend (backend-specific).
    prior_dict_C : dict, optional
        Additional prior mapping used by the HALS backend (backend-specific).
    gamma_W : float, default 0
        Additional prior/penalty strength for the sample-mode factor matrix ``W`` (HALS solver only).
    gamma_H : float, default 0
        Additional prior/penalty strength for the component matrix ``H`` (HALS solver only).
    gamma_A : float, default 0
        Additional prior/penalty strength for backend-specific prior term A (HALS solver only).
    gamma_B : float, default 0
        Additional prior/penalty strength for backend-specific prior term B (HALS solver only).
    gamma_C : float, default 0
        Additional prior/penalty strength for backend-specific prior term C (HALS solver only).
    ref_components : optional
        Reference component definitions used by the backend prior/regularization logic (backend-specific).
    kw_top : str, optional
        Keyword used to identify "top" EEM from ``eem_dataset.index`` during fitting. "Top" and "bot"
        EEMs are assumed to be paired one-to-one and aligned by selection order (first "top" ↔ first "bot", etc.).
        A recommended naming convention is "a_sharing_sample_name" + "kw_top" or "kw_bot" for the quenched and
        unquenched EEM derived from the same original sample, so the pair differs only by ``kw_top``/``kw_bot`` and
        alignment is preserved when selecting by keywords. An alternative approach is to provide ``idx_top`` and
        ``idx_bot`` to directly specify "top" and "bot" EEMs by positions.
    kw_bot : str, optional
        Keyword used to identify "bot" EEM from ``eem_dataset.index`` during fitting.
    idx_top : list of int, optional
        0-based integer positions of samples in eem_dataset used as the numerator ("top") group (e.g., [0, 1,
        2]).
    idx_bot : list of int, optional
        0-based integer positions of samples in eem_dataset used as the denominator ("bot") group (e.g., [3, 4,
        5]).
    lam : float, default 0
        Strength of ratio-based regularization between "top" and "bot" samples (HALS solver only).
    fit_rank_one : bool, default False
        Whether to enable a rank-one component constraint/penalty in the HALS backend (backend-specific).
    normalization : {'pixel_std', None}, default None
        Optional preprocessing applied to the unfolded data matrix before factorization.
        - ``None``: no normalization.
        - ``'pixel_std'``: divide each pixel (feature) by its standard deviation across samples.
    sort_components_by_em : bool, default True
        Whether to sort components by the emission peak position (ascending). If ``False``, components are
        kept in the solver output order.
    max_iter_als : int, default 100
        Maximum number of outer iterations for the HALS solver.
    max_iter_nnls : int, default 500
        Maximum number of iterations for NNLS subproblems (when used by the backend).
    tol : float, default 1e-5
        Convergence tolerance passed to the solver.
    random_state : int, default 42
        Random seed used by solvers that support it.

    Attributes
    ----------
    fmax : pandas.DataFrame or None
        Sample-mode component amplitudes computed from the fitted NMF ``W`` (and rescaling after component
        normalization). Columns follow the naming convention ``"component {i} NMF-Fmax"``.
    nnls_fmax : pandas.DataFrame or None
        Component amplitudes computed by refitting each EEM using NNLS with the fitted components.
        Columns follow the naming convention ``"component {i} NNLS-Fmax"``.
    components : numpy.ndarray or None
        Component EEMs with shape ``(n_components, n_ex, n_em)`` constructed from the unfolded ``H``.
        Each component is normalized by its maximum value (peak intensity equals 1), and the scaling is
        carried into ``fmax``.
    eem_stack_train : numpy.ndarray or None
        EEM stack used for model fitting, with shape ``(n_samples, n_ex, n_em)``.
    eem_stack_reconstructed : numpy.ndarray or None
        Reconstructed EEM stack from the fitted model, with shape ``(n_samples, n_ex, n_em)``.
    eem_stack_unfolded : numpy.ndarray or None
        Unfolded 2D matrix used by the solver, with shape ``(n_samples, n_pixels)``.
    normalization_factor_std : numpy.ndarray or None
        Per-pixel standard deviation used when ``normalization='pixel_std'``. Shape is ``(n_pixels,)``.
        ``None`` if no pixel-wise standard-deviation normalization was applied.
    normalization_factor_max : numpy.ndarray or None
        Per-component scaling factors (maximum value of each component in the unfolded space) used
        to normalize ``components`` and rescale reported amplitudes. Shape is ``(n_components,)``.
    ex_range : numpy.ndarray or None
        Excitation wavelength grid corresponding to ``components``.
    em_range : numpy.ndarray or None
        Emission wavelength grid corresponding to ``components``.
    beta : numpy.ndarray or None
        Component-wise ratio parameters used when ratio regularization / beta fitting is enabled
        (backend-specific).
    decomposer : object or None
        Underlying solver object when using scikit-learn NMF (e.g., fitted ``sklearn.decomposition.NMF``).
        May be ``None`` depending on the solver/backend implementation.
    reconstruction_error : float or None
        Reconstruction error if provided by the backend/solver; otherwise ``None``.
    objective_function_error : object or None
        Objective tracking information if provided by the backend/solver; otherwise ``None``.

    References
    ----------
    [1]  scikit-learn documentation for ``sklearn.decomposition.NMF`` (Coordinate Descent and Multiplicative Updates).
    [2]  Hu, Yongmin, Céline Jacquin, and Eberhard Morgenroth. "Fluorescence Quenching as a Diagnostic Tool for
         Prediction Reliability Assessment and Anomaly Detection in EEM-Based Water Quality Monitoring."
         Environmental Science & Technology 59.36 (2025): 19490-19501.
    """
    def __init__(
            self, n_components, solver='cd', init='nndsvda', custom_init=None, fixed_components=None,
            beta_loss='frobenius', alpha_sample=0, alpha_component=0, l1_ratio=1,
            prior_dict_W=None, prior_dict_H=None, prior_dict_A=None, prior_dict_B=None, prior_dict_C=None,
            gamma_W=0, gamma_H=0, gamma_A=0, gamma_B=0, gamma_C=0, ref_components=None,
            idx_top=None, idx_bot=None, kw_top=None, kw_bot=None, lam=0, fit_rank_one=False,
            normalization=None, sort_components_by_em=True, max_iter_als=100, max_iter_nnls=500, tol=1e-5,
            random_state=42
    ):
        # -----------Parameters-------------
        self.n_components = n_components
        self.solver = solver
        self.init = init
        self.custom_init = custom_init
        self.fixed_components = fixed_components
        self.beta_loss = beta_loss
        self.alpha_sample = alpha_sample
        self.alpha_component = alpha_component
        self.l1_ratio = l1_ratio
        self.prior_dict_W = prior_dict_W
        self.prior_dict_H = prior_dict_H
        self.prior_dict_A = prior_dict_A
        self.prior_dict_B = prior_dict_B
        self.prior_dict_C = prior_dict_C
        self.prior_ref_components = ref_components
        self.gamma_W = gamma_W
        self.gamma_H = gamma_H
        self.gamma_A = gamma_A
        self.gamma_B = gamma_B
        self.gamma_C = gamma_C
        self.idx_top = idx_top
        self.idx_bot = idx_bot
        self.kw_top = kw_top
        self.kw_bot = kw_bot
        self.lam = lam
        self.fit_rank_one = fit_rank_one
        self.normalization = normalization
        self.sort_components_by_em = sort_components_by_em
        self.max_iter_als = max_iter_als
        self.max_iter_nnls = max_iter_nnls
        self.tol = tol
        self.random_state = random_state
        # -----------Attributes-------------
        self.eem_stack_unfolded = None
        self.fmax = None
        self.nnls_fmax = None
        self.components = None
        self.decomposer = None
        self.normalization_factor_std = None
        self.normalization_factor_max = None
        self.reconstruction_error = None
        self.eem_stack_train = None
        self.eem_stack_reconstructed = None
        self.ex_range = None,
        self.em_range = None
        self.beta = None
        self.objective_function_error = None,


    def fit(self, eem_dataset):
        """
        Fit NMF model.

        Parameters
        ----------
        eem_dataset: eempy.dataset.EEMDataset
            The EEM dataset that the PARAFAC model establishes on.
        """
        if self.kw_top is not None and self.kw_bot is not None:
            assert eem_dataset.index is not None
            self.idx_top = [i for i in range(len(eem_dataset.index)) if self.kw_top in eem_dataset.index[i]]
            self.idx_bot = [i for i in range(len(eem_dataset.index)) if self.kw_bot in eem_dataset.index[i]]
        if self.solver == 'cd' or self.solver == 'mu':
            if self.prior_dict_W is not None or self.prior_dict_H is not None:
                raise ValueError("'cd' and 'mu' solver do not support prior knowledge input. Please use 'hals' solver "
                                 "instead")
            decomposer = NMF(
                n_components=self.n_components,
                solver=self.solver,
                init=self.init,
                beta_loss=self.beta_loss,
                alpha_W=self.alpha_sample,
                alpha_H=self.alpha_component,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
                tol=self.tol,
            )
            eem_dataset.threshold_masking(0, 0, 'smaller', inplace=True)
            n_samples = eem_dataset.eem_stack.shape[0]
            X = eem_dataset.eem_stack.reshape([n_samples, -1])
            X[np.isnan(X)] = 0
            if self.normalization == 'pixel_std':
                factor_std = np.std(X, axis=0)
                X = X / factor_std
                X[np.isnan(X)] = 0
                nmf_score = decomposer.fit_transform(X)
                components = decomposer.components_ * factor_std
            else:
                factor_std = None
                nmf_score = decomposer.fit_transform(X)
                components = decomposer.components_
            eem_stack_reconstructed = nmf_score @ components
            nmf_score = pd.DataFrame(nmf_score, index=eem_dataset.index,
                                     columns=["component {i} NMF-Fmax".format(i=i + 1) for i in
                                              range(self.n_components)])
        elif self.solver == 'hals':
            eem_dataset.threshold_masking(0, 0, 'smaller', inplace=True)
            n_samples = eem_dataset.eem_stack.shape[0]
            X = eem_dataset.eem_stack.reshape([n_samples, -1])
            X[np.isnan(X)] = 0
            if self.normalization == 'pixel_std':
                factor_std = np.std(X, axis=0)
                X = X / factor_std
                X[np.isnan(X)] = 0
                W, H, beta = nmf_with_prior_hals(
                    X,
                    rank=self.n_components,
                    init=self.init,
                    custom_init=self.custom_init,
                    fixed_components=self.fixed_components,
                    prior_dict_W=self.prior_dict_W,
                    prior_dict_H=self.prior_dict_H,
                    prior_dict_A=self.prior_dict_A,
                    prior_dict_B=self.prior_dict_B,
                    prior_dict_C=self.prior_dict_C,
                    alpha_W=self.alpha_sample,
                    alpha_H=self.alpha_component,
                    l1_ratio=self.l1_ratio,
                    gamma_W=self.gamma_W,
                    gamma_H=self.gamma_H,
                    gamma_A=self.gamma_A,
                    gamma_B=self.gamma_B,
                    gamma_C=self.gamma_C,
                    idx_top=self.idx_top,
                    idx_bot=self.idx_bot,
                    lam=self.lam,
                    fit_rank_one=self.fit_rank_one,
                    component_shape=[eem_dataset.eem_stack.shape[1], eem_dataset.eem_stack.shape[2]],
                    max_iter_als=self.max_iter_als,
                    max_iter_nnls=self.max_iter_nnls,
                    ref_components=self.prior_ref_components,
                    tol=self.tol,
                    random_state=self.random_state
                )
                self.beta = beta
                nmf_score = W
                components = H * factor_std
            else:
                factor_std = None
                W, H, beta = nmf_with_prior_hals(
                    X,
                    rank=self.n_components,
                    init=self.init,
                    custom_init=self.custom_init,
                    fixed_components=self.fixed_components,
                    prior_dict_W=self.prior_dict_W,
                    prior_dict_H=self.prior_dict_H,
                    prior_dict_A=self.prior_dict_A,
                    prior_dict_B=self.prior_dict_B,
                    prior_dict_C=self.prior_dict_C,
                    gamma_W=self.gamma_W,
                    gamma_H=self.gamma_H,
                    gamma_A=self.gamma_A,
                    gamma_B=self.gamma_B,
                    gamma_C=self.gamma_C,
                    alpha_W=self.alpha_sample,
                    alpha_H=self.alpha_component,
                    l1_ratio=self.l1_ratio,
                    idx_top=self.idx_top,
                    idx_bot=self.idx_bot,
                    lam=self.lam,
                    fit_rank_one=self.fit_rank_one,
                    component_shape=eem_dataset.eem_stack.shape[1:],
                    max_iter_als=self.max_iter_als,
                    max_iter_nnls=self.max_iter_nnls,
                    ref_components=self.prior_ref_components,
                    tol=self.tol,
                    random_state=self.random_state
                )
                self.beta = beta
                nmf_score = W
                components = H
            eem_stack_reconstructed = W @ H
            nmf_score = pd.DataFrame(nmf_score, index=eem_dataset.index,
                                     columns=["component {i} NMF-Fmax".format(i=i + 1) for i in
                                              range(self.n_components)])
        else:
            raise ValueError("Unknown solver name: choose 'mu', 'cd' or 'hals'.")
        factor_max = np.max(components, axis=1)
        components = components / factor_max[:, None]
        components = components.reshape([self.n_components, eem_dataset.eem_stack.shape[1],
                                         eem_dataset.eem_stack.shape[2]])
        nmf_score = nmf_score.mul(factor_max, axis=1)
        _, nnls_score, _ = eems_fit_components(eem_dataset.eem_stack, components,
                                               fit_intercept=False, positive=True)
        nnls_score = pd.DataFrame(nnls_score, index=eem_dataset.index,
                                  columns=["component {i} NNLS-Fmax".format(i=i + 1) for i in range(self.n_components)])
        if self.sort_components_by_em:
            em_peaks = []
            for i in range(self.n_components):
                flat_max_index = components[i].argmax()
                row_index, col_index = np.unravel_index(flat_max_index, components[i].shape)
                em_peaks.append(col_index)
            peak_rank = list(enumerate(stats.rankdata(em_peaks)))
            order = [i[0] for i in sorted(peak_rank, key=lambda x: x[1])]
            components = components[order]
            nmf_score = pd.DataFrame({'component {r} NMF-Fmax'.format(r=i + 1): nmf_score.iloc[:, order[i]]
                                      for i in range(self.n_components)})
            nnls_score = pd.DataFrame({'component {r} NNLS-Fmax'.format(r=i + 1): nnls_score.iloc[:, order[i]]
                                       for i in range(self.n_components)})
            self.beta = self.beta[order] if self.beta is not None else None
        self.fmax = nmf_score
        self.nnls_fmax = nnls_score
        self.components = components
        self.eem_stack_unfolded = X
        self.normalization_factor_std = factor_std
        self.normalization_factor_max = factor_max
        self.eem_stack_train = eem_dataset.eem_stack
        self.eem_stack_reconstructed = eem_stack_reconstructed.reshape(eem_dataset.eem_stack.shape)
        self.ex_range = eem_dataset.ex_range
        self.em_range = eem_dataset.em_range
        return self


    def component_peak_locations(self):
        """
        Get the ex/em of component peaks

        Returns
        -------
        max_exem: list
            A List of (ex, em) of component peaks.
        """
        max_exem = []
        for r in range(self.n_components):
            max_index = np.unravel_index(np.argmax(self.components[r, :, :]), self.components[r, :, :].shape)
            max_exem.append((self.ex_range[-(max_index[0] + 1)], self.em_range[max_index[1]]))
        return max_exem


    def residual(self):
        """
        Get the residual of the established PARAFAC model, i.e., the difference between the original EEM dataset and
        the reconstructed EEM dataset.

        Returns
        -------
        res: np.ndarray (3d)
            the residual
        """
        res = self.eem_stack_train - self.eem_stack_reconstructed
        return res


    def variance_explained(self):
        """
        Calculate the explained variance of the established NMF model

        Returns
        -------
        ev: float
            the explained variance
        """
        y_train = self.eem_stack_train.reshape(-1)
        y_pred = self.eem_stack_reconstructed.reshape(-1)
        ev = 100 * (1 - np.var(y_pred - y_train) / np.var(y_train))
        return ev


    def sample_rmse(self):
        """
        Calculate the root mean squared error (RMSE) of EEM of each sample.

        Returns
        -------
        sse: pandas.DataFrame
            Table of RMSE
        """
        res = self.residual()
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        rmse = pd.DataFrame(np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels), index=self.nnls_fmax.index,
                            columns=['RMSE'])
        return rmse


    def sample_normalized_rmse(self):
        """
        Calculate the normalized root mean squared error (normalized RMSE) of EEM of each sample. It is defined as the
        RMSE divided by the mean of original signal.

        Returns
        -------
        normalized_sse: pandas.DataFrame
            Table of normalized RMSE
        """
        res = self.residual()
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        normalized_sse = pd.DataFrame(sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels) /
                                      np.average(self.eem_stack_train, axis=(1, 2)),
                                      index=self.nnls_fmax.index, columns=['sample_normalized_rmse'])
        return normalized_sse


    def predict(self, eem_dataset: EEMDataset, fit_intercept=False, fit_beta=False, idx_top=None, idx_bot=None):
        """
        Predict the score and Fmax of a given EEM dataset using the component fitted. This method can be applied to a
        new EEM dataset independent of the one used in NMF model establishment.

        Parameters
        ----------
        eem_dataset : EEMDataset
            The EEM dataset to be predicted.
        fit_intercept : bool
            Whether to calculate the intercept.
        fit_beta: bool
            Whether to fit the beta parameter (the proportions between "top" and "bot" samples).
        idx_top : list, optional
            List of indices of samples serving as numerators in ratio calculation.
        idx_bot : list, optional
            List of indices of samples serving as denominators in ratio calculation.

        Returns
        -------
        score_sample : pd.DataFrame
            The fitted score.
        fmax_sample : pd.DataFrame
            The fitted Fmax.
        eem_stack_pred : np.ndarray (3d)
            The EEM dataset reconstructed.
        """
        if not fit_beta:
            score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset.eem_stack,
                                                                            self.components,
                                                                            fit_intercept=fit_intercept)
        else:
            assert self.beta is not None, "Parameter beta must be provided through fitting."
            assert idx_top is not None and idx_bot is not None, "idx_top and idx_bot must be provided."
            max_values = np.amax(self.components, axis=(1, 2))
            score_sample = np.zeros([eem_dataset.eem_stack.shape[0], self.n_components])
            score_sample_bot = solve_W(
                X1=eem_stack_to_2d(eem_dataset.eem_stack)[idx_bot],
                X2=eem_stack_to_2d(eem_dataset.eem_stack)[idx_top],
                H=self.components.reshape([self.n_components, -1]),
                beta=self.beta
            )
            score_sample[idx_bot] = score_sample_bot
            score_sample[idx_top] = score_sample_bot * self.beta
            fmax_sample = score_sample * max_values
            eem_stack_pred = score_sample @ self.components.reshape([self.n_components, -1])
            eem_stack_pred = eem_stack_pred.reshape(eem_dataset.eem_stack.shape)
        score_sample = pd.DataFrame(score_sample, index=eem_dataset.index, columns=self.fmax.columns)
        fmax_sample = pd.DataFrame(fmax_sample, index=eem_dataset.index, columns=self.fmax.columns)
        return score_sample, fmax_sample, eem_stack_pred

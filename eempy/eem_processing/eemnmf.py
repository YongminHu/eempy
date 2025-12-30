

from eempy.eem_processing.eem_dataset import *

class EEMNMF:
    """
    Non-negative matrix factorization (NMF) model for Excitation–Emission Matrix (EEM) datasets.

    This model decomposes an EEM dataset into a set of non-negative components and non-negative
    sample scores. The input EEM stack is unfolded into a 2D matrix with shape
    (n_samples, n_pixels), where n_pixels = n_ex_wavelengths * n_em_wavelengths.

    The fitted NMF components are reshaped back to EEM shape (n_components, n_ex_wavelengths,
    n_em_wavelengths). Component amplitudes are reported as Fmax values using (i) the NMF score
    matrix and (ii) a non-negative least squares (NNLS) refit of each EEM using the extracted
    components.

    Parameters
    ----------
    n_components : int
        Number of components to extract.
    solver : str, {"cd", "mu", "hals"}, default="cd"
        NMF solver.
        - "cd": Coordinate Descent solver (scikit-learn).
        - "mu": Multiplicative Update solver (scikit-learn).
        - "hals": Hierarchical Alternating Least Squares solver with optional priors/regularization.
    init : str, default="nndsvda"
        Initialization method. Passed to the selected solver.
    custom_init : object, optional
        Custom initialization used by the HALS solver when ``init="custom"``.
    fixed_components : object, optional
        Components to hold fixed during HALS updates (solver-specific behavior).
    beta_loss: str, {"frobenius", "kullback-leibler", "itakura-saito"}, default="frobenius"
        Beta divergence used by the "mu" solver. Ignored by "cd" and "hals".
    alpha_sample: float, default=0
        Strength of elastic-net regularization on the sample score matrix (W).
    alpha_component: float, default=0
        Strength of elastic-net regularization on the component matrix (H).
    l1_ratio: float, default=1
        Mixing ratio between L1 and L2 penalties for elastic-net regularization.
        - 0 corresponds to pure L2 penalty
        - 1 corresponds to pure L1 penalty
    prior_dict_W: dict, optional
        Prior information for sample scores (W). Keys are component indices (int) and values are
        1D arrays of length n_samples. Only supported by the "hals" solver.
    prior_dict_H: dict, optional
        Prior information for components (H). Keys are component indices (int) and values are
        1D arrays of length n_pixels. Only supported by the "hals" solver.
    prior_dict_A: dict, optional
        Additional prior mapping used by the HALS solver (solver-specific).
    prior_dict_B: dict, optional
        Additional prior mapping used by the HALS solver (solver-specific).
    prior_dict_C: dict, optional
        Additional prior mapping used by the HALS solver (solver-specific).
    gamma_W: float, default=0
        Strength of prior regularization on W for the "hals" solver.
    gamma_H: float, default=0
        Strength of prior regularization on H for the "hals" solver.
    gamma_A: float, default=0
        Strength of additional prior regularization term A for the "hals" solver.
    gamma_B: float, default=0
        Strength of additional prior regularization term B for the "hals" solver.
    gamma_C: float, default=0
        Strength of additional prior regularization term C for the "hals" solver.
    ref_components: object, optional
        Reference components used by the HALS solver (solver-specific behavior).
    idx_top: list, optional
        Indices of samples used as numerators in ratio regularization (HALS solver only).
    idx_bot: list, optional
        Indices of samples used as denominators in ratio regularization (HALS solver only).
    kw_top: str, optional
        Keyword used to automatically populate ``idx_top`` from ``eem_dataset.index`` during fitting.
    kw_bot: str, optional
        Keyword used to automatically populate ``idx_bot`` from ``eem_dataset.index`` during fitting.
    lam: float, default=0
        Strength of ratio regularization on W (HALS solver only).
    fit_rank_one: bool, default=False
        Whether to enable a rank-one component constraint/penalty in the HALS solver (solver-specific).
    normalization: None or str, {"pixel_std"}, default=None
        Optional preprocessing applied to the unfolded data matrix X before factorization.
        - None: no normalization
        - "pixel_std": divide each pixel (feature) by its standard deviation across samples.
    sort_components_by_em: bool, default=True
        If True, components are reordered by the emission peak position (lowest to highest emission
        wavelength index). If False, component order follows the solver output.
    max_iter_als: int, default=100
        Maximum number of outer iterations for the HALS solver.
    max_iter_nnls: int, default=500
        Maximum number of NNLS iterations used inside the HALS solver.
    tol: float, default=1e-5
        Convergence tolerance passed to the solver.
    random_state: int, default=42
        Random seed used by solvers that support it.

    Attributes
    ----------
    fmax: pandas.DataFrame
        Component amplitudes (Fmax) computed from the NMF score matrix. Columns follow the
        naming convention ``"component {i} NMF-Fmax"``.
    nnls_fmax: pandas.DataFrame
        Component amplitudes (Fmax) computed by refitting each EEM using NNLS with the fitted
        components. Columns follow the naming convention ``"component {i} NNLS-Fmax"``.
    components: np.ndarray
        Fitted components with shape (n_components, n_ex_wavelengths, n_em_wavelengths). Each
        component is normalized by its maximum value (peak intensity equals 1).
    eem_stack_train: np.ndarray
        The EEM stack used for fitting, with shape (n_samples, n_ex_wavelengths, n_em_wavelengths).
    eem_stack_reconstructed: np.ndarray
        The reconstructed EEM stack from the fitted model, with the same shape as ``eem_stack_train``.
    eem_stack_unfolded: np.ndarray
        The unfolded 2D matrix used by the solver, with shape (n_samples, n_pixels).
    normalization_factor_std: np.ndarray or None
        Per-pixel standard deviation used when ``normalization="pixel_std"``. Shape is (n_pixels,).
        None if no pixel-wise standard-deviation normalization was applied.
    normalization_factor_max: np.ndarray
        Per-component scaling factors (maximum value of each component in the unfolded space) used
        to normalize "components" and rescale the reported Fmax values. Shape is (n_components,).
    ex_range: np.ndarray or None
        Excitation wavelength axis copied from the fitted dataset.
    em_range: np.ndarray or None
        Emission wavelength axis copied from the fitted dataset.
    beta: np.ndarray or None
        Optional per-component ratio parameter returned by the HALS solver with ratio regularization
        (when applicable). None if not returned by the solver.
    decomposer: object or None
        Placeholder for an external decomposer object (e.g., scikit-learn NMF). Not populated in the
        current implementation.
    reconstruction_error: float or None
        Placeholder for reconstruction error. Not populated in the current implementation.
    objective_function_error: object or None
        Placeholder for solver objective tracking. Not populated in the current implementation.

    References
    ----------
    [1] scikit-learn decomposition.NMF documentation (for "cd" and "mu" solvers).
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

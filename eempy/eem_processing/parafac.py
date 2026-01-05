import numpy as np
import pandas as pd
import tensorly as tl

from scipy import stats
from typing import Optional

from tensorly.decomposition import parafac, non_negative_parafac
from scipy.sparse.linalg import ArpackError
from tlviz.model_evaluation import core_consistency
from tlviz.outliers import compute_leverage
from eempy.solver import parafac_with_prior_hals, update_beta_in_hals, solve_W
from .eem_dataset import EEMDataset
from .basic import eems_fit_components, eem_stack_to_2d


class PARAFAC:
    """
    Parallel factor analysis (PARAFAC) model for an excitation–emission matrix (EEM) dataset.

    This class fits a low-rank PARAFAC (CP) decomposition to a 3D EEM stack with shape ``(n_samples, n_ex,
    n_em)`` by factorizing it into:
        - A sample-mode score matrix ``A`` with shape ``(n_samples, n_components)``.
        - An excitation-mode loading matrix ``B`` with shape ``(n_ex, n_components)``.
        - An emission-mode loading matrix ``C`` with shape ``(n_em, n_components)``.

    Each component r corresponds to a rank-1 outer product A[:, r] ⊗ B[:, r] ⊗ C[:, r], and the reconstructed
    EEM stack is obtained by summing these rank-1 components over r = 1...n_components.

    This class fits a low-rank PARAFAC decomposition to a 3D EEM stack with optional regularization:
        - Non-negativity
        - Elastic-net regularization on any factor (L1/L2 mix).
        - Quadratic priors on ``A``, ``B``, and/or ``C`` (controlled by ``prior_dict_sample``, ``prior_dict_ex``,
          ``prior_dict_em and ``gamma_sample``, ``gamma_ex`` and ``gamma_em), with NaNs allowed to skip entries. This is
          useful when fitted scores or spectral components are desired to be close (but not necessarily identical) to
          prior knowledge. For example, if a component’s concentration is known for some samples, a prior vector of
          length n_samples can be passed with real values for known samples and NaN for unknown samples.
        - A ratio constraint on paired rows of ``A``: ``A[idx_top] ≈ beta * A[idx_bot]``. This is useful when
          the ratios of component amplitudes between two sets of samples are desired to be constant. For example,
          if each sample is measured both unquenched and quenched using a fixed quencher dosage, then for a given
          chemically consistent component the ratio between unquenched and quenched amplitudes may be approximately
          constant across samples (Hu et al., ES&T, 2025). In this case, passing the unquenched and quenched sample
          indices to ``idx_top`` and ``idx_bot`` encourages a constant ratio. ``lam`` controls the strength of this
          regularization.

    Parameters
    ----------
    n_components : int
        Number of PARAFAC components (rank of the CP decomposition).
    non_negativity : bool, default True
        Whether to enforce non-negativity constraints on the factor matrices.
    solver : {'mu', 'hals'}, default 'hals'
        Optimization algorithm used when ``non_negativity=True``.
        - ``'mu'``: Multiplicative Updates solver (tensorly.decomposition.non_negative_parafac).
        - ``'hals'``: Hierarchical Alternating Least Squares solver with optional priors/regularization(
        eempy.solver.parafac_with_prior_hals).
        if ``non_negativity=False``, a standard alternating least squares solver is used anyway (
        tensorly.decomposition.parafac).
    init : {'svd', 'random'} or tensorly.CPTensor, default 'svd'
        Initialization strategy for the factor matrices. If a ``tensorly.CPTensor`` is provided, it is used
        as the initialization.
    custom_init : optional
        Custom initialization passed to the HALS solver (when supported by the backend implementation).
    fixed_components : optional
        Component(s) to keep fixed during fitting (backend-specific behavior).
    tf_normalization : bool, default False
        Whether to normalize each EEM by its total fluorescence during model fitting.
    loadings_normalization : {'sd', 'maximum', None}, default 'maximum'
        Post-fit normalization applied to excitation/emission loadings, with the sample scores scaled
        accordingly.
        - 'sd': normalize each loading vector to unit standard deviation.
        - 'maximum': normalize each loading vector to unit maximum.
        - None: no loading normalization.
    sort_components_by_em : bool, default True
        Whether to sort components by the emission peak position (ascending). If ``False``, components are
        kept in the solver output order (which may correlate with variance contribution depending on the solver).
    alpha_sample : float, default 0
        Regularization strength applied to the sample-mode factor matrix (backend-specific).
    alpha_ex : float, default 0
        Regularization strength applied to the excitation-mode factor matrix (backend-specific).
    alpha_em : float, default 0
        Regularization strength applied to the emission-mode factor matrix (backend-specific).
    l1_ratio : float, default 1
        Elastic-net mixing parameter used by the backend (``1`` corresponds to L1 only; ``0`` to L2 only).
    prior_dict_sample : dict, optional
        Prior information for the sample-mode factor matrix (backend-specific).
    prior_dict_ex : dict, optional
        Prior information for the excitation-mode factor matrix (backend-specific).
    prior_dict_em : dict, optional
        Prior information for the emission-mode factor matrix (backend-specific).
    gamma_sample : float, default 0
        Additional prior/penalty strength for the sample-mode factor matrix (backend-specific).
    gamma_ex : float, default 0
        Additional prior/penalty strength for the excitation-mode factor matrix (backend-specific).
    gamma_em : float, default 0
        Additional prior/penalty strength for the emission-mode factor matrix (backend-specific).
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
        Strength of ratio-based regularization between "top" and "bot" samples (backend-specific).
    max_iter_als : int, default 100
        Maximum number of outer ALS iterations.
    tol : float, default 1e-6
        Convergence tolerance for the ALS loop.
    max_iter_nnls : int, default 500
        Maximum number of iterations for NNLS subproblems (when used by the backend).
    random_state : int or numpy.random.RandomState, optional
        Random seed or RNG used for reproducible initialization (when supported).
    mask : array-like, optional
        A ideally sparse mask array for missing values (backend-specific). When provided, masked entries are ignored in
        fitting.

    Attributes
    ----------
    score : pandas.DataFrame or None
        Sample scores (sample loadings).
    ex_loadings : pandas.DataFrame or None
        Excitation-mode loadings for each component.
    em_loadings : pandas.DataFrame or None
        Emission-mode loadings for each component.
    fmax : pandas.DataFrame or None
        The maximum fluorescence intensity of components. Fmax is calculated by multiplying the maximum excitation
        loading and maximum emission loading for each component by its score.
    nnls_fmax : pandas.DataFrame or None
        Fmax estimated from refitting PARAFAC components to the original EEMs using NNLS. It may be slightly
        different from ``fmax`` due to the non-exact fit.
    components : numpy.ndarray or None
        Component EEMs with shape ``(n_components, n_ex, n_em)`` constructed from excitation/emission loadings.
    cptensors : tensorly.cp_tensor.CPTensor or None
        Fitted CP/PARAFAC tensor representation returned by the underlying solver.
    eem_stack_train : numpy.ndarray or None
        EEM stack used for model fitting, with shape ``(n_samples, n_ex, n_em)``.
    eem_stack_reconstructed : numpy.ndarray or None
        Reconstructed EEM stack from the fitted model, with shape ``(n_samples, n_ex, n_em)``.
    ex_range : numpy.ndarray or None
        Excitation wavelength grid corresponding to ``ex_loadings`` and ``components``.
    em_range : numpy.ndarray or None
        Emission wavelength grid corresponding to ``em_loadings`` and ``components``.
    beta : numpy.ndarray or None
        Component-wise ratio parameters used when ratio regularization / beta fitting is enabled.

    References
    -----------
    [1]  Tensorly documentation for CP/PARAFAC deomposition.
    [2]  Hu, Yongmin, Céline Jacquin, and Eberhard Morgenroth. "Fluorescence Quenching as a Diagnostic Tool for
    Prediction Reliability Assessment and Anomaly Detection in EEM-Based Water Quality Monitoring." Environmental
    Science & Technology 59.36 (2025): 19490-19501.
    """
    def __init__(self, n_components, non_negativity=True, solver='hals', init='svd', custom_init=None, fixed_components=None,
                 tf_normalization=False, loadings_normalization: Optional[str] = 'maximum', sort_components_by_em=True,
                 alpha_sample=0, alpha_ex=0, alpha_em=0, l1_ratio=1,
                 prior_dict_sample=None, prior_dict_ex=None, prior_dict_em=None,
                 gamma_sample=0, gamma_ex=0, gamma_em=0, ref_components=None,
                 idx_top=None, idx_bot=None, kw_top=None, kw_bot=None, lam=0,
                 max_iter_als=100, tol=1e-06, max_iter_nnls=500, random_state=None, mask=None
                 ):
        # ----------parameters--------------
        self.n_components = n_components
        self.non_negativity = non_negativity
        self.init = init
        self.custom_init = custom_init
        self.fixed_components = fixed_components
        self.tf_normalization = tf_normalization
        self.loadings_normalization = loadings_normalization
        self.sort_components_by_em = sort_components_by_em
        self.solver = solver
        self.alpha_sample = alpha_sample
        self.alpha_ex = alpha_ex
        self.alpha_em = alpha_em
        self.l1_ratio = l1_ratio
        self.prior_dict_sample = prior_dict_sample
        self.prior_dict_ex = prior_dict_ex
        self.prior_dict_em = prior_dict_em
        self.gamma_sample = gamma_sample
        self.gamma_ex = gamma_ex
        self.gamma_em = gamma_em
        self.prior_ref_components = ref_components
        self.idx_top = idx_top
        self.idx_bot = idx_bot
        self.kw_top = kw_top
        self.kw_bot = kw_bot
        self.lam = lam
        self.max_iter_als = max_iter_als
        self.tol = tol
        self.max_iter_nnls = max_iter_nnls
        self.random_state = random_state
        self.mask = mask
        # -----------attributes---------------
        self.score = None
        self.ex_loadings = None
        self.em_loadings = None
        self.fmax = None
        self.nnls_fmax = None
        self.components = None
        self.cptensors = None
        self.eem_stack_train = None
        self.eem_stack_reconstructed = None
        self.ex_range = None
        self.em_range = None
        self.beta = None

    # --------------methods------------------


    def fit(self, eem_dataset: EEMDataset):
        """
        Establish a PARAFAC model based on a given EEM dataset

        Parameters
        ----------
        eem_dataset : EEMDataset
            The EEM dataset that the PARAFAC model establishes on.

        Returns
        -------
        self : object
            The established PARAFAC model
        """
        if self.kw_top is not None and self.kw_bot is not None:
            assert eem_dataset.index is not None
            self.idx_top = [i for i in range(len(eem_dataset.index)) if self.kw_top in eem_dataset.index[i]]
            self.idx_bot = [i for i in range(len(eem_dataset.index)) if self.kw_bot in eem_dataset.index[i]]
        if self.tf_normalization:
            if self.lam>0 and self.idx_bot is not None and self.idx_top is not None:
                Warning("Applying tf_normalization together with ratio regularization (lam>0) will lead to unreasonable results")
            eem_dataset_tf, tf_weights = eem_dataset.tf_normalization(inplace=True)
            eem_stack_tf = eem_dataset_tf.eem_stack
        else:
            eem_stack_tf = eem_dataset.eem_stack.copy()
        try:
            if not self.non_negativity:
                if np.isnan(eem_stack_tf).any():
                    mask = np.where(np.isnan(eem_stack_tf), 0, 1)
                    cptensors = parafac(eem_stack_tf, rank=self.n_components, mask=mask, init=self.init,
                                        n_iter_max=self.max_iter_als,
                                        tol=self.tol)
                else:
                    cptensors = parafac(eem_stack_tf, rank=self.n_components, init=self.init,
                                        n_iter_max=self.max_iter_als,
                                        tol=self.tol)
                a, b, c = cptensors[1]
            else:
                if self.solver == 'hals':
                    a, b, c, beta = parafac_with_prior_hals(
                        eem_stack_tf,
                        rank=self.n_components,
                        init=self.init,
                        custom_init=self.custom_init,
                        prior_dict_A=self.prior_dict_sample,
                        prior_dict_B=self.prior_dict_ex,
                        prior_dict_C=self.prior_dict_em,
                        alpha_A=self.alpha_sample,
                        alpha_B=self.alpha_ex,
                        alpha_C=self.alpha_em,
                        l1_ratio=self.l1_ratio,
                        gamma_A=self.gamma_sample,
                        gamma_B=self.gamma_ex,
                        gamma_C=self.gamma_em,
                        idx_top=self.idx_top,
                        idx_bot=self.idx_bot,
                        lam=self.lam,
                        max_iter_als=self.max_iter_als,
                        max_iter_nnls=self.max_iter_nnls,
                        ref_components=self.prior_ref_components,
                        random_state=self.random_state,
                        fixed_components=self.fixed_components,
                        mask=self.mask,
                    )
                    self.beta = beta
                    cptensors = tl.cp_tensor.CPTensor((np.ones(self.n_components), [a, b, c]))
                elif self.solver == 'mu':
                    cptensors = non_negative_parafac(eem_stack_tf, rank=self.n_components, init=self.init,
                                                     n_iter_max=self.max_iter_als, tol=self.tol)
                    a, b, c = cptensors[1]
        except ArpackError:
            print(
                "PARAFAC failed possibly due to the presence of large non-sparse missing values. Please consider cut or "
                "interpolate the nan values.")
        components = np.zeros([self.n_components, b.shape[0], c.shape[0]])
        for r in range(self.n_components):
            # when non_negativity is not applied, ensure the scores are generally positive
            if not self.non_negativity:
                if a[:, r].sum() < 0:
                    a[:, r] = -a[:, r]
                    if abs(b[:, r].min()) > b[:, r].max():
                        b[:, r] = -b[:, r]
                    else:
                        c[:, r] = -c[:, r]
                elif abs(b[:, r].min()) > b[:, r].max() and abs(c[:, r].min()) > c[:, r].max():
                    b[:, r] = -b[:, r]
                    c[:, r] = -c[:, r]
            if self.loadings_normalization == 'sd':
                stdb = b[:, r].std()
                stdc = c[:, r].std()
                b[:, r] = b[:, r] / stdb
                c[:, r] = c[:, r] / stdc
                a[:, r] = a[:, r] * stdb * stdc
            elif self.loadings_normalization == 'maximum':
                maxb = b[:, r].max()
                maxc = c[:, r].max()
                b[:, r] = b[:, r] / maxb
                c[:, r] = c[:, r] / maxc
                a[:, r] = a[:, r] * maxb * maxc
            component = np.array([b[:, r]]).T.dot(np.array([c[:, r]]))
            components[r, :, :] = component
        if self.tf_normalization:
            fmax = pd.DataFrame(a * tf_weights[:, np.newaxis])
            self.beta = update_beta_in_hals(fmax.to_numpy(), self.idx_top, self.idx_bot) if self.beta is not None else None
        else:
            fmax = pd.DataFrame(a)
        a, _, _ = eems_fit_components(eem_dataset.eem_stack, components, fit_intercept=False, positive=True)
        score = pd.DataFrame(a)
        nnls_fmax = a * components.max(axis=(1, 2))
        ex_loadings = pd.DataFrame(np.flipud(b), index=eem_dataset.ex_range)
        em_loadings = pd.DataFrame(c, index=eem_dataset.em_range)
        if self.sort_components_by_em:
            em_peaks = [c for c in em_loadings.idxmax()]
            peak_rank = list(enumerate(stats.rankdata(em_peaks)))
            order = [i[0] for i in sorted(peak_rank, key=lambda x: x[1])]
            components = components[order]
            ex_loadings = pd.DataFrame({'component {r} ex loadings'.format(r=i + 1): ex_loadings.iloc[:, order[i]]
                                        for i in range(self.n_components)})
            em_loadings = pd.DataFrame({'component {r} em loadings'.format(r=i + 1): em_loadings.iloc[:, order[i]]
                                        for i in range(self.n_components)})
            score = pd.DataFrame({'component {r} score'.format(r=i + 1): score.iloc[:, order[i]]
                                  for i in range(self.n_components)})
            fmax = pd.DataFrame({'component {r} score'.format(r=i + 1): fmax.iloc[:, order[i]]
                                 for i in range(self.n_components)})
            nnls_fmax = pd.DataFrame({'component {r} fmax'.format(r=i + 1): nnls_fmax[:, order[i]]
                                      for i in range(self.n_components)})
            self.beta = self.beta[order] if self.beta is not None else None
        else:
            column_labels = ['component {r}'.format(r=i + 1) for i in range(self.n_components)]
            ex_loadings.columns = column_labels
            em_loadings.columns = column_labels
            score.columns = ['component {r} PARAFAC-score'.format(r=i + 1) for i in range(self.n_components)]
            nnls_fmax = pd.DataFrame(nnls_fmax, columns=['component {r} PARAFAC-Fmax'.format(r=i + 1) for i in
                                                         range(self.n_components)])
        ex_loadings.index = eem_dataset.ex_range.tolist()
        em_loadings.index = eem_dataset.em_range.tolist()
        if eem_dataset.index:
            score.index = eem_dataset.index
            fmax.index = eem_dataset.index
            nnls_fmax.index = eem_dataset.index
        else:
            score.index = [i + 1 for i in range(a.shape[0])]
            fmax.index = [i + 1 for i in range(a.shape[0])]
            nnls_fmax.index = [i + 1 for i in range(a.shape[0])]
        eem_stack_reconstructed = np.tensordot(score.to_numpy(), components, axes=(1, 0))
        self.score = score
        self.ex_loadings = ex_loadings
        self.em_loadings = em_loadings
        self.fmax = fmax
        self.nnls_fmax = nnls_fmax
        self.components = components
        self.cptensors = cptensors
        self.eem_stack_train = eem_dataset.eem_stack
        self.ex_range = eem_dataset.ex_range
        self.em_range = eem_dataset.em_range
        self.eem_stack_reconstructed = eem_stack_reconstructed
        return self


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
        fit_beta : bool
            Whether to fit the beta parameter (the proportions between "top" and "bot" samples).
        idx_top : list, optional
            List of indices of samples serving as numerators in ratio calculation.
        idx_bot  list, optional
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
            score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset.eem_stack, self.components,
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
        score_sample = pd.DataFrame(score_sample, index=eem_dataset.index, columns=self.nnls_fmax.columns)
        fmax_sample = pd.DataFrame(fmax_sample, index=eem_dataset.index, columns=self.nnls_fmax.columns)
        return score_sample, fmax_sample, eem_stack_pred


    def component_peak_locations(self):
        """
        Get the ex/em of component peaks
        Returns
        -------
        max_exem : list
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
        res : np.ndarray (3d)
            the residual
        """
        res = self.eem_stack_train - self.eem_stack_reconstructed
        return res


    def variance_explained(self):
        """
        Calculate the explained variance of the established PARAFAC model
        Returns
        -------
        ev : float
            the explained variance
        """
        ss_total = tl.norm(self.eem_stack_train) ** 2
        ss_residual = tl.norm(self.eem_stack_train - self.eem_stack_reconstructed) ** 2
        variance_explained = (ss_total - ss_residual) / ss_total * 100
        return variance_explained


    def core_consistency(self):
        """
        Calculate the core consistency of the established PARAFAC model
        Returns
        -------
        cc : float
            core consistency
        """
        cc = core_consistency(self.cptensors, self.eem_stack_train)
        return cc


    def leverage(self, mode: str = 'sample'):
        """
        Calculate the leverage of a selected mode.
        Parameters
        ----------
        mode : str, {'ex', 'em', 'sample'}
            The mode of which the leverage is calculated.
        Returns
        -------
        lvr : pandas.DataFrame
            The table of leverage
        """
        if mode == 'ex':
            lvr = compute_leverage(self.ex_loadings)
            lvr.columns = ['leverage-ex']
        elif mode == 'em':
            lvr = compute_leverage(self.em_loadings)
            lvr.columns = ['leverage-em']
        elif mode == 'sample':
            lvr = compute_leverage(self.score)
            lvr.columns = ['leverage-sample']
        else:
            raise ValueError("'mode' should be 'ex' or 'em' or 'sample'.")
        # lvr.index = lvr.index.set_levels(['leverage of {m}'.format(m=mode)] * len(lvr.index.levels[0]), level=0)
        return lvr


    def sample_rmse(self):
        """
        Calculate the root mean squared error (RMSE) of EEM of each sample.
        Returns
        -------
        rmse : pandas.DataFrame
            Table of RMSE
        """
        res = self.residual()
        # res = process_eem_stack(res, eem_rayleigh_scattering_removal, ex_range=self.ex_range, em_range=self.em_range)
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        rmse = pd.DataFrame(np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels),
                            index=self.nnls_fmax.index, columns=['RMSE'])
        return rmse


    def sample_relative_rmse(self):
        """
        Calculate the normalized root mean squared error (normalized RMSE) of EEM of each sample. It is defined as the
        RMSE divided by the mean of original signal.
        Returns
        -------
        relative_rmse : pandas.DataFrame
            Table of normalized RMSE
        """
        res = self.residual()
        # res = process_eem_stack(res, eem_rayleigh_scattering_removal, ex_range=self.ex_range, em_range=self.em_range)
        n_pixels = self.eem_stack_train.shape[1] * self.eem_stack_train.shape[2]
        relative_rmse = pd.DataFrame(
            np.sqrt(np.sum(res ** 2, axis=(1, 2)) / n_pixels) / np.average(self.eem_stack_train, axis=(1, 2)),
            index=self.score.index, columns=['Relative RMSE'])
        return relative_rmse


    def sample_summary(self):
        """
        Get a table showing the score, Fmax, leverage, RMSE and normalized RMSE for each sample.
        Returns
        -------
        summary : pandas.DataFrame
            Table of samples' score, Fmax, leverage, RMSE and normalized RMSE.
        """
        lvr = self.leverage()
        rmse = self.sample_rmse()
        normalized_rmse = self.sample_normalized_rmse()
        summary = pd.concat([self.score, self.nnls_fmax, lvr, rmse, normalized_rmse], axis=1)
        return summary


    def export(self, filepath, info_dict):
        """
        Export the PARAFAC model to a text file that can be uploaded to the online PARAFAC model database Openfluor
        (https://openfluor.lablicate.com/#).
        Parameters
        ----------
        filepath : str
            Location of the saved text file. Please specify the ".csv" extension.
        info_dict: dict
            A dictionary containing the model information. Possible keys include: name, creator
            date, email, doi, reference, unit, toolbox, fluorometer, nSample, decomposition_method, validation,
            dataset_calibration, preprocess, sources, description
        Returns
        -------
        info_dict : dict
            A dictionary containing the information of the PARAFAC model.
        """
        ex_column = ["Ex"] * self.ex_range.shape[0]
        em_column = ["Em"] * self.em_range.shape[0]
        score_column = ["Score"] * self.score.shape[0]
        exl, eml, score = (self.ex_loadings.copy(), self.em_loadings.copy(), self.score.copy())
        exl.index = pd.MultiIndex.from_tuples(list(zip(*[ex_column, self.ex_range.tolist()])),
                                              names=('type', 'wavelength'))
        eml.index = pd.MultiIndex.from_tuples(list(zip(*[em_column, self.em_range.tolist()])),
                                              names=('type', 'wavelength'))
        score.index = pd.MultiIndex.from_tuples(list(zip(*[score_column, self.score.index])),
                                                names=('type', 'time'))
        with open(filepath, 'w') as f:
            f.write('# \n# Fluorescence Model \n# \n')
            for key, value in info_dict.items():
                f.write(key + '\t' + value)
                f.write('\n')
            f.write('# \n# Excitation/Emission (Ex, Em), wavelength [nm], component_n [loading] \n# \n')
            f.close()
        with pd.option_context('display.multi_sparse', False):
            exl.to_csv(filepath, mode='a', sep="\t", header=None)
            eml.to_csv(filepath, mode='a', sep="\t", header=None)
        with open(filepath, 'a') as f:
            f.write('# \n# timestamp, component_n [Score] \n# \n')
            f.close()
        score.to_csv(filepath, mode='a', sep="\t", header=None)
        with open(filepath, 'a') as f:
            f.write('# end #')
        return info_dict

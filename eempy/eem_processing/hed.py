from .eemnmf import EEMNMF
from .basic import random_split_columns

class HED:
    """
    Hierarchical EEM Decomposition (HED).

    Parameters
    ----------
    n_components: int
        The number of components
    solver: str, {'cd', 'mu', 'hals'}
        The numerical solver of NMF. 'cd' is a Coordinate Descent solver. 'mu' is a Multiplicative Update solver. 'hals'
        is a Hierarchical Alternating Least Squares solver.
    prior_dict_W: dict, optional
        A dictionary mapping component indices (int) to prior vectors (1D array of length n_features) for regularizing
        sample loadings. The k-th loading of sample loadings will be penalized to be close to prior_dict_sample[k] if
        present. Only applied to 'hals' solver.
    prior_dict_H: dict, optional
        A dictionary mapping component indices (int) to prior vectors (1D array of length n_samples) for regularizing
        component loadings. The k-th loading of components loadings will be penalized to be close to
        prior_dict_component[k] if present. Only applied to 'hals' solver.
    gamma_W: float, default=0
        Strength of the prior regularization on sample loadings. Only applied to 'hals' solver.
    gamma_H: float ,default=0
        Strength of the prior regularization on component loadings. Only applied to 'hals' solver.
    idx_top: list, optional
        List of indices of samples serving as numerators in ratio calculation.
    idx_bot: list, optional
        List of indices of samples serving as denominators in ratio calculation.
    lam: float, default=0
        Strength of the ratio regularization on sample loadings. Only applied to 'hals' solver.
    normalization: str, {'pixel_std'}:
        The normalization of EEMs before conducting NMF. 'pixel_std' normalizes the intensities of each pixel across
        all samples by standard deviation.
    Attributes
    ----------
    fmax: pandas.DataFrame
        Fmax table calculated using the score matrix of NMF.
    nnls_fmax: pandas.DataFrame
        Fmax table calculated by fitting EEMs with NMF components using non-negative least square (NNLS).
    components: np.ndarray
        NMF components.
    eem_stack_train: np.ndarray
        EEMs used for PARAFAC model establishment.
    eem_stack_reconstructed: np.ndarray
        EEMs reconstructed by the established PARAFAC model.
    """
    def __init__(
            self, parent_components, parent_scores, child_structure_dict, solver='hals',
            prior_dict_W=None, gamma_W=0, gamma_H=0,
            idx_top=None, idx_bot=None, kw_top=None, kw_bot=None, lam=None,
            normalization=None, max_iter_als=300, max_iter_nnls=300, tol=1e-6, random_state=None,
                 ):
        # -----------Parameters-------------
        assert len(child_structure_dict) == parent_components.shape[0], \
            "The length of child_structure_dict must be the same as the number of parent components."
        self.parent_components = parent_components
        self.parent_scores = parent_scores
        self.child_structure_dict = child_structure_dict
        self.solver = solver
        self.prior_dict_W = prior_dict_W
        self.gamma_W = gamma_W
        self.gamma_H = gamma_H
        self.idx_top = idx_top
        self.idx_bot = idx_bot
        self.kw_top = kw_top
        self.kw_bot = kw_bot
        self.lam = lam
        self.normalization = normalization
        self.max_iter_als = max_iter_als
        self.max_iter_nnls = max_iter_nnls
        self.tol = tol
        self.random_state = random_state
        # -----------Attributes-------------
        self.fmax = None
        self.nnls_fmax = None
        self.components = None
        self.eem_stack_train = None
        self.eem_stack_reconstructed = None
        self.ex_range = None,
        self.em_range = None
        self.beta = None


    def fit(self, eem_dataset):
        prior_components = {k: self.parent_components[k].reshape(-1) for k in
                                       range(self.parent_components.shape[0])}
        new_order = []
        for k, n in self.child_structure_dict.items():
            if n <= 1:
                new_order.append(k)
            else:
                new_order.extend(([k] * n))
        fixed_components = [key for key, value in self.child_structure_dict.items() if value == 0]
        fmax_init = random_split_columns(self.parent_scores.to_numpy(), self.child_structure_dict)
        components_init = self.parent_components.reshape((self.parent_components.shape[0], -1))[new_order, :]
        if not isinstance(self.gamma_H, dict):
            gamma_H = {i: self.gamma_H for i in range(len(new_order))}
        else:
            gamma_H = self.gamma_H
        if not isinstance(self.gamma_W, dict):
            gamma_W = {i: self.gamma_W for i in range(len(new_order))}
        else:
            gamma_W = self.gamma_W
        model_child = EEMNMF(
            n_components=len(new_order),
            max_iter_nnls=self.max_iter_nnls,
            max_iter_als=self.max_iter_als,
            random_state=self.random_state,
            solver=self.solver,
            sort_components_by_em=False,
            tol=1e-8,
            init='custom',
            custom_init=[fmax_init, components_init],
            fixed_components=[i for i in range(len(new_order)) if new_order[i] in fixed_components],
            prior_dict_W={new_order.index(key): value for key, value in self.prior_dict_W.items()} if self.prior_dict_W is not None else None,
            prior_dict_H={i: prior_components[k] for i, k in enumerate(new_order)},
            gamma_H=gamma_H,
            gamma_W=gamma_W,
        )
        model_child.fit(eem_dataset)
        self.fmax = model_child.fmax
        self.nnls_fmax = model_child.nnls_fmax
        self.components = model_child.components
        self.eem_stack_train = eem_dataset
        self.eem_stack_reconstructed = model_child.eem_stack_reconstructed
        self.ex_range = eem_dataset.ex_range
        self.em_range = eem_dataset.em_range
        self.beta = model_child.beta

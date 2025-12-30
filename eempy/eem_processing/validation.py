from eempy.eem_processing.eem_dataset import *

class SplitValidation:
    """
    Conduct PARAFAC model validation by evaluating the consistency of PARAFAC models established on EEM sub-datasets.

    Parameters
    ----------
    base_model: PARAFAC or EEMNMF
        The base PARAFAC or NMF model to be used for validation.
    n_splits: int
        Number of splits.
    combination_size: int or str, {int, 'half'}
        The number of splits assembled into one combination. If 'half' is passed, each combination will include
        half of the splits (i.e., the split-half validation).
    rule: str, {'random', 'sequential'}
        Whether to split the EEM dataset randomly. If 'sequential' is passed, the dataset will be split according
        to index order.
    random_state: int, optional
        Random seed for reproducibility. Only used if `rule` is 'random'.
    Attributes
    -----------
    eem_subsets: dict
        Dictionary of EEM sub-datasets.
    subset_specific_models: dict
        Dictionary of PARAFAC models established on sub-datasets.
    """
    def __init__(self, base_model, n_splits=4, combination_size='half', rule='random',
                 random_state=None):
        # ---------------Parameters-------------------
        self.base_model = base_model
        self.n_split = n_splits
        self.combination_size = combination_size
        self.rule = rule
        self.random_state = random_state
        # ----------------Attributes------------------
        self.eem_subsets = None
        self.subset_specific_models = None
        self.eem_dataset_full = None


    def fit(self, eem_dataset: EEMDataset):
        split_set = eem_dataset.splitting(
            n_split=self.n_split, rule=self.rule, random_state=self.random_state,
            kw_top=self.base_model.kw_top, kw_bot=self.base_model.kw_bot,
            idx_top=self.base_model.idx_top, idx_bot=self.base_model.idx_bot
        )
        if self.combination_size == 'half':
            cs = int(self.n_split) / 2
        else:
            cs = int(self.combination_size)
        elements = list(itertools.combinations([i for i in range(self.n_split)], int(cs)))
        codes = list(itertools.combinations(list(string.ascii_uppercase)[0:self.n_split], int(cs)))
        model_complete = copy.deepcopy(self.base_model)
        model_complete.fit(eem_dataset=eem_dataset)
        sims_ex, sims_em, models, subsets = ({}, {}, {}, {})
        for e, c in zip(elements, codes):
            label = ''.join(c)
            subdataset = combine_eem_datasets([split_set[i] for i in e])
            model_subdataset = copy.deepcopy(self.base_model)
            if model_subdataset.init == "custom":
                init0 = model_subdataset.custom_init[0]
                idx_in_split = [eem_dataset.index.index(idx) for idx in subdataset.index]
                model_subdataset.custom_init[0] = init0[idx_in_split]
            if isinstance(model_subdataset, EEMNMF):
                if self.base_model.prior_dict_W is not None:
                    idx_in_split = [eem_dataset.index.index(idx) for idx in subdataset.index]
                    for r in list(self.base_model.prior_dict_W.keys()):
                        model_subdataset.prior_dict_W[r] = self.base_model.prior_dict_W[r][idx_in_split]
            elif isinstance(model_subdataset, PARAFAC):
                if self.base_model.prior_dict_sample is not None:
                    idx_in_split = [eem_dataset.index.index(idx) for idx in subdataset.index]
                    for r in list(self.base_model.prior_dict_W.keys()):
                        model_subdataset.prior_dict_sample[r] = self.base_model.prior_dict_sample[r][idx_in_split]
            model_subdataset.fit(subdataset)
            models[label] = model_subdataset
            subsets[label] = subdataset
        models = align_components_by_components(
            models,
            {f'C{i + 1}': model_complete.components[i] for i in range(model_complete.n_components)},
        )
        self.eem_subsets = subsets
        self.subset_specific_models = models
        self.eem_dataset_full = eem_dataset
        return self


    def compare_parafac_loadings(self):
        """
        Calculate the similarities of ex/em loadings between PARAFAC models established on sub-datasets.
        Returns
        -------
        similarities_ex: pandas.DataFrame
            Similarities in excitation loadings.
        similarities_em: pandas.DataFrame
            Similarities in emission loadings.
        """
        labels = sorted(self.subset_specific_models.keys())
        similarities_ex = {}
        similarities_em = {}
        for k in range(int(len(labels) / 2)):
            m1 = self.subset_specific_models[labels[k]]
            m2 = self.subset_specific_models[labels[-1 - k]]
            sims_ex = loadings_similarity(m1.ex_loadings, m2.ex_loadings).to_numpy().diagonal()
            sims_em = loadings_similarity(m1.em_loadings, m2.em_loadings).to_numpy().diagonal()
            pair_labels = '{m1} vs. {m2}'.format(m1=labels[k], m2=labels[-1 - k])
            similarities_ex[pair_labels] = sims_ex
            similarities_em[pair_labels] = sims_em
        similarities_ex = pd.DataFrame.from_dict(
            similarities_ex, orient='index',
            columns=['Similarities in C{i}-ex'.format(i=i + 1) for i in range(self.base_model.n_components)]
        )
        similarities_ex.index.name_train = 'Test'
        similarities_em = pd.DataFrame.from_dict(
            similarities_em, orient='index',
            columns=['Similarities in C{i}-em'.format(i=i + 1) for i in range(self.base_model.n_components)]
        )
        similarities_em.index.name_train = 'Test'
        return similarities_ex, similarities_em


    def compare_components(self):
        """
        Calculate the similarities of components between PARAFAC or NMF models established on sub-datasets.
        Returns
        -------
        similarities_components: pandas.DataFrame
            Similarities in components.
        """
        labels = sorted(self.subset_specific_models.keys())
        similarities_components = {}
        for k in range(int(len(labels) / 2)):
            m1 = self.subset_specific_models[labels[k]]
            m2 = self.subset_specific_models[labels[-1 - k]]
            sims = component_similarity(m1.components, m2.components).to_numpy().diagonal()
            pair_labels = '{m1} vs. {m2}'.format(m1=labels[k], m2=labels[-1 - k])
            similarities_components[pair_labels] = sims
        similarities_components = pd.DataFrame.from_dict(
            similarities_components, orient='index',
            columns=['Similarities in C{i}'.format(i=i + 1) for i in range(self.base_model.n_components)]
        )
        similarities_components.index.name_train = 'Test'
        return similarities_components


    def correlation_cv(self, ref_col):
        assert ref_col in self.eem_dataset_full.ref.columns, f"'{ref_col}' is not found in reference."
        labels = sorted(self.subset_specific_models.keys())
        tbl = {}
        for k in range(int(len(labels) / 2)):
            m1 = self.subset_specific_models[labels[k]]
            m2 = self.subset_specific_models[labels[-1 - k]]
            d1 = self.eem_subsets[labels[k]]
            d2 = self.eem_subsets[labels[-1 - k]]
            pair_labels_12 = 'train: {m1} / test: {m2}'.format(m1=labels[k], m2=labels[-1 - k])
            pair_labels_21 = 'train: {m2} / test: {m1}'.format(m1=labels[k], m2=labels[-1 - k])
            mask_ref1 = ~np.isnan(d1.ref[ref_col].to_numpy())
            y_d1 = d1.ref[ref_col].to_numpy()[mask_ref1]
            mask_ref2 = ~np.isnan(d2.ref[ref_col].to_numpy())
            y_d2 = d2.ref[ref_col].to_numpy()[mask_ref2]
            _, fmax_train_d1, _ = m1.predict(d1)
            _, fmax_test_d2, _ = m1.predict(d2)
            _, fmax_train_d2, _ = m2.predict(d2)
            _, fmax_test_d1, _ = m2.predict(d1)
            tbl_12 = {}
            for r in range(self.base_model.n_components):
                x_train = fmax_train_d1.iloc[mask_ref1, [r]].to_numpy()
                x_test = fmax_test_d2.iloc[mask_ref2, [r]].to_numpy()
                lr = LinearRegression(fit_intercept=True)
                lr.fit(x_train, y_d1)
                y_pred_train = lr.predict(x_train)
                r2_train = lr.score(x_train, y_d1)
                rmse_train = np.sqrt(mean_squared_error(y_d1, y_pred_train))
                tbl_12[ref_col + '-' + f'C{r + 1} Fmax' + '-r2-training'] = r2_train
                tbl_12[ref_col + '-' + f'C{r + 1} Fmax' + '-rmse-training'] = rmse_train
                y_pred_test = lr.predict(x_test)
                r2_test = lr.score(x_test, y_d2)
                rmse_test = np.sqrt(mean_squared_error(y_d2, y_pred_test))
                tbl_12[ref_col + '-' + f'C{r + 1} Fmax' + '-r2-test'] = r2_test
                tbl_12[ref_col + '-' + f'C{r + 1} Fmax' + '-rmse-test'] = rmse_test
            tbl[pair_labels_12] = tbl_12
            tbl_21 = {}
            for r in range(self.base_model.n_components):
                x_train = fmax_train_d2.iloc[mask_ref2, [r]].to_numpy()
                x_test = fmax_test_d1.iloc[mask_ref1, [r]].to_numpy()
                lr = LinearRegression(fit_intercept=True)
                lr.fit(x_train, y_d2)
                y_pred_train = lr.predict(x_train)
                r2_train = lr.score(x_train, y_d2)
                rmse_train = np.sqrt(mean_squared_error(y_d2, y_pred_train))
                tbl_21[ref_col + '-' + f'C{r + 1} Fmax' + '-r2-training'] = r2_train
                tbl_21[ref_col + '-' + f'C{r + 1} Fmax' + '-rmse-training'] = rmse_train
                y_pred_test = lr.predict(x_test)
                r2_test = lr.score(x_test, y_d1)
                rmse_test = np.sqrt(mean_squared_error(y_d1, y_pred_test))
                tbl_21[ref_col + '-' + f'C{r + 1} Fmax' + '-r2-test'] = r2_test
                tbl_21[ref_col + '-' + f'C{r + 1} Fmax' + '-rmse-test'] = rmse_test
            tbl[pair_labels_21] = tbl_21
        tbl_df = pd.DataFrame(tbl).T
        return tbl_df

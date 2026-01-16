from .shared import *
from ..config import COLORS, MARKER_SHAPES
from ..ids import IDS


def register_callbacks(app):
    #   -------------Callbacks of page #6

    #       ---------------Step 1: Calculate consensus

    @app.callback(
        [
            Output(IDS.KMETHOD_BASE_MODEL_MESSAGE, 'children')
        ],
        [
            Input(IDS.KMETHOD_BASE_MODEL, 'value')
        ]
    )
    def on_kmethod_base_clustering_message(base_clustering):
        if base_clustering == 'parafac':
            message = ['Parameters "initialization", "non negativity" and "total fluorescence normalization"'
                       'are set in tab "PARAFAC".']
        elif base_clustering == 'nmf':
            message = ['Parameters "Initialization", "solver", "normalization" and "alpha_w", "alpha_h", "l1_ratio" are set in tab "NMF".']
        else:
            message = [None]
        return message


    @app.callback(
        [
            Output(IDS.KMETHOD_EEM_DATASET_ESTABLISHMENT_MESSAGE, 'children'),
            Output(IDS.KMETHOD_CONSENSUS_MATRIX, 'children'),
            Output(IDS.KMETHOD_ERROR_HISTORY, 'children'),
            Output(IDS.KMETHOD_STEP1_SPINNER, 'children'),
            Output(IDS.KMETHOD_CONSENSUS_MATRIX_DATA, 'data'),
            Output(IDS.KMETHOD_EEM_DATASET_ESTABLISH, 'data'),
            Output(IDS.KMETHOD_BASE_CLUSTERING_PARAMETERS, 'data')
        ],
        [
            Input(IDS.BUILD_KMETHOD_CONSENSUS, 'n_clicks'),
            State(IDS.EEM_GRAPH_OPTIONS, 'value'),
            State(IDS.KMETHOD_EEM_DATASET_ESTABLISHMENT_PATH_INPUT, 'value'),
            State(IDS.KMETHOD_ESTABLISHMENT_INDEX_KW_MANDATORY, 'value'),
            State(IDS.KMETHOD_ESTABLISHMENT_INDEX_KW_OPTIONAL, 'value'),
            State(IDS.KMETHOD_CLUSTER_FROM_FILE_CHECKBOX, 'value'),
            State(IDS.KMETHOD_RANK, 'value'),
            State(IDS.KMETHOD_BASE_MODEL, 'value'),
            State(IDS.KMETHOD_NUM_INIT_SPLITS, 'value'),
            State(IDS.KMETHOD_NUM_BASE_CLUSTERINGS, 'value'),
            State(IDS.KMETHOD_NUM_ITERATIONS, 'value'),
            State(IDS.KMETHOD_CONVERGENCE_TOLERANCE, 'value'),
            State(IDS.KMETHOD_ELIMINATION, 'value'),
            State(IDS.KMETHOD_SUBSAMPLING_PORTION, 'value'),
            State(IDS.PARAFAC_INIT_METHOD, 'value'),
            State(IDS.PARAFAC_NN_CHECKBOX, 'value'),
            State(IDS.PARAFAC_TF_CHECKBOX, 'value'),
            State(IDS.NMF_SOLVER, 'value'),
            State(IDS.NMF_INIT, 'value'),
            State(IDS.NMF_NORMALIZATION_CHECKBOX, 'value'),
            State(IDS.NMF_ALPHA_W, 'value'),
            State(IDS.NMF_ALPHA_H, 'value'),
            State(IDS.NMF_L1_RATIO, 'value'),
            State(IDS.EEM_DATASET, 'data')
        ]
    )
    def on_build_consensus(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, cluster_from_file,
                           rank, base_clustering, n_init_splits, n_base_clusterings, n_iterations, tol, elimination,
                           subsampling_portion,
                           parafac_init, parafac_nn, parafac_tf,
                           nmf_solver, nmf_init, nmf_normalization, nmf_alpha_w, nmf_alpha_h, nmf_l1_ratio,
                           eem_dataset_dict):
        if n_clicks is None:
            return None, None, None, 'Calculate consensus', None, None, None
        if not path_establishment:
            if eem_dataset_dict is None:
                message = (
                    'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing"'
                    'section, or import an EEM dataset from file.')
                return message, None, None, 'Calculate consensus', None, None, None
            eem_dataset_establishment = EEMDataset(
                eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                    in eem_dataset_dict['eem_stack']]),
                ex_range=np.array(eem_dataset_dict['ex_range']),
                em_range=np.array(eem_dataset_dict['em_range']),
                index=eem_dataset_dict['index'],
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                 index=eem_dataset_dict['index']) if eem_dataset_dict['ref'] is not None else None,
                cluster=eem_dataset_dict['cluster']
            )
        else:
            if not os.path.exists(path_establishment):
                message = ('Error: No such file or directory: ' + path_establishment)
                return message, None, None, 'Calculate consensus', None, None, None
            else:
                _, file_extension = os.path.splitext(path_establishment)

                if file_extension == '.json':
                    with open(path_establishment, 'r') as file:
                        eem_dataset_dict = json.load(file)
                elif file_extension == '.pkl':
                    with open(path_establishment, 'rb') as file:
                        eem_dataset_dict = pickle.load(file)
                else:
                    raise ValueError("Unsupported file extension: {}".format(file_extension))
                eem_dataset_establishment = EEMDataset(
                    eem_stack=np.array(
                        [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                         in eem_dataset_dict['eem_stack']]),
                    ex_range=np.array(eem_dataset_dict['ex_range']),
                    em_range=np.array(eem_dataset_dict['em_range']),
                    index=eem_dataset_dict['index'],
                    ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                     index=eem_dataset_dict['index']) if eem_dataset_dict['ref'] is not None else None,
                    cluster=eem_dataset_dict['cluster']
                )
        kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
        kw_optional = str_string_to_list(kw_optional) if kw_optional else None
        eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                                  inplace=True)

        eem_dataset_establishment_json_dict = {
            'eem_stack': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for sublist in
                          eem_dataset_establishment.eem_stack.tolist()],
            'ex_range': eem_dataset_establishment.ex_range.tolist(),
            'em_range': eem_dataset_establishment.em_range.tolist(),
            'index': eem_dataset_establishment.index,
            'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
            if eem_dataset_establishment.ref is not None else None,
            'cluster': None,
        }

        if base_clustering == 'parafac':
            base_clustering_parameters = {
                'n_components': rank, 'init': parafac_init,
                'non_negativity': True if 'non_negative' in parafac_nn else False,
                'tf_normalization': True if 'tf_normalization' in parafac_tf else False,
                'sort_components_by_em': True
            }
            base_model = PARAFAC(**base_clustering_parameters)
        elif base_clustering == 'nmf':
            base_clustering_parameters = {
                'n_components': rank, 'solver': nmf_solver, 'init': nmf_init,
                'normalization': nmf_normalization[0],
                'alpha_H': nmf_alpha_h, 'alpha_W': nmf_alpha_w, 'l1_ratio': nmf_l1_ratio
            }
            base_model = EEMNMF(**base_clustering_parameters)

        kmethod = KMethod(base_model=base_model, n_initial_splits=n_init_splits, max_iter=n_iterations, tol=tol,
                          elimination=elimination, distance_metric="reconstruction_error", kw_top="B1C1", kw_bot="B1C2")
        consensus_matrix, _, error_history = kmethod.calculate_consensus(eem_dataset_establishment, n_base_clusterings,
                                                                         subsampling_portion)
        consensus_matrix_tabs = dbc.Card(children=[])
        error_history_tabs = dbc.Card(children=[])

        fig_consensus_matrix = go.Figure(
            data=go.Heatmap(
                z=consensus_matrix,
                x=eem_dataset_establishment.index if eem_dataset_establishment.index
                else [i for i in range(consensus_matrix.shape[1])],
                y=eem_dataset_establishment.index if eem_dataset_establishment.index
                else [i for i in range(consensus_matrix.shape[0])],
                colorscale='reds',
                hoverongaps=False,
                hovertemplate='sample1: %{y}<br>sample2: %{x}<br>consensus coefficient: %{z}<extra></extra>'
            )
        )
        fig_consensus_matrix.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

        consensus_matrix_tabs.children.append(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    figure=fig_consensus_matrix,
                                    config={'autosizable': False},
                                    style={'width': '40vw',
                                           'height': '70vh'}
                                )
                            ]
                        )
                    ]
                ),
            ]),
        )

        error_means = [df.mean(axis=0) for df in error_history]
        error_means = pd.concat(error_means, axis=1).T
        error_means.columns = [f'iteration {i + 1}' for i in range(error_means.shape[1])]
        error_means.index = [f'base clustering {i + 1}' for i in range(len(error_history))]
        error_means_melted = error_means.melt(var_name='Column', value_name='Value')

        fig_error = px.box(error_means_melted, x='Column', y='Value', points='all')
        fig_error.update_layout(
            xaxis_title='Iterations',
            yaxis_title='Error'
        )

        error_history_tabs.children.append(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    figure=fig_error
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Table.from_dataframe(error_means,
                                                         bordered=True, hover=True, index=True,

                                                         )
                            ]
                        )
                    ]
                )
            ])
        )

        return (None, consensus_matrix_tabs, error_history_tabs, 'Calculate consensus', consensus_matrix.tolist(),
                eem_dataset_establishment_json_dict, base_clustering_parameters)


    #   -----------------Step 2: Hierarchical clustering

    @app.callback(
        [
            Output(IDS.KMETHOD_DENDROGRAM, 'children'),
            Output(IDS.KMETHOD_SORTED_CONSENSUS_MATRIX, 'children'),
            Output(IDS.KMETHOD_SILHOUETTE_SCORE, 'children'),
            Output(IDS.KMETHOD_RECONSTRUCTION_ERROR_REDUCTION, 'children'),
            Output(IDS.KMETHOD_FMAX, 'children'),
            Output(IDS.KMETHOD_STEP2_SPINNER, 'children'),
            Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION, 'options'),
            Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION, 'value'),
            # Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION, 'options'),
            # Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION, 'value'),
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION, 'options'),
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION, 'value'),
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION, 'options'),
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION, 'value'),
            # Output(IDS.KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION, 'options'),
            # Output(IDS.KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION, 'value'),
            Output(IDS.KMETHOD_TEST_MODEL_SELECTION, 'options'),
            Output(IDS.KMETHOD_TEST_MODEL_SELECTION, 'value'),
            Output(IDS.KMETHOD_TEST_PRED_REF_SELECTION, 'options'),
            Output(IDS.KMETHOD_TEST_PRED_REF_SELECTION, 'value'),
            Output(IDS.KMETHOD_MODELS, 'data')
        ],
        [
            Input(IDS.BUILD_KMETHOD_CLUSTERING, 'n_clicks'),
            State(IDS.KMETHOD_BASE_MODEL, 'value'),
            State(IDS.KMETHOD_NUM_FINAL_CLUSTERS, 'value'),
            State(IDS.KMETHOD_CONSENSUS_CONVERSION, 'value'),
            State(IDS.KMETHOD_VALIDATIONS, 'value'),
            State(IDS.KMETHOD_CONSENSUS_MATRIX_DATA, 'data'),
            State(IDS.KMETHOD_EEM_DATASET_ESTABLISH, 'data'),
            State(IDS.KMETHOD_BASE_CLUSTERING_PARAMETERS, 'data')
        ]
    )
    def on_hierarchical_clustering(n_clicks, base_clustering, n_final_clusters, conversion, validations,
                                   consensus_matrix, eem_dataset_establish_dict, base_clustering_parameters):
        if n_clicks is None:
            return None, None, None, None, None, 'Clustering', [], None, [], None, [], None, [], None, [], None, None
        else:
            if eem_dataset_establish_dict is None:
                return None, None, None, None, None, 'Clustering', [], None, [], None, [], None, [], None, [], None, None
            else:
                eem_dataset_establish = EEMDataset(
                    eem_stack=np.array(
                        [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                         in eem_dataset_establish_dict['eem_stack']]),
                    ex_range=np.array(eem_dataset_establish_dict['ex_range']),
                    em_range=np.array(eem_dataset_establish_dict['em_range']),
                    index=eem_dataset_establish_dict['index'],
                    ref=pd.DataFrame(eem_dataset_establish_dict['ref'][1:],
                                     columns=eem_dataset_establish_dict['ref'][0],
                                     index=eem_dataset_establish_dict['index'])
                    if eem_dataset_establish_dict['ref'] is not None else None,
                    cluster=eem_dataset_establish_dict['cluster']
                )

        n_clusters_list = num_string_to_list(n_final_clusters)
        kmethod_fit_params = {}

        if eem_dataset_establish.ref is not None:
            valid_ref = eem_dataset_establish.ref.columns[~eem_dataset_establish.ref.isna().all()].tolist()
        else:
            valid_ref = []

        if base_clustering == 'parafac':
            base_model = PARAFAC(**base_clustering_parameters)
            unified_model = PARAFAC(**base_clustering_parameters)
        elif base_clustering == 'nmf':
            base_model = EEMNMF(**base_clustering_parameters)
            unified_model = EEMNMF(**base_clustering_parameters)

        unified_model.fit(eem_dataset_establish)

        dendrogram_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        sorted_consensus_matrix_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        silhouette_score_tabs = dbc.Card([])
        reconstruction_error_reduction_tabs = dbc.Card(
            [dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        fmax_establishment_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        slt = []
        kmethod_models = {}

        for k in n_clusters_list:
            kmethod = KMethod(base_model=base_model, n_initial_splits=None)
            kmethod.consensus_matrix = np.array(consensus_matrix)
            try:
                kmethod.hierarchical_clustering(eem_dataset_establish, k, conversion)
            except ValueError:
                continue
            cluster_specific_models = kmethod.cluster_specific_models
            eem_clusters = kmethod.eem_clusters
            cluster_labels_combined = []
            for sub_dataset in eem_clusters.values():
                cluster_labels_combined += sub_dataset.cluster

            fmax_combined = pd.concat([model.nnls_fmax for model in cluster_specific_models.values()], axis=0)
            fmax_combined_sorted = fmax_combined.sort_index()
            cluster_labels_combined_sorted = [x for _, x in sorted(zip(fmax_combined.index, cluster_labels_combined))]
            fig_fmax = plot_fmax(table=fmax_combined_sorted, display=False, labels=cluster_labels_combined_sorted)
            fmax_combined_sorted['Cluster'] = cluster_labels_combined_sorted

            kmethod_fit_params_k = {}

            if base_clustering == 'parafac':
                component_names = unified_model.nnls_fmax.columns.tolist()
            elif base_clustering == 'nmf':
                component_names = unified_model.nnls_fmax.columns.tolist()
            for ref_var in valid_ref + component_names:
                kmethod_fit_params_k[ref_var] = []

            if eem_dataset_establish.ref is not None:
                for i in range(k):
                    cluster_specific_model = kmethod.cluster_specific_models[i + 1]
                    eem_cluster = kmethod.eem_clusters[i + 1]
                    valid_ref_cluster = eem_cluster.ref.columns[~eem_cluster.ref.isna().all()].tolist()
                    for ref_var in valid_ref_cluster:
                        stats = []
                        x = eem_cluster.ref[ref_var]
                        if base_clustering == 'parafac':
                            model_var = copy.copy(cluster_specific_model.nnls_fmax)
                        elif base_clustering == 'nmf':
                            model_var = copy.copy(cluster_specific_model.nnls_fmax)
                        model_var.columns = [f'Cluster {i + 1}-' + col for col in model_var.columns]
                        nan_rows = x[x.isna()].index
                        x = x.drop(nan_rows)
                        if x.shape[0] <= 1:
                            continue
                        for f_col in model_var.columns:
                            y = model_var[f_col]
                            y = y.drop(nan_rows)
                            x_reshaped = np.array(x).reshape(-1, 1)
                            lm = LinearRegression().fit(x_reshaped, y)
                            r_squared = lm.score(x_reshaped, y)
                            intercept = lm.intercept_
                            slope = lm.coef_[0]
                            pearson_corr, pearson_p = pearsonr(x, y)
                            stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                        kmethod_fit_params_k[ref_var] = kmethod_fit_params_k[ref_var] + stats

                    for c_var in component_names:
                        if base_clustering == 'parafac':
                            x = cluster_specific_model.nnls_fmax[c_var]
                            model_var = cluster_specific_model.nnls_fmax
                        elif base_clustering == 'nmf':
                            x = cluster_specific_model.nnls_fmax[c_var]
                            model_var = cluster_specific_model.nnls_fmax
                        stats = []
                        nan_rows = x[x.isna()].index
                        x = x.drop(nan_rows)
                        if x.shape[0] <= 1:
                            continue
                        for f_col in model_var.columns:
                            y = model_var[f_col]
                            y = y.drop(nan_rows)
                            x_reshaped = np.array(x).reshape(-1, 1)
                            lm = LinearRegression().fit(x_reshaped, y)
                            r_squared = lm.score(x_reshaped, y)
                            intercept = lm.intercept_
                            slope = lm.coef_[0]
                            pearson_corr, pearson_p = pearsonr(x, y)
                            stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                        kmethod_fit_params_k[c_var] = kmethod_fit_params_k[c_var] + stats

            fig_dendrogram = plot_dendrogram(kmethod.linkage_matrix, kmethod.threshold_r,
                                             eem_dataset_establish_dict['index'])
            dendrogram_tabs.children[0].children.append(
                dcc.Tab(label=f'{k}-cluster',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    figure=fig_dendrogram,
                                                    config={'autosizable': False},
                                                    style={'width': '50vw',
                                                           'height': '70vh'}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                            ]),
                        ],
                        style={'padding': '0', 'line-width': '100%'},
                        selected_style={'padding': '0', 'line-width': '100%'}
                        )
            )

            fig_sorted_consensus_matrix = go.Figure(
                data=go.Heatmap(
                    z=kmethod.consensus_matrix_sorted,
                    x=kmethod.index_sorted if eem_dataset_establish.index
                    else [i for i in range(consensus_matrix.shape[1])],
                    y=kmethod.index_sorted if eem_dataset_establish.index
                    else [i for i in range(consensus_matrix.shape[0])],
                    colorscale='reds',
                    hoverongaps=False,
                    hovertemplate='sample1: %{y}<br>sample2: %{x}<br>consensus coefficient: %{z}<extra></extra>'
                )
            )
            for j in range(max(kmethod.labels)):
                idx = np.where(np.sort(kmethod.labels) == j + 1)[0]
                fig_sorted_consensus_matrix.add_shape(type="rect", x0=min(idx), y0=min(idx),
                                                      x1=min(idx) + len(idx), y1=min(idx) + len(idx),
                                                      line=dict(color="black", width=3),
                                                      )
            fig_sorted_consensus_matrix.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

            sorted_consensus_matrix_tabs.children[0].children.append(
                dcc.Tab(
                    label=f'{k}-cluster',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(
                                                figure=fig_sorted_consensus_matrix,
                                                config={'autosizable': False},
                                                style={'width': '50vw',
                                                       'height': '70vh'}
                                            )
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ],
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                )
            )

            if 'silhouette_score' in validations:
                slt.append(kmethod.silhouette_score)

            if 'RER' in validations:
                rmse_combined = pd.concat([model.sample_rmse() for model in cluster_specific_models.values()], axis=0)
                rmse_combined_sorted = rmse_combined.sort_index()
                rmse_combined_sorted = pd.concat([rmse_combined_sorted, unified_model.sample_rmse()], axis=1)
                rmse_combined_sorted.columns = ['RMSE with cluster-specific models', 'RMSE with unified model']
                rmse_combined_sorted['RMSE reduction (%)'] = (
                        100 * (1 - (rmse_combined_sorted.iloc[:, 0] / rmse_combined_sorted.iloc[:, 1])))
                fig_rer = plot_reconstruction_error(
                    rmse_combined_sorted, display=False, bar_col_name='RMSE reduction (%)',
                    labels=cluster_labels_combined_sorted
                )
                rmse_combined_sorted['Cluster'] = cluster_labels_combined_sorted

                reconstruction_error_reduction_tabs.children[0].children.append(
                    dcc.Tab(label=f'{k}-cluster',
                            children=[
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(figure=fig_rer,
                                                              style={'width': '80vw',
                                                                     'height': '70vh'}
                                                              )
                                                ]
                                            )
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(rmse_combined_sorted,
                                                                             bordered=True, hover=True, index=True)
                                                ]
                                            )
                                        ]
                                    )
                                ]),
                            ],
                            style={'padding': '0', 'line-width': '100%'},
                            selected_style={'padding': '0', 'line-width': '100%'}
                            )
                )

            fmax_establishment_tabs.children[0].children.append(
                dcc.Tab(label=f'{k}-cluster',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(figure=fig_fmax,
                                                          style={'width': '80vw',
                                                                 'height': '70vh'}
                                                          )
                                            ]
                                        )
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(fmax_combined_sorted,
                                                                         bordered=True, hover=True, index=True)
                                            ]
                                        )
                                    ]
                                )
                            ]),
                        ],
                        style={'padding': '0', 'line-width': '100%'},
                        selected_style={'padding': '0', 'line-width': '100%'}
                        )
            )

            kmethod_models_n = {}
            for i in range(k):
                cluster_specific_model = kmethod.cluster_specific_models[i + 1]
                eem_cluster = kmethod.eem_clusters[i + 1]
                if base_clustering == 'parafac':
                    kmethod_models_n[i+1] = {
                        'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                                       sublist in cluster_specific_model.components.tolist()],
                        'Fmax': [
                                    cluster_specific_model.nnls_fmax.columns.tolist()] + cluster_specific_model.nnls_fmax.values.tolist(),
                        'index': eem_cluster.index,
                        'ref': [eem_cluster.ref.columns.tolist()] + eem_cluster.ref.values.tolist()
                        if eem_cluster.ref is not None else None,
                        'fitting_params': kmethod_fit_params_k
                    }
                elif base_clustering == 'nmf':
                    kmethod_models_n[i+1] = {
                        'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                                       sublist in cluster_specific_model.components.tolist()],
                        'NNLS_Fmax': [
                                         cluster_specific_model.nnls_fmax.columns.tolist()] + cluster_specific_model.nnls_fmax.values.tolist(),
                        'nmf_Fmax': [
                                        cluster_specific_model.nnls_fmax.columns.tolist()] + cluster_specific_model.nnls_fmax.values.tolist(),
                        'index': eem_cluster.index,
                        'ref': [eem_cluster.ref.columns.tolist()] + eem_cluster.ref.values.tolist()
                        if eem_cluster.ref is not None else None,
                        'fitting_params': kmethod_fit_params_k
                    }

            kmethod_models[k] = kmethod_models_n

        if 'silhouette_score' in validations:
            slt_table = pd.DataFrame({'Number of clusters': list(kmethod_models.keys()), 'Silhouette score': slt})
            silhouette_score_tabs.children.append(
                html.Div([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=px.line(
                                            x=slt_table['Number of clusters'],
                                            y=slt_table['Silhouette score'],
                                            markers=True,
                                            labels={'x': 'Number of cluster', 'y': 'Silhouette score'},
                                        ),
                                        config={'autosizable': False},
                                    )
                                ]
                            )
                        ]
                    ),

                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Table.from_dataframe(slt_table,
                                                             bordered=True, hover=True,
                                                             )
                                ]
                            ),
                        ]
                    ),
                ]),
            )

        model_options = [{'label': '{r}-cluster'.format(r=r), 'value': r} for r in kmethod_models.keys()]
        ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
            (eem_dataset_establish.ref is not None) else None

        if base_clustering == 'parafac':
            indicator_options = ['Fmax']
        elif base_clustering == 'nmf':
            indicator_options = ['NNLS-Fmax', 'NMF-Fmax']

        return (dendrogram_tabs, sorted_consensus_matrix_tabs, silhouette_score_tabs, reconstruction_error_reduction_tabs,
                fmax_establishment_tabs, 'Clustering', model_options, None, model_options, None, indicator_options, None,
                model_options, None, ref_options, None, kmethod_models)


    # ---------Update cluster dropdown in components section-------------
    @app.callback(
        [
            Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION, 'options'),
            Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION, 'value'),
        ],
        [
            Input(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION, 'value')
        ]
    )
    def on_update_kmethod_component_cluster_list(k):
        if k is not None:
            options = [{'label': f'Cluster {n + 1}', 'value': n + 1} for n in range(int(k))]
            return options, None
        else:
            return [], None


    # ---------Update reference dropdown in correlation section-----------
    @app.callback(
        [
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION, 'options'),
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION, 'value'),
        ],
        [
            Input(IDS.KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION, 'value'),
            State(IDS.KMETHOD_MODELS, 'data')
        ]
    )
    def update_reference_dropdown_by_selected_model(k, model):
        if all([k, model]):
            options = []
            for c in list(model[str(k)].values()):
                options += list(c['fitting_params'].keys())
            options = list(set(options))
            options = sorted(options)
            return options, None
        else:
            return [], None


    # ---------Plot components-----------
    @app.callback(
        [
            Output(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_GRAPH, 'children')
        ],
        [
            Input(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION, 'value'),
            Input(IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION, 'value'),
            State(IDS.EEM_GRAPH_OPTIONS, 'value'),
            State(IDS.KMETHOD_MODELS, 'data'),
            State(IDS.KMETHOD_EEM_DATASET_ESTABLISH, 'data'),
        ],
    )
    def on_plot_kmethod_components(k, cluster_i, eem_graph_options, kmethod_models, eem_dataset_establish):
        if all([k, cluster_i, kmethod_models, eem_dataset_establish]):
            ex_range = np.array(eem_dataset_establish['ex_range'])
            em_range = np.array(eem_dataset_establish['em_range'])
            cluster_model = kmethod_models[str(k)][str(cluster_i)]
            r = len(cluster_model['components'])
            n_rows = (r - 1) // 3 + 1
            graphs = [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=plot_eem(np.array(cluster_model['components'][3 * i]),
                                                ex_range=ex_range,
                                                em_range=em_range,
                                                vmin=0 if np.min(
                                                    cluster_model['components'][3 * i]) > -1e-3 else None,
                                                vmax=None,
                                                auto_intensity_range=False,
                                                plot_tool='plotly',
                                                display=False,
                                                figure_size=(5, 3.5),
                                                axis_label_font_size=14,
                                                cbar_font_size=12,
                                                title_font_size=16,
                                                title=f'C{3 * i + 1}',
                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                else False,
                                                rotate=True if 'rotate' in eem_graph_options else False,
                                                ) if 3 * i + 1 <= r else go.Figure(
                                    layout={'width': 400, 'height': 300}),
                                # style={'width': '30vw'}
                            ),
                            width={'size': 4},
                        ),

                        dbc.Col(
                            dcc.Graph(
                                figure=plot_eem(np.array(cluster_model['components'][3 * i + 1]),
                                                ex_range=ex_range,
                                                em_range=em_range,
                                                vmin=0 if np.min(
                                                    cluster_model['components'][
                                                        3 * i + 1]) > -1e-3 else None,
                                                vmax=None,
                                                auto_intensity_range=False,
                                                plot_tool='plotly',
                                                display=False,
                                                figure_size=(5, 3.5),
                                                axis_label_font_size=14,
                                                cbar_font_size=12,
                                                title_font_size=16,
                                                title=f'C{3 * i + 2}',
                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                else False,
                                                rotate=True if 'rotate' in eem_graph_options else False,
                                                ) if 3 * i + 2 <= r else go.Figure(
                                    layout={'width': 400, 'height': 300}),
                                # style={'width': '30vw'}
                            ),
                            width={'size': 4},
                        ),

                        dbc.Col(
                            dcc.Graph(
                                figure=plot_eem(np.array(cluster_model['components'][3 * i + 2]),
                                                ex_range=ex_range,
                                                em_range=em_range,
                                                vmin=0 if np.min(
                                                    cluster_model['components'][
                                                        3 * i + 2]) > -1e-3 else None,
                                                vmax=None,
                                                auto_intensity_range=False,
                                                plot_tool='plotly',
                                                display=False,
                                                figure_size=(5, 3.5),
                                                axis_label_font_size=14,
                                                cbar_font_size=12,
                                                title_font_size=16,
                                                title=f'C{3 * i + 3}',
                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                else False,
                                                rotate=True if 'rotate' in eem_graph_options else False,
                                                ) if 3 * i + 3 <= r else go.Figure(),
                                # style={'width': '30vw'}
                            ),
                            width={'size': 4},
                        ),
                    ], style={'width': '90vw'}
                ) for i in range(n_rows)
            ]
            return [graphs]
        else:
            return [None]


    # -----------Analyze correlations between Fmax and reference variables in model establishment

    @app.callback(
        [
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_GRAPH, 'figure'),
            Output(IDS.KMETHOD_ESTABLISHMENT_CORR_TABLE, 'children'),
        ],
        [
            Input(IDS.KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION, 'value'),
            Input(IDS.KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION, 'value'),
            Input(IDS.KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION, 'value'),
            State(IDS.KMETHOD_MODELS, 'data')
        ]
    )
    def on_kmethod_establishment_correlations(k, indicator, ref_var, kmethod_models):
        if all([k, indicator, ref_var, kmethod_models]):
            fig = go.Figure()
            for n in range(1, k + 1):
                cluster_specific_model = kmethod_models[str(k)][str(n)]
                r = len(cluster_specific_model['components'])
                ref_df = pd.DataFrame(cluster_specific_model['ref'][1:], columns=cluster_specific_model['ref'][0],
                                      index=cluster_specific_model['index'])
                if 'NNLS_Fmax' in list(cluster_specific_model.keys()):
                    nnls_fmax_df = pd.DataFrame(cluster_specific_model['NNLS-Fmax'][1:],
                                                columns=cluster_specific_model['NNLS-Fmax'][0],
                                                index=cluster_specific_model['index'])
                    nmf_fmax_df = pd.DataFrame(cluster_specific_model['NMF-Fmax'][1:],
                                               columns=cluster_specific_model['NMF-Fmax'][0],
                                               index=cluster_specific_model['index'])
                    fmax_df = pd.concat([nnls_fmax_df, nmf_fmax_df], axis=1)
                elif 'Fmax' in list(cluster_specific_model.keys()):
                    fmax_df = pd.DataFrame(cluster_specific_model['Fmax'][1:],
                                           columns=cluster_specific_model['Fmax'][0],
                                           index=cluster_specific_model['index'])

                ref_df = pd.concat([ref_df, fmax_df], axis=1)
                reference_variable = ref_df[ref_var]
                fluorescence_indicators = pd.DataFrame(cluster_specific_model[indicator][1:],
                                                       columns=cluster_specific_model[indicator][0],
                                                       index=cluster_specific_model['index'])

                stats_k = cluster_specific_model['fitting_params']

                for i, col in enumerate(fluorescence_indicators.columns):
                    x = reference_variable
                    y = fluorescence_indicators[col]
                    nan_rows = x[x.isna()].index
                    x = x.drop(nan_rows)
                    y = y.drop(nan_rows)
                    if x.shape[0] < 1:
                        return go.Figure(), None
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                             name=f'Cluster {n}-{col}',
                                             text=[i for i in x.index],
                                             marker=dict(color=COLORS[i % len(COLORS)],
                                                         symbol=MARKER_SHAPES[n % len(MARKER_SHAPES)]),
                                             hoverinfo='text+x+y'))
                    fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                             y=stats_k[ref_var][r*n-r+i][1] * np.array([x.min(), x.max()]) + stats_k[ref_var][r*n-r+i][2],
                                             mode='lines', name=f'Cluster {n}-{col}-Linear Regression Line',
                                             line=dict(dash='dash', color=COLORS[i % len(COLORS)])))
                fig.update_xaxes(title_text=ref_var)
                fig.update_yaxes(title_text=indicator)


            fig.update_layout(legend=dict(y=-0.3, x=0.5, xanchor='center', yanchor='top'))

            tbl = pd.DataFrame(
                stats_k[ref_var],
                columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
            )
            tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
            return fig, tbl
        else:
            return go.Figure(), None
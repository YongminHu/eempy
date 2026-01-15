from .shared import *


#   -------------Callbacks of page #3

#   -------------Establish NMF model

@app.callback(
    [
        Output('nmf-eem-dataset-establishment-message', 'children'),
        Output('nmf-components', 'children'),
        Output('nmf-fmax', 'children'),
        Output('nmf-establishment-reconstruction-error', 'children'),
        Output('nmf-split-half', 'children'),
        Output('nmf-spinner', 'children'),
        Output('nmf-establishment-corr-model-selection', 'options'),
        Output('nmf-establishment-corr-model-selection', 'value'),
        # Output('nmf-establishment-corr-ref-selection', 'options'),
        # Output('nmf-establishment-corr-ref-selection', 'value'),
        Output('nmf-test-pred-ref-selection', 'options'),
        Output('nmf-test-pred-ref-selection', 'value'),
        Output('nmf-test-model-selection', 'options'),
        Output('nmf-test-model-selection', 'value'),
        Output('nmf-models', 'data'),
    ],
    [
        Input('build-nmf-model', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('nmf-eem-dataset-establishment-path-input', 'value'),
        State('nmf-establishment-index-kw-mandatory', 'value'),
        State('nmf-establishment-index-kw-optional', 'value'),
        State('nmf-rank', 'value'),
        State('nmf-solver', 'value'),
        State('nmf-init', 'value'),
        State('nmf-normalization-checkbox', 'value'),
        State('nmf-alpha-w', 'value'),
        State('nmf-alpha-h', 'value'),
        State('nmf-l1-ratio', 'value'),
        State('nmf-max-iter-als', 'value'),
        State('nmf-max-iter-nnls', 'value'),
        State('nmf-validations', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_nmf_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, rank, solver, init,
                       normalization, alpha_w, alpha_h, l1_ratio, n_iter_als, n_iter_nnls, validations, eem_dataset_dict):
    if n_clicks is None:
        return None, None, None, None, None, 'Build model', [], None, [], None, [], None, None
    if not path_establishment:
        if eem_dataset_dict is None:
            message = (
                'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                'section, or import an EEM dataset from file.')
            return message, None, None, None, None, 'Build model', [], None, [], None, [], None, None
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                             index=eem_dataset_dict['index'])
            if eem_dataset_dict['ref'] is not None else None,
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return message, None, None, None, None, 'Build model', [], None, [], None, [], None, None
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
                                 index=eem_dataset_dict['index'])
                if eem_dataset_dict['ref'] is not None else None,
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              inplace=True)
    rank_list = num_string_to_list(rank)
    nmf_models = {}
    components_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    fmax_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    reconstruction_error_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    split_half_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])

    if eem_dataset_establishment.ref is not None:
        valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
    else:
        valid_ref = None

    for r in rank_list:
        nmf_r = EEMNMF(
            n_components=r, solver=solver, init=init, normalization=normalization[0] if normalization else None,
            alpha_component=alpha_h, alpha_sample=alpha_w, l1_ratio=l1_ratio, max_iter_als=n_iter_als,
            max_iter_nnls=n_iter_nnls,
        )
        nmf_r.fit(eem_dataset_establishment)
        nmf_fit_params_r = {}
        if eem_dataset_establishment.ref is not None:
            for ref_var in valid_ref:
                x = eem_dataset_establishment.ref[ref_var]
                stats = []
                nmf_var = nmf_r.nnls_fmax
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                if x.shape[0] < 1:
                    continue
                for f_col in nmf_var.columns:
                    y = nmf_var[f_col]
                    y = y.drop(nan_rows)
                    x_reshaped = np.array(x).reshape(-1, 1)
                    lm = LinearRegression().fit(x_reshaped, y)
                    r_squared = lm.score(x_reshaped, y)
                    intercept = lm.intercept_
                    slope = lm.coef_[0]
                    pearson_corr, pearson_p = pearsonr(x, y)
                    stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                nmf_fit_params_r[ref_var] = stats
        for c_var in nmf_r.nnls_fmax.columns:
            x = nmf_r.nnls_fmax[c_var]
            parafac_var = nmf_r.nnls_fmax
            stats = []
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            if x.shape[0] < 1:
                continue
            for f_col in parafac_var.columns:
                y = parafac_var[f_col]
                y = y.drop(nan_rows)
                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                r_squared = lm.score(x_reshaped, y)
                intercept = lm.intercept_
                slope = lm.coef_[0]
                pearson_corr, pearson_p = pearsonr(x, y)
                stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
            nmf_fit_params_r[c_var] = stats
        nmf_models[r] = {
            'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                           sublist in nmf_r.components.tolist()],
            'NNLS-Fmax': [nmf_r.nnls_fmax.columns.tolist()] + nmf_r.nnls_fmax.values.tolist(),
            'NMF-Fmax': [nmf_r.fmax.columns.tolist()] + nmf_r.fmax.values.tolist(),
            'index': eem_dataset_establishment.index,
            'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
            if eem_dataset_establishment.ref is not None else None,
            'fitting_params': nmf_fit_params_r
        }

        # for component graphs, determine the layout according to the number of components
        n_rows = (r - 1) // 3 + 1

        # components
        components_tabs.children[0].children.append(
            # html.Div(
            dcc.Tab(label=f'{r}-component',
                    children=
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[4 * i],
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[4 * i]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{4 * i + 1}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 4 * i + 1 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 3},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[4 * i + 1],
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    4 * i + 1]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{4 * i + 2}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 4 * i + 2 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 3},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[4 * i + 2],
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    4 * i + 2]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{4 * i + 3}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 4 * i + 3 <= r else go.Figure(),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 3},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[4 * i + 3],
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    4 * i + 3]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{4 * i + 4}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 4 * i + 4 <= r else go.Figure(),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 3},
                                    ),
                                ]
                            ) for i in range(n_rows)
                        ],
                        style={'width': '90vw'}
                    ),
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        # scores
        fmax_tabs.children[0].children.append(
            dcc.Tab(label=f'{r}-component',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_fmax(nmf_r.fmax,
                                                                       display=False
                                                                       ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(nmf_r.fmax,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_fmax(nmf_r.nnls_fmax,
                                                                       display=False
                                                                       ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(nmf_r.nnls_fmax,
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

        reconstruction_error_tabs.children[0].children.append(
            dcc.Tab(label=f'{r}-component',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_error(nmf_r.sample_rmse(),
                                                                        display=False
                                                                        ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(nmf_r.sample_rmse(),
                                                                     bordered=True, hover=True, index=True)
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

        if 'split_half' in validations:
            model_sv = copy.deepcopy(nmf_r)
            split_validation = SplitValidation(base_model=model_sv)
            split_validation.fit(eem_dataset_establishment)
            subset_specific_models = split_validation.subset_specific_models
            similarities_components = split_validation.compare_components()
            split_half_tabs.children[0].children.append(
                dcc.Tab(label=f'{r}-component',
                        children=[
                            html.Div([
                                # dbc.Row(
                                #     [
                                #         dbc.Col(
                                #             [
                                #                 dcc.Graph(
                                #                     figure=plot_loadings(subset_specific_models,
                                #                                          n_cols=3,
                                #                                          plot_tool='plotly',
                                #                                          display=False,
                                #                                          legend_pad=0.2),
                                #                     config={'autosizable': False},
                                #                 )
                                #             ]
                                #         )
                                #     ]
                                # ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(similarities_components,
                                                                         bordered=True, hover=True, index=True,
                                                                         )
                                            ]
                                        ),
                                    ]
                                ),

                            ]),
                        ],
                        style={'padding': '0', 'line-width': '100%'},
                        selected_style={'padding': '0', 'line-width': '100%'}
                        )
            )


    model_options = [{'label': '{r}-component'.format(r=r), 'value': r} for r in nmf_models.keys()]
    ref_options = [{'label': var, 'value': var} for var in valid_ref] if (
            eem_dataset_establishment.ref is not None) else []

    return (None, components_tabs, fmax_tabs, reconstruction_error_tabs, split_half_tabs, 'Build model', model_options, None,
            ref_options, None, model_options, None, nmf_models)


# -----------Update reference selection dropdown

@app.callback(
    [
        Output('nmf-establishment-corr-ref-selection', 'options'),
        Output('nmf-establishment-corr-ref-selection', 'value'),
    ],
    [
        Input('nmf-establishment-corr-model-selection', 'value'),
        State('nmf-models', 'data')
    ]
)
def update_nmf_reference_dropdown_by_selected_model(r, nmf_model):
    if all([r, nmf_model]):
        options = list(nmf_model[str(r)]['fitting_params'].keys())
        return options, None
    else:
        return [], None


# -----------Analyze correlations between score/Fmax and reference variables in model establishment

@app.callback(
    [
        Output('nmf-establishment-corr-graph', 'figure'),  # size, intervals?
        Output('nmf-establishment-corr-table', 'children'),
    ],
    [
        Input('nmf-establishment-corr-model-selection', 'value'),
        Input('nmf-establishment-corr-indicator-selection', 'value'),
        Input('nmf-establishment-corr-ref-selection', 'value'),
        State('nmf-models', 'data')
    ]
)
def on_nmf_establishment_correlations(r, indicator, ref_var, nmf_models):
    if all([r, indicator, ref_var, nmf_models]):
        ref_df = pd.DataFrame(nmf_models[str(r)]['ref'][1:], columns=nmf_models[str(r)]['ref'][0],
                              index=nmf_models[str(r)]['index'])
        nnls_fmax_df = pd.DataFrame(nmf_models[str(r)]['NNLS-Fmax'][1:], columns=nmf_models[str(r)]['NNLS-Fmax'][0],
                                    index=nmf_models[str(r)]['index'])
        nmf_fmax_df = pd.DataFrame(nmf_models[str(r)]['NMF-Fmax'][1:], columns=nmf_models[str(r)]['NMF-Fmax'][0],
                                   index=nmf_models[str(r)]['index'])
        ref_df = pd.concat([ref_df, nnls_fmax_df, nmf_fmax_df], axis=1)
        var = ref_df[ref_var]
        nmf_var = pd.DataFrame(nmf_models[str(r)][indicator][1:], columns=nmf_models[str(r)][indicator][0],
                               index=nmf_models[str(r)]['index'])
        fig = go.Figure()

        stats = nmf_models[str(r)]['fitting_params']

        for i, col in enumerate(nmf_var.columns):
            x = var
            y = nmf_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     marker=dict(color=colors[i % 10]), hoverinfo='text+x+y'))
            fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                     y=stats[ref_var][i][1] * np.array([x.min(), x.max()]) + stats[ref_var][i][2],
                                     mode='lines', name=f'{col}-Linear Regression Line',
                                     line=dict(dash='dash', color=colors[i % 10])))
        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text=indicator)

        tbl = pd.DataFrame(
            stats[ref_var],
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Fit a test EEM dataset using the established nmf model components
@app.callback(
    [
        Output('nmf-eem-dataset-predict-message', 'children'),  # size, intervals?
        Output('nmf-test-fmax', 'children'),
        Output('nmf-test-error', 'children'),
        Output('nmf-test-corr-ref-selection', 'options'),
        Output('nmf-test-corr-ref-selection', 'value'),
        Output('nmf-test-corr-indicator-selection', 'options'),
        Output('nmf-test-corr-indicator-selection', 'value'),
        Output('nmf-predict-spinner', 'children'),
        Output('nmf-test-results', 'data'),
    ],
    [
        Input('predict-nmf-model', 'n_clicks'),
        State('nmf-eem-dataset-predict-path-input', 'value'),
        State('nmf-test-index-kw-mandatory', 'value'),
        State('nmf-test-index-kw-optional', 'value'),
        State('nmf-test-model-selection', 'value'),
        State('nmf-models', 'data')
    ]
)
def on_nmf_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, model_r, nmf_models):
    if n_clicks is None:
        return (None, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    if path_predict is None:
        return (None, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    if not os.path.exists(path_predict):
        message = ('Error: No such file: ' + path_predict)
        return (message, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    else:
        _, file_extension = os.path.splitext(path_predict)

        if file_extension == '.json':
            with open(path_predict, 'r') as file:
                eem_dataset_dict = json.load(file)
        elif file_extension == '.pkl':
            with open(path_predict, 'rb') as file:
                eem_dataset_dict = pickle.load(file)
        else:
            raise ValueError("Unsupported file extension: {}".format(file_extension))
        eem_dataset_predict = EEMDataset(
            eem_stack=np.array(
                [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                 in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], index=eem_dataset_dict['index'],
                             columns=eem_dataset_dict['ref'][0]) if eem_dataset_dict['ref'] is not None else None,
        )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, inplace=True)

    if eem_dataset_predict.ref is not None:
        valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
    else:
        valid_ref = None

    _, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset_predict.eem_stack,
                                                         np.array(
                                                             [
                                                                 [
                                                                     [np.nan if x is None else x for x in
                                                                      subsublist] for subsublist in sublist
                                                                 ]
                                                                 for sublist in
                                                                 nmf_models[str(model_r)][
                                                                     'components']
                                                             ]
                                                         ),
                                                         fit_intercept=False)
    fmax_sample = pd.DataFrame(
        fmax_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i + 1) for i in range(model_r)]
    )

    pred = {}
    if eem_dataset_predict.ref is not None:
        for ref_var in valid_ref:
            if ref_var in nmf_models[str(model_r)]['fitting_params'].keys():
                params = nmf_models[str(model_r)]['fitting_params'][ref_var]
                pred_var = fmax_sample.copy()
                for i, f_col in enumerate(fmax_sample):
                    pred_r = fmax_sample[f_col] - params[i][2]
                    pred_r = pred_r / params[i][1]
                    pred_var[f_col] = pred_r
                pred[ref_var] = [pred_var.columns.tolist()] + pred_var.values.tolist()
            else:
                pred[ref_var] = None

    fmax_tab = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(figure=plot_fmax(fmax_sample,
                                                   display=False
                                                   ),
                                  config={'autosizable': False},
                                  style={'width': 1700, 'height': 800}
                                  )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Table.from_dataframe(fmax_sample,
                                                 bordered=True, hover=True, index=True
                                                 )
                    ]
                ),
            ]
        ),
    ])

    error_tab = html.Div(children=[])

    if eem_dataset_predict.ref is not None:
        indicator_options = [{'label': 'Fmax', 'value': 'Fmax'},
                             {'label': 'Prediction of reference', 'value': 'Prediction of reference'}]
    else:
        indicator_options = [{'label': 'Fmax', 'value': 'Fmax'}]

    ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

    test_results = {
        'Fmax': [fmax_sample.columns.tolist()] + fmax_sample.values.tolist(),
        'Prediction of reference': pred,
        'ref': [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
        if eem_dataset_predict.ref is not None else None,
        'index': eem_dataset_predict.index
    }

    return (None, fmax_tab, error_tab, ref_options, None,
            indicator_options, None, 'predict', test_results)


# -----------Predict the corresponding reference variables for the test EEM datasets using the model fitted in the model
#            establishment step
@app.callback(
    [
        Output('nmf-test-pred-graph', 'figure'),  # size, intervals?
        Output('nmf-test-pred-table', 'children'),
    ],
    [
        Input('predict-parafac-model', 'n_clicks'),
        Input('nmf-test-pred-ref-selection', 'value'),
        Input('nmf-test-pred-model-selection', 'value'),
        State('nmf-test-results', 'data'),
    ]
)
def on_nmf_predict_reference_test(n_clicks, ref_var, pred_model, nmf_test_results):
    if all([ref_var, pred_model, nmf_test_results]):
        pred = nmf_test_results['Prediction of reference'][ref_var]
        pred = pd.DataFrame(pred[1:], columns=pred[0], index=nmf_test_results['index'])
        fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
        if nmf_test_results['ref'] is not None:
            ref = pd.DataFrame(nmf_test_results['ref'][1:], columns=nmf_test_results['ref'][0],
                               index=nmf_test_results['index'])
            true_value = ref[ref_var]
            nan_rows = true_value[true_value.isna()].index
            true_value = true_value.drop(nan_rows)
            pred_overlap = pred.drop(nan_rows)
            fig.add_trace(go.Scatter(x=true_value.index, y=true_value.values, mode='markers',
                                     marker=dict(color='black', size=5), name='True value'))
            rmse_results = []
            for f_col in pred.columns:
                rmse = sqrt(sum((true_value.values - pred_overlap[f_col]) ** 2) / len(true_value.values))
                rmse_results.append(rmse)
            tbl = html.Div(
                [
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.DataFrame([rmse_results], columns=pred.columns,
                                                              index=pd.Index(['RMSE'], name='Error metric')),
                                                 bordered=True, hover=True, index=True)
                    ),
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.concat([pred, ref[ref_var]], axis=1),
                                                 bordered=True, hover=True, index=True)
                    )
                ]
            )
        else:
            tbl = dbc.Table.from_dataframe(pred, bordered=True, hover=True, index=True)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Analyze correlations between score/Fmax and reference variables in model testing
@app.callback(
    [
        Output('nmf-test-corr-graph', 'figure'),  # size, intervals?
        Output('nmf-test-corr-table', 'children'),
    ],
    [
        Input('nmf-test-corr-indicator-selection', 'value'),
        Input('nmf-test-corr-ref-selection', 'value'),
        State('nmf-test-results', 'data'),
    ]
)
def on_nmf_test_correlations(indicator, ref_var, nmf_test_results):
    if all([indicator, ref_var, nmf_test_results]):
        ref_df = pd.DataFrame(nmf_test_results['ref'][1:], columns=nmf_test_results['ref'][0],
                              index=nmf_test_results['index'])
        var = ref_df[ref_var]
        if indicator != 'Prediction of reference':
            nmf_var = pd.DataFrame(nmf_test_results[indicator][1:], columns=nmf_test_results[indicator][0],
                                   index=nmf_test_results['index'])
        else:
            if nmf_test_results[indicator][ref_var] is not None:
                nmf_var = pd.DataFrame(nmf_test_results[indicator][ref_var][1:],
                                       columns=nmf_test_results[indicator][ref_var][0],
                                       index=nmf_test_results['index'])
            else:
                return go.Figure(), None

        fig = go.Figure()
        stats = []
        for i, col in enumerate(nmf_var.columns):
            x = var
            y = nmf_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            predictions = lm.predict(x_reshaped)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     hoverinfo='text+x+y', marker=dict(color=colors[i % 10])))
            fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name=f'{col} fit',
                                     line=dict(dash='dash', color=colors[i % 10])))
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([col, slope, intercept, r_squared, pearson_corr, pearson_p])

        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text="Prediction of " + ref_var if indicator == 'Prediction of reference' else indicator)

        tbl = pd.DataFrame(
            stats,
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None

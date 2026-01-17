from .shared import *
from ..config import COLORS
from ..ids import IDS
from ..serialization import eem_dataset_from_serializable

def register_callbacks(app):
    #   -------------Callbacks of page #2

    #  ----------Establish PARAFAC model
    @app.callback(
        [
            Output(IDS.PARAFAC_EEM_DATASET_ESTABLISHMENT_MESSAGE, 'children'),
            Output(IDS.PARAFAC_LOADINGS, 'children'),
            Output(IDS.PARAFAC_COMPONENTS, 'children'),
            Output(IDS.PARAFAC_FMAX, 'children'),
            Output(IDS.PARAFAC_ESTABLISHMENT_RECONSTRUCTION_ERROR, 'children'),
            Output(IDS.PARAFAC_VARIANCE_EXPLAINED, 'children'),
            Output(IDS.PARAFAC_CORE_CONSISTENCY, 'children'),
            Output(IDS.PARAFAC_LEVERAGE, 'children'),
            Output(IDS.PARAFAC_SPLIT_HALF, 'children'),
            Output(IDS.BUILD_PARAFAC_SPINNER, 'children'),
            Output(IDS.PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION, 'options'),
            Output(IDS.PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION, 'value'),
            Output(IDS.PARAFAC_TEST_MODEL_SELECTION, 'options'),
            Output(IDS.PARAFAC_TEST_MODEL_SELECTION, 'value'),
            Output(IDS.PARAFAC_TEST_PRED_REF_SELECTION, 'options'),
            Output(IDS.PARAFAC_TEST_PRED_REF_SELECTION, 'value'),
            Output(IDS.PARAFAC_MODELS, 'data'),
        ],
        [
            Input(IDS.BUILD_PARAFAC_MODEL, 'n_clicks'),
            State(IDS.EEM_GRAPH_OPTIONS, 'value'),
            State(IDS.PARAFAC_EEM_DATASET_ESTABLISHMENT_PATH_INPUT, 'value'),
            State(IDS.PARAFAC_ESTABLISHMENT_INDEX_KW_MANDATORY, 'value'),
            State(IDS.PARAFAC_ESTABLISHMENT_INDEX_KW_OPTIONAL, 'value'),
            State(IDS.PARAFAC_RANK, 'value'),
            State(IDS.PARAFAC_INIT_METHOD, 'value'),
            State(IDS.PARAFAC_NN_CHECKBOX, 'value'),
            State(IDS.PARAFAC_TF_CHECKBOX, 'value'),
            State(IDS.PARAFAC_SOLVER, 'value'),
            State(IDS.PARAFAC_VALIDATIONS, 'value'),
            State(IDS.EEM_DATASET, 'data')
        ]
    )
    def on_build_parafac_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, rank, init, nn,
                               tf, optimizer, validations, eem_dataset_dict):
        if n_clicks is None:
            return (None, None, None, None, None, None, None, None, None, 'Build model', [], None, [], None, [], None,
                    None)
        if not path_establishment:
            if eem_dataset_dict is None:
                message = ("Error: No built EEM dataset detected. Please build an EEM dataset first in EEM "
                           "pre-processing section, or import an EEM dataset from file.")
                return (message, None, None, None, None, None, None, None, None, 'Build model', [], None, [], None, [], None, None)
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
                return (message, None, None, None, None, None, None, None, None, 'Build model', [], None, [], None, [], None, None)
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
                eem_dataset_establishment = eem_dataset_from_serializable(eem_dataset_dict)
        kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
        kw_optional = str_string_to_list(kw_optional) if kw_optional else None
        eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                                  inplace=True)

        rank_list = num_string_to_list(rank)
        parafac_models = {}
        loadings_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        components_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        scores_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        fmax_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        reconstruction_error_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        variance_explained_tabs = dbc.Card(children=[])
        core_consistency_tabs = dbc.Card(children=[])
        leverage_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        split_half_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
        cc = []
        ve = []

        if eem_dataset_establishment.ref is not None:
            valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
        else:
            valid_ref = None

        for r in rank_list:
            parafac_r = PARAFAC(n_components=r, init=init, non_negativity=True if 'non_negative' in nn else False,
                                tf_normalization=True if 'tf_normalization' in tf else False, solver=optimizer,
                                sort_components_by_em=True, loadings_normalization='maximum')
            parafac_r.fit(eem_dataset_establishment)
            parafac_fit_params_r = {}
            if eem_dataset_establishment.ref is not None:
                for ref_var in valid_ref:
                    x = eem_dataset_establishment.ref[ref_var]
                    parafac_var = parafac_r.nnls_fmax
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
                    parafac_fit_params_r[ref_var] = stats
            for c_var in parafac_r.nnls_fmax.columns:
                x = parafac_r.nnls_fmax[c_var]
                parafac_var = parafac_r.nnls_fmax
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
                parafac_fit_params_r[c_var] = stats

            parafac_models[r] = {
                'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                               sublist in parafac_r.components.tolist()],
                'score': [parafac_r.score.columns.tolist()] + parafac_r.score.values.tolist(),
                'Fmax': [parafac_r.fmax.columns.tolist()] + parafac_r.fmax.values.tolist(),
                'NNLS_Fmax': [parafac_r.nnls_fmax.columns.tolist()] + parafac_r.nnls_fmax.values.tolist(),
                'index': eem_dataset_establishment.index,
                'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
                if eem_dataset_establishment.ref is not None else None,
                'fitting_params': parafac_fit_params_r
            }

            # for component graphs, determine the layout according to the number of components
            n_rows = (r - 1) // 3 + 1

            # ex/em loadings
            loadings_tabs.children[0].children.append(
                dcc.Tab(label=f'{r}-component',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(figure=plot_loadings({f'{r}-component': parafac_r},
                                                                               plot_tool='plotly', n_cols=4,
                                                                               display=False),
                                                          config={'autosizable': False}
                                                          )
                                            ]
                                        )
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(parafac_r.ex_loadings,
                                                                         bordered=True, hover=True, index=True,

                                                                         )
                                            ]
                                        ),

                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(parafac_r.em_loadings,
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
                                                figure=plot_eem(parafac_r.components[4 * i],
                                                                ex_range=parafac_r.ex_range,
                                                                em_range=parafac_r.em_range,
                                                                vmin=0 if np.min(
                                                                    parafac_r.components[4 * i]) > -1e-3 else None,
                                                                vmax=None,
                                                                auto_intensity_range=False,
                                                                plot_tool='plotly',
                                                                display=False,
                                                                figure_size=(5, 3.5),
                                                                axis_label_font_size=14,
                                                                cbar_font_size=12,
                                                                title_font_size=16,
                                                                title=f'C{4 * i + 1}',
                                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                                rotate=True if 'rotate' in eem_graph_options else False,
                                                                ) if 4 * i + 1 <= r else go.Figure(
                                                    layout={'width': 400, 'height': 300}),
                                                style={'width': '500', 'height': '500'}
                                            ),
                                            width={'size': 3},
                                        ),

                                        dbc.Col(
                                            dcc.Graph(
                                                figure=plot_eem(parafac_r.components[4 * i + 1],
                                                                ex_range=parafac_r.ex_range,
                                                                em_range=parafac_r.em_range,
                                                                vmin=0 if np.min(
                                                                    parafac_r.components[
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
                                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                                rotate=True if 'rotate' in eem_graph_options else False,
                                                                ) if 4 * i + 2 <= r else go.Figure(
                                                    layout={'width': 400, 'height': 300}),
                                                style={'width': '500', 'height': '500'}
                                            ),
                                            width={'size': 3},
                                        ),

                                        dbc.Col(
                                            dcc.Graph(
                                                figure=plot_eem(parafac_r.components[4 * i + 2],
                                                                ex_range=parafac_r.ex_range,
                                                                em_range=parafac_r.em_range,
                                                                vmin=0 if np.min(
                                                                    parafac_r.components[
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
                                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                                rotate=True if 'rotate' in eem_graph_options else False,
                                                                ) if 4 * i + 3 <= r else go.Figure(),
                                                style={'width': '500', 'height': '500'}
                                            ),
                                            width={'size': 3},
                                        ),

                                        dbc.Col(
                                            dcc.Graph(
                                                figure=plot_eem(parafac_r.components[4 * i + 3],
                                                                ex_range=parafac_r.ex_range,
                                                                em_range=parafac_r.em_range,
                                                                vmin=0 if np.min(
                                                                    parafac_r.components[
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
                                                                fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
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

            # fmax
            fmax_tabs.children[0].children.append(
                dcc.Tab(label=f'{r}-component',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(figure=plot_fmax(parafac_r.nnls_fmax,
                                                                           display=False,
                                                                           yaxis_title='Fmax'
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
                                                dbc.Table.from_dataframe(parafac_r.nnls_fmax,
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

            # scores
            reconstruction_error_tabs.children[0].children.append(
                dcc.Tab(label=f'{r}-component',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(figure=plot_error(parafac_r.sample_rmse(),
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
                                                dbc.Table.from_dataframe(parafac_r.sample_rmse(),
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

            # variance explained
            if 'variance_explained' in validations:
                ve.append(parafac_r.variance_explained())

            # core consistency
            if 'core_consistency' in validations:
                cc.append(parafac_r.core_consistency())

            # leverage
            if 'leverage' in validations:
                lvr_sample = parafac_r.leverage('sample')
                lvr_ex = parafac_r.leverage('ex')
                lvr_em = parafac_r.leverage('em')
                leverage_tabs.children[0].children.append(
                    dcc.Tab(label=f'{r}-component',
                            children=[
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        figure=px.line(
                                                            x=lvr_sample.index,
                                                            y=lvr_sample.iloc[:, 0],
                                                            markers=True,
                                                            labels={'x': 'index-sample', 'y': 'leverage-sample'},
                                                        ),
                                                        config={'autosizable': False},
                                                        style={'width': 1700, 'height': 400}
                                                    )
                                                ]
                                            )
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        figure=px.line(
                                                            x=lvr_ex.index,
                                                            y=lvr_ex.iloc[:, 0],
                                                            markers=True,
                                                            labels={'x': 'index-ex', 'y': 'leverage-ex'},
                                                        ),
                                                        config={'autosizable': False},
                                                        style={'width': 1700, 'height': 400}
                                                    )
                                                ]
                                            )
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        figure=px.line(
                                                            x=lvr_em.index,
                                                            y=lvr_em.iloc[:, 0],
                                                            markers=True,
                                                            labels={'x': 'index-em', 'y': 'leverage-em'},
                                                        ),
                                                        config={'autosizable': False},
                                                        style={'width': 1700, 'height': 400}
                                                    )
                                                ]
                                            )
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(lvr_sample,
                                                                             bordered=True, hover=True, index=True,
                                                                             )
                                                ]
                                            ),

                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(lvr_ex,
                                                                             bordered=True, hover=True, index=True,
                                                                             )
                                                ]
                                            ),

                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(lvr_em,
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

            if 'split_half' in validations:
                model_sv = copy.deepcopy(parafac_r)
                split_validation = SplitValidation(base_model=model_sv)
                split_validation.fit(eem_dataset_establishment)
                subset_specific_models = split_validation.subset_specific_models
                similarities_ex, similarities_em = split_validation.compare_parafac_loadings()
                split_half_tabs.children[0].children.append(
                    dcc.Tab(label=f'{r}-component',
                            children=[
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        figure=plot_loadings(subset_specific_models,
                                                                             n_cols=3,
                                                                             plot_tool='plotly',
                                                                             display=False,
                                                                             legend_pad=0.2),
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
                                                    dbc.Table.from_dataframe(similarities_ex,
                                                                             bordered=True, hover=True, index=True,
                                                                             )
                                                ]
                                            ),
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(similarities_em,
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

        if 'variance_explained' in validations:
            ve_table = pd.DataFrame({'Number of components': rank_list, 'Variance explained': ve})
            variance_explained_tabs.children.append(
                html.Div([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=px.line(
                                            x=ve_table['Number of components'],
                                            y=ve_table['Variance explained'],
                                            markers=True,
                                            labels={'x': 'Number of components', 'y': 'Core consistency'},
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
                                    dbc.Table.from_dataframe(ve_table,
                                                             bordered=True, hover=True,
                                                             )
                                ]
                            ),
                        ]
                    ),
                ]),
            )


        if 'core_consistency' in validations:
            cc_table = pd.DataFrame({'Number of components': rank_list, 'Core consistency': cc})
            core_consistency_tabs.children.append(
                html.Div([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=px.line(
                                            x=cc_table['Number of components'],
                                            y=cc_table['Core consistency'],
                                            markers=True,
                                            labels={'x': 'Number of components', 'y': 'Core consistency'},
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
                                    dbc.Table.from_dataframe(cc_table,
                                                             bordered=True, hover=True,
                                                             )
                                ]
                            ),
                        ]
                    ),
                ]),
            )

        model_options = [{'label': '{r}-component'.format(r=r), 'value': r} for r in parafac_models.keys()]
        ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
            (eem_dataset_establishment.ref is not None) else None

        return (None, loadings_tabs, components_tabs, fmax_tabs, reconstruction_error_tabs, variance_explained_tabs,
                core_consistency_tabs, leverage_tabs, split_half_tabs, 'Build model', model_options, None, model_options,
                None, ref_options, None, parafac_models)


    # -----------Update reference selection dropdown

    @app.callback(
        [
            Output(IDS.PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION, 'options'),
            Output(IDS.PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION, 'value'),
        ],
        [
            Input(IDS.PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION, 'value'),
            State(IDS.PARAFAC_MODELS, 'data')
        ]
    )
    def update_parafac_reference_dropdown_by_selected_model(r, parafac_model):
        if all([r, parafac_model]):
            options = list(parafac_model[str(r)]['fitting_params'].keys())
            return options, None
        else:
            return [], None


    # -----------Analyze correlations between score/Fmax and reference variables in model establishment

    @app.callback(
        [
            Output(IDS.PARAFAC_ESTABLISHMENT_CORR_GRAPH, 'figure'),  # size, intervals?
            Output(IDS.PARAFAC_ESTABLISHMENT_CORR_TABLE, 'children'),
        ],
        [
            Input(IDS.PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION, 'value'),
            Input(IDS.PARAFAC_ESTABLISHMENT_CORR_INDICATOR_SELECTION, 'value'),
            Input(IDS.PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION, 'value'),
            State(IDS.PARAFAC_MODELS, 'data')
        ]
    )
    def on_parafac_establishment_correlations(r, indicator, ref_var, parafac_models):
        if all([r, indicator, ref_var, parafac_models]):
            ## future fix: make it work for model without external ref
            ref_df = pd.DataFrame(parafac_models[str(r)]['ref'][1:], columns=parafac_models[str(r)]['ref'][0],
                                  index=parafac_models[str(r)]['index'])
            fmax_df = pd.DataFrame(parafac_models[str(r)]['Fmax'][1:], columns=parafac_models[str(r)]['Fmax'][0],
                                   index=parafac_models[str(r)]['index'])
            ref_df = pd.concat([ref_df, fmax_df], axis=1)
            var = ref_df[ref_var]
            parafac_var = pd.DataFrame(parafac_models[str(r)][indicator][1:], columns=parafac_models[str(r)][indicator][0],
                                       index=parafac_models[str(r)]['index'])
            fig = go.Figure()

            stats = parafac_models[str(r)]['fitting_params']

            for i, col in enumerate(parafac_var.columns):
                x = var
                y = parafac_var[col]
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                y = y.drop(nan_rows)
                if x.shape[0] < 1:
                    return go.Figure(), None
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                         marker=dict(color=COLORS[i % 10]), hoverinfo='text+x+y'))
                fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                         y=stats[ref_var][i][1] * np.array([x.min(), x.max()]) + stats[ref_var][i][2],
                                         mode='lines', name=f'{col}-Linear Regression Line',
                                         line=dict(dash='dash', color=COLORS[i % 10])))
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


    # -----------Fit a test EEM dataset using the established PARAFAC model components
    @app.callback(
        [
            Output(IDS.PARAFAC_EEM_DATASET_PREDICT_MESSAGE, 'children'),  # size, intervals?
            # Output(IDS.PARAFAC_TEST_SCORE, 'children'),
            Output(IDS.PARAFAC_TEST_FMAX, 'children'),
            Output(IDS.PARAFAC_TEST_ERROR, 'children'),
            Output(IDS.PARAFAC_TEST_CORR_REF_SELECTION, 'options'),
            Output(IDS.PARAFAC_TEST_CORR_REF_SELECTION, 'value'),
            Output(IDS.PARAFAC_TEST_CORR_INDICATOR_SELECTION, 'options'),
            Output(IDS.PARAFAC_TEST_CORR_INDICATOR_SELECTION, 'value'),
            Output(IDS.PARAFAC_PREDICT_SPINNER, 'children'),
            Output(IDS.PARAFAC_TEST_RESULTS, 'data'),
        ],
        [
            Input(IDS.PREDICT_PARAFAC_MODEL, 'n_clicks'),
            State(IDS.PARAFAC_EEM_DATASET_PREDICT_PATH_INPUT, 'value'),
            State(IDS.PARAFAC_TEST_INDEX_KW_MANDATORY, 'value'),
            State(IDS.PARAFAC_TEST_INDEX_KW_OPTIONAL, 'value'),
            State(IDS.PARAFAC_TEST_MODEL_SELECTION, 'value'),
            State(IDS.PARAFAC_MODELS, 'data')
        ]
    )
    def on_parafac_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, model_r, parafac_models):
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
            eem_dataset_predict = eem_dataset_from_serializable(eem_dataset_dict)
        kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
        kw_optional = str_string_to_list(kw_optional) if kw_optional else None
        eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, inplace=True)

        if eem_dataset_predict.ref is not None:
            valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
        else:
            valid_ref = None

        score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset_predict.eem_stack,
                                                                        np.array(
                                                                            [
                                                                                [
                                                                                    [np.nan if x is None else x for x in
                                                                                     subsublist] for subsublist in sublist
                                                                                ]
                                                                                for sublist in
                                                                                parafac_models[str(model_r)][
                                                                                    'components']
                                                                            ]
                                                                        ),
                                                                        fit_intercept=False)
        score_sample = pd.DataFrame(
            score_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i + 1) for i in range(model_r)]
        )
        fmax_sample = pd.DataFrame(
            fmax_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i + 1) for i in range(model_r)]
        )

        pred = {}
        if eem_dataset_predict.ref is not None:
            for ref_var in valid_ref:
                if ref_var in parafac_models[str(model_r)]['fitting_params'].keys():
                    params = parafac_models[str(model_r)]['fitting_params'][ref_var]
                    pred_var = fmax_sample.copy()
                    for i, f_col in enumerate(fmax_sample):
                        pred_r = fmax_sample[f_col] - params[i][2]
                        pred_r = pred_r / params[i][1]
                        pred_var[f_col] = pred_r
                    pred[ref_var] = [pred_var.columns.tolist()] + pred_var.values.tolist()
                else:
                    pred[ref_var] = None

        score_tab = html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(figure=plot_fmax(score_sample,
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
                            dbc.Table.from_dataframe(score_sample,
                                                     bordered=True, hover=True, index=True
                                                     )
                        ]
                    ),
                ]
            ),
        ])

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
            indicator_options = [
                # {'label': 'Score', 'value': 'Score'},
                {'label': 'Fmax', 'value': 'Fmax'},
                {'label': 'Prediction of reference', 'value': 'Prediction of reference'}
            ]
        else:
            indicator_options = [
                # {'label': 'Score', 'value': 'Score'},
                {'label': 'Fmax', 'value': 'Fmax'}
            ]

        ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

        test_results = {
            'Fmax': [fmax_sample.columns.tolist()] + fmax_sample.values.tolist(),
            'Score': [score_sample.columns.tolist()] + score_sample.values.tolist(),
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
            Output(IDS.PARAFAC_TEST_PRED_GRAPH, 'figure'),  # size, intervals?
            Output(IDS.PARAFAC_TEST_PRED_TABLE, 'children'),
        ],
        [
            Input(IDS.PREDICT_PARAFAC_MODEL, 'n_clicks'),
            Input(IDS.PARAFAC_TEST_PRED_REF_SELECTION, 'value'),
            Input(IDS.PARAFAC_TEST_PRED_MODEL_SELECTION, 'value'),
            State(IDS.PARAFAC_TEST_RESULTS, 'data'),
        ]
    )
    def on_parafac_test_predict_reference(n_clicks, ref_var, pred_model, parafac_test_results):
        if all([ref_var, pred_model, parafac_test_results]):
            pred = parafac_test_results['Prediction of reference'][ref_var]
            pred = pd.DataFrame(pred[1:], columns=pred[0], index=parafac_test_results['index'])
            fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
            if parafac_test_results['ref'] is not None:
                ref = pd.DataFrame(parafac_test_results['ref'][1:], columns=parafac_test_results['ref'][0],
                                   index=parafac_test_results['index'])
                true_value = ref[ref_var]
                nan_rows = true_value[true_value.isna()].index
                true_value = true_value.drop(nan_rows)
                pred_overlap = pred.drop(nan_rows)
                fig.add_trace(go.Scatter(x=true_value.index, y=true_value.values, mode='markers',
                                         marker=dict(color='black', size=5),
                                         name='True value'))
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


    # -----------Analyze correlations between Fmax and reference variables in model testing
    @app.callback(
        [
            Output(IDS.PARAFAC_TEST_CORR_GRAPH, 'figure'),  # size, intervals?
            Output(IDS.PARAFAC_TEST_CORR_TABLE, 'children'),
        ],
        [
            Input(IDS.PARAFAC_TEST_CORR_INDICATOR_SELECTION, 'value'),
            Input(IDS.PARAFAC_TEST_CORR_REF_SELECTION, 'value'),
            State(IDS.PARAFAC_TEST_RESULTS, 'data'),
        ]
    )
    def on_parafac_test_correlations(indicator, ref_var, parafac_test_results):
        if all([indicator, ref_var, parafac_test_results]):
            ref_df = pd.DataFrame(parafac_test_results['ref'][1:], columns=parafac_test_results['ref'][0],
                                  index=parafac_test_results['index'])
            var = ref_df[ref_var]
            if indicator != 'Prediction of reference':
                parafac_var = pd.DataFrame(parafac_test_results[indicator][1:], columns=parafac_test_results[indicator][0],
                                           index=parafac_test_results['index'])
            else:
                if parafac_test_results[indicator][ref_var] is not None:
                    parafac_var = pd.DataFrame(parafac_test_results[indicator][ref_var][1:],
                                               columns=parafac_test_results[indicator][ref_var][0],
                                               index=parafac_test_results['index'])
                else:
                    return go.Figure(), None
            fig = go.Figure()
            stats = []
            for i, col in enumerate(parafac_var.columns):
                x = var
                y = parafac_var[col]
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                y = y.drop(nan_rows)
                if x.shape[0] < 1:
                    return go.Figure(), None
                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                predictions = lm.predict(x_reshaped)
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                         hoverinfo='text+x+y', marker=dict(color=COLORS[i % 10])))
                fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name=f'{col} fit',
                                         line=dict(dash='dash', color=COLORS[i % 10])))
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

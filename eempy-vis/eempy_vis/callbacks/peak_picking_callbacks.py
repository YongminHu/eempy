from .shared import *
from ..config import COLORS
from ..serialization import eem_dataset_from_serializable


def register_callbacks(app):
    @app.callback(
        [
            Output('pp-eem-dataset-establishment-message', 'children'),
            Output('pp-intensities', 'children'),
            Output('build-pp-spinner', 'children'),
            Output('pp-establishment-corr-ref-selection', 'options'),
            Output('pp-establishment-corr-ref-selection', 'value'),
            Output('pp-test-pred-ref-selection', 'options'),
            Output('pp-test-pred-ref-selection', 'value'),
            Output('pp-model', 'data'),
        ],
        [
            Input('build-pp-model', 'n_clicks'),
            State('eem-graph-options', 'value'),
            State('pp-eem-dataset-establishment-path-input', 'value'),
            State('pp-establishment-index-kw-mandatory', 'value'),
            State('pp-establishment-index-kw-optional', 'value'),
            State('pp-excitation', 'value'),
            State('pp-emission', 'value'),
            State('eem-dataset', 'data')
        ]
    )
    def on_build_pp_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, ex_target,
                          em_target,
                          eem_dataset_dict):
        if n_clicks is None:
            return None, None, 'Build model', [], None, [], None, None
        if not path_establishment:
            if eem_dataset_dict is None:
                message = (
                    'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                    'section, or import an EEM dataset from file.')
                return None, None, 'Build model', [], None, [], None, None
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
        else:
            if not os.path.exists(path_establishment):
                message = ('Error: No such file or directory: ' + path_establishment)
                return message, None, 'Build model', [], None, [], None, None
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

        pp_model = {}
        intensities_tabs = dbc.Card([])

        if eem_dataset_establishment.ref is not None:
            valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
        else:
            valid_ref = None

        fi, ex_actual, em_actual = eem_dataset_establishment.peak_picking(ex=ex_target, em=em_target)
        fi_name = f'Intensity (ex={ex_actual} nm, em={em_actual} nm)'

        pp_fit_params = {}
        if eem_dataset_establishment.ref is not None:
            for ref_var in valid_ref:
                x = eem_dataset_establishment.ref[ref_var]
                stats = []
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                if x.shape[0] < 1:
                    continue
                y = fi.squeeze()
                y = y.drop(nan_rows)
                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                r_squared = lm.score(x_reshaped, y)
                intercept = lm.intercept_
                slope = lm.coef_[0]
                pearson_corr, pearson_p = pearsonr(x, y)
                stats.append([fi_name, slope, intercept, r_squared, pearson_corr, pearson_p])
                pp_fit_params[ref_var] = stats

            pp_model = {
                'intensities': [fi.columns.tolist()] + fi.values.tolist(),
                'index': eem_dataset_establishment.index,
                'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
                if eem_dataset_establishment.ref is not None else None,
                'ex_actual': ex_actual,
                'em_actual': em_actual,
                'fitting_params': pp_fit_params
            }

        # fmax
        intensities_tabs.children.append(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(figure=plot_fmax(fi,
                                                           display=False,
                                                           yaxis_title=fi_name
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
                                dbc.Table.from_dataframe(fi,
                                                         bordered=True, hover=True, index=True)
                            ]
                        )
                    ]
                )
            ]),
        )

        ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
            (eem_dataset_establishment.ref is not None) else None

        return None, intensities_tabs, 'Build model', ref_options, None, ref_options, None, pp_model

    # -----------Analyze correlations between score/Fmax and reference variables in model establishment

    @app.callback(
        [
            Output('pp-establishment-corr-graph', 'figure'),  # size, intervals?
            Output('pp-establishment-corr-table', 'children'),
        ],
        [
            Input('pp-establishment-corr-ref-selection', 'value'),
            State('pp-model', 'data')
        ]
    )
    def on_pp_establishment_correlations(ref_var, pp_model):
        if all([ref_var, pp_model]):
            ref_df = pd.DataFrame(pp_model['ref'][1:], columns=pp_model['ref'][0],
                                  index=pp_model['index'])
            intensities_df = pd.DataFrame(pp_model['intensities'][1:], columns=pp_model['intensities'][0],
                                          index=pp_model['index'])
            ref_df = pd.concat([ref_df, intensities_df], axis=1)
            var = ref_df[ref_var]
            fig = go.Figure()

            stats = pp_model['fitting_params']

            x = var
            y = intensities_df.iloc[:, 0]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=stats[ref_var][0][0], text=[i for i in x.index],
                                     marker=dict(color=COLORS[0]), hoverinfo='text+x+y'))
            fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                     y=stats[ref_var][0][1] * np.array([x.min(), x.max()]) + stats[ref_var][0][2],
                                     mode='lines', name=f'{stats[ref_var][0][0]}-Linear Regression Line',
                                     line=dict(dash='dash', color=COLORS[0])))
            fig.update_xaxes(title_text=ref_var)
            fig.update_yaxes(title_text='Intensity')

            tbl = pd.DataFrame(
                stats[ref_var],
                columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
            )
            tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
            return fig, tbl
        else:
            return go.Figure(), None

    # -----------Fit a test EEM dataset using the established Linear model
    @app.callback(
        [
            Output('pp-eem-dataset-predict-message', 'children'),  # size, intervals?
            Output('pp-test-intensities', 'children'),
            Output('pp-test-error', 'children'),
            Output('pp-test-corr-indicator-selection', 'options'),
            Output('pp-test-corr-indicator-selection', 'value'),
            Output('pp-test-corr-ref-selection', 'options'),
            Output('pp-test-corr-ref-selection', 'value'),
            Output('pp-predict-spinner', 'children'),
            Output('pp-test-results', 'data'),
        ],
        [
            Input('predict-pp-model', 'n_clicks'),
            State('pp-eem-dataset-predict-path-input', 'value'),
            State('pp-test-index-kw-mandatory', 'value'),
            State('pp-test-index-kw-optional', 'value'),
            State('pp-model', 'data')
        ]
    )
    def on_pp_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, pp_model):
        if n_clicks is None:
            return None, None, None, [], None, [], None, 'predict', None
        if path_predict is None:
            return None, None, None, [], None, [], None, 'predict', None
        if not os.path.exists(path_predict):
            message = ('Error: No such file: ' + path_predict)
            return message, None, None, [], None, [], None, 'predict', None
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
        eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                            inplace=True)

        if eem_dataset_predict.ref is not None:
            valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
        else:
            valid_ref = None

        fi_test, ex_actual_test, em_actual_test = eem_dataset_predict.peak_picking(ex=pp_model['ex_actual'],
                                                                                   em=pp_model['em_actual'])

        pred = {}
        if eem_dataset_predict.ref is not None:
            for ref_var in valid_ref:
                if ref_var in pp_model['fitting_params'].keys():
                    params = pp_model['fitting_params'][ref_var]
                    pred_sample = fi_test.copy()
                    pred_r = fi_test.iloc[:, 0] - params[0][2]
                    pred_r = pred_r / params[0][1]
                    pred_sample.iloc[:, 0] = pred_r
                    pred[ref_var] = [pred_sample.columns.tolist()] + pred_sample.values.tolist()
                else:
                    pred[ref_var] = None

        intensities_tab = html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(figure=plot_fmax(fi_test,
                                                       display=False,
                                                       yaxis_title='Intensities of test dataset'
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
                            dbc.Table.from_dataframe(fi_test,
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
                {'label': 'Intensities of test dataset', 'value': 'Intensities of test dataset'},
                {'label': 'Prediction of reference', 'value': 'Prediction of reference'}
            ]
        else:
            indicator_options = [
                {'label': 'Intensities of test dataset', 'value': 'Intensities of test dataset'}
            ]

        ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

        test_results = {
            'Intensities of test dataset': [fi_test.columns.tolist()] + fi_test.values.tolist(),
            'Prediction of reference': pred,
            'ref': [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
            if eem_dataset_predict.ref is not None else None,
            'index': eem_dataset_predict.index
        }

        return None, intensities_tab, error_tab, indicator_options, None, ref_options, None, 'predict', test_results

    # -----------Predict the corresponding reference variables for the test EEM datasets using the model fitted in the model
    #            establishment step
    @app.callback(
        [
            Output('pp-test-pred-graph', 'figure'),  # size, intervals?
            Output('pp-test-pred-table', 'children'),
        ],
        [
            Input('predict-pp-model', 'n_clicks'),
            Input('pp-test-pred-ref-selection', 'value'),
            State('pp-test-results', 'data'),
        ]
    )
    def on_pp_test_predict_reference(n_clicks, ref_var, pp_test_results):
        if all([ref_var, pp_test_results]):
            pred = pp_test_results['Prediction of reference'][ref_var]
            pred = pd.DataFrame(pred[1:], columns=pred[0], index=pp_test_results['index'])
            fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
            if pp_test_results['ref'] is not None:
                ref = pd.DataFrame(pp_test_results['ref'][1:], columns=pp_test_results['ref'][0],
                                   index=pp_test_results['index'])
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
            Output('pp-test-corr-graph', 'figure'),  # size, intervals?
            Output('pp-test-corr-table', 'children'),
        ],
        [
            Input('pp-test-corr-indicator-selection', 'value'),
            Input('pp-test-corr-ref-selection', 'value'),
            State('pp-test-results', 'data'),
        ]
    )
    def on_pp_test_correlations(indicator, ref_var, pp_test_results):
        if all([indicator, ref_var, pp_test_results]):
            ref_df = pd.DataFrame(pp_test_results['ref'][1:], columns=pp_test_results['ref'][0],
                                  index=pp_test_results['index'])
            var = ref_df[ref_var]
            if indicator != 'Prediction of reference':
                pp_var = pd.DataFrame(pp_test_results[indicator][1:], columns=pp_test_results[indicator][0],
                                      index=pp_test_results['index'])
            else:
                if pp_test_results[indicator][ref_var] is not None:
                    pp_var = pd.DataFrame(pp_test_results[indicator][ref_var][1:],
                                          columns=pp_test_results[indicator][ref_var][0],
                                          index=pp_test_results['index'])
                else:
                    return go.Figure(), None
            fig = go.Figure()
            stats = []
            for i, col in enumerate(pp_var.columns):
                x = var
                y = pp_var[col]
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
            fig.update_yaxes(
                title_text="Prediction of " + ref_var if indicator == 'Prediction of reference' else indicator)

            tbl = pd.DataFrame(
                stats,
                columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
            )
            tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
            return fig, tbl
        else:
            return go.Figure(), None

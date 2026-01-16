from .shared import *

from dash import Input, Output, State
from ..ids import IDS
from ..config import COLORS


def register_callbacks(app):

    @app.callback(
        [
            Output(IDS.RI_EEM_DATASET_ESTABLISHMENT_MESSAGE, "children"),
            Output(IDS.RI_INTENSITIES, "children"),
            Output(IDS.BUILD_RI_SPINNER, "children"),
            Output(IDS.RI_ESTABLISHMENT_CORR_REF_SELECTION, "options"),
            Output(IDS.RI_ESTABLISHMENT_CORR_REF_SELECTION, "value"),
            Output(IDS.RI_TEST_PRED_REF_SELECTION, "options"),
            Output(IDS.RI_TEST_PRED_REF_SELECTION, "value"),
            Output(IDS.RI_MODEL, "data"),
        ],
        [
            Input(IDS.BUILD_RI_MODEL, "n_clicks"),
            State(IDS.EEM_GRAPH_OPTIONS, "value"),
            State(IDS.RI_EEM_DATASET_ESTABLISHMENT_PATH_INPUT, "value"),
            State(IDS.RI_ESTABLISHMENT_INDEX_KW_MANDATORY, "value"),
            State(IDS.RI_ESTABLISHMENT_INDEX_KW_OPTIONAL, "value"),
            State(IDS.RI_EX_MIN, "value"),
            State(IDS.RI_EX_MAX, "value"),
            State(IDS.RI_EM_MIN, "value"),
            State(IDS.RI_EM_MAX, "value"),
            State(IDS.EEM_DATASET, "data"),
        ],
    )
    def on_build_ri_model(
        n_clicks,
        eem_graph_options,
        path_establishment,
        kw_mandatory,
        kw_optional,
        ex_min,
        ex_max,
        em_min,
        em_max,
        eem_dataset_dict,
    ):
        if n_clicks is None:
            return None, None, "Build model", [], None, [], None, None
        if not path_establishment:
            if eem_dataset_dict is None:
                message = (
                    'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                    "section, or import an EEM dataset from file."
                )
                return message, None, "Build model", [], None, [], None, None
            eem_dataset_establishment = EEMDataset(
                eem_stack=np.array(
                    [
                        [[np.nan if x is None else x for x in subsublist] for subsublist in sublist]
                        for sublist in eem_dataset_dict["eem_stack"]
                    ]
                ),
                ex_range=np.array(eem_dataset_dict["ex_range"]),
                em_range=np.array(eem_dataset_dict["em_range"]),
                index=eem_dataset_dict["index"],
                ref=pd.DataFrame(
                    eem_dataset_dict["ref"][1:],
                    columns=eem_dataset_dict["ref"][0],
                    index=eem_dataset_dict["index"],
                )
                if eem_dataset_dict["ref"] is not None
                else None,
            )
        else:
            if not os.path.exists(path_establishment):
                message = "Error: No such file or directory: " + path_establishment
                return message, None, "Build model", [], None, [], None, None
            else:
                _, file_extension = os.path.splitext(path_establishment)

                if file_extension == ".json":
                    with open(path_establishment, "r") as file:
                        eem_dataset_dict = json.load(file)
                elif file_extension == ".pkl":
                    with open(path_establishment, "rb") as file:
                        eem_dataset_dict = pickle.load(file)
                else:
                    raise ValueError("Unsupported file extension: {}".format(file_extension))

                eem_dataset_establishment = EEMDataset(
                    eem_stack=np.array(
                        [
                            [[np.nan if x is None else x for x in subsublist] for subsublist in sublist]
                            for sublist in eem_dataset_dict["eem_stack"]
                        ]
                    ),
                    ex_range=np.array(eem_dataset_dict["ex_range"]),
                    em_range=np.array(eem_dataset_dict["em_range"]),
                    index=eem_dataset_dict["index"],
                    ref=pd.DataFrame(
                        eem_dataset_dict["ref"][1:],
                        columns=eem_dataset_dict["ref"][0],
                        index=eem_dataset_dict["index"],
                    )
                    if eem_dataset_dict["ref"] is not None
                    else None,
                )

        kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
        kw_optional = str_string_to_list(kw_optional) if kw_optional else None
        eem_dataset_establishment.filter_by_index(
            mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, inplace=True
        )

        intensities_tabs = dbc.Card([])

        if eem_dataset_establishment.ref is not None:
            valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
        else:
            valid_ref = None

        ri = eem_dataset_establishment.regional_integration(ex_min=ex_min, ex_max=ex_max, em_min=em_min, em_max=em_max)
        ri_name = f"RI (ex=[{ex_min}, {ex_max}] nm, em=[{em_min}, {em_max}] nm)"

        ri_fit_params = {}
        if eem_dataset_establishment.ref is not None:
            for ref_var in valid_ref:
                x = eem_dataset_establishment.ref[ref_var]
                stats = []
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                if x.shape[0] < 1:
                    continue
                y = ri.squeeze()
                y = y.drop(nan_rows)
                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                r_squared = lm.score(x_reshaped, y)
                intercept = lm.intercept_
                slope = lm.coef_[0]
                pearson_corr, pearson_p = pearsonr(x, y)
                stats.append([ri_name, slope, intercept, r_squared, pearson_corr, pearson_p])
                ri_fit_params[ref_var] = stats

            ri_model = {
                "intensities": [ri.columns.tolist()] + ri.values.tolist(),
                "index": eem_dataset_establishment.index,
                "ref": [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
                if eem_dataset_establishment.ref is not None
                else None,
                "ex_min": ex_min,
                "ex_max": ex_max,
                "em_min": em_min,
                "em_max": em_max,
                "fitting_params": ri_fit_params,
            }
        else:
            ri_model = {}

        intensities_tabs.children.append(
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=plot_fmax(ri, display=False, yaxis_title=ri_name),
                                        config={"autosizable": False},
                                        style={"width": 1700, "height": 800},
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.Row([dbc.Col([dbc.Table.from_dataframe(ri, bordered=True, hover=True, index=True)])]),
                ]
            ),
        )

        ref_options = [{"label": var, "value": var} for var in valid_ref] if (eem_dataset_establishment.ref is not None) else None
        return None, intensities_tabs, "Build model", ref_options, None, ref_options, None, ri_model

    # -----------Analyze correlations between score/Fmax and reference variables in model establishment

    @app.callback(
        [
            Output(IDS.RI_ESTABLISHMENT_CORR_GRAPH, "figure"),
            Output(IDS.RI_ESTABLISHMENT_CORR_TABLE, "children"),
        ],
        [
            Input(IDS.RI_ESTABLISHMENT_CORR_REF_SELECTION, "value"),
            State(IDS.RI_MODEL, "data"),
        ],
    )
    def on_ri_establishment_correlations(ref_var, ri_model):
        if all([ref_var, ri_model]):
            ref_df = pd.DataFrame(ri_model["ref"][1:], columns=ri_model["ref"][0], index=ri_model["index"])
            intensities_df = pd.DataFrame(
                ri_model["intensities"][1:], columns=ri_model["intensities"][0], index=ri_model["index"]
            )
            ref_df = pd.concat([ref_df, intensities_df], axis=1)
            var = ref_df[ref_var]
            fig = go.Figure()

            stats = ri_model["fitting_params"]

            x = var
            y = intensities_df.iloc[:, 0]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=stats[ref_var][0][0],
                    text=[i for i in x.index],
                    marker=dict(color=COLORS[0]),
                    hoverinfo="text+x+y",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.array([x.min(), x.max()]),
                    y=stats[ref_var][0][1] * np.array([x.min(), x.max()]) + stats[ref_var][0][2],
                    mode="lines",
                    name=f"{stats[ref_var][0][0]}-Linear Regression Line",
                    line=dict(dash="dash", color=COLORS[0]),
                )
            )
            fig.update_xaxes(title_text=ref_var)
            fig.update_yaxes(title_text="Intensity")

            tbl = pd.DataFrame(
                stats[ref_var],
                columns=["Variable", "slope", "intercept", "R²", "Pearson Correlation", "Pearson p-value"],
            )
            tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
            return fig, tbl
        else:
            return go.Figure(), None

    # -----------Fit a test EEM dataset using the established RFI model
    @app.callback(
        [
            Output(IDS.RI_EEM_DATASET_PREDICT_MESSAGE, "children"),
            Output(IDS.RI_TEST_INTENSITIES, "children"),
            Output(IDS.RI_TEST_ERROR, "children"),
            Output(IDS.RI_TEST_CORR_INDICATOR_SELECTION, "options"),
            Output(IDS.RI_TEST_CORR_INDICATOR_SELECTION, "value"),
            Output(IDS.RI_TEST_CORR_REF_SELECTION, "options"),
            Output(IDS.RI_TEST_CORR_REF_SELECTION, "value"),
            Output(IDS.RI_PREDICT_SPINNER, "children"),
            Output(IDS.RI_TEST_RESULTS, "data"),
        ],
        [
            Input(IDS.PREDICT_RI_MODEL, "n_clicks"),
            State(IDS.RI_EEM_DATASET_PREDICT_PATH_INPUT, "value"),
            State(IDS.RI_TEST_INDEX_KW_MANDATORY, "value"),
            State(IDS.RI_TEST_INDEX_KW_OPTIONAL, "value"),
            State(IDS.RI_MODEL, "data"),
        ],
    )
    def on_ri_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, ri_model):
        if n_clicks is None:
            return None, None, None, [], None, [], None, "predict", None
        if path_predict is None:
            return None, None, None, [], None, [], None, "predict", None
        if not os.path.exists(path_predict):
            message = "Error: No such file: " + path_predict
            return message, None, None, [], None, [], None, "predict", None

        _, file_extension = os.path.splitext(path_predict)
        if file_extension == ".json":
            with open(path_predict, "r") as file:
                eem_dataset_dict = json.load(file)
        elif file_extension == ".pkl":
            with open(path_predict, "rb") as file:
                eem_dataset_dict = pickle.load(file)
        else:
            raise ValueError("Unsupported file extension: {}".format(file_extension))

        eem_dataset_predict = EEMDataset(
            eem_stack=np.array(
                [
                    [[np.nan if x is None else x for x in subsublist] for subsublist in sublist]
                    for sublist in eem_dataset_dict["eem_stack"]
                ]
            ),
            ex_range=np.array(eem_dataset_dict["ex_range"]),
            em_range=np.array(eem_dataset_dict["em_range"]),
            index=eem_dataset_dict["index"],
            ref=pd.DataFrame(
                eem_dataset_dict["ref"][1:],
                index=eem_dataset_dict["index"],
                columns=eem_dataset_dict["ref"][0],
            )
            if eem_dataset_dict["ref"] is not None
            else None,
        )

        kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
        kw_optional = str_string_to_list(kw_optional) if kw_optional else None
        eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, inplace=True)

        if eem_dataset_predict.ref is not None:
            valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
        else:
            valid_ref = None

        ri_test = eem_dataset_predict.regional_integration(
            ex_min=ri_model["ex_min"],
            ex_max=ri_model["ex_max"],
            em_min=ri_model["em_min"],
            em_max=ri_model["em_max"],
        )

        pred = {}
        if eem_dataset_predict.ref is not None:
            for ref_var in valid_ref:
                if ref_var in ri_model["fitting_params"].keys():
                    params = ri_model["fitting_params"][ref_var]
                    pred_sample = ri_test.copy()
                    pred_r = ri_test.iloc[:, 0] - params[0][2]
                    pred_r = pred_r / params[0][1]
                    pred_sample.iloc[:, 0] = pred_r
                    pred[ref_var] = [pred_sample.columns.tolist()] + pred_sample.values.tolist()
                else:
                    pred[ref_var] = None

        intensities_tab = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    figure=plot_fmax(ri_test, display=False, yaxis_title="Intensities of test dataset"),
                                    config={"autosizable": False},
                                    style={"width": 1700, "height": 800},
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row([dbc.Col([dbc.Table.from_dataframe(ri_test, bordered=True, hover=True, index=True)])]),
            ]
        )

        error_tab = html.Div(children=[])

        if eem_dataset_predict.ref is not None:
            indicator_options = [
                {"label": "RI of test dataset", "value": "RI of test dataset"},
                {"label": "Prediction of reference", "value": "Prediction of reference"},
            ]
        else:
            indicator_options = [{"label": "RI of test dataset", "value": "RI of test dataset"}]

        ref_options = [{"label": var, "value": var} for var in eem_dataset_predict.ref.columns]

        test_results = {
            "RI of test dataset": [ri_test.columns.tolist()] + ri_test.values.tolist(),
            "Prediction of reference": pred,
            "ref": [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
            if eem_dataset_predict.ref is not None
            else None,
            "index": eem_dataset_predict.index,
        }

        return None, intensities_tab, error_tab, indicator_options, None, ref_options, None, "predict", test_results

    # -----------Predict the corresponding reference variables for the test EEM datasets

    @app.callback(
        [
            Output(IDS.RI_TEST_PRED_GRAPH, "figure"),
            Output(IDS.RI_TEST_PRED_TABLE, "children"),
        ],
        [
            Input(IDS.PREDICT_RI_MODEL, "n_clicks"),
            Input(IDS.RI_TEST_PRED_REF_SELECTION, "value"),
            State(IDS.RI_TEST_RESULTS, "data"),
        ],
    )
    def on_ri_test_predict_reference(n_clicks, ref_var, ri_test_results):
        if all([ref_var, ri_test_results]):
            pred = ri_test_results["Prediction of reference"][ref_var]
            pred = pd.DataFrame(pred[1:], columns=pred[0], index=ri_test_results["index"])
            fig = plot_fmax(pred, display=False, yaxis_title=ref_var)

            if ri_test_results["ref"] is not None:
                ref = pd.DataFrame(ri_test_results["ref"][1:], columns=ri_test_results["ref"][0], index=ri_test_results["index"])
                true_value = ref[ref_var]
                nan_rows = true_value[true_value.isna()].index
                true_value = true_value.drop(nan_rows)
                pred_overlap = pred.drop(nan_rows)

                fig.add_trace(
                    go.Scatter(
                        x=true_value.index,
                        y=true_value.values,
                        mode="markers",
                        marker=dict(color="black", size=5),
                        name="True value",
                    )
                )
                rmse_results = []
                for f_col in pred.columns:
                    rmse = sqrt(sum((true_value.values - pred_overlap[f_col]) ** 2) / len(true_value.values))
                    rmse_results.append(rmse)

                tbl = html.Div(
                    [
                        dbc.Row(
                            dbc.Table.from_dataframe(
                                pd.DataFrame([rmse_results], columns=pred.columns, index=pd.Index(["RMSE"], name="Error metric")),
                                bordered=True,
                                hover=True,
                                index=True,
                            )
                        ),
                        dbc.Row(
                            dbc.Table.from_dataframe(pd.concat([pred, ref[ref_var]], axis=1), bordered=True, hover=True, index=True)
                        ),
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
            Output(IDS.RI_TEST_CORR_GRAPH, "figure"),
            Output(IDS.RI_TEST_CORR_TABLE, "children"),
        ],
        [
            Input(IDS.RI_TEST_CORR_INDICATOR_SELECTION, "value"),
            Input(IDS.RI_TEST_CORR_REF_SELECTION, "value"),
            State(IDS.RI_TEST_RESULTS, "data"),
        ],
    )
    def on_ri_test_correlations(indicator, ref_var, ri_test_results):
        if all([indicator, ref_var, ri_test_results]):
            ref_df = pd.DataFrame(ri_test_results["ref"][1:], columns=ri_test_results["ref"][0], index=ri_test_results["index"])
            var = ref_df[ref_var]

            if indicator != "Prediction of reference":
                ri_var = pd.DataFrame(ri_test_results[indicator][1:], columns=ri_test_results[indicator][0], index=ri_test_results["index"])
            else:
                if ri_test_results[indicator][ref_var] is not None:
                    ri_var = pd.DataFrame(
                        ri_test_results[indicator][ref_var][1:],
                        columns=ri_test_results[indicator][ref_var][0],
                        index=ri_test_results["index"],
                    )
                else:
                    return go.Figure(), None

            fig = go.Figure()
            stats = []
            for i, col in enumerate(ri_var.columns):
                x = var
                y = ri_var[col]
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                y = y.drop(nan_rows)
                if x.shape[0] < 1:
                    return go.Figure(), None

                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                predictions = lm.predict(x_reshaped)

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name=col,
                        text=[i for i in x.index],
                        hoverinfo="text+x+y",
                        marker=dict(color=COLORS[i % 10]),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=predictions,
                        mode="lines",
                        name=f"{col} fit",
                        line=dict(dash="dash", color=COLORS[i % 10]),
                    )
                )
                r_squared = lm.score(x_reshaped, y)
                intercept = lm.intercept_
                slope = lm.coef_[0]
                pearson_corr, pearson_p = pearsonr(x, y)
                stats.append([col, slope, intercept, r_squared, pearson_corr, pearson_p])

            fig.update_xaxes(title_text=ref_var)
            fig.update_yaxes(title_text=("Prediction of " + ref_var) if indicator == "Prediction of reference" else indicator)

            tbl = pd.DataFrame(
                stats,
                columns=["Variable", "slope", "intercept", "R²", "Pearson Correlation", "Pearson p-value"],
            )
            tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
            return fig, tbl
        else:
            return go.Figure(), None

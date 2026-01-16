from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..ids import IDS


# -----------Page #3: Regional integration--------------
# First setup the dbc cards, then the general layout in the end.

#   -------------Setting up the dbc cards

#       -----------------dbc card for regional-integration parameters
card_ri_param = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Import EEM dataset for model establishment", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                dcc.Input(
                                    id=IDS.RI_EEM_DATASET_ESTABLISHMENT_PATH_INPUT,
                                    type="text",
                                    value=None,
                                    placeholder=(
                                        "Please enter the eem dataset path (.json and .pkl are supported). "
                                        'If empty, the model built in "eem pre-processing" would be used'
                                    ),
                                    style={"width": "97%", "height": "30px"},
                                    debounce=True,
                                ),
                                justify="center",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [],
                                            id=IDS.RI_EEM_DATASET_ESTABLISHMENT_MESSAGE,
                                            style={"width": "80vw"},
                                        ),
                                        width={"size": 12, "offset": 0},
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("Index mandatory keywords"), width={"size": 2}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.RI_ESTABLISHMENT_INDEX_KW_MANDATORY,
                                            type="text",
                                            placeholder="",
                                            style={"width": "100%", "height": "30px"},
                                            debounce=True,
                                            value="",
                                        ),
                                        width={"offset": 0, "size": 2},
                                    ),
                                    dbc.Col(dbc.Label("Index optional keywords"), width={"size": 2, "offset": 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.RI_ESTABLISHMENT_INDEX_KW_OPTIONAL,
                                            type="text",
                                            placeholder="",
                                            style={"width": "100%", "height": "30px"},
                                            debounce=True,
                                            value="",
                                        ),
                                        width={"offset": 0, "size": 2},
                                    ),
                                ]
                            ),
                        ],
                        gap=2,
                    )
                ]
            ),
            html.H5("Parameters selection", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("Ex range min"), width={"size": 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.RI_EX_MIN,
                                            type="number",
                                            placeholder="nm",
                                            style={"width": "100px", "height": "30px"},
                                            debounce=True,
                                        ),
                                        width={"size": 2},
                                    ),
                                    dbc.Col(dbc.Label("Ex range max"), width={"size": 1, "offset": 0}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.RI_EX_MAX,
                                            type="number",
                                            placeholder="nm",
                                            style={"width": "100px", "height": "30px"},
                                            debounce=True,
                                        ),
                                        width={"size": 2},
                                    ),
                                    dbc.Col(dbc.Label("Em range min"), width={"size": 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.RI_EM_MIN,
                                            type="number",
                                            placeholder="nm",
                                            style={"width": "100px", "height": "30px"},
                                            debounce=True,
                                        ),
                                        width={"size": 2},
                                    ),
                                    dbc.Col(dbc.Label("Em range max"), width={"size": 1, "offset": 0}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.RI_EM_MAX,
                                            type="number",
                                            placeholder="nm",
                                            style={"width": "100px", "height": "30px"},
                                            debounce=True,
                                        ),
                                        width={"size": 2},
                                    ),
                                ]
                            ),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [dbc.Spinner(size="sm", id=IDS.BUILD_RI_SPINNER)],
                                        id=IDS.BUILD_RI_MODEL,
                                        className="col-2",
                                    )
                                )
                            ),
                        ],
                        gap=2,
                    )
                ]
            ),
        ]
    ),
    className="w-100",
)


layout = html.Div(
    [
        dbc.Stack(
            [
                dbc.Row(card_ri_param),
                dbc.Row(
                    dcc.Tabs(
                        id=IDS.RI_RESULTS,
                        children=[
                            dcc.Tab(label="Intensities", id=IDS.RI_INTENSITIES),
                            dcc.Tab(
                                children=[
                                    html.Div(
                                        [
                                            dbc.Card(
                                                dbc.Stack(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Label("Select reference variable"),
                                                                    width={"size": 1, "offset": 1},
                                                                ),
                                                                dbc.Col(
                                                                    dcc.Dropdown(
                                                                        options=[],
                                                                        id=IDS.RI_ESTABLISHMENT_CORR_REF_SELECTION,
                                                                    ),
                                                                    width={"size": 2},
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dcc.Graph(
                                                                    id=IDS.RI_ESTABLISHMENT_CORR_GRAPH,
                                                                    style={"width": "45vw", "height": "60vh"},
                                                                )
                                                            ]
                                                        ),
                                                        dbc.Row(
                                                            html.Div(
                                                                [],
                                                                id=IDS.RI_ESTABLISHMENT_CORR_TABLE,
                                                            )
                                                        ),
                                                    ],
                                                    gap=2,
                                                    style={"margin": "20px"},
                                                )
                                            )
                                        ],
                                        style={"width": "90vw"},
                                    )
                                ],
                                label="Correlations",
                                id=IDS.RI_ESTABLISHMENT_CORR,
                            ),
                            dcc.Tab(
                                children=[
                                    html.Div(
                                        [
                                            dbc.Card(
                                                dbc.Stack(
                                                    [
                                                        html.H5("Import EEM dataset to be predicted"),
                                                        dbc.Row(
                                                            dcc.Input(
                                                                id=IDS.RI_EEM_DATASET_PREDICT_PATH_INPUT,
                                                                type="text",
                                                                placeholder=(
                                                                    "Please enter the eem dataset path (.json and .pkl are supported)."
                                                                ),
                                                                style={"width": "97%", "height": "30px"},
                                                                debounce=True,
                                                            ),
                                                            justify="center",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    html.Div(
                                                                        [],
                                                                        id=IDS.RI_EEM_DATASET_PREDICT_MESSAGE,
                                                                        style={"width": "1000px"},
                                                                    ),
                                                                    width={"size": 12, "offset": 0},
                                                                )
                                                            ]
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(dbc.Label("Index mandatory keywords"), width={"size": 2}),
                                                                dbc.Col(
                                                                    dcc.Input(
                                                                        id=IDS.RI_TEST_INDEX_KW_MANDATORY,
                                                                        type="text",
                                                                        placeholder="",
                                                                        style={"width": "100%", "height": "30px"},
                                                                        debounce=True,
                                                                        value="",
                                                                    ),
                                                                    width={"offset": 0, "size": 2},
                                                                ),
                                                                dbc.Col(dbc.Label("Index optional keywords"), width={"size": 2, "offset": 1}),
                                                                dbc.Col(
                                                                    dcc.Input(
                                                                        id=IDS.RI_TEST_INDEX_KW_OPTIONAL,
                                                                        type="text",
                                                                        placeholder="",
                                                                        style={"width": "100%", "height": "30px"},
                                                                        debounce=True,
                                                                        value="",
                                                                    ),
                                                                    width={"offset": 0, "size": 2},
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Button(
                                                                        [dbc.Spinner(size="sm", id=IDS.RI_PREDICT_SPINNER)],
                                                                        id=IDS.PREDICT_RI_MODEL,
                                                                        className="col-2",
                                                                    )
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    gap=2,
                                                    style={"margin": "20px"},
                                                ),
                                            ),
                                            dbc.Card(
                                                [
                                                    dbc.Tabs(
                                                        children=[
                                                            dcc.Tab(label="Intensities", children=[], id=IDS.RI_TEST_INTENSITIES),
                                                            dcc.Tab(label="Error", children=[], id=IDS.RI_TEST_ERROR),
                                                            dcc.Tab(
                                                                label="Prediction of reference",
                                                                children=[
                                                                    dbc.Stack(
                                                                        [
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Col(
                                                                                        dbc.Label("Select reference variable"),
                                                                                        width={"size": 2, "offset": 0},
                                                                                    ),
                                                                                    dbc.Col(
                                                                                        dcc.Dropdown(
                                                                                            options=[],
                                                                                            id=IDS.RI_TEST_PRED_REF_SELECTION,
                                                                                        ),
                                                                                        width={"size": 2},
                                                                                    ),
                                                                                    dbc.Col(
                                                                                        dcc.Dropdown(
                                                                                            options=[
                                                                                                {
                                                                                                    "label": "Linear least squares",
                                                                                                    "value": "linear_least_squares",
                                                                                                },
                                                                                            ],
                                                                                            id=IDS.RI_TEST_PRED_MODEL_SELECTION,
                                                                                        ),
                                                                                        width={"size": 2},
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Row(
                                                                                        [
                                                                                            dcc.Graph(
                                                                                                id=IDS.RI_TEST_PRED_GRAPH,
                                                                                                config={"autosizable": False},
                                                                                                style={"width": 1700, "height": 800},
                                                                                            )
                                                                                        ]
                                                                                    ),
                                                                                    dbc.Row(
                                                                                        html.Div(
                                                                                            children=[],
                                                                                            id=IDS.RI_TEST_PRED_TABLE,
                                                                                        )
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                        ],
                                                                        gap=3,
                                                                        style={"margin": "20px"},
                                                                    )
                                                                ],
                                                            ),
                                                            dcc.Tab(
                                                                label="Correlations",
                                                                children=[
                                                                    dbc.Stack(
                                                                        [
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Col(
                                                                                        dbc.Label("Select indicator"),
                                                                                        width={"size": 2, "offset": 0},
                                                                                    ),
                                                                                    dbc.Col(
                                                                                        dcc.Dropdown(
                                                                                            options=[{"label": "Intensities", "value": "Intensities"}],
                                                                                            id=IDS.RI_TEST_CORR_INDICATOR_SELECTION,
                                                                                        ),
                                                                                        width={"size": 2},
                                                                                    ),
                                                                                    dbc.Col(
                                                                                        dbc.Label("Select reference variable"),
                                                                                        width={"size": 2, "offset": 2},
                                                                                    ),
                                                                                    dbc.Col(
                                                                                        dcc.Dropdown(
                                                                                            options=[],
                                                                                            id=IDS.RI_TEST_CORR_REF_SELECTION,
                                                                                        ),
                                                                                        width={"size": 2},
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                            dbc.Row(
                                                                                [
                                                                                    dbc.Row(
                                                                                        [
                                                                                            dcc.Graph(
                                                                                                id=IDS.RI_TEST_CORR_GRAPH,
                                                                                                style={"width": "700", "height": "900"},
                                                                                            )
                                                                                        ]
                                                                                    ),
                                                                                    dbc.Row(
                                                                                        html.Div(
                                                                                            children=[],
                                                                                            id=IDS.RI_TEST_CORR_TABLE,
                                                                                        )
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                        ],
                                                                        gap=3,
                                                                        style={"margin": "20px"},
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                        persistence=True,
                                                        persistence_type="session",
                                                    )
                                                ],
                                            ),
                                        ],
                                        style={"width": "90vw"},
                                    )
                                ],
                                label="Predict",
                                id=IDS.RI_PREDICT,
                            ),
                        ],
                        vertical=True,
                    )
                ),
            ],
            gap=3,
        )
    ]
)

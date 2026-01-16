from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..ids import IDS


# -----------Page #4: PARAFAC--------------

#   -------------Setting up the dbc cards

#       -----------------dbc card for PARAFAC parameters
card_parafac_param = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Import EEM dataset for model establishment", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                dcc.Input(
                                    id=IDS.PARAFAC_EEM_DATASET_ESTABLISHMENT_PATH_INPUT,
                                    type='text',
                                    value=None,
                                    placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                ' If empty, the model built in "eem pre-processing" '
                                                'would be used',
                                    style={'width': '97%', 'height': '30px'},
                                    debounce=True,
                                ),
                                justify="center"
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div(
                                        [],
                                        id=IDS.PARAFAC_EEM_DATASET_ESTABLISHMENT_MESSAGE,
                                        style={'width': '80vw'}
                                    ),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 2}
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.PARAFAC_ESTABLISHMENT_INDEX_KW_MANDATORY,
                                            type='text',
                                            placeholder='',
                                            style={'width': '100%', 'height': '30px'},
                                            debounce=True,
                                            value='',
                                        ),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 2, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.PARAFAC_ESTABLISHMENT_INDEX_KW_OPTIONAL,
                                            type='text',
                                            placeholder='',
                                            style={'width': '100%', 'height': '30px'},
                                            debounce=True,
                                            value='',
                                        ),
                                        width={"offset": 0, "size": 2}
                                    )
                                ]
                            ),
                        ],
                        gap=2
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
                                    dbc.Col(
                                        dbc.Label("Num. components"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.PARAFAC_RANK,
                                            type='text',
                                            placeholder='Multiple values possible, e.g., 3, 4',
                                            style={'width': '250px', 'height': '30px'},
                                            debounce=True,
                                        ),
                                        width={'size': 1}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Initialization"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options=[
                                                {'label': 'SVD', 'value': 'svd'},
                                                {'label': 'random', 'value': 'random'}
                                            ],
                                            value='svd',
                                            style={'width': '150px'},
                                            id=IDS.PARAFAC_INIT_METHOD,
                                        ),
                                        width={'size': 1}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(
                                            options=[{
                                                'label': html.Span(
                                                    "Apply non-negativity constraint",
                                                    style={"font-size": 15, "padding-left": 10}
                                                ),
                                                'value': 'non_negative'
                                            }],
                                            id=IDS.PARAFAC_NN_CHECKBOX,
                                            switch=True,
                                            value=['non_negative'],
                                        ),
                                        width={"size": 2, 'offset': 1}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(
                                            options=[{
                                                'label': html.Span(
                                                    "Normalize EEM by total fluorescence",
                                                    style={"font-size": 15, "padding-left": 10}
                                                ),
                                                'value': 'tf_normalization'
                                            }],
                                            id=IDS.PARAFAC_TF_CHECKBOX,
                                            switch=True,
                                            value=['tf_normalization'],
                                        ),
                                        width={"size": 3, 'offset': 1}
                                    ),
                                ]
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("Solver"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Multiplicative update', 'value': 'mu'},
                                            {'label': 'Hierarchical ALS', 'value': 'hals'},
                                        ],
                                        id=IDS.PARAFAC_SOLVER,
                                        value='hals'
                                    ),
                                    width={'size': 2}
                                ),
                                dbc.Col(
                                    dbc.Label("Validations"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Variance explained', 'value': 'variance_explained'},
                                            {'label': 'Core consistency', 'value': 'core_consistency'},
                                            {'label': 'Leverage', 'value': 'leverage'},
                                            {'label': 'Split-half validation', 'value': 'split_half'},
                                        ],
                                        multi=True,
                                        id=IDS.PARAFAC_VALIDATIONS,
                                        value=[],
                                    ),
                                    width={'size': 4}
                                ),
                            ]),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [dbc.Spinner(size="sm", id=IDS.BUILD_PARAFAC_SPINNER)],
                                        id=IDS.BUILD_PARAFAC_MODEL,
                                        className='col-2'
                                    )
                                )
                            )
                        ],
                        gap=2
                    )
                ]
            ),
        ]
    ),
    className='w-100'
)

#   -------------Layout of page parafac------------

layout = html.Div([
    dbc.Stack(
        [
            dbc.Row(card_parafac_param),
            dbc.Row(
                dcc.Tabs(
                    id=IDS.PARAFAC_RESULTS,
                    children=[
                        dcc.Tab(label='ex/em loadings', id=IDS.PARAFAC_LOADINGS),
                        dcc.Tab(label='Components', id=IDS.PARAFAC_COMPONENTS),
                        dcc.Tab(label='Fmax', id=IDS.PARAFAC_FMAX),
                        dcc.Tab(label='Reconstruction error', id=IDS.PARAFAC_ESTABLISHMENT_RECONSTRUCTION_ERROR),
                        dcc.Tab(label='Variance explained', id=IDS.PARAFAC_VARIANCE_EXPLAINED),
                        dcc.Tab(label='Core consistency', id=IDS.PARAFAC_CORE_CONSISTENCY),
                        dcc.Tab(label='Leverage', id=IDS.PARAFAC_LEVERAGE),
                        dcc.Tab(label='Split-half validation', id=IDS.PARAFAC_SPLIT_HALF),
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
                                                                dbc.Label("Select PARAFAC model"),
                                                                width={'size': 1, 'offset': 0}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.PARAFAC_ESTABLISHMENT_CORR_MODEL_SELECTION,
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                            dbc.Col(
                                                                dbc.Label("Select indicator"),
                                                                width={'size': 1, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[
                                                                        {'label': 'Fmax', 'value': 'Fmax'}
                                                                    ],
                                                                    id=IDS.PARAFAC_ESTABLISHMENT_CORR_INDICATOR_SELECTION,
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                            dbc.Col(
                                                                dbc.Label("Select reference variable"),
                                                                width={'size': 1, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.PARAFAC_ESTABLISHMENT_CORR_REF_SELECTION,
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(
                                                            id=IDS.PARAFAC_ESTABLISHMENT_CORR_GRAPH,
                                                            style={'width': '45vw', 'height': '60vh'}
                                                        ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div(
                                                            [],
                                                            id=IDS.PARAFAC_ESTABLISHMENT_CORR_TABLE
                                                        )
                                                    ),
                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Correlations',
                            id=IDS.PARAFAC_ESTABLISHMENT_CORR
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
                                                            id=IDS.PARAFAC_EEM_DATASET_PREDICT_PATH_INPUT,
                                                            type='text',
                                                            placeholder='Please enter the eem dataset path (.json'
                                                                        ' and .pkl are supported).',
                                                            style={'width': '97%', 'height': '30px'},
                                                            debounce=True
                                                        ),
                                                        justify="center"
                                                    ),
                                                    dbc.Row([
                                                        dbc.Col(
                                                            html.Div(
                                                                [],
                                                                id=IDS.PARAFAC_EEM_DATASET_PREDICT_MESSAGE,
                                                                style={'width': '1000px'}
                                                            ),
                                                            width={"size": 12, "offset": 0}
                                                        )
                                                    ]),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dbc.Label("Index mandatory keywords"), width={'size': 2}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Input(
                                                                    id=IDS.PARAFAC_TEST_INDEX_KW_MANDATORY,
                                                                    type='text',
                                                                    placeholder='',
                                                                    style={'width': '100%', 'height': '30px'},
                                                                    debounce=True,
                                                                    value='',
                                                                ),
                                                                width={"offset": 0, "size": 2}
                                                            ),
                                                            dbc.Col(
                                                                dbc.Label("Index optional keywords"),
                                                                width={'size': 2, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Input(
                                                                    id=IDS.PARAFAC_TEST_INDEX_KW_OPTIONAL,
                                                                    type='text',
                                                                    placeholder='',
                                                                    style={'width': '100%', 'height': '30px'},
                                                                    debounce=True,
                                                                    value='',
                                                                ),
                                                                width={"offset": 0, "size": 2}
                                                            )
                                                        ]
                                                    ),
                                                    html.H5("Select established model"),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.PARAFAC_TEST_MODEL_SELECTION,
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm", id=IDS.PARAFAC_PREDICT_SPINNER)],
                                                                    id=IDS.PREDICT_PARAFAC_MODEL,
                                                                    className='col-2'
                                                                )
                                                            )
                                                        ]
                                                    )
                                                ],
                                                gap=2, style={"margin": "20px"}
                                            ),
                                        ),
                                        dbc.Card(
                                            [
                                                dbc.Tabs(
                                                    children=[
                                                        dcc.Tab(
                                                            label='Fmax',
                                                            children=[],
                                                            id=IDS.PARAFAC_TEST_FMAX
                                                        ),
                                                        dcc.Tab(
                                                            label='Reconstruction error',
                                                            children=[],
                                                            id=IDS.PARAFAC_TEST_ERROR
                                                        ),
                                                        dcc.Tab(
                                                            label='Prediction of reference',
                                                            children=[
                                                                dbc.Stack(
                                                                    [
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(
                                                                                    dbc.Label("Select reference variable"),
                                                                                    width={'size': 2, 'offset': 0}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[],
                                                                                        id=IDS.PARAFAC_TEST_PRED_REF_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dbc.Label(
                                                                                        "Select model to fit reference "
                                                                                        "variable with fmax"
                                                                                    ),
                                                                                    width={'size': 2, 'offset': 2}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[
                                                                                            {
                                                                                                'label': 'Linear least squares',
                                                                                                'value': 'linear_least_squares'
                                                                                            },
                                                                                        ],
                                                                                        id=IDS.PARAFAC_TEST_PRED_MODEL_SELECTION,
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Row([
                                                                                    dcc.Graph(
                                                                                        id=IDS.PARAFAC_TEST_PRED_GRAPH,
                                                                                        config={'autosizable': False},
                                                                                        style={'width': 1700, 'height': 800}
                                                                                    ),
                                                                                ]),
                                                                                dbc.Row(
                                                                                    html.Div(
                                                                                        children=[],
                                                                                        id=IDS.PARAFAC_TEST_PRED_TABLE
                                                                                    )
                                                                                ),
                                                                            ]
                                                                        )
                                                                    ],
                                                                    gap=3, style={"margin": "20px"}
                                                                ),
                                                            ],
                                                        ),
                                                        dcc.Tab(
                                                            label='Correlations',
                                                            children=[
                                                                dbc.Stack(
                                                                    [
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(
                                                                                    dbc.Label("Select indicator"),
                                                                                    width={'size': 2, 'offset': 0}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[
                                                                                            {'label': 'Fmax', 'value': 'Fmax'},
                                                                                        ],
                                                                                        id=IDS.PARAFAC_TEST_CORR_INDICATOR_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dbc.Label("Select reference variable"),
                                                                                    width={'size': 2, 'offset': 2}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[],
                                                                                        id=IDS.PARAFAC_TEST_CORR_REF_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Row([
                                                                                    dcc.Graph(
                                                                                        id=IDS.PARAFAC_TEST_CORR_GRAPH,
                                                                                        style={'width': '700', 'height': '900'}
                                                                                    ),
                                                                                ]),
                                                                                dbc.Row(
                                                                                    html.Div(
                                                                                        children=[],
                                                                                        id=IDS.PARAFAC_TEST_CORR_TABLE
                                                                                    )
                                                                                ),
                                                                            ]
                                                                        )
                                                                    ],
                                                                    gap=3, style={"margin": "20px"}
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                    persistence=True,
                                                    persistence_type='session'
                                                ),
                                            ],
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Predict',
                            id=IDS.PARAFAC_PREDICT
                        )
                    ],
                    vertical=True
                )
            ),
        ],
        gap=3
    )
])

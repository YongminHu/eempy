from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..ids import IDS

# -----------Page #3: NMF--------------

#   -------------Setting up the dbc cards
card_nmf_param = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Parameters selection", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                dcc.Input(
                                    id=IDS.NMF_EEM_DATASET_ESTABLISHMENT_PATH_INPUT,
                                    type='text',
                                    value=None,
                                    placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                ' If empty, the model built in "eem pre-processing" '
                                                'would be used',
                                    style={'width': '97%', 'height': '30px'},
                                    debounce=True
                                ),
                                justify="center"
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div(
                                        [],
                                        id=IDS.NMF_EEM_DATASET_ESTABLISHMENT_MESSAGE,
                                        style={'width': '80vw'}
                                    ),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.NMF_ESTABLISHMENT_INDEX_KW_MANDATORY,
                                            type='text',
                                            placeholder='',
                                            style={'width': '100%', 'height': '30px'},
                                            debounce=True,
                                            value=''
                                        ),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.NMF_ESTABLISHMENT_INDEX_KW_OPTIONAL,
                                            type='text',
                                            placeholder='',
                                            style={'width': '100%', 'height': '30px'},
                                            debounce=True,
                                            value=''
                                        ),
                                        width={"offset": 0, "size": 2}
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Num. components"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.NMF_RANK,
                                            type='text',
                                            placeholder='Multiple values possible, e.g., 3, 4',
                                            style={'width': '250px', 'height': '30px'},
                                            debounce=True
                                        ),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Solver"), width={'size': 1, 'offset': 0}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options=[
                                                {'label': 'Coordinate Descent', 'value': 'cd'},
                                                {'label': 'Multiplicative Update', 'value': 'mu'},
                                                {'label': 'Hierarchical ALS', 'value': 'hals'},
                                            ],
                                            value='hals',
                                            style={'width': '300px'},
                                            id=IDS.NMF_SOLVER
                                        ),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Initialization"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options=[
                                                {'label': 'random', 'value': 'random'},
                                                {'label': 'nndsvd', 'value': 'nndsvd'},
                                                {'label': 'nndsvda', 'value': 'nndsvda'},
                                                {'label': 'nndsvdar', 'value': 'nndsvdar'},
                                            ],
                                            value='nndsvda',
                                            style={'width': '100px'},
                                            id=IDS.NMF_INIT
                                        ),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(
                                            options=[{
                                                'label': html.Span(
                                                    "Normalize pixels by STD",
                                                    style={"font-size": 15, "padding-left": 10}
                                                ),
                                                'value': 'pixel_std'
                                            }],
                                            id=IDS.NMF_NORMALIZATION_CHECKBOX,
                                            switch=True,
                                            value=['pixel_std']
                                        ),
                                        width={"size": 2, 'offset': 0}
                                    ),
                                ]
                            ),
                            dbc.Row([
                                dbc.Col(dbc.Label("alpha_W"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.NMF_ALPHA_W,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0
                                    ),
                                    width={'size': 2},
                                ),

                                dbc.Col(dbc.Label("alpha_H"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.NMF_ALPHA_H,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0
                                    ),
                                    width={'size': 3},
                                ),

                                dbc.Col(dbc.Label("l1 ratio"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.NMF_L1_RATIO,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0
                                    ),
                                    width={'size': 2},
                                ),
                            ]),

                            dbc.Row([
                                dbc.Col(dbc.Label("max_iter_als"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.NMF_MAX_ITER_ALS,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=500
                                    ),
                                    width={'size': 2},
                                ),

                                dbc.Col(dbc.Label("max_iter_nnls"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.NMF_MAX_ITER_NNLS,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=200
                                    ),
                                    width={'size': 3},
                                ),
                            ]),

                            dbc.Row([
                                dbc.Col(dbc.Label("Validations"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Split-half validation', 'value': 'split_half'},
                                        ],
                                        multi=True,
                                        id=IDS.NMF_VALIDATIONS,
                                        value=[]
                                    ),
                                    width={'size': 4}
                                ),
                            ]),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [dbc.Spinner(size="sm", id=IDS.NMF_SPINNER)],
                                        id=IDS.BUILD_NMF_MODEL,
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

#   -------------Layout of page #3

layout = html.Div([
    dbc.Stack(
        [
            dbc.Row(card_nmf_param),
            dbc.Row(
                dcc.Tabs(
                    id=IDS.NMF_RESULTS,
                    children=[
                        dcc.Tab(label='Components', id=IDS.NMF_COMPONENTS),
                        dcc.Tab(label='Fmax', id=IDS.NMF_FMAX),
                        dcc.Tab(label='Reconstruction error', id=IDS.NMF_ESTABLISHMENT_RECONSTRUCTION_ERROR),
                        dcc.Tab(label='Split-half validation', id=IDS.NMF_SPLIT_HALF),
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
                                                                dbc.Label("Select nmf model"),
                                                                width={'size': 1, 'offset': 0}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.NMF_ESTABLISHMENT_CORR_MODEL_SELECTION
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
                                                                        {'label': 'NMF-Fmax', 'value': 'NMF-Fmax'},
                                                                        {'label': 'NNLS-Fmax', 'value': 'NNLS-Fmax'}
                                                                    ],
                                                                    id=IDS.NMF_ESTABLISHMENT_CORR_INDICATOR_SELECTION
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
                                                                    id=IDS.NMF_ESTABLISHMENT_CORR_REF_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(
                                                            id=IDS.NMF_ESTABLISHMENT_CORR_GRAPH,
                                                            style={'width': '45vw', 'height': '60vh'}
                                                        ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div(
                                                            [],
                                                            id=IDS.NMF_ESTABLISHMENT_CORR_TABLE
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
                            id=IDS.NMF_ESTABLISHMENT_CORR
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
                                                            id=IDS.NMF_EEM_DATASET_PREDICT_PATH_INPUT,
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
                                                                id=IDS.NMF_EEM_DATASET_PREDICT_MESSAGE,
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
                                                                    id=IDS.NMF_TEST_INDEX_KW_MANDATORY,
                                                                    type='text',
                                                                    placeholder='',
                                                                    style={'width': '100%', 'height': '30px'},
                                                                    debounce=True,
                                                                    value=''
                                                                ),
                                                                width={"offset": 0, "size": 2}
                                                            ),
                                                            dbc.Col(
                                                                dbc.Label("Index optional keywords"),
                                                                width={'size': 2, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Input(
                                                                    id=IDS.NMF_TEST_INDEX_KW_OPTIONAL,
                                                                    type='text',
                                                                    placeholder='',
                                                                    style={'width': '100%', 'height': '30px'},
                                                                    debounce=True,
                                                                    value=''
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
                                                                    id=IDS.NMF_TEST_MODEL_SELECTION
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm", id=IDS.NMF_PREDICT_SPINNER)],
                                                                    id=IDS.PREDICT_NMF_MODEL,
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
                                                            id=IDS.NMF_TEST_FMAX
                                                        ),
                                                        dcc.Tab(
                                                            label='Reconstruction error',
                                                            children=[],
                                                            id=IDS.NMF_TEST_ERROR
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
                                                                                        id=IDS.NMF_TEST_PRED_REF_SELECTION
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
                                                                                        id=IDS.NMF_TEST_PRED_MODEL_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Row([
                                                                                    dcc.Graph(
                                                                                        id=IDS.NMF_TEST_PRED_GRAPH,
                                                                                        config={'autosizable': False},
                                                                                        style={'width': 1700, 'height': 800}
                                                                                    ),
                                                                                ]),
                                                                                dbc.Row(
                                                                                    html.Div(
                                                                                        children=[],
                                                                                        id=IDS.NMF_TEST_PRED_TABLE
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
                                                                                            {'label': 'NMF-Fmax', 'value': 'NMF-Fmax'},
                                                                                            {'label': 'NNLS-Fmax', 'value': 'NNLS-Fmax'},
                                                                                        ],
                                                                                        id=IDS.NMF_TEST_CORR_INDICATOR_SELECTION
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
                                                                                        id=IDS.NMF_TEST_CORR_REF_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Row([
                                                                                    dcc.Graph(
                                                                                        id=IDS.NMF_TEST_CORR_GRAPH,
                                                                                        style={'width': '700', 'height': '900'}
                                                                                    ),
                                                                                ]),
                                                                                dbc.Row(
                                                                                    html.Div(
                                                                                        children=[],
                                                                                        id=IDS.NMF_TEST_CORR_TABLE
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
                            id=IDS.NMF_PREDICT
                        )
                    ],
                    vertical=True
                )
            ),
        ],
        gap=3
    )
])

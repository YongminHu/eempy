from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..ids import IDS


# -----------Page #6: K-method--------------

#   -------------Setting up the dbc cards

card_kmethod_param1 = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Step 1: calculate consensus", className="card-title"),
            html.H6("Parameters selection", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                dcc.Input(
                                    id=IDS.KMETHOD_EEM_DATASET_ESTABLISHMENT_PATH_INPUT,
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
                            dbc.Row(
                                dbc.Checklist(
                                    options=[{
                                        'label': html.Span(
                                            "Read clustering output from the file",
                                            style={"font-size": 15, "padding-left": 10}
                                        ),
                                        'value': True
                                    }],
                                    id=IDS.KMETHOD_CLUSTER_FROM_FILE_CHECKBOX,
                                    switch=True,
                                    value=[True]
                                ),
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div(
                                        [],
                                        id=IDS.KMETHOD_EEM_DATASET_ESTABLISHMENT_MESSAGE,
                                        style={'width': '80vw'}
                                    ),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("Index mandatory keywords"), width={'size': 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.KMETHOD_ESTABLISHMENT_INDEX_KW_MANDATORY,
                                            type='text',
                                            placeholder='',
                                            style={'width': '100%', 'height': '30px'},
                                            debounce=True,
                                            value=''
                                        ),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(dbc.Label("Index optional keywords"), width={'size': 1, 'offset': 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.KMETHOD_ESTABLISHMENT_INDEX_KW_OPTIONAL,
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
                                    dbc.Col(dbc.Label("Base model"), width={'size': 1, 'offset': 0}),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options=[
                                                {'label': 'PARAFAC', 'value': 'parafac'},
                                                {'label': 'NMF', 'value': 'nmf'}
                                            ],
                                            value=None,
                                            style={'width': '300px'},
                                            id=IDS.KMETHOD_BASE_MODEL
                                        ),
                                        width={'size': 2}
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [],
                                            id=IDS.KMETHOD_BASE_MODEL_MESSAGE,
                                            style={'width': '80vw'}
                                        ),
                                        width={"size": 8, "offset": 1}
                                    )
                                ]
                            ),

                            dbc.Row([
                                dbc.Col(dbc.Label("Num. components"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_RANK,
                                        type='number',
                                        placeholder='Please enter only one number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0
                                    ),
                                    width={'size': 1},
                                ),
                                dbc.Col(dbc.Label("Number of initial splits"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_NUM_INIT_SPLITS,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0
                                    ),
                                    width={'size': 1},
                                ),
                                dbc.Col(dbc.Label("Number of base clustering runs"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_NUM_BASE_CLUSTERINGS,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0
                                    ),
                                    width={'size': 1},
                                ),
                                dbc.Col(
                                    dbc.Label("Maximum iterations for one time base clustering"),
                                    width={'size': 2}
                                ),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_NUM_ITERATIONS,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=10
                                    ),
                                    width={'size': 1},
                                ),
                            ]),

                            dbc.Row([
                                dbc.Col(dbc.Label("Convergence tolerance"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_CONVERGENCE_TOLERANCE,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0.01
                                    ),
                                    width={'size': 1},
                                ),
                                dbc.Col(dbc.Label("Elimination"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_ELIMINATION,
                                        type='text',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value='default'
                                    ),
                                    width={'size': 1},
                                ),
                                dbc.Col(dbc.Label("Subsampling portion"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Input(
                                        id=IDS.KMETHOD_SUBSAMPLING_PORTION,
                                        type='number',
                                        style={'width': '100px', 'height': '30px'},
                                        debounce=True,
                                        value=0.8
                                    ),
                                    width={'size': 1},
                                ),
                            ]),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [dbc.Spinner(size="sm", id=IDS.KMETHOD_STEP1_SPINNER)],
                                        id=IDS.BUILD_KMETHOD_CONSENSUS,
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

card_kmethod_param2 = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Step 2: clustering", className="card-title"),
            html.H6("Parameters selection", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("Number of final clusters"), width={'size': 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.KMETHOD_NUM_FINAL_CLUSTERS,
                                            type='text',
                                            placeholder='Multiple values possible, e.g., 3, 4',
                                            style={'width': '250px', 'height': '30px'},
                                            debounce=True
                                        ),
                                        width={"offset": 0, "size": 3}
                                    ),
                                    dbc.Col(dbc.Label("Consensus conversion factor"), width={'size': 1}),
                                    dbc.Col(
                                        dcc.Input(
                                            id=IDS.KMETHOD_CONSENSUS_CONVERSION,
                                            type='number',
                                            value=1,
                                            style={'width': '250px', 'height': '30px'},
                                            debounce=True
                                        ),
                                        width={"offset": 0, "size": 1}
                                    ),
                                ]
                            ),

                            dbc.Row([
                                dbc.Col(dbc.Label("Validations"), width={'size': 1}),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'silhouette score', 'value': 'silhouette_score'},
                                            {'label': 'reconstruction error reduction', 'value': 'RER'}
                                        ],
                                        multi=True,
                                        id=IDS.KMETHOD_VALIDATIONS,
                                        value=['silhouette_score', 'RER']
                                    ),
                                    width={'size': 4}
                                ),
                            ]),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button(
                                        [dbc.Spinner(size="sm", id=IDS.KMETHOD_STEP2_SPINNER)],
                                        id=IDS.BUILD_KMETHOD_CLUSTERING,
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

#   -------------Layout of page #6

layout = html.Div([
    dbc.Stack(
        [
            dbc.Row(card_kmethod_param1),
            dbc.Row(
                dcc.Tabs(
                    id=IDS.KMETHOD_RESULTS1,
                    children=[
                        dcc.Tab(label='Consensus matrix', id=IDS.KMETHOD_CONSENSUS_MATRIX),
                        dcc.Tab(label='Label history', id=IDS.KMETHOD_LABEL_HISTORY),
                        dcc.Tab(label='Error history', id=IDS.KMETHOD_ERROR_HISTORY),
                    ],
                    vertical=True
                )
            ),
            dbc.Row(card_kmethod_param2),
            dbc.Row(
                dcc.Tabs(
                    id=IDS.KMETHOD_RESULTS2,
                    children=[
                        dcc.Tab(label='Dendrogram', id=IDS.KMETHOD_DENDROGRAM),
                        dcc.Tab(label='Sorted consensus matrix', id=IDS.KMETHOD_SORTED_CONSENSUS_MATRIX),
                        dcc.Tab(label='Silhouette score', id=IDS.KMETHOD_SILHOUETTE_SCORE),
                        dcc.Tab(label='Reconstruction error reduction', id=IDS.KMETHOD_RECONSTRUCTION_ERROR_REDUCTION),
                        dcc.Tab(
                            children=[
                                html.Div(
                                    [
                                        dbc.Card(
                                            dbc.Stack(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dbc.Label("Select K-method model"), width={'size': 1, 'offset': 0}),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_MODEL_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                            dbc.Col(dbc.Label("Select cluster"), width={'size': 1, 'offset': 1}),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_CLUSTER_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.Row([
                                                        html.Div(
                                                            children=[],
                                                            id=IDS.KMETHOD_ESTABLISHMENT_COMPONENTS_GRAPH,
                                                            style={'width': '45vw', 'height': '60vh'}
                                                        )
                                                    ]),
                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Components',
                            id=IDS.KMETHOD_COMPONENTS
                        ),
                        dcc.Tab(label='Fmax', id=IDS.KMETHOD_FMAX),
                        dcc.Tab(
                            children=[
                                html.Div(
                                    [
                                        dbc.Card(
                                            dbc.Stack(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dbc.Label("Select K-method model"), width={'size': 1, 'offset': 0}),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.KMETHOD_ESTABLISHMENT_CORR_MODEL_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                            dbc.Col(dbc.Label("Select indicator"), width={'size': 1, 'offset': 1}),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[{'label': 'Fmax', 'value': 'Fmax'}],
                                                                    id=IDS.KMETHOD_ESTABLISHMENT_CORR_INDICATOR_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                            dbc.Col(dbc.Label("Select reference variable"), width={'size': 1, 'offset': 1}),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.KMETHOD_ESTABLISHMENT_CORR_REF_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.Row([
                                                        dcc.Graph(
                                                            id=IDS.KMETHOD_ESTABLISHMENT_CORR_GRAPH,
                                                            style={'width': '45vw', 'height': '60vh'}
                                                        ),
                                                    ]),
                                                    dbc.Row(
                                                        html.Div(
                                                            [],
                                                            id=IDS.KMETHOD_ESTABLISHMENT_CORR_TABLE
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
                            id=IDS.KMETHOD_ESTABLISHMENT_CORR
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
                                                            id=IDS.KMETHOD_EEM_DATASET_PREDICT_PATH_INPUT,
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
                                                                id=IDS.KMETHOD_EEM_DATASET_PREDICT_MESSAGE,
                                                                style={'width': '1000px'}
                                                            ),
                                                            width={"size": 12, "offset": 0}
                                                        )
                                                    ]),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dbc.Label("Index mandatory keywords"), width={'size': 2}),
                                                            dbc.Col(
                                                                dcc.Input(
                                                                    id=IDS.KMETHOD_TEST_INDEX_KW_MANDATORY,
                                                                    type='text',
                                                                    placeholder='',
                                                                    style={'width': '100%', 'height': '30px'},
                                                                    debounce=True,
                                                                    value=''
                                                                ),
                                                                width={"offset": 0, "size": 2}
                                                            ),
                                                            dbc.Col(dbc.Label("Index optional keywords"), width={'size': 2, 'offset': 1}),
                                                            dbc.Col(
                                                                dcc.Input(
                                                                    id=IDS.KMETHOD_TEST_INDEX_KW_OPTIONAL,
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
                                                                    id=IDS.KMETHOD_TEST_MODEL_SELECTION
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm", id=IDS.KMETHOD_PREDICT_SPINNER)],
                                                                    id=IDS.PREDICT_KMETHOD_MODEL,
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
                                                        dcc.Tab(label='Fmax', children=[], id=IDS.KMETHOD_TEST_FMAX),
                                                        dcc.Tab(label='Reconstruction error', children=[], id=IDS.KMETHOD_TEST_ERROR),
                                                        dcc.Tab(
                                                            label='Prediction of reference',
                                                            children=[
                                                                dbc.Stack(
                                                                    [
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Col(dbc.Label("Select reference variable"), width={'size': 2, 'offset': 0}),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[],
                                                                                        id=IDS.KMETHOD_TEST_PRED_REF_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dbc.Label("Select model to fit reference variable with fmax"),
                                                                                    width={'size': 2, 'offset': 2}
                                                                                ),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[{'label': 'Linear least squares', 'value': 'linear_least_squares'}],
                                                                                        id=IDS.KMETHOD_TEST_PRED_MODEL_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Row([
                                                                                    dcc.Graph(
                                                                                        id=IDS.KMETHOD_TEST_PRED_GRAPH,
                                                                                        config={'autosizable': False},
                                                                                        style={'width': 1700, 'height': 800}
                                                                                    ),
                                                                                ]),
                                                                                dbc.Row(
                                                                                    html.Div(
                                                                                        children=[],
                                                                                        id=IDS.KMETHOD_TEST_PRED_TABLE
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
                                                                                dbc.Col(dbc.Label("Select indicator"), width={'size': 2, 'offset': 0}),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[{'label': 'Fmax', 'value': 'Fmax'}],
                                                                                        id=IDS.KMETHOD_TEST_CORR_INDICATOR_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                                dbc.Col(dbc.Label("Select reference variable"), width={'size': 2, 'offset': 2}),
                                                                                dbc.Col(
                                                                                    dcc.Dropdown(
                                                                                        options=[],
                                                                                        id=IDS.KMETHOD_TEST_CORR_REF_SELECTION
                                                                                    ),
                                                                                    width={'size': 2}
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        dbc.Row(
                                                                            [
                                                                                dbc.Row([
                                                                                    dcc.Graph(
                                                                                        id=IDS.KMETHOD_TEST_CORR_GRAPH,
                                                                                        style={'width': '700', 'height': '900'}
                                                                                    ),
                                                                                ]),
                                                                                dbc.Row(
                                                                                    html.Div(
                                                                                        children=[],
                                                                                        id=IDS.KMETHOD_TEST_CORR_TABLE
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
                            id=IDS.KMETHOD_PREDICT
                        ),
                        dcc.Tab(
                            children=[
                                html.Div(
                                    [
                                        dbc.Card(
                                            dbc.Stack(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dbc.Label("Select K-method model"), width={'size': 1, 'offset': 0}),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id=IDS.KMETHOD_CLUSTER_EXPORT_MODEL_SELECTION
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Export',
                            id=IDS.KMETHOD_CLUSTER_EXPORT
                        ),
                    ],
                    vertical=True
                )
            ),
        ],
        gap=3
    )
])

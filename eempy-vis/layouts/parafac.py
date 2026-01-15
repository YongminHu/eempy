from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc


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
                                dcc.Input(id='parafac-eem-dataset-establishment-path-input', type='text', value=None,
                                          placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                      ' If empty, the model built in "eem pre-processing" '
                                                      'would be used',
                                          style={'width': '97%', 'height': '30px'}, debounce=True),
                                justify="center"
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div([],
                                             id='parafac-eem-dataset-establishment-message', style={'width': '80vw'}),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 2}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='parafac-establishment-index-kw-mandatory', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 2, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='parafac-establishment-index-kw-optional', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
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
                                        dcc.Input(id='parafac-rank', type='text',
                                                  placeholder='Multiple values possible, e.g., 3, 4',
                                                  style={'width': '250px', 'height': '30px'}, debounce=True),
                                        width={'size': 1}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Initialization"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(options=[
                                            {'label': 'SVD', 'value': 'svd'},
                                            {'label': 'random', 'value': 'random'}
                                        ],
                                            value='svd', style={'width': '150px'}, id='parafac-init-method'
                                        ),
                                        width={'size': 1}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(options=[{'label': html.Span("Apply non-negativity constraint",
                                                                                   style={"font-size": 15,
                                                                                          "padding-left": 10}),
                                                                'value': 'non_negative'}],
                                                      id='parafac-nn-checkbox', switch=True, value=['non_negative']
                                                      ),
                                        width={"size": 2, 'offset': 1}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(
                                            options=[{'label': html.Span("Normalize EEM by total fluorescence",
                                                                         style={"font-size": 15,
                                                                                "padding-left": 10}),
                                                      'value': 'tf_normalization'}],
                                            id='parafac-tf-checkbox', switch=True,
                                            value=['tf_normalization']
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
                                        id='parafac-solver', value='hals'),
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
                                        multi=True, id='parafac-validations', value=[]),
                                    width={'size': 4}
                                ),

                            ]),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='build-parafac-spinner')],
                                               id='build-parafac-model', className='col-2')
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
            dbc.Row(
                card_parafac_param
            ),
            dbc.Row(
                dcc.Tabs(
                    id='parafac-results',
                    children=[
                        dcc.Tab(label='ex/em loadings', id='parafac-loadings'),
                        dcc.Tab(label='Components', id='parafac-components'),
                        dcc.Tab(label='Fmax', id='parafac-fmax'),
                        dcc.Tab(label='Reconstruction error', id='parafac-establishment-reconstruction-error'),
                        dcc.Tab(label='Variance explained', id='parafac-variance-explained'),
                        dcc.Tab(label='Core consistency', id='parafac-core-consistency'),
                        dcc.Tab(label='Leverage', id='parafac-leverage'),
                        dcc.Tab(label='Split-half validation', id='parafac-split-half'),
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
                                                                    id='parafac-establishment-corr-model-selection'
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
                                                                        # {'label': 'Score', 'value': 'Score'},
                                                                        {'label': 'Fmax', 'value': 'Fmax'}
                                                                    ],
                                                                    id='parafac-establishment-corr-indicator-selection'
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
                                                                    id='parafac-establishment-corr-ref-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(id='parafac-establishment-corr-graph',
                                                                  # config={'responsive': 'auto'},
                                                                  style={'width': '45vw', 'height': '60vh'}
                                                                  ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div([],
                                                                 id='parafac-establishment-corr-table')
                                                    ),

                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Correlations', id='parafac-establishment-corr'
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
                                                        dcc.Input(id='parafac-eem-dataset-predict-path-input',
                                                                  type='text',
                                                                  placeholder='Please enter the eem dataset path (.json'
                                                                              ' and .pkl are supported).',
                                                                  style={'width': '97%', 'height': '30px'},
                                                                  debounce=True),
                                                        justify="center"
                                                    ),
                                                    dbc.Row([
                                                        dbc.Col(
                                                            html.Div([],
                                                                     id='parafac-eem-dataset-predict-message',
                                                                     style={'width': '1000px'}),
                                                            width={"size": 12, "offset": 0}
                                                        )
                                                    ]),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dbc.Label("Index mandatory keywords"), width={'size': 2}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Input(id='parafac-test-index-kw-mandatory',
                                                                          type='text',
                                                                          placeholder='',
                                                                          style={'width': '100%', 'height': '30px'},
                                                                          debounce=True, value=''),
                                                                width={"offset": 0, "size": 2}
                                                            ),
                                                            dbc.Col(
                                                                dbc.Label("Index optional keywords"),
                                                                width={'size': 2, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Input(id='parafac-test-index-kw-optional',
                                                                          type='text',
                                                                          placeholder='',
                                                                          style={'width': '100%', 'height': '30px'},
                                                                          debounce=True, value=''),
                                                                width={"offset": 0, "size": 2}
                                                            )
                                                        ]
                                                    ),
                                                    html.H5("Select established model"),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[], id='parafac-test-model-selection'
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm",
                                                                                 id='parafac-predict-spinner')],
                                                                    id='predict-parafac-model', className='col-2')
                                                            )
                                                        ]
                                                    )
                                                ],
                                                gap=2, style={"margin": "20px"}
                                            ),
                                        ),
                                        dbc.Card(
                                            [
                                                dbc.Tabs(children=[
                                                    # dcc.Tab(
                                                    #     label='Score',
                                                    #     children=[],
                                                    #     id='parafac-test-score'
                                                    # ),
                                                    dcc.Tab(
                                                        label='Fmax',
                                                        children=[],
                                                        id='parafac-test-fmax'
                                                    ),
                                                    dcc.Tab(
                                                        label='Reconstruction error',
                                                        children=[],
                                                        id='parafac-test-error'
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
                                                                                    id='parafac-test-pred-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                            dbc.Col(
                                                                                dbc.Label(
                                                                                    "Select model to fit reference "
                                                                                    "variable with fmax"),
                                                                                width={'size': 2, 'offset': 2}
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Dropdown(
                                                                                    options=[
                                                                                        {
                                                                                            'label': 'Linear least squares',
                                                                                            'value': 'linear_least_squares'},
                                                                                    ],
                                                                                    id='parafac-test-pred-model'
                                                                                       '-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Row([
                                                                                dcc.Graph(
                                                                                    id='parafac-test-pred-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    config={'autosizable': False},
                                                                                    style={'width': 1700, 'height': 800}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='parafac-test-pred-table'
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
                                                                                        # {'label': 'Score',
                                                                                        #  'value': 'Score'},
                                                                                        {'label': 'Fmax',
                                                                                         'value': 'Fmax'},
                                                                                    ],
                                                                                    id='parafac-test-corr-indicator-selection'
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
                                                                                    id='parafac-test-corr-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Row([
                                                                                dcc.Graph(
                                                                                    id='parafac-test-corr-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    style={'width': '700',
                                                                                           'height': '900'}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='parafac-test-corr-table'
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
                                                    persistence_type='session'),

                                            ],
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Predict', id='parafac-predict'
                        )
                    ],
                    # style={
                    #     'width': '100%'
                    # },
                    vertical=True
                )
            ),
        ],
        gap=3
    )

])

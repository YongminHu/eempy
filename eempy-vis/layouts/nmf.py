from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

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
                                dcc.Input(id='nmf-eem-dataset-establishment-path-input', type='text', value=None,
                                          placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                      ' If empty, the model built in "eem pre-processing" '
                                                      'would be used',
                                          style={'width': '97%', 'height': '30px'}, debounce=True),
                                justify="center"
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div([],
                                             id='nmf-eem-dataset-establishment-message', style={'width': '80vw'}),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='nmf-establishment-index-kw-mandatory', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='nmf-establishment-index-kw-optional', type='text', placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
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
                                        dcc.Input(id='nmf-rank', type='text',
                                                  placeholder='Multiple values possible, e.g., 3, 4',
                                                  style={'width': '250px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Solver"), width={'size': 1, 'offset': 0}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(options=[
                                            {'label': 'Coordinate Descent', 'value': 'cd'},
                                            {'label': 'Multiplicative Update', 'value': 'mu'},
                                            {'label': 'Hierarchical ALS', 'value': 'hals'},
                                        ],
                                            value='hals', style={'width': '300px'}, id='nmf-solver'
                                        ),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Initialization"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(options=[
                                            {'label': 'random', 'value': 'random'},
                                            {'label': 'nndsvd', 'value': 'nndsvd'},
                                            {'label': 'nndsvda', 'value': 'nndsvda'},
                                            {'label': 'nndsvdar', 'value': 'nndsvdar'},
                                        ],
                                            value='nndsvda', style={'width': '100px'}, id='nmf-init'
                                        ),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(options=[{'label': html.Span("Normalize pixels by STD",
                                                                                   style={"font-size": 15,
                                                                                          "padding-left": 10}),
                                                                'value': 'pixel_std'}],
                                                      id='nmf-normalization-checkbox', switch=True, value=['pixel_std']
                                                      ),
                                        width={"size": 2, 'offset': 0}
                                    ),

                                ]
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("alpha_W"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='nmf-alpha-w', type='number',
                                              # placeholder='Multiple values possible, e.g., 3, 4',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 2},
                                ),

                                dbc.Col(
                                    dbc.Label("alpha_H"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='nmf-alpha-h', type='number',
                                              # placeholder='Multiple values possible, e.g., 3, 4',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 3},
                                ),

                                dbc.Col(
                                    dbc.Label("l1 ratio"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='nmf-l1-ratio', type='number',
                                              # placeholder='Multiple values possible, e.g., 3, 4',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 2},
                                ),
                            ]),

                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("max_iter_als"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='nmf-max-iter-als', type='number',
                                              # placeholder='Multiple values possible, e.g., 3, 4',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=500),
                                    width={'size': 2},
                                ),

                                dbc.Col(
                                    dbc.Label("max_iter_nnls"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='nmf-max-iter-nnls', type='number',
                                              # placeholder='Multiple values possible, e.g., 3, 4',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=200),
                                    width={'size': 3},
                                ),
                            ]),

                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("Validations"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Split-half validation', 'value': 'split_half'},
                                        ],
                                        multi=True, id='nmf-validations', value=[]),
                                    width={'size': 4}
                                ),
                            ]),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='nmf-spinner')],
                                               id='build-nmf-model', className='col-2')
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
            dbc.Row(
                card_nmf_param
            ),
            dbc.Row(
                dcc.Tabs(
                    id='nmf-results',
                    children=[
                        dcc.Tab(label='Components', id='nmf-components'),
                        dcc.Tab(label='Fmax', id='nmf-fmax'),
                        dcc.Tab(label='Reconstruction error', id='nmf-establishment-reconstruction-error'),
                        dcc.Tab(label='Split-half validation', id='nmf-split-half'),
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
                                                                    id='nmf-establishment-corr-model-selection'
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
                                                                    id='nmf-establishment-corr-indicator-selection'
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
                                                                    id='nmf-establishment-corr-ref-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(id='nmf-establishment-corr-graph',
                                                                  # config={'responsive': 'auto'},
                                                                  style={'width': '45vw', 'height': '60vh'}
                                                                  ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div([],
                                                                 id='nmf-establishment-corr-table')
                                                    ),

                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Correlations', id='nmf-establishment-corr'
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
                                                        dcc.Input(id='nmf-eem-dataset-predict-path-input',
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
                                                                     id='nmf-eem-dataset-predict-message',
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
                                                                dcc.Input(id='nmf-test-index-kw-mandatory',
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
                                                                dcc.Input(id='nmf-test-index-kw-optional',
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
                                                                    options=[], id='nmf-test-model-selection'
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm",
                                                                                 id='nmf-predict-spinner')],
                                                                    id='predict-nmf-model', className='col-2')
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
                                                    dcc.Tab(
                                                        label='Fmax',
                                                        children=[],
                                                        id='nmf-test-fmax'
                                                    ),
                                                    dcc.Tab(
                                                        label='Reconstruction error',
                                                        children=[],
                                                        id='nmf-test-error'
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
                                                                                    id='nmf-test-pred-ref-selection'
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
                                                                                    id='nmf-test-pred-model'
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
                                                                                    id='nmf-test-pred-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    config={'autosizable': False},
                                                                                    style={'width': 1700, 'height': 800}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='nmf-test-pred-table'
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
                                                                                        {'label': 'NMF-Fmax',
                                                                                         'value': 'NMF-Fmax'},
                                                                                        {'label': 'NNLS-Fmax',
                                                                                         'value': 'NNLS-Fmax'},
                                                                                    ],
                                                                                    id='nmf-test-corr-indicator-selection'
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
                                                                                    id='nmf-test-corr-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Row([
                                                                                dcc.Graph(
                                                                                    id='nmf-test-corr-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    style={'width': '700',
                                                                                           'height': '900'}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='nmf-test-corr-table'
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
                            label='Predict', id='nmf-predict'
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

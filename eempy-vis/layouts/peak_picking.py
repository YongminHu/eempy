from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc


# -----------Page #2: Peak picking--------------
# First setup the dbc cards, then the general layout in the end.

#   -------------Setting up the dbc cards

#       -----------------dbc card for peak-picking parameters
card_pp_param = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Import EEM dataset for model establishment", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                dcc.Input(id='pp-eem-dataset-establishment-path-input', type='text', value=None,
                                          placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                      ' If empty, the model built in "eem pre-processing" '
                                                      'would be used',
                                          style={'width': '97%', 'height': '30px'}, debounce=True),
                                justify="center"
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div([],
                                             id='pp-eem-dataset-establishment-message', style={'width': '80vw'}),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 2}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='pp-establishment-index-kw-mandatory', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 2, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='pp-establishment-index-kw-optional', type='text',
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
                                        dbc.Label("Excitation wavelength"), width={'size': 2}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='pp-excitation', type='number',
                                                  placeholder='nm',
                                                  style={'width': '250px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Emission wavelength"), width={'size': 2, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='pp-emission', type='number',
                                                  placeholder='nm',
                                                  style={'width': '250px', 'height': '30px'}, debounce=True),
                                        width={'size': 1}
                                    ),
                                ]
                            ),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='build-pp-spinner')],
                                               id='build-pp-model', className='col-2')
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


layout = html.Div([
    dbc.Stack(
        [
            dbc.Row(
                card_pp_param
            ),
            dbc.Row(
                dcc.Tabs(
                    id='pp-results',
                    children=[
                        dcc.Tab(label='Intensities', id='pp-intensities'),
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
                                                                width={'size': 1, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id='pp-establishment-corr-ref-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(id='pp-establishment-corr-graph',
                                                                  # config={'responsive': 'auto'},
                                                                  style={'width': '45vw', 'height': '60vh'}
                                                                  ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div([],
                                                                 id='pp-establishment-corr-table')
                                                    ),

                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Correlations', id='pp-establishment-corr'
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
                                                        dcc.Input(id='pp-eem-dataset-predict-path-input',
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
                                                                     id='pp-eem-dataset-predict-message',
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
                                                                dcc.Input(id='pp-test-index-kw-mandatory',
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
                                                                dcc.Input(id='pp-test-index-kw-optional',
                                                                          type='text',
                                                                          placeholder='',
                                                                          style={'width': '100%', 'height': '30px'},
                                                                          debounce=True, value=''),
                                                                width={"offset": 0, "size": 2}
                                                            )
                                                        ]
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm",
                                                                                 id='pp-predict-spinner')],
                                                                    id='predict-pp-model', className='col-2')
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
                                                        label='Intensities',
                                                        children=[],
                                                        id='pp-test-intensities'
                                                    ),
                                                    dcc.Tab(
                                                        label='Error',
                                                        children=[],
                                                        id='pp-test-error'
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
                                                                                    id='pp-test-pred-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Dropdown(
                                                                                    options=[
                                                                                        {
                                                                                            'label': 'Linear least squares',
                                                                                            'value': 'linear_least_squares'},
                                                                                    ],
                                                                                    id='pp-test-pred-model'
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
                                                                                    id='pp-test-pred-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    config={'autosizable': False},
                                                                                    style={'width': 1700, 'height': 800}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='pp-test-pred-table'
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
                                                                                        {'label': 'Intensities',
                                                                                         'value': 'Intensities'},
                                                                                    ],
                                                                                    id='pp-test-corr-indicator-selection'
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
                                                                                    id='pp-test-corr-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Row([
                                                                                dcc.Graph(
                                                                                    id='pp-test-corr-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    style={'width': '700',
                                                                                           'height': '900'}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='pp-test-corr-table'
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
                            label='Predict', id='pp-predict'
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
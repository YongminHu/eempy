from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc

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
                                dcc.Input(id='ri-eem-dataset-establishment-path-input', type='text', value=None,
                                          placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                      ' If empty, the model built in "eem pre-processing" '
                                                      'would be used',
                                          style={'width': '97%', 'height': '30px'}, debounce=True),
                                justify="center"
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div([],
                                             id='ri-eem-dataset-establishment-message', style={'width': '80vw'}),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 2}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='ri-establishment-index-kw-mandatory', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 2, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='ri-establishment-index-kw-optional', type='text',
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
                                        dbc.Label("Ex range min"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='ri-ex-min', type='number',
                                                  placeholder='nm',
                                                  style={'width': '100px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Ex range max"), width={'size': 1, 'offset': 0}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='ri-ex-max', type='number',
                                                  placeholder='nm',
                                                  style={'width': '100px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Em range min"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='ri-em-min', type='number',
                                                  placeholder='nm',
                                                  style={'width': '100px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Em range max"), width={'size': 1, 'offset': 0}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='ri-em-max', type='number',
                                                  placeholder='nm',
                                                  style={'width': '100px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),
                                ]
                            ),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='build-ri-spinner')],
                                               id='build-ri-model', className='col-2')
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


page_regional_integration = html.Div([
    dbc.Stack(
        [
            dbc.Row(
                card_ri_param
            ),
            dbc.Row(
                dcc.Tabs(
                    id='ri-results',
                    children=[
                        dcc.Tab(label='Intensities', id='ri-intensities'),
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
                                                                    id='ri-establishment-corr-ref-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(id='ri-establishment-corr-graph',
                                                                  # config={'responsive': 'auto'},
                                                                  style={'width': '45vw', 'height': '60vh'}
                                                                  ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div([],
                                                                 id='ri-establishment-corr-table')
                                                    ),

                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Correlations', id='ri-establishment-corr'
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
                                                        dcc.Input(id='ri-eem-dataset-predict-path-input',
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
                                                                     id='ri-eem-dataset-predict-message',
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
                                                                dcc.Input(id='ri-test-index-kw-mandatory',
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
                                                                dcc.Input(id='ri-test-index-kw-optional',
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
                                                                                 id='ri-predict-spinner')],
                                                                    id='predict-ri-model', className='col-2')
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
                                                        id='ri-test-intensities'
                                                    ),
                                                    dcc.Tab(
                                                        label='Error',
                                                        children=[],
                                                        id='ri-test-error'
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
                                                                                    id='ri-test-pred-ref-selection'
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
                                                                                    id='ri-test-pred-model'
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
                                                                                    id='ri-test-pred-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    config={'autosizable': False},
                                                                                    style={'width': 1700, 'height': 800}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='ri-test-pred-table'
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
                                                                                    id='ri-test-corr-indicator-selection'
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
                                                                                    id='ri-test-corr-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Row([
                                                                                dcc.Graph(
                                                                                    id='ri-test-corr-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    style={'width': '700',
                                                                                           'height': '900'}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='ri-test-corr-table'
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
                            label='Predict', id='ri-predict'
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
import os
import dash
import datetime
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

from eempy.plot import plot_eem, plot_abs
from eempy.read_data import *
from eempy.eem_processing import *
from eempy.utils import string_to_list
from matplotlib import pyplot as plt

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# -----------Global variables--------------

# eem_dataset_working = None  # The EEM dataset that the user would build and analyse

# -----------Page #0: Homepage


# -----------Page #1: EEM pre-processing--------------

#   -------------Setting up the dbc cards

#       -----------dbc card for Selecting working folder and choosing EEM for previewing

card_selecting_files = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Select files", className="card-title"),
            dbc.Stack(
                [
                    dbc.Row(
                        dcc.Input(id='folder-path-input', type='text',
                                  placeholder='Please enter the data folder path...',
                                  style={'width': '97%', 'height': '30px'}, debounce=True),
                        justify="center"
                    ),

                    html.H6("Data format"),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("EEM file format"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='eem-data-format',
                                options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                value=None,
                                style={'width': '100%'},
                                optionHeight=50
                            ),
                            width={'offset': 0, 'size': 3},
                        ),
                        dbc.Col(
                            dbc.Label("Absorbance file format"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='abs-data-format',
                                options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                value=None,
                                style={'width': '100%'},
                                optionHeight=50
                            ),
                            width={"offset": 0, "size": 3},

                        ),
                    ]),

                    html.H6("File searching keywords"),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Main body (mandatory)"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Input(id='file-keyword-mandatory', type='text', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                            width={"offset": 0, "size": 3}
                        ),
                        dbc.Col(
                            dbc.Label("Main body (optional)"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Input(id='file-keyword-optional', type='text', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                            width={"offset": 0, "size": 3}
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Sample EEM"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Input(id='file-keyword-sample', type='text', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                            width={"offset": 0, "size": 3}
                        ),
                        dbc.Col(
                            dbc.Label("Absorbance"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Input(id='file-keyword-absorbance', type='text', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                            width={"offset": 0, "size": 3}
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Blank EEM"),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dcc.Input(id='file-keyword-blank', type='text', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                            width={"offset": 0, "size": 3}
                        ),
                    ]),

                    html.H6("Index extraction from filenames"),

                    dbc.Row([

                        dbc.Col(
                            dbc.Label("Index starting position"),
                        ),
                        dbc.Col(
                            dcc.Input(id='index-pos-left', type='number', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True),
                        ),
                        dbc.Col(
                            dbc.Label("Index end position"),
                        ),
                        dbc.Col(
                            dcc.Input(id='index-pos-right', type='number', placeholder='',
                                      style={'width': '100%', 'height': '30px'}, debounce=True),
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Checklist(options=[{'label': html.Span("Read index as timestamp",
                                                                       style={"font-size": 15, "padding-left": 10}),
                                                    'value': 'timestamp'}],
                                          id='timestamp-checkbox', switch=True),
                            width={"size": 6}
                        ),
                        dbc.Col(
                            dbc.Label("timestamp format"),
                        ),
                        dbc.Col(
                            dcc.Input(id='timestamp-format', type='text', placeholder='e.g., %Y-%m-%d-%H-%M',
                                      style={'width': '100%', 'height': '30px'}, debounce=True),
                        )
                    ]),
                ],
                gap=1
            ),
        ],
        className="w-100")
)

#       -----------dbc card for EEM display

card_eem_display = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Data preview", className="card-title"),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(id='filename-sample-dropdown', options=[], placeholder='Please select a '
                                                                                                    'eem file...'),
                            ),
                        ],
                        justify='end'
                    ),

                    dbc.Row([
                        dcc.Graph(id='eem-graph'),
                    ]),

                    dbc.Row([
                        dcc.Graph(id='absorbance-graph'),
                    ])

                ]
            ),
        ]
    ),
    className='w-100'
)

#       -----------dbc card for Selecting excitation/emission/intensity display ranges

card_range_selection = dbc.Card(
    dbc.CardBody(
        [
            dbc.Stack(
                [
                    html.H5("Ex/Em/intensity ranges"),
                    html.H6("Excitation wavelength", className="card-title"),
                    html.Div(
                        [
                            dbc.Row([
                                dbc.Col(
                                    dcc.Input(id='excitation-wavelength-min', type='number', placeholder='min',
                                              style={'width': '120%', 'height': '30px'}, debounce=True, value=240),
                                    width={'size': 3}
                                ),
                                dbc.Col(
                                    dbc.Label("nm"),
                                    width={'size': 1}
                                ),
                                dbc.Col(
                                    html.Span("-",
                                              style={"font-size": 15, "padding-left": 20}),
                                    width={'offset': 0, 'size': 2}
                                ),
                                dbc.Col(
                                    dcc.Input(id='excitation-wavelength-max', type='number', placeholder='max',
                                              style={'width': '120%', 'height': '30px'}, debounce=True, value=500),
                                    width={'size': 3}
                                ),
                                dbc.Col(
                                    dbc.Label("nm"),
                                    width={'size': 1}
                                )
                            ],
                                justify='start'
                            ),
                        ],
                    ),
                    html.H6("Emission wavelength", className="card-title"),
                    html.Div(
                        [
                            dbc.Row([
                                dbc.Col(
                                    dcc.Input(id='emission-wavelength-min', type='number', placeholder='min',
                                              style={'width': '120%', 'height': '30px'}, debounce=True, value=300),
                                    width={'size': 3}
                                ),
                                dbc.Col(
                                    dbc.Label("nm"),
                                    width={'size': 1}
                                ),
                                dbc.Col(
                                    html.Span("-",
                                              style={"font-size": 15, "padding-left": 20}),
                                    width={'offset': 0, 'size': 2}
                                ),
                                dbc.Col(
                                    dcc.Input(id='emission-wavelength-max', type='number', placeholder='max',
                                              style={'width': '120%', 'height': '30px'}, debounce=True, value=800),
                                    width={'size': 3}
                                ),
                                dbc.Col(
                                    dbc.Label("nm"),
                                    width={'size': 1}
                                )
                            ],
                                justify='start'
                            ),
                        ],
                    ),

                    html.H6("Fluorescence intensity", className="card-title"),
                    html.Div(
                        [
                            dbc.Row([
                                dbc.Col(
                                    dcc.Input(id='fluorescence-intensity-min', type='number', placeholder='min',
                                              style={'width': '120%', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 3}
                                ),
                                dbc.Col(
                                    dbc.Label("a.u."),
                                    width={'size': 1}
                                ),
                                dbc.Col(
                                    html.Span("-",
                                              style={"font-size": 15, "padding-left": 20}),
                                    width={'offset': 0, 'size': 2}
                                ),
                                dbc.Col(
                                    dcc.Input(id='fluorescence-intensity-max', type='number', placeholder='max',
                                              style={'width': '120%', 'height': '30px'}, debounce=True, value=1000),
                                    width={'size': 3}
                                ),
                                dbc.Col(
                                    dbc.Label("a.u."),
                                    width={'size': 1}
                                )
                            ],
                                justify='start'
                            ),
                        ],
                    ),
                ],
                gap=1)

        ]
    ),
    className="w-100",
)

#       -----------dbc card for automatic Raman scattering unit normalization

card_su = dbc.Card(
    dbc.CardBody(
        [
            dbc.Stack(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Raman scattering unit (RSU) normalization from blank"),
                                ],
                                width={'size': 10}),
                            dbc.Col([
                                dbc.Checklist(id='su-button', options=[{'label': ' ', 'value': "su"}], switch=True,
                                              style={'transform': 'scale(1.3)'})
                            ], width={'size': 2})
                        ],
                        align='start'
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("RS Ex"),
                            width={'size': 6}
                        ),
                        dbc.Col(
                            dcc.Input(id='su-excitation', type='number', placeholder='max',
                                      style={'width': '120%', 'height': '30px'}, debounce=True, value=350),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dbc.Label('nm'),
                        )
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("RS Em bandwidth"),
                            width={'size': 6}
                        ),
                        dbc.Col(
                            dbc.Col(
                                dcc.Input(id='su-emission-width', type='number', placeholder='max',
                                          style={'width': '120%', 'height': '30px'}, debounce=True, value=5)
                            )
                        ),
                        dbc.Col(
                            dbc.Label("nm"),
                        )
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Normalization factor")
                        ),
                        dbc.Col(
                            dbc.Col(
                                dcc.Input(id='su-normalization-factor', type='number', placeholder='max',
                                          style={'width': '100%', 'height': '30px'}, debounce=True, value=1000),
                                width={'size': 6}
                            )
                        ),
                    ]),

                    dbc.Row([
                        html.Div(id='rsu-display')
                    ])

                ], gap=1
            ),
        ]
    ),
    className="w-100"
)

#       -----------dbc card for IFE correction

card_ife = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Inner filter effect correction"),
                                ],
                                width={'size': 10}),
                            dbc.Col([
                                dbc.Checklist(id='ife-button', options=[{'label': ' ', 'value': "ife"}], switch=True,
                                              style={'transform': 'scale(1.3)'})
                            ], width={'size': 2})
                        ],
                        align='start'
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("method"),
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='ife-methods', options=[{'label': 'default', 'value': 'default'}],
                                         value='default', placeholder='Select IFE correction method',
                                         style={'width': '100%'})
                        )
                    ]),

                ]
            ),
        ]
    ),
    className="w-100"
)

#       -----------dbc card for Raman scattering interpolation

card_raman = dbc.Card(
    dbc.CardBody(
        [
            dbc.Stack(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Raman scattering removal"),
                                ],
                                width={'size': 10}),
                            dbc.Col([
                                dbc.Checklist(id='raman-button', options=[{'label': ' ', 'value': "raman"}],
                                              switch=True,
                                              style={'transform': 'scale(1.3)'})
                            ], width={'size': 2})
                        ],
                        align='start'
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("method"),
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='raman-methods',
                                         options=[{'label': 'linear', 'value': 'linear'},
                                                  {'label': 'cubic', 'value': 'cubic'},
                                                  {'label': 'nan', 'value': 'nan'}],
                                         value='linear', placeholder='', style={'width': '100%'})
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("dimension")
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='raman-dimension',
                                         options=[{'label': '1d-ex', 'value': '1d-ex'},
                                                  {'label': '1d-em', 'value': '1d-em'},
                                                  {'label': '2d', 'value': '2d'}],
                                         value='2d', placeholder='', style={'width': '100%'})
                        )
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("width")
                        ),
                        dbc.Col(
                            dcc.Input(id='raman-width',
                                      type='number', placeholder='max',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=15)
                        ),
                    ]),
                ],
                gap=1
            )
        ]
    )
)

#       -----------dbc card for Rayleigh scattering interpolation

card_rayleigh = dbc.Card(
    dbc.CardBody(
        [
            dbc.Stack(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Rayleigh scattering removal"),
                                ],
                                width={'size': 10}),
                            dbc.Col([
                                dbc.Checklist(id='rayleigh-button', options=[{'label': ' ', 'value': "rayleigh"}],
                                              switch=True,
                                              style={'transform': 'scale(1.3)'})
                            ], width={'size': 2})
                        ],
                        align='start'
                    ),

                    dbc.Row([
                        html.H6("First order")
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("method"),
                            ),
                            dbc.Col(
                                dcc.Dropdown(id='rayleigh-o1-methods',
                                             options=[{'label': 'zero', 'value': 'zero'},
                                                      {'label': 'linear', 'value': 'linear'},
                                                      {'label': 'cubic', 'value': 'cubic'},
                                                      {'label': 'nan', 'value': 'nan'},
                                                      {'label': 'none', 'value': 'none'}],
                                             value='zero', placeholder='', style={'width': '100%'})
                            ),
                        ]
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("dimension")
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='rayleigh-o1-dimension',
                                options=[{'label': '1d-ex', 'value': '1d-ex'},
                                         {'label': '1d-em', 'value': '1d-em'},
                                         {'label': '2d', 'value': '2d'}],
                                value='2d', placeholder='', style={'width': '100%'})
                        )
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("width")
                            ),
                            dbc.Col(
                                dcc.Input(id='rayleigh-o1-width',
                                          type='number', placeholder='max',
                                          style={'width': '100%', 'height': '30px'}, debounce=True, value=20)
                            ),
                        ]
                    ),

                    dbc.Row([
                        html.H6("Second order")
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("method"),
                            ),
                            dbc.Col(
                                dcc.Dropdown(id='rayleigh-o2-methods',
                                             options=[{'label': 'zero', 'value': 'zero'},
                                                      {'label': 'linear', 'value': 'linear'},
                                                      {'label': 'cubic', 'value': 'cubic'},
                                                      {'label': 'nan', 'value': 'nan'},
                                                      {'label': 'none', 'value': 'none'}],
                                             value='linear', placeholder='', style={'width': '100%'})
                            ),
                        ]
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("dimension")
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='rayleigh-o2-dimension',
                                         options=[{'label': '1d-ex', 'value': '1d-ex'},
                                                  {'label': '1d-em', 'value': '1d-em'},
                                                  {'label': '2d', 'value': '2d'}],
                                         value='2d', placeholder='', style={'width': '100%'})
                        )
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("width")
                            ),
                            dbc.Col(
                                dcc.Input(id='rayleigh-o2-width',
                                          type='number', placeholder='max',
                                          style={'width': '100%', 'height': '30px'}, debounce=True, value=20)
                            ),
                        ]
                    )

                ], gap=1
            ),
        ]
    ),
    className="w-100"
)

#       -----------dbc card for Gaussian smoothing

card_smoothing = dbc.Card(
    dbc.CardBody(
        [
            dbc.Stack(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Gaussian Smoothing"),
                                ],
                                width={'size': 10}),
                            dbc.Col([
                                dbc.Checklist(id='gaussian-button', options=[{'label': ' ', 'value': "gaussian"}],
                                              switch=True,
                                              style={'transform': 'scale(1.3)'})
                            ], width={'size': 2})
                        ],
                        align='start'
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("sigma"),
                        ),
                        dbc.Col(
                            dcc.Input(id='gaussian-sigma',
                                      type='number', placeholder='max',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=1),
                        ),
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("truncate")
                            ),
                            dbc.Col(
                                dcc.Input(id='gaussian-truncate',
                                          type='number', placeholder='max',
                                          style={'width': '100%', 'height': '30px'}, debounce=True, value=3),
                            )
                        ])
                ],
                gap=1
            )
        ]
    ),
    className="w-100"
)

#       -----------dbc card for downloading pre-processed EEM

card_eem_downloading = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Export pre-processed EEM", className="card-title"),
            dbc.Stack([
                dbc.Row([
                    dbc.Col(
                        [
                            dbc.Label("format"),
                        ],
                        width={'size': 5}
                    ),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="eem-downloading-format",
                                options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                value='aqualog', placeholder='Select EEM data file format',
                                style={'width': '100%'}, optionHeight=50)
                        ]
                    ),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Button("Export", id='export-eem', className='col-5')
                            )])
            ], gap=2)
        ]
    ),
    className="w-100"
)

#       -----------dbc card for building EEM dataset

card_built_eem_dataset = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Build EEM dataset", className="card-title"),
            dbc.Stack([
                dbc.Row([
                    dcc.Checklist(
                        id="align-exem",
                        options=[{
                            'label': html.Span('Align Ex/Em (select if the Ex/Em intervals are different between EEMs)',
                                               style={"font-size": 15, "padding-left": 10}),
                            'value': 'align'}])
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Button([dbc.Spinner(size="sm", id='build-eem-dataset-spinner')],
                                   id='build-eem-dataset', className='col-5')
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        html.Div(id='info-eem-dataset', style={'width': '300px'}),
                        width={"size": 12, "offset": 0}
                    )
                ])
            ], gap=2)
        ]
    ),
    className="w-100"
)

#   -------------Layout of page #1

page1 = html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Stack(
                        [
                            card_selecting_files,
                            card_eem_display
                        ],
                        gap=3
                    )
                ],
                width={"size": 6, "offset": 0},
            ),
            dbc.Col(
                [
                    dbc.Stack(
                        [
                            card_range_selection,
                            card_su,
                            card_ife,
                            card_raman
                        ],
                        gap=3)
                ],
                width={"size": 3}
            ),
            dbc.Col(
                [
                    dbc.Stack(
                        [
                            card_rayleigh,
                            card_smoothing,
                            card_eem_downloading,
                            card_built_eem_dataset
                        ],
                        gap=3)
                ],
                width={"size": 3}
            )
        ]
    )
])


# -------------Callbacks of page #1

#   ---------------Update file list according to input data folder path


@app.callback(
    [
        Output('filename-sample-dropdown', 'options'),
        Output('filename-sample-dropdown', 'value')
    ],
    [
        Input('folder-path-input', 'value'),
        Input('file-keyword-mandatory', 'value'),
        Input('file-keyword-optional', 'value'),
        Input('file-keyword-sample', 'value'),
    ]
)
def update_filenames(folder_path, kw_mandatory, kw_optional, kw_sample):
    if folder_path:
        try:
            filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if kw_mandatory or kw_optional or kw_sample:
                kw_mandatory = string_to_list(kw_mandatory) if kw_mandatory else []
                kw_optional = string_to_list(kw_optional) if kw_optional else []
                kw_sample = string_to_list(kw_sample) if kw_sample else []
                filenames = get_filelist(folder_path, kw_mandatory + kw_sample, kw_optional)
            options = [{'label': f, 'value': f} for f in filenames]
            return options, None
        except FileNotFoundError:
            return ['Files not found - please check the settings'], None
    else:
        return [], None


#   ---------------Update Data plot with changes of parameters

@app.callback(
    [
        Output('eem-graph', 'figure'),
        Output('absorbance-graph', 'figure'),
        Output('rsu-display', 'children')
    ],
    [
        Input('folder-path-input', 'value'),
        Input('filename-sample-dropdown', 'value'),
        Input('eem-data-format', 'value'),
        Input('abs-data-format', 'value'),
        Input('file-keyword-sample', 'value'),
        Input('file-keyword-absorbance', 'value'),
        Input('file-keyword-blank', 'value'),
        Input('index-pos-left', 'value'),
        Input('index-pos-right', 'value'),
        Input('excitation-wavelength-min', 'value'),
        Input('excitation-wavelength-max', 'value'),
        Input('emission-wavelength-min', 'value'),
        Input('emission-wavelength-max', 'value'),
        Input('fluorescence-intensity-min', 'value'),
        Input('fluorescence-intensity-max', 'value'),
        Input('su-button', 'value'),
        Input('su-excitation', 'value'),
        Input('su-emission-width', 'value'),
        Input('su-normalization-factor', 'value'),
        Input('ife-button', 'value'),
        Input('ife-methods', 'value'),
        Input('raman-button', 'value'),
        Input('raman-methods', 'value'),
        Input('raman-dimension', 'value'),
        Input('raman-width', 'value'),
        Input('rayleigh-button', 'value'),
        Input('rayleigh-o1-methods', 'value'),
        Input('rayleigh-o1-dimension', 'value'),
        Input('rayleigh-o1-width', 'value'),
        Input('rayleigh-o2-methods', 'value'),
        Input('rayleigh-o2-dimension', 'value'),
        Input('rayleigh-o2-width', 'value'),
        Input('gaussian-button', 'value'),
        Input('gaussian-sigma', 'value'),
        Input('gaussian-truncate', 'value'),
    ]
)
def update_eem_plot(folder_path, file_name_sample,
                    eem_data_format, abs_data_format,
                    file_kw_sample, file_kw_abs, file_kw_blank,
                    index_pos_left, index_pos_right,
                    ex_range_min, ex_range_max, em_range_min, em_range_max, intensity_range_min, intensity_range_max,
                    su, su_ex, su_em_width, su_normalization_factor,
                    ife, ife_method,
                    raman, raman_method, raman_dimension, raman_width,
                    rayleigh, rayleigh_o1_method, rayleigh_o1_dimension, rayleigh_o1_width,
                    rayleigh_o2_method, rayleigh_o2_dimension, rayleigh_o2_width,
                    gaussian, gaussian_sigma, gaussian_truncate):
    try:
        full_path_sample = os.path.join(folder_path, file_name_sample)
        intensity, ex_range, em_range, index = read_eem(full_path_sample,
                                                        index_pos=(index_pos_left, index_pos_right)
                                                        if (index_pos_left and index_pos_right) else None,
                                                        data_format=eem_data_format)
        # Cut EEM
        if (ex_range_min and ex_range_max) or (em_range_min and em_range_max):
            ex_range_min = ex_range_min if ex_range_min else np.min(ex_range)
            ex_range_max = ex_range_max if ex_range_max else np.max(ex_range)
            em_range_min = em_range_min if em_range_min else np.min(em_range)
            em_range_max = em_range_max if em_range_max else np.max(em_range)
            intensity, ex_range, em_range = eem_cutting(intensity, ex_range, em_range, ex_range_min, ex_range_max,
                                                        em_range_min, em_range_max)

        # Scattering unit normalization
        if file_kw_blank and su:
            file_name_blank = file_name_sample.replace(file_kw_sample, file_kw_blank)
            full_path_blank = os.path.join(folder_path, file_name_blank)
            intensity_blank, ex_range_blank, em_range_blank, _ = read_eem(full_path_blank, data_format=eem_data_format)
            intensity, rsu_value = eem_raman_normalization(intensity, blank=intensity_blank,
                                                           ex_range_blank=ex_range_blank,
                                                           em_range_blank=em_range_blank, from_blank=True,
                                                           ex_target=su_ex,
                                                           bandwidth=su_em_width, rsu_standard=su_normalization_factor)
        else:
            rsu_value = None

        # IFE correction
        if file_kw_abs and ife:
            file_name_abs = file_name_sample.replace(file_kw_sample, file_kw_abs)
            full_path_abs = os.path.join(folder_path, file_name_abs)
            absorbance, ex_range_abs, _ = read_abs(full_path_abs, data_format=abs_data_format)
            intensity = eem_ife_correction(intensity, ex_range, em_range, absorbance, ex_range_abs)

        # Raman scattering removal
        if all([raman, raman_method, raman_width, raman_dimension]):
            intensity, _ = eem_raman_scattering_removal(intensity, ex_range, em_range,
                                                        interpolation_method=raman_method,
                                                        width=raman_width, interpolation_dimension=raman_dimension)

        # Rayleigh scattering removal
        if all([rayleigh, rayleigh_o1_method, rayleigh_o1_dimension, rayleigh_o1_width,
                rayleigh_o2_method, rayleigh_o2_dimension, rayleigh_o2_width]):
            intensity, _, _ = eem_rayleigh_scattering_removal(intensity, ex_range, em_range,
                                                              width_o1=rayleigh_o1_width, width_o2=rayleigh_o2_width,
                                                              interpolation_method_o1=rayleigh_o1_method,
                                                              interpolation_method_o2=rayleigh_o2_method,
                                                              interpolation_dimension_o1=rayleigh_o1_dimension,
                                                              interpolation_dimension_o2=rayleigh_o2_dimension)

        # Gaussian smoothing
        if all([gaussian, gaussian_sigma, gaussian_truncate]):
            intensity = eem_gaussian_filter(intensity, gaussian_sigma, gaussian_truncate)

        # Plot EEM
        fig_eem = plot_eem(intensity, ex_range, em_range, vmin=intensity_range_min, vmax=intensity_range_max,
                           plot_tool='plotly', display=False, auto_intensity_range=False, cmap='jet',
                           figure_size=(7, 4), fix_aspect_ratio=True, title=index if index else None)

        # Plot absorbance (if exists)
    except:
        # Create an empty scatter plot
        fig_eem = go.Figure()

        # Add a black border
        fig_eem.update_layout(
            xaxis=dict(showline=False, linewidth=0, linecolor="black"),
            yaxis=dict(showline=False, linewidth=0, linecolor="black"),
            width=700,
            height=400,
            margin=dict(l=50, r=50, b=50, t=50),
        )

        # Add centered text annotation
        fig_eem.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="EEM file or parameters missing",
            showarrow=False,
            font=dict(size=16),
        )

        rsu_value = None

    try:
        file_name_abs = file_name_sample.replace(file_kw_sample, file_kw_abs)
        full_path_abs = os.path.join(folder_path, file_name_abs)
        absorbance, ex_range_abs, _ = read_abs(full_path_abs, data_format=abs_data_format)
        fig_abs = plot_abs(absorbance, ex_range_abs, figure_size=(7, 2.5), plot_tool='plotly', display=False)
    except:
        fig_abs = go.Figure()

        fig_abs.update_layout(
            xaxis=dict(showline=False, linewidth=0, linecolor="black"),
            yaxis=dict(showline=False, linewidth=0, linecolor="black"),
            width=700,
            height=200,
            margin=dict(l=50, r=50, b=50, t=50)
        )

        # Add centered text annotation
        fig_abs.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="Absorbance file or parameters missing",
            showarrow=False,
            font=dict(size=16),
        )

    return fig_eem, fig_abs, f'RSU: {rsu_value}'


#   ---------------Export EEM


#   ---------------Build EEM dataset

@app.callback(
    [
        Output('eem-dataset', 'data'),
        Output('info-eem-dataset', 'children'),
        Output('build-eem-dataset-spinner', 'children')
    ],
    [
        Input('build-eem-dataset', 'n_clicks'),
        State('folder-path-input', 'value'),
        State('eem-data-format', 'value'),
        State('abs-data-format', 'value'),
        State('filename-sample-dropdown', 'options'),
        State('file-keyword-sample', 'value'),
        State('file-keyword-absorbance', 'value'),
        State('file-keyword-blank', 'value'),
        State('index-pos-left', 'value'),
        State('index-pos-right', 'value'),
        State('timestamp-checkbox', 'value'),
        State('timestamp-format', 'value'),
        State('excitation-wavelength-min', 'value'),
        State('excitation-wavelength-max', 'value'),
        State('emission-wavelength-min', 'value'),
        State('emission-wavelength-max', 'value'),
        State('su-button', 'value'),
        State('su-excitation', 'value'),
        State('su-emission-width', 'value'),
        State('su-normalization-factor', 'value'),
        State('ife-button', 'value'),
        State('ife-methods', 'value'),
        State('raman-button', 'value'),
        State('raman-methods', 'value'),
        State('raman-dimension', 'value'),
        State('raman-width', 'value'),
        State('rayleigh-button', 'value'),
        State('rayleigh-o1-methods', 'value'),
        State('rayleigh-o1-dimension', 'value'),
        State('rayleigh-o1-width', 'value'),
        State('rayleigh-o2-methods', 'value'),
        State('rayleigh-o2-dimension', 'value'),
        State('rayleigh-o2-width', 'value'),
        State('gaussian-button', 'value'),
        State('gaussian-sigma', 'value'),
        State('gaussian-truncate', 'value'),
        State('align-exem', 'value'),
    ]
)
def on_build_eem_dataset(n_clicks,
                         folder_path,
                         eem_data_format, abs_data_format,
                         file_name_sample_options, file_kw_sample, file_kw_abs, file_kw_blank,
                         index_pos_left, index_pos_right, timestamp, timestamp_format,
                         ex_range_min, ex_range_max, em_range_min, em_range_max,
                         su, su_ex, su_em_width, su_normalization_factor,
                         ife, ife_method,
                         raman, raman_method, raman_dimension, raman_width,
                         rayleigh, rayleigh_o1_method, rayleigh_o1_dimension, rayleigh_o1_width,
                         rayleigh_o2_method, rayleigh_o2_dimension, rayleigh_o2_width,
                         gaussian, gaussian_sigma, gaussian_truncate,
                         align_exem
                         ):
    if n_clicks is None:
        return None, None, "Build"
        # raise PreventUpdate
    try:
        file_name_sample_options = file_name_sample_options or {}
        file_name_sample_list = [f['value'] for f in file_name_sample_options]
        eem_stack, ex_range, em_range, indexes = read_eem_dataset(
            folder_path=folder_path, mandatory_keywords=[], optional_keywords=[], data_format=eem_data_format,
            index_pos=(index_pos_left, index_pos_right) if index_pos_left and index_pos_right else None,
            custom_filename_list=file_name_sample_list, wavelength_alignment=True if align_exem else False,
            interpolation_method='linear'
        )
    except (UnboundLocalError, IndexError) as e:
        error_message = ("EEM dataset building failed. Are there any non-EEM files mixed in? "
                         "Please check data folder path and file searching keywords settings. If the Ex/Em "
                         "ranges/intervals are different between EEMs, make sure you select the 'Align Ex/Em' checkbox.")
        return None, error_message, "Build"

    steps_track = [
        "EEM dataset building successful!\n",
        "Number of EEMs: {n}\n".format(n=eem_stack.shape[0]),
        "Pre-processing steps implemented:\n",
    ]
    # EEM cutting
    eem_dataset = EEMDataset(eem_stack, ex_range, em_range, index=indexes)
    if any([np.min(ex_range) != ex_range_min, np.max(ex_range) != ex_range_max,
            np.min(em_range) != em_range_min, np.max(em_range) != em_range_max]):
        eem_dataset.cutting(ex_min=ex_range_min, ex_max=ex_range_max,
                            em_min=em_range_min, em_max=em_range_max, copy=False)
        steps_track += "- EEM cutting \n"

    # RSU normalization
    if file_kw_blank and su:
        file_name_blank_list = [f.replace(file_kw_sample, file_kw_blank) for f in file_name_sample_list]
        blank_stack, ex_range_blank, em_range_blank, _ = read_eem_dataset(
            folder_path=folder_path, mandatory_keywords=[], optional_keywords=[], data_format=eem_data_format,
            index_pos=None,
            custom_filename_list=file_name_blank_list, wavelength_alignment=True if align_exem else False,
            interpolation_method='linear'
        )
        eem_dataset.raman_normalization(blank=blank_stack,
                                        ex_range_blank=ex_range_blank,
                                        em_range_blank=em_range_blank, from_blank=True,
                                        ex_target=su_ex,
                                        bandwidth=su_em_width, rsu_standard=su_normalization_factor,
                                        copy=False)
        steps_track += ["- Raman scattering unit normalization\n"]

    # IFE correction
    if file_kw_abs and ife:
        file_name_abs_list = [f.replace(file_kw_sample, file_kw_abs) for f in file_name_sample_list]
        abs_stack, ex_range_abs, _ = read_abs_dataset(
            folder_path=folder_path, mandatory_keywords=[], optional_keywords=[], data_format=abs_data_format,
            index_pos=None,
            custom_filename_list=file_name_abs_list, wavelength_alignment=True if align_exem else False,
            interpolation_method='linear'
        )
        eem_dataset.ife_correction(absorbance=abs_stack, ex_range_abs=ex_range_abs, copy=False)
        steps_track += ["- Inner filter effect correction\n"]

    # Raman scattering removal
    if all([raman, raman_method, raman_width, raman_dimension]):
        eem_dataset.raman_scattering_removal(interpolation_method=raman_method, width=raman_width,
                                             interpolation_dimension=raman_dimension, copy=False)
        steps_track += ["- Raman scattering removal\n"]

    # Rayleigh scattering removal
    if all([rayleigh, rayleigh_o1_method, rayleigh_o1_dimension, rayleigh_o1_width,
            rayleigh_o2_method, rayleigh_o2_dimension, rayleigh_o2_width]):
        eem_dataset.rayleigh_scattering_removal(width_o1=rayleigh_o1_width, width_o2=rayleigh_o2_width,
                                                interpolation_method_o1=rayleigh_o1_method,
                                                interpolation_method_o2=rayleigh_o2_method,
                                                interpolation_dimension_o1=rayleigh_o1_dimension,
                                                interpolation_dimension_o2=rayleigh_o2_dimension,
                                                copy=False)
        steps_track += ["- Rayleigh scattering removal\n"]

    # Gaussian smoothing
    if all([gaussian, gaussian_sigma, gaussian_truncate]):
        eem_dataset.gaussian_filter(sigma=gaussian_sigma, truncate=gaussian_truncate)
        steps_track += ["- Gaussian smoothing\n"]

    # convert eem_dataset to a dict whose values are json serializable
    eem_dataset_json_dict = {
        'eem_stack': eem_dataset.eem_stack.tolist(),
        'ex_range': eem_dataset.ex_range.tolist(),
        'em_range': eem_dataset.em_range.tolist(),
        'index': eem_dataset.index,
        'ref': eem_dataset.ref.tolist() if eem_dataset.ref is not None else None
    }

    return eem_dataset_json_dict, dbc.Label(steps_track, style={'whiteSpace': 'pre'}), "Build"


# -----------Page #2: PARAFAC--------------

#   -------------Setting up the dbc cards

#       -----------------dbc card for PARAFAC parameters
card_parafac_param = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Parameters selection", className="card-title"),
            html.Div(
                [
                    dbc.Stack(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Rank"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='parafac-rank', type='number',
                                                  placeholder='i.e., number of components',
                                                  style={'width': '300px', 'height': '30px'}, debounce=True),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Label("Initialization"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(options=[
                                            {'label': 'SVD', 'value': 'svd'},
                                            {'label': 'random', 'value': 'random'}
                                        ],
                                            value='svd', style={'width': '300px'}, id='parafac-init-method'
                                        ),
                                        width={'size': 2}
                                    ),

                                    dbc.Col(
                                        dbc.Checklist(options=[{'label': html.Span("Apply non-negativity constraint",
                                                                                   style={"font-size": 15,
                                                                                          "padding-left": 10}),
                                                                'value': 'non_negative'}],
                                                      id='parafac-nn-checkbox', switch=True, value='non_negative'),
                                        width={"size": 2, 'offset': 1}
                                    ),
                                ]
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("Additional analysis"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Core consistency', 'value': 'core_consistency'},
                                            {'label': 'Leverage', 'value': 'leverage'},
                                            {'label': 'Split-half validation', 'value': 'split_half'},
                                        ],
                                        multi=True, id='parafac-additional-analysis'),
                                    width={'size': 4}
                                ),
                            ]),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='parafac-spinner')],
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

#   -------------Layout of page #2

page2 = html.Div([
    dbc.Stack(
        [
            dbc.Row(
                card_parafac_param
            ),
            dbc.Row(
                dcc.Tabs(
                    id='parafac-results',
                    children=[
                        dcc.Tab(label='Excitation loadings', id='parafac-excitation-loadings'),
                        dcc.Tab(label='Emission loadings', id='parafac-emission-loadings'),
                        dcc.Tab(label='Components', id='parafac-components'),
                        dcc.Tab(label='Scores', id='parafac-scores'),
                        dcc.Tab(label='Fmax', id='parafac-fmax'),
                        dcc.Tab(label='Core consistency', id='parafac-core-consistency'),
                        dcc.Tab(label='Leverage', id='parafac-leverage'),
                        dcc.Tab(label='Split-half validation', id='parafac-split-half'),
                    ],
                    vertical=True
                )
            ),
        ],
        gap=3
    )

])

#   -------------Callbacks of page #2

@app.callback(
    [
        Output('parafac-excitation-loadings', 'children'),
        Output('parafac-emission-loadings', 'children'),
        Output('parafac-components', 'children'),
        Output('parafac-scores', 'children'),
        Output('parafac-fmax', 'children'),
        Output('parafac-core-consistency', 'children'),
        Output('parafac-leverage', 'children'),
        Output('parafac-split-half', 'children'),
        Output('parafac-spinner', 'children'),
        Output('parafac-model', 'data'),
    ],
    [
        Input('build-parafac-model', 'n_clicks'),
        State('parafac-rank', 'value'),
        State('parafac-init-method', 'value'),
        State('parafac-nn-checkbox', 'value'),
        State('parafac-additional-analysis', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_parafac_model(n_clicks, rank, init, nn, additional_analysis, eem_dataset):

    return


# -----------Page #3: K-PARAFACs--------------

#   -------------Setting up the dbc cards

#   -------------Layout of page #2

#   -------------Callbacks of page #2


# -----------Page #4: NMF--------------

#   -------------Setting up the dbc cards

#   -------------Layout of page #2

#   -------------Callbacks of page #2


# -----------Setup the sidebar-----------------

content = html.Div(
    [
        html.H2("eempy-vis", className="display-5"),
        html.Hr(),
        html.P(
            "An interactive python toolkit for EEM analysis", className="lead"
        ),
        dbc.Tabs(
            id='tabs-content',
            children=[
                dcc.Tab(label='Homepage', id='homepage', children=html.P('Homepage')),
                dcc.Tab(label='EEM pre-processing', id='eem-pre-processing', children=html.P(page1)),
                dcc.Tab(label='PARAFAC', id='parafac', children=html.P(page2)),
                dcc.Tab(label='K-PARAFACs', id='k-parafacs', children=html.P('K-PARAFAC')),
            ],
            # value="homepage",
            persistence=True,
            persistence_type='session',
        ),
    ],
)


def serve_layout():
    return html.Div([
        dcc.Store(id='pre-processed-eem'),
        dcc.Store(id='eem-dataset'),
        dcc.Store(id='parafac-model'),
        dcc.Store(id='k-parafacs-model'),
        dcc.Store(id='nmf-model'),
        content])


app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=True)

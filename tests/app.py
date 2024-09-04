import math
import os.path

import dash
import json
import pickle

import numpy as np
import pandas as pd
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr

from eempy.plot import plot_eem, plot_abs, plot_loadings, plot_score
from eempy.read_data import *
from eempy.eem_processing import *
from eempy.utils import str_string_to_list, num_string_to_list

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

                    html.H6("File filtering keywords"),

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
                                          value=[], id='timestamp-checkbox', switch=True),
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

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Checklist(options=[
                                    {'label': 'Fix aspect ratio to 1', 'value': 'aspect_one'},
                                    {'label': 'Rotate EEM', 'value': 'rotate'},
                                ],
                                    id='eem-graph-options', inline=True, value=[]),
                                width={"size": 12}
                            )
                        ]
                    ),

                    dbc.Row([
                        dcc.Graph(id='eem-graph',
                                  # config={'responsive': 'auto'},
                                  style={'width': '700', 'height': '900'}
                                  ),
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

#       -----------dbc card for median filter

card_median_filter = dbc.Card(
    dbc.CardBody(
        [
            dbc.Stack(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("Median filter"),
                                ],
                                width={'size': 10}),
                            dbc.Col([
                                dbc.Checklist(id='median-filter-button',
                                              options=[{'label': ' ', 'value': "median_filter"}],
                                              switch=True,
                                              style={'transform': 'scale(1.3)'})
                            ], width={'size': 2})
                        ],
                        align='start'
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("window size-ex"),
                        ),
                        dbc.Col(
                            dcc.Input(id='median-filter-window-ex',
                                      type='number', placeholder='max',
                                      style={'width': '100%', 'height': '30px'}, debounce=True, value=3),
                        ),
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("window size-em")
                            ),
                            dbc.Col(
                                dcc.Input(id='median-filter-window-em',
                                          type='number', placeholder='max',
                                          style={'width': '100%', 'height': '30px'}, debounce=True, value=3),
                            )
                        ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("edge processing mode")
                            ),
                            dbc.Col(
                                dcc.Dropdown(id='median-filter-mode',
                                             options=[
                                                 {'label': 'reflect', 'value': 'reflect'},
                                                 {'label': 'constant', 'value': 'constant'},
                                                 {'label': 'nearest', 'value': 'nearest'},
                                                 {'label': 'mirror', 'value': 'mirror'},
                                                 {'label': 'wrap', 'value': 'wrap'}
                                             ],
                                             value='reflect', placeholder='', style={'width': '100%'})
                            )
                        ]
                    )
                ],
                gap=1
            )
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
                dbc.Row(
                    dcc.Input(id='path-reference', type='text',
                              placeholder='Please enter the reference file path (optional)',
                              style={'width': '97%', 'height': '30px'}, debounce=True),
                    justify="center"
                ),
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

#       -----------dbc card for exporting pre-processed EEM

card_eem_dataset_downloading = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Export Processed EEM Dataset", className="card-title"),
            dbc.Stack([
                dbc.Row(
                    dcc.Input(id='folder-path-export-eem-dataset', type='text',
                              placeholder='Please enter the output folder path...',
                              style={'width': '97%', 'height': '30px'}, debounce=True),
                    justify="center"
                ),
                dbc.Row(
                    dcc.Input(id='filename-export-eem-dataset', type='text',
                              placeholder='Please enter the output filename (without extension)...',
                              style={'width': '97%', 'height': '30px'}, debounce=True),
                    justify="center"
                ),
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
                                id="eem-dataset-export-format",
                                options=[{'label': '.json', 'value': 'json'},
                                         {'label': '.pkl', 'value': 'pkl'}],
                                value='json', placeholder='Select EEM dateset file format',
                                style={'width': '100%'}, optionHeight=50)
                        ]
                    ),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Button("Export", id='export-eem-dataset', className='col-5')
                            )]),
                dbc.Row([
                    dbc.Col(
                        html.Div(id='message-eem-dataset-export', style={'width': '300px'}),
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
                            card_rayleigh
                        ],
                        gap=3)
                ],
                width={"size": 3}
            ),
            dbc.Col(
                [
                    dbc.Stack(
                        [
                            card_raman,
                            card_median_filter,
                            card_smoothing,
                            card_built_eem_dataset,
                            card_eem_dataset_downloading,
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
                kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else []
                kw_optional = str_string_to_list(kw_optional) if kw_optional else []
                kw_sample = str_string_to_list(kw_sample) if kw_sample else []
                filenames = get_filelist(folder_path, kw_mandatory + kw_sample, kw_optional)
            options = [{'label': f, 'value': f} for f in filenames]
            return options, None
        except FileNotFoundError as e:
            return [f'{e}'], None
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
        Input('eem-graph-options', 'value'),
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
        Input('median-filter-button', 'value'),
        Input('median-filter-window-ex', 'value'),
        Input('median-filter-window-em', 'value'),
        Input('median-filter-mode', 'value')
    ]
)
def update_eem_plot(folder_path, file_name_sample, graph_options,
                    eem_data_format, abs_data_format,
                    file_kw_sample, file_kw_abs, file_kw_blank,
                    index_pos_left, index_pos_right,
                    ex_range_min, ex_range_max, em_range_min, em_range_max, intensity_range_min, intensity_range_max,
                    su, su_ex, su_em_width, su_normalization_factor,
                    ife, ife_method,
                    raman, raman_method, raman_dimension, raman_width,
                    rayleigh, rayleigh_o1_method, rayleigh_o1_dimension, rayleigh_o1_width,
                    rayleigh_o2_method, rayleigh_o2_dimension, rayleigh_o2_width,
                    gaussian, gaussian_sigma, gaussian_truncate,
                    median_filter, median_filter_window_ex, median_filter_window_em, median_filter_mode):
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

        # Median filter
        if all([median_filter, median_filter_window_ex, median_filter_window_em, median_filter_mode]):
            intensity = eem_median_filter(intensity, footprint=(median_filter_window_ex, median_filter_window_em),
                                          mode=median_filter_mode)

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
                           fix_aspect_ratio=True if 'aspect_one' in graph_options else False,
                           rotate=True if 'rotate' in graph_options else False,
                           title=index if index else None)

        # Plot absorbance (if exists)
    except:
        # Create an empty scatter plot
        fig_eem = go.Figure()

        # Add a black border
        fig_eem.update_layout(
            xaxis=dict(showline=False, linewidth=0, linecolor="black"),
            yaxis=dict(showline=False, linewidth=0, linecolor="black"),
            # width=700,
            # height=400,
            margin=dict(l=50, r=50, b=50, t=50),
        )

        # Add centered text annotation
        fig_eem.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            text="EEM file or parameters unspecified",
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
            text="Absorbance file or parameters unspecified",
            showarrow=False,
            font=dict(size=16),
        )

    return fig_eem, fig_abs, f'RSU: {rsu_value}'


#   ---------------Build EEM dataset

@app.callback(
    [
        Output('eem-dataset', 'data'),
        Output('info-eem-dataset', 'children'),
        Output('build-eem-dataset-spinner', 'children'),
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
        State('path-reference', 'value'),
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
        State('median-filter-button', 'value'),
        State('median-filter-window-ex', 'value'),
        State('median-filter-window-em', 'value'),
        State('median-filter-mode', 'value'),
        State('align-exem', 'value'),
    ],
)
def on_build_eem_dataset(n_clicks,
                         folder_path,
                         eem_data_format, abs_data_format,
                         file_name_sample_options, file_kw_sample, file_kw_abs, file_kw_blank,
                         index_pos_left, index_pos_right, timestamp, timestamp_format, reference_path,
                         ex_range_min, ex_range_max, em_range_min, em_range_max,
                         su, su_ex, su_em_width, su_normalization_factor,
                         ife, ife_method,
                         raman, raman_method, raman_dimension, raman_width,
                         rayleigh, rayleigh_o1_method, rayleigh_o1_dimension, rayleigh_o1_width,
                         rayleigh_o2_method, rayleigh_o2_dimension, rayleigh_o2_width,
                         gaussian, gaussian_sigma, gaussian_truncate,
                         median_filter, median_filter_ex, median_filter_em, median_filter_mode,
                         align_exem
                         ):
    if n_clicks is None:
        return None, None, "Build"
        # raise PreventUpdate
    try:
        file_name_sample_options = file_name_sample_options or {}
        file_name_sample_list = [f['value'] for f in file_name_sample_options]
        if index_pos_left and index_pos_right:
            index = (index_pos_left, index_pos_right)
        eem_stack, ex_range, em_range, indexes = read_eem_dataset(
            folder_path=folder_path, mandatory_keywords=[], optional_keywords=[], data_format=eem_data_format,
            index_pos=(index_pos_left, index_pos_right) if index_pos_left and index_pos_right else None,
            custom_filename_list=file_name_sample_list, wavelength_alignment=True if align_exem else False,
            interpolation_method='linear', as_timestamp=True if 'timestamp' in timestamp else False,
            timestamp_format=timestamp_format
        )
    except (UnboundLocalError, IndexError, TypeError) as e:
        error_message = ("EEM dataset building failed. Possible causes: (1) There are non-EEM files mixed in the EEM "
                         "data file list. Please check data folder path and file filtering keywords settings. "
                         "(2) There are necessary parameter boxes that has not been filled in. Please check the "
                         "parameter boxes. "
                         "(3) The Ex/Em ranges/intervals are different between EEMs, make sure you select the "
                         "'Align Ex/Em' checkbox.")
        return None, error_message, "Build"

    steps_track = []
    if reference_path is not None:
        if os.path.exists(reference_path):
            if reference_path.endswith('.csv'):
                refs_from_file = pd.read_csv(reference_path, index_col=0, header=0)
            elif reference_path.endswith('.xlsx'):
                refs_from_file = pd.read_excel(reference_path, index_col=0, header=0)
            else:
                return None, ("Unsupported file format. Please provide a .csv or .xlsx file."), "build"

            if index_pos_left and index_pos_right:
                # Check for missing indices
                extra_indices = [
                    index_from_file for index_from_file in refs_from_file.index if index_from_file not in indexes
                ]
                missing_indices = [index for index in indexes if index not in refs_from_file.index]
                if extra_indices or missing_indices:
                    steps_track += ["Warning: indices of EEM dataset and reference file are not \nexactly the "
                                            "same. The reference value of unmatched \nindices would be set as NaN\n"]
                refs = np.array(
                    [refs_from_file.loc[indexes[i]] if indexes[i] in refs_from_file.index
                     else np.full(shape=(refs_from_file.shape[1]), fill_value=np.nan) for i in range(len(indexes))]
                )
                refs = pd.DataFrame(refs, index=indexes, columns=refs_from_file.columns)
            else:
                if refs_from_file.shape[0] != len(indexes):
                    return None, ('Error: number of samples in reference file is not the same as the EEM dataset'), "build"
                refs = refs_from_file
        else:
            return None, ('Error: No such file or directory: ' + reference_path), "build"
    else:
        refs = None

    steps_track += [
        "EEM dataset building successful!\n",
        "Number of EEMs: {n}\n".format(n=eem_stack.shape[0]),
        "Pre-processing steps implemented:\n",
    ]

    # EEM cutting
    eem_dataset = EEMDataset(eem_stack, ex_range, em_range, index=indexes, ref=refs)
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

    # Median filter
    if all([median_filter, median_filter_ex, median_filter_em, median_filter_mode]):
        eem_dataset.median_filter(footprint=(median_filter_ex, median_filter_em), mode=median_filter_mode, copy=False)
        steps_track += ["- Median filter\n"]

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
        eem_dataset.gaussian_filter(sigma=gaussian_sigma, truncate=gaussian_truncate, copy=False)
        steps_track += ["- Gaussian smoothing\n"]

    # convert eem_dataset to a dict whose values are json serializable
    eem_dataset_json_dict = {
        'eem_stack': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for sublist in
                      eem_dataset.eem_stack.tolist()],
        'ex_range': eem_dataset.ex_range.tolist(),
        'em_range': eem_dataset.em_range.tolist(),
        'index': eem_dataset.index,
        'ref': [refs.columns.tolist()] + refs.values.tolist() if eem_dataset.ref is not None else None
    }

    return eem_dataset_json_dict, dbc.Label(steps_track, style={'whiteSpace': 'pre'}), "Build"


#   ---------------Export EEM

@app.callback(
    [
        Output('message-eem-dataset-export', 'children')
    ],
    [
        Input('export-eem-dataset', 'n_clicks'),
        Input('build-eem-dataset', 'n_clicks'),
        State('eem-dataset', 'data'),
        State('folder-path-export-eem-dataset', 'value'),
        State('filename-export-eem-dataset', 'value'),
        State('eem-dataset-export-format', 'value'),
    ]
)
def on_export_eem_dataset(n_clicks_export, n_clicks_build, eem_dataset_json_dict, export_folder_path, export_filename,
                          export_format):
    if ctx.triggered_id == "build-eem-dataset":
        return [None]
    if eem_dataset_json_dict is None:
        message = ['Please first build the eem dataset.']
        return message
    if not os.path.isdir(export_folder_path):
        message = ['Error: No such file or directory: ' + export_folder_path]
        return message
    else:
        path = export_folder_path + '/' + export_filename + '.' + export_format
    with open(path, 'w') as file:
        json.dump(eem_dataset_json_dict, file)

    return ["EEM dataset exported."]


# -----------Page #2: PARAFAC--------------

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
                                             id='parafac-eem-dataset-establishment-message', style={'width': '1000px'}),
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
                                    dbc.Label("Validations"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
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
                        dcc.Tab(label='ex/em loadings', id='parafac-loadings'),
                        dcc.Tab(label='Components', id='parafac-components'),
                        dcc.Tab(label='Scores', id='parafac-scores'),
                        dcc.Tab(label='Fmax', id='parafac-fmax'),
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
                                                                        {'label': 'scores', 'value': 'scores'},
                                                                        {'label': 'Fmax', 'value': 'fmax'}
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
                                                                  style={'width': '700', 'height': '900'}
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
                                                                dcc.Input(id='parafac-predict-index-kw-mandatory',
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
                                                                dcc.Input(id='parafac-predict-index-kw-optional',
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
                                                                    options=[], id='parafac-predict-model-selection'
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm", id='predict-parafac-spinner')],
                                                                    id='predict-parafac-model', className='col-2')
                                                            )
                                                        ]
                                                    )
                                                ],
                                                gap=2, style={"margin": "20px"}
                                            ),
                                        ),
                                        dbc.Card(
                                            children=None,
                                            id='parafac-test-result-card'
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


#   -------------Callbacks of page #2

@app.callback(
    [
        Output('parafac-eem-dataset-establishment-message', 'children'),
        Output('parafac-loadings', 'children'),
        Output('parafac-components', 'children'),
        Output('parafac-scores', 'children'),
        Output('parafac-fmax', 'children'),
        Output('parafac-core-consistency', 'children'),
        Output('parafac-leverage', 'children'),
        Output('parafac-split-half', 'children'),
        Output('build-parafac-spinner', 'children'),
        Output('parafac-establishment-corr-model-selection', 'options'),
        Output('parafac-establishment-corr-model-selection', 'value'),
        Output('parafac-establishment-corr-ref-selection', 'options'),
        Output('parafac-establishment-corr-ref-selection', 'value'),
        Output('parafac-predict-model-selection', 'options'),
        Output('parafac-predict-model-selection', 'value'),
        Output('parafac-models', 'data'),
    ],
    [
        Input('build-parafac-model', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('parafac-eem-dataset-establishment-path-input', 'value'),
        State('parafac-establishment-index-kw-mandatory', 'value'),
        State('parafac-establishment-index-kw-optional', 'value'),
        State('parafac-rank', 'value'),
        State('parafac-init-method', 'value'),
        State('parafac-nn-checkbox', 'value'),
        State('parafac-tf-checkbox', 'value'),
        State('parafac-validations', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_parafac_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, rank, init, nn,
                           tf, validations, eem_dataset_dict):
    if n_clicks is None:
        return None, None, None, None, None, None, None, None, 'build model', [], None, [], None, [], None, None
    if not path_establishment:
        if eem_dataset_dict is None:
            message = ('Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                       'section, or import an EEM dataset from file.')
            return message, None, None, None, None, None, None, None, 'build model', [], None, [], None, [], None, None
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0]),
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return message, None, None, None, None, None, None, None, 'build model', [], None, [], None, [], None, None
        else:
            _, file_extension = os.path.splitext(path_establishment)

            if file_extension == '.json':
                with open(path_establishment, 'r') as file:
                    eem_dataset_dict = json.load(file)
            elif file_extension == '.pkl':
                with open(path_establishment, 'rb') as file:
                    eem_dataset_dict = pickle.load(file)
            else:
                raise ValueError("Unsupported file extension: {}".format(file_extension))
            eem_dataset_establishment = EEMDataset(
                eem_stack=np.array(
                    [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                     in eem_dataset_dict['eem_stack']]),
                ex_range=np.array(eem_dataset_dict['ex_range']),
                em_range=np.array(eem_dataset_dict['em_range']),
                index=eem_dataset_dict['index'],
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0]),
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else []
    kw_optional = str_string_to_list(kw_optional) if kw_optional else []
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              copy=False)

    rank_list = num_string_to_list(rank)
    parafac_components_dict = {}
    loadings_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    components_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    scores_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    fmax_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    core_consistency_tabs = dbc.Card(children=[])
    leverage_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    split_half_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    cc = []

    for r in rank_list:
        parafac_r = PARAFAC(rank=r, init=init, non_negativity=True if 'non_negative' in nn else False,
                            tf_normalization=True if 'tf_normalization' in tf else False,
                            sort_em=True)
        parafac_r.fit(eem_dataset_establishment)

        parafac_components_dict[r] = {
            'component_stack': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                                      sublist in parafac_r.component_stack.tolist()],
            'score': [parafac_r.score.columns.tolist()] + parafac_r.score.values.tolist(),
            'fmax': [parafac_r.fmax.columns.tolist()] + parafac_r.fmax.values.tolist(),
        }

        # for component graphs, determine the layout according to the number of components
        n_rows = (r - 1) // 3 + 1

        # ex/em loadings
        loadings_tabs.children[0].children.append(
            dcc.Tab(label=f'{r}-component',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_loadings({f'{r}-component': parafac_r},
                                                                           plot_tool='plotly', n_cols=3,
                                                                           display=False),
                                                      config={'autosizable': False}
                                                      )
                                        ]
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(parafac_r.ex_loadings,
                                                                     bordered=True, hover=True, index=True,

                                                                     )
                                        ]
                                    ),

                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(parafac_r.em_loadings,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            )
                        ]),
                    ],
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        # components
        components_tabs.children[0].children.append(
            # html.Div(
            dcc.Tab(label=f'{r}-component',
                    children=
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(parafac_r.component_stack[3 * i],
                                                            ex_range=parafac_r.ex_range,
                                                            em_range=parafac_r.em_range,
                                                            vmin=0 if np.min(
                                                                parafac_r.component_stack[3 * i]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 1}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 1 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 4},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(parafac_r.component_stack[3 * i + 1],
                                                            ex_range=parafac_r.ex_range,
                                                            em_range=parafac_r.em_range,
                                                            vmin=0 if np.min(
                                                                parafac_r.component_stack[
                                                                    3 * i + 1]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 2}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 2 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 4},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(parafac_r.component_stack[3 * i + 2],
                                                            ex_range=parafac_r.ex_range,
                                                            em_range=parafac_r.em_range,
                                                            vmin=0 if np.min(
                                                                parafac_r.component_stack[
                                                                    3 * i + 2]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 3}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 3 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 4},
                                    ),
                                ]
                            ) for i in range(n_rows)
                        ],
                        style={'width': '90vw'}
                    ),
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        # scores
        scores_tabs.children[0].children.append(
            dcc.Tab(label=f'{r}-component',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_score(parafac_r.score,
                                                                        display=False
                                                                        ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(parafac_r.score,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            )
                        ]),
                    ],
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        # fmax
        fmax_tabs.children[0].children.append(
            dcc.Tab(label=f'{r}-component',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_score(parafac_r.fmax,
                                                                        display=False,
                                                                        yaxis_title='Fmax'
                                                                        ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(parafac_r.fmax,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            )
                        ]),
                    ],
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        # core consistency
        if 'core_consistency' in validations:
            cc.append(parafac_r.core_consistency())

        # leverage
        if 'leverage' in validations:
            lvr_sample = parafac_r.leverage('sample')
            lvr_ex = parafac_r.leverage('ex')
            lvr_em = parafac_r.leverage('em')
            leverage_tabs.children[0].children.append(
                dcc.Tab(label=f'{r}-component',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    figure=px.line(
                                                        x=lvr_sample.index,
                                                        y=lvr_sample.iloc[:, 0],
                                                        markers=True,
                                                        labels={'x': 'index-sample', 'y': 'leverage-sample'},
                                                        tickangle=90
                                                    ),
                                                    config={'autosizable': False},
                                                    style={'width': 1700, 'height': 400}
                                                )
                                            ]
                                        )
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    figure=px.line(
                                                        x=lvr_ex.index,
                                                        y=lvr_ex.iloc[:, 0],
                                                        markers=True,
                                                        labels={'x': 'index-ex', 'y': 'leverage-ex'},
                                                    ),
                                                    config={'autosizable': False},
                                                    style={'width': 1700, 'height': 400}
                                                )
                                            ]
                                        )
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    figure=px.line(
                                                        x=lvr_em.index,
                                                        y=lvr_em.iloc[:, 0],
                                                        markers=True,
                                                        labels={'x': 'index-em', 'y': 'leverage-em'},
                                                    ),
                                                    config={'autosizable': False},
                                                    style={'width': 1700, 'height': 400}
                                                )
                                            ]
                                        )
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(lvr_sample,
                                                                         bordered=True, hover=True, index=True,
                                                                         )
                                            ]
                                        ),

                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(lvr_ex,
                                                                         bordered=True, hover=True, index=True,
                                                                         )
                                            ]
                                        ),

                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(lvr_em,
                                                                         bordered=True, hover=True, index=True,
                                                                         )
                                            ]
                                        ),

                                    ]
                                ),
                            ]),
                        ],
                        style={'padding': '0', 'line-width': '100%'},
                        selected_style={'padding': '0', 'line-width': '100%'}
                        )
            )

            if 'split_half' in validations:
                split_validation = SplitValidation(rank=r,
                                                   non_negativity=True if 'non_negative' in nn else False,
                                                   tf_normalization=True if 'tf_normalization' in tf else False)
                split_validation.fit(eem_dataset_establishment)
                subset_specific_models = split_validation.subset_specific_models
                similarities_ex, similarities_em = split_validation.compare()
                split_half_tabs.children[0].children.append(
                    dcc.Tab(label=f'{r}-component',
                            children=[
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        figure=plot_loadings(subset_specific_models,
                                                                             n_cols=3,
                                                                             plot_tool='plotly',
                                                                             display=False,
                                                                             legend_pad=0.2),
                                                        config={'autosizable': False},
                                                    )
                                                ]
                                            )
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(similarities_ex,
                                                                             bordered=True, hover=True, index=True,
                                                                             )
                                                ]
                                            ),
                                        ]
                                    ),

                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Table.from_dataframe(similarities_em,
                                                                             bordered=True, hover=True, index=True,
                                                                             )
                                                ]
                                            ),
                                        ]
                                    ),

                                ]),
                            ],
                            style={'padding': '0', 'line-width': '100%'},
                            selected_style={'padding': '0', 'line-width': '100%'}
                            )
                )

    if 'core_consistency' in validations:
        cc_table = pd.DataFrame({'Number of components': rank_list, 'Core consistency': cc})
        core_consistency_tabs.children.append(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    figure=px.line(
                                        x=cc_table['Number of components'],
                                        y=cc_table['Core consistency'],
                                        markers=True,
                                        labels={'x': 'Number of components', 'y': 'Core consistency'},
                                    ),
                                    config={'autosizable': False},
                                )
                            ]
                        )
                    ]
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Table.from_dataframe(cc_table,
                                                         bordered=True, hover=True,
                                                         )
                            ]
                        ),
                    ]
                ),
            ]),
        )

    model_options = [{'label': 'component {r}'.format(r=r), 'value': r} for r in parafac_components_dict.keys()]
    ref_options = [{'label': var, 'value': var} for var in eem_dataset_establishment.ref.columns]

    return (None, loadings_tabs, components_tabs, scores_tabs, fmax_tabs, core_consistency_tabs, leverage_tabs,
            split_half_tabs, 'build model', model_options, None, ref_options, None, model_options, None,
            parafac_components_dict)


# # -----------Update parafac model dropdown list
# @app.callback(
#     [
#         Output('parafac-predict-model-selection', 'options')
#     ],
#     [
#         Input('parafac-models', 'data')
#     ]
# )
# def update_parafac_models_options(parafac_components_all):
#     if parafac_components_all is None:
#         return []
#     options = []
#     for r in parafac_components_all.keys():
#         options.append({'label': 'component {r}'.format(r=r), 'value': r})
#     return options


# -----------Analyze correlations between score/Fmax and reference variables

@app.callback(
    [
        Output('parafac-establishment-corr-graph', 'figure'), # size, intervals?
        Output('parafac-establishment-corr-table', 'children'),
    ],
    [
        Input('parafac-establishment-corr-model-selection', 'value'),
        Input('parafac-establishment-corr-indicator-selection', 'value'),
        Input('parafac-establishment-corr-ref-selection', 'value'),
        State('eem-dataset', 'data'),
        State('parafac-models', 'data')
    ]
)
def on_parafac_establishment_correlations(r, indicator, ref_var, eem_dataset_establishment, parafac_models):
    if all([r, indicator, ref_var, eem_dataset_establishment, parafac_models]):
        ref_df = pd.DataFrame(eem_dataset_establishment['ref'][1:], columns=eem_dataset_establishment[r]['ref'][0],
                              index=eem_dataset_establishment['index'])
        ref_var = ref_df[ref_var]
        parafac_var = pd.DataFrame(parafac_models[r][indicator][1:], columns=parafac_models[r][indicator][0],
                                   index=eem_dataset_establishment['index'])
        fig = go.Figure()
        tbl = pd.DataFrame(columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value'])
        for col in parafac_var.columns:
            x = ref_var
            y = parafac_var[col]
            x = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x, y)
            predictions = lm.predict(x)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col))
            fig.add_trace(go.Scatter(x=x.flatten(), y=predictions, mode='lines', name=f'{col} fit'))
            r_squared = lm.score(x, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(ref_var, y)
            tbl = tbl.append({'Variable': col, 'R²': r_squared, 'slope': slope, 'intercept': intercept,
                                      'Pearson Correlation': pearson_corr, 'Pearson p-value': pearson_p},
                                     ignore_index=True)

        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return None, None



# -----------Make prediction on an EEM dataset using an established PARAFAC model
@app.callback(
    [
        Output('parafac-eem-dataset-predict-message', 'children'), # size, intervals?
        Output('parafac-test-result-card', 'children'),
        Output('predict-parafac-spinner', 'children'),
    ],
    [
        Input('predict-parafac-model', 'n_clicks'),
        State('parafac-eem-dataset-predict-path-input', 'value'),
        State('parafac-predict-index-kw-mandatory', 'value'),
        State('parafac-predict-index-kw-optional', 'value'),
        State('parafac-predict-model-selection', 'value'),
        State('parafac-models', 'data')
    ]
)
def on_parafac_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, model_r, parafac_models):
    if n_clicks is None:
        return None, None, 'predict'
    if path_predict is None:
        return None, None, 'predict'
    if not os.path.exists(path_predict):
        message = ('Error: No such file: ' + path_predict)
        return message, None, 'predict'
    else:
        _, file_extension = os.path.splitext(path_predict)

        if file_extension == '.json':
            with open(path_predict, 'r') as file:
                eem_dataset_dict = json.load(file)
        elif file_extension == '.pkl':
            with open(path_predict, 'rb') as file:
                eem_dataset_dict = pickle.load(file)
        else:
            raise ValueError("Unsupported file extension: {}".format(file_extension))
        eem_dataset_predict = EEMDataset(
            eem_stack=np.array(
                [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                 in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0]),
        )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else []
    kw_optional = str_string_to_list(kw_optional) if kw_optional else []
    eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, copy=False)

    score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset_predict.eem_stack,
                                                                    np.array(
                                                                        [
                                                                            [
                                                                                [np.nan if x is None else x for x in
                                                                                subsublist] for subsublist in sublist
                                                                            ]
                                                                            for sublist in
                                                                            parafac_models[str(model_r)]['component_stack']
                                                                        ]
                                                                    ),
                                                                    fit_intercept=False)
    score_sample = pd.DataFrame(
        score_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i+1) for i in range(model_r)]
    )
    fmax_sample = pd.DataFrame(
        fmax_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i+1) for i in range(model_r)]
    )

    prediction_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    prediction_tabs.children[0].children.append(dcc.Tab(
        label='Score',
        children=[
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(figure=plot_score(score_sample,
                                                            display=False
                                                            ),
                                          config={'autosizable': False},
                                          style={'width': 1700, 'height': 800}
                                          )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Table.from_dataframe(score_sample,
                                                         bordered=True, hover=True, index=True
                                                         )
                            ]
                        ),
                    ]
                ),
            ]),
        ]
    ))
    prediction_tabs.children[0].children.append(dcc.Tab(
        label='Fmax',
        children=[
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(figure=plot_score(fmax_sample,
                                                            display=False
                                                            ),
                                          config={'autosizable': False},
                                          style={'width': 1700, 'height': 800}
                                          )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Table.from_dataframe(fmax_sample,
                                                         bordered=True, hover=True, index=True
                                                         )
                            ]
                        ),
                    ]
                ),
            ]),
        ]
    ))
    prediction_tabs.children[0].children.append(dcc.Tab(
        label='Error',
        children=[

        ]
    ))
    return None, prediction_tabs, 'predict'

# -----------Page #3: K-PARAFACs--------------

#   -------------Setting up the dbc cards

#   -------------Layout of page #2

#   -------------Callbacks of page #2


# -----------Page #4: NMF--------------

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
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='nmf-sample-kw-mandatory', type='text', placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='nmf-sample-kw-optional', type='text', placeholder='',
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
                                            {'label': 'Coordinate Descent solver', 'value': 'cd'},
                                            {'label': 'Multiplicative Update solver', 'value': 'mu'}
                                        ],
                                            value='cd', style={'width': '300px'}, id='nmf-solver'
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
                                        width={"size": 2, 'offset': 1}
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
                                    dbc.Label("Validations"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Residual', 'value': 'leverage'},
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

#   -------------Layout of page #4

page4 = html.Div([
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
                        dcc.Tab(label='Residual', id='nmf-residual'),
                        dcc.Tab(label='Split-half validation', id='nmf-split-half'),
                        dcc.Tab(
                            children=[
                                html.Div(
                                    [
                                        dbc.Card(
                                            dbc.Stack(
                                                [
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
                                                                    [dbc.Spinner(size="sm", id='test-nmf-spinner')],
                                                                    id='test-nmf-model', className='col-2')
                                                            )
                                                        ]
                                                    )
                                                ], gap=2
                                            ),
                                        ),
                                        dbc.Card(
                                            children=None,
                                            id='nmf-test-result-card'
                                        )
                                    ],
                                    style={'width': '90vw'}
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


#   -------------Callbacks of page #4

@app.callback(
    [
        Output('nmf-components', 'children'),
        Output('nmf-fmax', 'children'),
        Output('nmf-residual', 'children'),
        Output('nmf-split-half', 'children'),
        Output('nmf-spinner', 'children'),
        Output('nmf-models', 'data'),
    ],
    [
        Input('build-nmf-model', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('nmf-sample-kw-mandatory', 'value'),
        State('nmf-sample-kw-optional', 'value'),
        State('nmf-rank', 'value'),
        State('nmf-solver', 'value'),
        State('nmf-normalization-checkbox', 'value'),
        State('nmf-alpha-w', 'value'),
        State('nmf-alpha-h', 'value'),
        State('nmf-l1-ratio', 'value'),
        State('nmf-validations', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_nmf_model(n_clicks, eem_graph_options, kw_mandatory, kw_optional, rank, solver, normalization, alpha_w,
                       alpha_h, l1_ratio, validations, eem_dataset_dict):
    if n_clicks is None:
        return None, None, None, None, 'build model', None
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else []
    kw_optional = str_string_to_list(kw_optional) if kw_optional else []
    eem_dataset = EEMDataset(
        eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                            in eem_dataset_dict['eem_stack']]),
        ex_range=np.array(eem_dataset_dict['ex_range']),
        em_range=np.array(eem_dataset_dict['em_range']),
        index=eem_dataset_dict['index']
    )
    eem_dataset.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, copy=False)
    rank_list = num_string_to_list(rank)
    nmfs_dict = {}
    components_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    fmax_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    residual_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    split_half_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])

    for r in rank_list:
        nmf_r = EEMNMF(
            n_components=r, solver=solver, normalization=normalization[0], alpha_H=alpha_h, alpha_W=alpha_w,
            l1_ratio=l1_ratio
        )
        nmf_r.fit(eem_dataset)
        nmfs_dict[r] = nmf_r

        # for component graphs, determine the layout according to the number of components
        n_rows = (r - 1) // 3 + 1

        # components
        components_tabs.children[0].children.append(
            # html.Div(
            dcc.Tab(label=f'{r}-component',
                    children=
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[3 * i],
                                                            ex_range=eem_dataset.ex_range,
                                                            em_range=eem_dataset.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[3 * i]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 1}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 1 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 4},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[3 * i + 1],
                                                            ex_range=eem_dataset.ex_range,
                                                            em_range=eem_dataset.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    3 * i + 1]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 2}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 2 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 4},
                                    ),

                                    dbc.Col(
                                        dcc.Graph(
                                            figure=plot_eem(nmf_r.components[3 * i + 2],
                                                            ex_range=eem_dataset.ex_range,
                                                            em_range=eem_dataset.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    3 * i + 2]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 3}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 3 <= r else go.Figure(
                                                layout={'width': 400, 'height': 300}),
                                            style={'width': '500', 'height': '500'}
                                        ),
                                        width={'size': 4},
                                    ),
                                ]
                            ) for i in range(n_rows)
                        ],
                        style={'width': '90vw'}
                    ),
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        # scores
        fmax_tabs.children[0].children.append(
            dcc.Tab(label=f'{r}-component',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_score(nmf_r.nnls_score,
                                                                        display=False
                                                                        ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(nmf_r.nnls_score,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_score(nmf_r.nmf_score,
                                                                        display=False
                                                                        ),
                                                      config={'autosizable': False},
                                                      style={'width': 1700, 'height': 800}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(nmf_r.nmf_score,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            )
                        ]),
                    ],
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )
    return components_tabs, fmax_tabs, residual_tabs, split_half_tabs, 'build model', None


# -----------Setup the sidebar-----------------

content = html.Div(
    [
        html.H2("eempy-vis", className="display-5"),
        html.Hr(),
        html.P(
            "An open-source, interactive toolkit for EEM analysis", className="lead"
        ),
        dbc.Tabs(
            id='tabs-content',
            children=[
                dcc.Tab(label='Homepage', id='homepage', children=html.P('Homepage')),
                dcc.Tab(label='EEM pre-processing', id='eem-pre-processing', children=html.P(page1)),
                dcc.Tab(label='PARAFAC', id='parafac', children=html.P(page2)),
                dcc.Tab(label='K-PARAFACs', id='k-parafacs', children=html.P('K-PARAFAC')),
                dcc.Tab(label='NMF', id='nmf', children=html.P(page4)),
            ],
            # value="homepage",
            # persistence=True,
            # persistence_type='session',
        ),
    ],
)


def serve_layout():
    return html.Div([
        dcc.Store(id='pre-processed-eem'),
        dcc.Store(id='eem-dataset'),
        dcc.Store(id='parafac-models'),
        dcc.Store(id='k-parafacs-models'),
        dcc.Store(id='nmf-models'),
        content])


app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=True)

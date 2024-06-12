import os
import dash
import datetime
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from eempy.plot import plot_eem, plot_abs
from eempy.read_data import read_eem, read_abs, get_filelist
from eempy.eem_processing import (eem_cutting, eem_raman_normalization, eem_ife_correction,
                                  eem_raman_scattering_removal, eem_rayleigh_scattering_removal, eem_gaussian_filter)
from eempy.utils import string_to_list
from matplotlib import pyplot as plt

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# -----------Global variables--------------

eem_dataset_working = None  # The EEM dataset that the user would build and analyse

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
                            dbc.Label("RS Ex lower bound"),
                            width={'size': 6}
                        ),
                        dbc.Col(
                            dcc.Input(id='su-excitation-lower-bound', type='number', placeholder='max',
                                      style={'width': '120%', 'height': '30px'}, debounce=True, value=349),
                            width={'size': 3}
                        ),
                        dbc.Col(
                            dbc.Label('nm')
                        )
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("RS Ex upper bound"),
                            width={'size': 6}
                        ),
                        dbc.Col(
                            dcc.Input(id='su-excitation-upper-bound', type='number', placeholder='max',
                                      style={'width': '120%', 'height': '30px'}, debounce=True, value=351),
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
                        dbc.Button([dbc.Spinner(size="sm", id='build-eem-dataset-spinner'), " Build"],
                                   id='build-eem-dataset', className='col-5')
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
def update_filenames(folder_path, kw_mandatory, kw_optional, kw_eem):
    if folder_path:
        try:
            filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if kw_mandatory or kw_optional or kw_eem:
                kw_mandatory = string_to_list(kw_mandatory) if kw_mandatory else []
                kw_optional = string_to_list(kw_optional) if kw_optional else []
                kw_eem = string_to_list(kw_eem) if kw_eem else []
                filenames = get_filelist(folder_path, kw_mandatory + kw_eem, kw_optional)
            options = [{'label': f, 'value': f} for f in filenames]
            return options, None
        except FileNotFoundError:
            return ['Folder not found - please check the folder path'], None
    else:
        return [], None


#   ---------------Update Data plot with changes of parameters

@app.callback(
    [
        Output('eem-graph', 'figure'),
        Output('absorbance-graph', 'figure')
    ],
    [
        Input('folder-path-input', 'value'),
        Input('filename-sample-dropdown', 'value'),
        Input('eem-data-format', 'value'),
        Input('abs-data-format', 'value'),
        Input('file-keyword-mandatory', 'value'),
        Input('file-keyword-optional', 'value'),
        Input('file-keyword-sample', 'value'),
        Input('file-keyword-absorbance', 'value'),
        Input('file-keyword-blank', 'value'),
        Input('index-pos-left', 'value'),
        Input('index-pos-right', 'value'),
        Input('timestamp-checkbox', 'value'),
        Input('timestamp-format', 'value'),
        Input('excitation-wavelength-min', 'value'),
        Input('excitation-wavelength-max', 'value'),
        Input('emission-wavelength-min', 'value'),
        Input('emission-wavelength-max', 'value'),
        Input('fluorescence-intensity-min', 'value'),
        Input('fluorescence-intensity-max', 'value'),
        Input('su-button', 'value'),
        Input('su-excitation-lower-bound', 'value'),
        Input('su-excitation-upper-bound', 'value'),
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
                    file_kw_mandatory, file_kw_optional, file_kw_sample, file_kw_abs, file_kw_blank,
                    index_pos_left, index_pos_right, timestamp, timestamp_format,
                    ex_range_min, ex_range_max, em_range_min, em_range_max, intensity_range_min, intensity_range_max,
                    su, su_ex_lb, su_ex_ub, su_em_width, su_normalization_factor,
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
            intensity, _ = eem_raman_normalization(intensity, blank=intensity_blank, ex_range_blank=ex_range_blank,
                                                   em_range_blank=em_range_blank, from_blank=True, ex_lb=su_ex_lb,
                                                   ex_ub=su_ex_ub,
                                                   bandwidth=su_em_width, bandwidth_type='wavelength',
                                                   rsu_standard=su_normalization_factor)

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
        if gaussian:
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

    return fig_eem, fig_abs


#   ---------------Export EEM


#   ---------------Build EEM dataset

#


# @app.callback(
#     Output('index-display', 'children'),
#     [Input('filename-dropdown', 'value'),
#      Input('index-pos-left', 'value'),
#      Input('index-pos-right', 'value'),
#      Input('timestamp-checkbox', 'value'),
#      Input('timestamp-format', 'value')]
# )
# def update_index_display(filename, pos_left, pos_right, ts_option, ts_format):
#     if pos_left and pos_right:
#         index = filename[pos_left:pos_right]
#         if ts_option:
#             index = datetime.strptime(index, ts_format)
#         return index
#     return None
#
#


# -----------Page #2: PARAFAC--------------

#   -------------Setting up the dbc cards

#   -------------Layout of page #2

#   -------------Callbacks of page #2


# -----------Setup the sidebar-----------------

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("eempy-interactive", className="display-4"),
        html.Hr(),
        html.P(
            "A interactive python toolkit for EEM analysis", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Homepage", href="/", active="exact"),
                dbc.NavLink("EEM pre-processing", href="/eem-pre-processing", active="exact"),
                dbc.NavLink("PARAFAC", href="/parafac", active="exact"),
                dbc.NavLink("K-PARAFACs", href="/k-parafacs", active="exact"),
            ],
            vertical=True,
            pills=True
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")
    elif pathname == "/eem-pre-processing":
        return page1
    elif pathname == "/parafac":
        return html.P("Oh cool, this is page 2!")
    elif pathname == "/k-parafacs":
        return html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == '__main__':
    app.run_server(debug=True)

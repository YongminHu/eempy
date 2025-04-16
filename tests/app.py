
import os.path

import dash
import pickle

from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr

from eempy.plot import plot_eem, plot_abs, plot_loadings, plot_fmax, plot_dendrogram, plot_reconstruction_error
from eempy.read_data import *
from eempy.eem_processing import *
from eempy.utils import str_string_to_list, num_string_to_list

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
colors = px.colors.qualitative.Plotly
marker_shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left',
                 'triangle-right', 'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw']

# -----------Page #0: Homepage

homepage = html.Div([
    dbc.Row(
        [
            html.Div(
                [
                    "Author: Yongmin Hu"
                ]
            )
        ]
    ),
    dbc.Row(
        [
            html.Div(
                [
                    "Github page: https://github.com/YongminHu/eempy"
                ]
            )
        ]
    )
])

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
                              value=None,
                              style={'width': '97%', 'height': '30px'}, debounce=False),
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
                        html.Div(id='info-eem-dataset', style={'width': '22vw'}),
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
                              style={'width': '97%', 'height': '30px'}, debounce=False),
                    justify="center"
                ),
                dbc.Row(
                    dcc.Input(id='filename-export-eem-dataset', type='text',
                              placeholder='Please enter the output filename (without extension)...',
                              style={'width': '97%', 'height': '30px'}, debounce=False),
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
                    dbc.Col(dbc.Button([dbc.Spinner(size="sm", id='export-eem-dataset-spinner')],
                                       id='export-eem-dataset', className='col-5')
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

page_eem_processing = html.Div([
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
    if reference_path is not None and reference_path != []:
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
                    steps_track += ["Warning: indices of EEM dataset and reference file are \nnot exactly the "
                                    "same. The reference value of samples \nwith unmatched indices would be "
                                    "set as NaN.\n"]
                refs = np.array(
                    [refs_from_file.loc[indexes[i]] if indexes[i] in refs_from_file.index
                     else np.full(shape=(refs_from_file.shape[1]), fill_value=np.nan) for i in range(len(indexes))]
                )
                refs = pd.DataFrame(refs, index=indexes, columns=refs_from_file.columns)
            else:
                if refs_from_file.shape[0] != len(indexes):
                    return None, (
                        'Error: number of samples in reference file is not the same as the EEM dataset. This error '
                        'occurs also when index starting/ending positions are not specified.'), "build"
                refs = refs_from_file
        else:
            return None, ('Error: No such file or directory: ' + reference_path), "Build"
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
        'ref': [refs.columns.tolist()] + refs.values.tolist() if eem_dataset.ref is not None else None,
        'cluster': None,
    }

    return eem_dataset_json_dict, dbc.Label(steps_track, style={'whiteSpace': 'pre'}), "Build"


#   ---------------Export EEM

@app.callback(
    [
        Output('message-eem-dataset-export', 'children'),
        Output('export-eem-dataset-spinner', 'children'),
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
        return [None], "Export"
    if eem_dataset_json_dict is None:
        message = ['Please first build the eem dataset.']
        return message, "Export"
    if not os.path.isdir(export_folder_path):
        message = ['Error: No such file or directory: ' + export_folder_path]
        return message, "Export"
    else:
        path = export_folder_path + '/' + export_filename + '.' + export_format
    with open(path, 'w') as file:
        json.dump(eem_dataset_json_dict, file)

    return ["EEM dataset exported."], "Export"


# -----------Page #2: Peak picking--------------

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


page_peak_picking = html.Div([
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

@app.callback(
    [
        Output('pp-eem-dataset-establishment-message', 'children'),
        Output('pp-intensities', 'children'),
        Output('build-pp-spinner', 'children'),
        Output('pp-establishment-corr-ref-selection', 'options'),
        Output('pp-establishment-corr-ref-selection', 'value'),
        Output('pp-test-pred-ref-selection', 'options'),
        Output('pp-test-pred-ref-selection', 'value'),
        Output('pp-model', 'data'),
    ],
    [
        Input('build-pp-model', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('pp-eem-dataset-establishment-path-input', 'value'),
        State('pp-establishment-index-kw-mandatory', 'value'),
        State('pp-establishment-index-kw-optional', 'value'),
        State('pp-excitation', 'value'),
        State('pp-emission', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_pp_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, ex_target, em_target,
                      eem_dataset_dict):
    if n_clicks is None:
        return None, None, 'Build model', [], None, [], None, None
    if not path_establishment:
        if eem_dataset_dict is None:
            message = (
                'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                'section, or import an EEM dataset from file.')
            return None, None, 'Build model', [], None, [], None, None
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                             index=eem_dataset_dict['index'])
            if eem_dataset_dict['ref'] is not None else None,
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return message, None, 'Build model', [], None, [], None, None
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
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                 index=eem_dataset_dict['index'])
                if eem_dataset_dict['ref'] is not None else None,
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              copy=False)

    pp_model = {}
    intensities_tabs = dbc.Card([])

    if eem_dataset_establishment.ref is not None:
        valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
    else:
        valid_ref = None

    fi, ex_actual, em_actual = eem_dataset_establishment.peak_picking(ex=ex_target, em=em_target)
    fi_name = f'Intensity (ex={ex_actual} nm, em={em_actual} nm)'

    pp_fit_params = {}
    if eem_dataset_establishment.ref is not None:
        for ref_var in valid_ref:
            x = eem_dataset_establishment.ref[ref_var]
            stats = []
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            if x.shape[0] < 1:
                continue
            y = fi.squeeze()
            y = y.drop(nan_rows)
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([fi_name, slope, intercept, r_squared, pearson_corr, pearson_p])
            pp_fit_params[ref_var] = stats

        pp_model = {
            'intensities': [fi.columns.tolist()] + fi.values.tolist(),
            'index': eem_dataset_establishment.index,
            'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
            if eem_dataset_establishment.ref is not None else None,
            'ex_actual': ex_actual,
            'em_actual': em_actual,
            'fitting_params': pp_fit_params
        }

    # fmax
    intensities_tabs.children.append(
        html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(figure=plot_fmax(fi,
                                                       display=False,
                                                       yaxis_title=fi_name
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
                            dbc.Table.from_dataframe(fi,
                                                     bordered=True, hover=True, index=True)
                        ]
                    )
                ]
            )
        ]),
    )

    ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
        (eem_dataset_establishment.ref is not None) else None

    return None, intensities_tabs, 'Build model', ref_options, None, ref_options, None, pp_model


# -----------Analyze correlations between score/Fmax and reference variables in model establishment

@app.callback(
    [
        Output('pp-establishment-corr-graph', 'figure'),  # size, intervals?
        Output('pp-establishment-corr-table', 'children'),
    ],
    [
        Input('pp-establishment-corr-ref-selection', 'value'),
        State('pp-model', 'data')
    ]
)
def on_pp_establishment_correlations(ref_var, pp_model):
    if all([ref_var, pp_model]):
        ref_df = pd.DataFrame(pp_model['ref'][1:], columns=pp_model['ref'][0],
                              index=pp_model['index'])
        intensities_df = pd.DataFrame(pp_model['intensities'][1:], columns=pp_model['intensities'][0],
                               index=pp_model['index'])
        ref_df = pd.concat([ref_df, intensities_df], axis=1)
        var = ref_df[ref_var]
        fig = go.Figure()

        stats = pp_model['fitting_params']

        x = var
        y = intensities_df.iloc[:, 0]
        nan_rows = x[x.isna()].index
        x = x.drop(nan_rows)
        y = y.drop(nan_rows)
        if x.shape[0] < 1:
            return go.Figure(), None
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=stats[ref_var][0][0], text=[i for i in x.index],
                                 marker=dict(color=colors[0]), hoverinfo='text+x+y'))
        fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                 y=stats[ref_var][0][1] * np.array([x.min(), x.max()]) + stats[ref_var][0][2],
                                 mode='lines', name=f'{stats[ref_var][0][0]}-Linear Regression Line',
                                 line=dict(dash='dash', color=colors[0])))
        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text='Intensity')

        tbl = pd.DataFrame(
            stats[ref_var],
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Fit a test EEM dataset using the established Linear model
@app.callback(
    [
        Output('pp-eem-dataset-predict-message', 'children'),  # size, intervals?
        Output('pp-test-intensities', 'children'),
        Output('pp-test-error', 'children'),
        Output('pp-test-corr-indicator-selection', 'options'),
        Output('pp-test-corr-indicator-selection', 'value'),
        Output('pp-test-corr-ref-selection', 'options'),
        Output('pp-test-corr-ref-selection', 'value'),
        Output('pp-predict-spinner', 'children'),
        Output('pp-test-results', 'data'),
    ],
    [
        Input('predict-pp-model', 'n_clicks'),
        State('pp-eem-dataset-predict-path-input', 'value'),
        State('pp-test-index-kw-mandatory', 'value'),
        State('pp-test-index-kw-optional', 'value'),
        State('pp-model', 'data')
    ]
)
def on_pp_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, pp_model):
    if n_clicks is None:
        return None, None, None, [], None, [], None, 'predict', None
    if path_predict is None:
        return None, None, None, [], None, [], None, 'predict', None
    if not os.path.exists(path_predict):
        message = ('Error: No such file: ' + path_predict)
        return message, None, None, [], None, [], None, 'predict', None
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
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], index=eem_dataset_dict['index'],
                             columns=eem_dataset_dict['ref'][0]) if eem_dataset_dict['ref'] is not None else None,
        )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, copy=False)

    if eem_dataset_predict.ref is not None:
        valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
    else:
        valid_ref = None

    fi_test, ex_actual_test, em_actual_test = eem_dataset_predict.peak_picking(ex=pp_model['ex_actual'], em=pp_model['em_actual'])

    pred = {}
    if eem_dataset_predict.ref is not None:
        for ref_var in valid_ref:
            if ref_var in pp_model['fitting_params'].keys():
                params = pp_model['fitting_params'][ref_var]
                pred_sample = fi_test.copy()
                pred_r = fi_test.iloc[:, 0] - params[0][2]
                pred_r = pred_r / params[0][1]
                pred_sample.iloc[:,0] = pred_r
                pred[ref_var] = [pred_sample.columns.tolist()] + pred_sample.values.tolist()
            else:
                pred[ref_var] = None

    intensities_tab = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(figure=plot_fmax(fi_test,
                                                   display=False,
                                                   yaxis_title='Intensities of test dataset'
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
                        dbc.Table.from_dataframe(fi_test,
                                                 bordered=True, hover=True, index=True
                                                 )
                    ]
                ),
            ]
        ),
    ])

    error_tab = html.Div(children=[])

    if eem_dataset_predict.ref is not None:
        indicator_options = [
            {'label': 'Intensities of test dataset', 'value': 'Intensities of test dataset'},
            {'label': 'Prediction of reference', 'value': 'Prediction of reference'}
        ]
    else:
        indicator_options = [
            {'label': 'Intensities of test dataset', 'value': 'Intensities of test dataset'}
        ]

    ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

    test_results = {
        'Intensities of test dataset': [fi_test.columns.tolist()] + fi_test.values.tolist(),
        'Prediction of reference': pred,
        'ref': [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
        if eem_dataset_predict.ref is not None else None,
        'index': eem_dataset_predict.index
    }

    return None, intensities_tab, error_tab, indicator_options, None, ref_options, None, 'predict', test_results


# -----------Predict the corresponding reference variables for the test EEM datasets using the model fitted in the model
#            establishment step
@app.callback(
    [
        Output('pp-test-pred-graph', 'figure'),  # size, intervals?
        Output('pp-test-pred-table', 'children'),
    ],
    [
        Input('predict-pp-model', 'n_clicks'),
        Input('pp-test-pred-ref-selection', 'value'),
        State('pp-test-results', 'data'),
    ]
)
def on_pp_test_predict_reference(n_clicks, ref_var, pp_test_results):
    if all([ref_var, pp_test_results]):
        pred = pp_test_results['Prediction of reference'][ref_var]
        pred = pd.DataFrame(pred[1:], columns=pred[0], index=pp_test_results['index'])
        fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
        if pp_test_results['ref'] is not None:
            ref = pd.DataFrame(pp_test_results['ref'][1:], columns=pp_test_results['ref'][0],
                               index=pp_test_results['index'])
            true_value = ref[ref_var]
            nan_rows = true_value[true_value.isna()].index
            true_value = true_value.drop(nan_rows)
            pred_overlap = pred.drop(nan_rows)
            fig.add_trace(go.Scatter(x=true_value.index, y=true_value.values, mode='markers',
                                     marker=dict(color='black', size=5),
                                     name='True value'))
            rmse_results = []
            for f_col in pred.columns:
                rmse = sqrt(sum((true_value.values - pred_overlap[f_col]) ** 2) / len(true_value.values))
                rmse_results.append(rmse)
            tbl = html.Div(
                [
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.DataFrame([rmse_results], columns=pred.columns,
                                                              index=pd.Index(['RMSE'], name='Error metric')),
                                                 bordered=True, hover=True, index=True)
                    ),
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.concat([pred, ref[ref_var]], axis=1),
                                                 bordered=True, hover=True, index=True)
                    )
                ]
            )
        else:
            tbl = dbc.Table.from_dataframe(pred, bordered=True, hover=True, index=True)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Analyze correlations between Fmax and reference variables in model testing
@app.callback(
    [
        Output('pp-test-corr-graph', 'figure'),  # size, intervals?
        Output('pp-test-corr-table', 'children'),
    ],
    [
        Input('pp-test-corr-indicator-selection', 'value'),
        Input('pp-test-corr-ref-selection', 'value'),
        State('pp-test-results', 'data'),
    ]
)
def on_pp_test_correlations(indicator, ref_var, pp_test_results):
    if all([indicator, ref_var, pp_test_results]):
        ref_df = pd.DataFrame(pp_test_results['ref'][1:], columns=pp_test_results['ref'][0],
                              index=pp_test_results['index'])
        var = ref_df[ref_var]
        if indicator != 'Prediction of reference':
            pp_var = pd.DataFrame(pp_test_results[indicator][1:], columns=pp_test_results[indicator][0],
                                       index=pp_test_results['index'])
        else:
            if pp_test_results[indicator][ref_var] is not None:
                pp_var = pd.DataFrame(pp_test_results[indicator][ref_var][1:],
                                           columns=pp_test_results[indicator][ref_var][0],
                                           index=pp_test_results['index'])
            else:
                return go.Figure(), None
        fig = go.Figure()
        stats = []
        for i, col in enumerate(pp_var.columns):
            x = var
            y = pp_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            predictions = lm.predict(x_reshaped)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     hoverinfo='text+x+y', marker=dict(color=colors[i % 10])))
            fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name=f'{col} fit',
                                     line=dict(dash='dash', color=colors[i % 10])))
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([col, slope, intercept, r_squared, pearson_corr, pearson_p])

        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text="Prediction of " + ref_var if indicator == 'Prediction of reference' else indicator)

        tbl = pd.DataFrame(
            stats,
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None



# -----------Page #3: Regional integration--------------

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

@app.callback(
    [
        Output('ri-eem-dataset-establishment-message', 'children'),
        Output('ri-intensities', 'children'),
        Output('build-ri-spinner', 'children'),
        Output('ri-establishment-corr-ref-selection', 'options'),
        Output('ri-establishment-corr-ref-selection', 'value'),
        Output('ri-test-pred-ref-selection', 'options'),
        Output('ri-test-pred-ref-selection', 'value'),
        Output('ri-model', 'data'),
    ],
    [
        Input('build-ri-model', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('ri-eem-dataset-establishment-path-input', 'value'),
        State('ri-establishment-index-kw-mandatory', 'value'),
        State('ri-establishment-index-kw-optional', 'value'),
        State('ri-ex-min', 'value'),
        State('ri-ex-max', 'value'),
        State('ri-em-min', 'value'),
        State('ri-em-max', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_ri_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, ex_min, ex_max,
                      em_min, em_max, eem_dataset_dict):
    if n_clicks is None:
        return None, None, 'Build model', [], None, [], None, None
    if not path_establishment:
        if eem_dataset_dict is None:
            message = (
                'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                'section, or import an EEM dataset from file.')
            return None, None, 'Build model', [], None, [], None, None
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                             index=eem_dataset_dict['index'])
            if eem_dataset_dict['ref'] is not None else None,
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return message, None, 'Build model', [], None, [], None, None
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
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                 index=eem_dataset_dict['index'])
                if eem_dataset_dict['ref'] is not None else None,
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              copy=False)

    ri_model = {}
    intensities_tabs = dbc.Card([])

    if eem_dataset_establishment.ref is not None:
        valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
    else:
        valid_ref = None

    ri = eem_dataset_establishment.regional_integration(ex_min=ex_min, ex_max=ex_max,
                                                        em_min=em_min, em_max=em_max)
    ri_name = f'RI (ex=[{ex_min}, {ex_max}] nm, em=[{em_min}, {em_max}] nm)'

    ri_fit_params = {}
    if eem_dataset_establishment.ref is not None:
        for ref_var in valid_ref:
            x = eem_dataset_establishment.ref[ref_var]
            stats = []
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            if x.shape[0] < 1:
                continue
            y = ri.squeeze()
            y = y.drop(nan_rows)
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([ri_name, slope, intercept, r_squared, pearson_corr, pearson_p])
            ri_fit_params[ref_var] = stats

        ri_model = {
            'intensities': [ri.columns.tolist()] + ri.values.tolist(),
            'index': eem_dataset_establishment.index,
            'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
            if eem_dataset_establishment.ref is not None else None,
            'ex_min': ex_min,
            'ex_max': ex_max,
            'em_min': em_min,
            'em_max': em_max,
            'fitting_params': ri_fit_params
        }

        # fmax
    intensities_tabs.children.append(
        html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(figure=plot_fmax(ri,
                                                       display=False,
                                                       yaxis_title=ri_name
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
                            dbc.Table.from_dataframe(ri,
                                                     bordered=True, hover=True, index=True)
                        ]
                    )
                ]
            )
        ]),
    )

    ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
        (eem_dataset_establishment.ref is not None) else None

    return None, intensities_tabs, 'Build model', ref_options, None, ref_options, None, ri_model


# -----------Analyze correlations between score/Fmax and reference variables in model establishment

@app.callback(
    [
        Output('ri-establishment-corr-graph', 'figure'),  # size, intervals?
        Output('ri-establishment-corr-table', 'children'),
    ],
    [
        Input('ri-establishment-corr-ref-selection', 'value'),
        State('ri-model', 'data')
    ]
)
def on_ri_establishment_correlations(ref_var, ri_model):
    if all([ref_var, ri_model]):
        ref_df = pd.DataFrame(ri_model['ref'][1:], columns=ri_model['ref'][0],
                              index=ri_model['index'])
        intensities_df = pd.DataFrame(ri_model['intensities'][1:], columns=ri_model['intensities'][0],
                               index=ri_model['index'])
        ref_df = pd.concat([ref_df, intensities_df], axis=1)
        var = ref_df[ref_var]
        fig = go.Figure()

        stats = ri_model['fitting_params']

        x = var
        y = intensities_df.iloc[:, 0]
        nan_rows = x[x.isna()].index
        x = x.drop(nan_rows)
        y = y.drop(nan_rows)
        if x.shape[0] < 1:
            return go.Figure(), None
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=stats[ref_var][0][0], text=[i for i in x.index],
                                 marker=dict(color=colors[0]), hoverinfo='text+x+y'))
        fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                 y=stats[ref_var][0][1] * np.array([x.min(), x.max()]) + stats[ref_var][0][2],
                                 mode='lines', name=f'{stats[ref_var][0][0]}-Linear Regression Line',
                                 line=dict(dash='dash', color=colors[0])))
        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text='Intensity')

        tbl = pd.DataFrame(
            stats[ref_var],
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Fit a test EEM dataset using the established RFI model
@app.callback(
    [
        Output('ri-eem-dataset-predict-message', 'children'),  # size, intervals?
        Output('ri-test-intensities', 'children'),
        Output('ri-test-error', 'children'),
        Output('ri-test-corr-indicator-selection', 'options'),
        Output('ri-test-corr-indicator-selection', 'value'),
        Output('ri-test-corr-ref-selection', 'options'),
        Output('ri-test-corr-ref-selection', 'value'),
        Output('ri-predict-spinner', 'children'),
        Output('ri-test-results', 'data'),
    ],
    [
        Input('predict-ri-model', 'n_clicks'),
        State('ri-eem-dataset-predict-path-input', 'value'),
        State('ri-test-index-kw-mandatory', 'value'),
        State('ri-test-index-kw-optional', 'value'),
        State('ri-model', 'data')
    ]
)
def on_ri_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, ri_model):
    if n_clicks is None:
        return None, None, None, [], None, [], None, 'predict', None
    if path_predict is None:
        return None, None, None, [], None, [], None, 'predict', None
    if not os.path.exists(path_predict):
        message = ('Error: No such file: ' + path_predict)
        return message, None, None, [], None, [], None, 'predict', None
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
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], index=eem_dataset_dict['index'],
                             columns=eem_dataset_dict['ref'][0]) if eem_dataset_dict['ref'] is not None else None,
        )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, copy=False)

    if eem_dataset_predict.ref is not None:
        valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
    else:
        valid_ref = None

    ri_test = eem_dataset_predict.regional_integration(
        ex_min=ri_model['ex_min'], ex_max=ri_model['ex_max'],
        em_min=ri_model['em_min'], em_max=ri_model['em_max']
    )

    pred = {}
    if eem_dataset_predict.ref is not None:
        for ref_var in valid_ref:
            if ref_var in ri_model['fitting_params'].keys():
                params = ri_model['fitting_params'][ref_var]
                pred_sample = ri_test.copy()
                pred_r = ri_test.iloc[:, 0] - params[0][2]
                pred_r = pred_r / params[0][1]
                pred_sample.iloc[:,0] = pred_r
                pred[ref_var] = [pred_sample.columns.tolist()] + pred_sample.values.tolist()
            else:
                pred[ref_var] = None

    intensities_tab = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(figure=plot_fmax(ri_test,
                                                   display=False,
                                                   yaxis_title='Intensities of test dataset'
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
                        dbc.Table.from_dataframe(ri_test,
                                                 bordered=True, hover=True, index=True
                                                 )
                    ]
                ),
            ]
        ),
    ])

    error_tab = html.Div(children=[])

    if eem_dataset_predict.ref is not None:
        indicator_options = [
            {'label': 'RI of test dataset', 'value': 'RI of test dataset'},
            {'label': 'Prediction of reference', 'value': 'Prediction of reference'}
        ]
    else:
        indicator_options = [
            {'label': 'RI of test dataset', 'value': 'RI of test dataset'}
        ]

    ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

    test_results = {
        'RI of test dataset': [ri_test.columns.tolist()] + ri_test.values.tolist(),
        'Prediction of reference': pred,
        'ref': [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
        if eem_dataset_predict.ref is not None else None,
        'index': eem_dataset_predict.index
    }

    return None, intensities_tab, error_tab, indicator_options, None, ref_options, None, 'predict', test_results


# -----------Predict the corresponding reference variables for the test EEM datasets using the model fitted in the model
#            establishment step
@app.callback(
    [
        Output('ri-test-pred-graph', 'figure'),  # size, intervals?
        Output('ri-test-pred-table', 'children'),
    ],
    [
        Input('predict-ri-model', 'n_clicks'),
        Input('ri-test-pred-ref-selection', 'value'),
        State('ri-test-results', 'data'),
    ]
)
def on_ri_test_predict_reference(n_clicks, ref_var, ri_test_results):
    if all([ref_var, ri_test_results]):
        pred = ri_test_results['Prediction of reference'][ref_var]
        pred = pd.DataFrame(pred[1:], columns=pred[0], index=ri_test_results['index'])
        fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
        if ri_test_results['ref'] is not None:
            ref = pd.DataFrame(ri_test_results['ref'][1:], columns=ri_test_results['ref'][0],
                               index=ri_test_results['index'])
            true_value = ref[ref_var]
            nan_rows = true_value[true_value.isna()].index
            true_value = true_value.drop(nan_rows)
            pred_overlap = pred.drop(nan_rows)
            fig.add_trace(go.Scatter(x=true_value.index, y=true_value.values, mode='markers',
                                     marker=dict(color='black', size=5),
                                     name='True value'))
            rmse_results = []
            for f_col in pred.columns:
                rmse = sqrt(sum((true_value.values - pred_overlap[f_col]) ** 2) / len(true_value.values))
                rmse_results.append(rmse)
            tbl = html.Div(
                [
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.DataFrame([rmse_results], columns=pred.columns,
                                                              index=pd.Index(['RMSE'], name='Error metric')),
                                                 bordered=True, hover=True, index=True)
                    ),
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.concat([pred, ref[ref_var]], axis=1),
                                                 bordered=True, hover=True, index=True)
                    )
                ]
            )
        else:
            tbl = dbc.Table.from_dataframe(pred, bordered=True, hover=True, index=True)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Analyze correlations between Fmax and reference variables in model testing
@app.callback(
    [
        Output('ri-test-corr-graph', 'figure'),  # size, intervals?
        Output('ri-test-corr-table', 'children'),
    ],
    [
        Input('ri-test-corr-indicator-selection', 'value'),
        Input('ri-test-corr-ref-selection', 'value'),
        State('ri-test-results', 'data'),
    ]
)
def on_ri_test_correlations(indicator, ref_var, ri_test_results):
    if all([indicator, ref_var, ri_test_results]):
        ref_df = pd.DataFrame(ri_test_results['ref'][1:], columns=ri_test_results['ref'][0],
                              index=ri_test_results['index'])
        var = ref_df[ref_var]
        if indicator != 'Prediction of reference':
            ri_var = pd.DataFrame(ri_test_results[indicator][1:], columns=ri_test_results[indicator][0],
                                       index=ri_test_results['index'])
        else:
            if ri_test_results[indicator][ref_var] is not None:
                ri_var = pd.DataFrame(ri_test_results[indicator][ref_var][1:],
                                           columns=ri_test_results[indicator][ref_var][0],
                                           index=ri_test_results['index'])
            else:
                return go.Figure(), None
        fig = go.Figure()
        stats = []
        for i, col in enumerate(ri_var.columns):
            x = var
            y = ri_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            predictions = lm.predict(x_reshaped)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     hoverinfo='text+x+y', marker=dict(color=colors[i % 10])))
            fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name=f'{col} fit',
                                     line=dict(dash='dash', color=colors[i % 10])))
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([col, slope, intercept, r_squared, pearson_corr, pearson_p])

        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text="Prediction of " + ref_var if indicator == 'Prediction of reference' else indicator)

        tbl = pd.DataFrame(
            stats,
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None




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
                                    dbc.Label("Optimizer"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'Multiplicative update', 'value': 'mu'},
                                            {'label': 'Hierarchical ALS', 'value': 'hals'},
                                        ],
                                        id='parafac-optimizer', value='mu'),
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

#   -------------Layout of page #3

page_parafac = html.Div([
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
                        # dcc.Tab(label='Scores', id='parafac-scores'),
                        dcc.Tab(label='Fmax', id='parafac-fmax'),
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


#   -------------Callbacks of page #2


#  ----------Establish PARAFAC model
@app.callback(
    [
        Output('parafac-eem-dataset-establishment-message', 'children'),
        Output('parafac-loadings', 'children'),
        Output('parafac-components', 'children'),
        # Output('parafac-scores', 'children'),
        Output('parafac-fmax', 'children'),
        Output('parafac-variance-explained', 'children'),
        Output('parafac-core-consistency', 'children'),
        Output('parafac-leverage', 'children'),
        Output('parafac-split-half', 'children'),
        Output('build-parafac-spinner', 'children'),
        Output('parafac-establishment-corr-model-selection', 'options'),
        Output('parafac-establishment-corr-model-selection', 'value'),
        # Output('parafac-establishment-corr-ref-selection', 'options'),
        # Output('parafac-establishment-corr-ref-selection', 'value'),
        Output('parafac-test-model-selection', 'options'),
        Output('parafac-test-model-selection', 'value'),
        Output('parafac-test-pred-ref-selection', 'options'),
        Output('parafac-test-pred-ref-selection', 'value'),
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
        State('parafac-optimizer', 'value'),
        State('parafac-validations', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_parafac_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, rank, init, nn,
                           tf, optimizer, validations, eem_dataset_dict):
    if n_clicks is None:
        return (None, None, None, None, None, None, None, None, 'Build model', [], None, [], None, [], None,
                None)
    if not path_establishment:
        if eem_dataset_dict is None:
            message = (
                'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                'section, or import an EEM dataset from file.')
            return (message, None, None, None, None, None, None, None, 'Build model', [], None, [], None, [], None, None)
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                             index=eem_dataset_dict['index'])
            if eem_dataset_dict['ref'] is not None else None,
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return (message, None, None, None, None, None, None, None, 'Build model', [], None, [], None, [], None, None)
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
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                 index=eem_dataset_dict['index'])
                if eem_dataset_dict['ref'] is not None else None,
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              copy=False)

    rank_list = num_string_to_list(rank)
    parafac_models = {}
    loadings_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    components_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    scores_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    fmax_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    variance_explained_tabs = dbc.Card(children=[])
    core_consistency_tabs = dbc.Card(children=[])
    leverage_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    split_half_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    cc = []
    ve = []

    if eem_dataset_establishment.ref is not None:
        valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
    else:
        valid_ref = None

    for r in rank_list:
        parafac_r = PARAFAC(n_components=r, init=init, non_negativity=True if 'non_negative' in nn else False,
                            tf_normalization=True if 'tf_normalization' in tf else False, optimizer=optimizer,
                            sort_em=True, loadings_normalization='maximum')
        parafac_r.fit(eem_dataset_establishment)
        parafac_fit_params_r = {}
        if eem_dataset_establishment.ref is not None:
            for ref_var in valid_ref:
                x = eem_dataset_establishment.ref[ref_var]
                parafac_var = parafac_r.fmax
                stats = []
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                if x.shape[0] < 1:
                    continue
                for f_col in parafac_var.columns:
                    y = parafac_var[f_col]
                    y = y.drop(nan_rows)
                    x_reshaped = np.array(x).reshape(-1, 1)
                    lm = LinearRegression().fit(x_reshaped, y)
                    r_squared = lm.score(x_reshaped, y)
                    intercept = lm.intercept_
                    slope = lm.coef_[0]
                    pearson_corr, pearson_p = pearsonr(x, y)
                    stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                parafac_fit_params_r[ref_var] = stats
        for c_var in parafac_r.fmax.columns:
            x = parafac_r.fmax[c_var]
            parafac_var = parafac_r.fmax
            stats = []
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            if x.shape[0] < 1:
                continue
            for f_col in parafac_var.columns:
                y = parafac_var[f_col]
                y = y.drop(nan_rows)
                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                r_squared = lm.score(x_reshaped, y)
                intercept = lm.intercept_
                slope = lm.coef_[0]
                pearson_corr, pearson_p = pearsonr(x, y)
                stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
            parafac_fit_params_r[c_var] = stats
        parafac_models[r] = {
            'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                           sublist in parafac_r.components.tolist()],
            'score': [parafac_r.score.columns.tolist()] + parafac_r.score.values.tolist(),
            'Fmax': [parafac_r.fmax.columns.tolist()] + parafac_r.fmax.values.tolist(),
            'index': eem_dataset_establishment.index,
            'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
            if eem_dataset_establishment.ref is not None else None,
            'fitting_params': parafac_fit_params_r
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
                                            figure=plot_eem(parafac_r.components[3 * i],
                                                            ex_range=parafac_r.ex_range,
                                                            em_range=parafac_r.em_range,
                                                            vmin=0 if np.min(
                                                                parafac_r.components[3 * i]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
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
                                            figure=plot_eem(parafac_r.components[3 * i + 1],
                                                            ex_range=parafac_r.ex_range,
                                                            em_range=parafac_r.em_range,
                                                            vmin=0 if np.min(
                                                                parafac_r.components[
                                                                    3 * i + 1]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
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
                                            figure=plot_eem(parafac_r.components[3 * i + 2],
                                                            ex_range=parafac_r.ex_range,
                                                            em_range=parafac_r.em_range,
                                                            vmin=0 if np.min(
                                                                parafac_r.components[
                                                                    3 * i + 2]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 3}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 3 <= r else go.Figure(),
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
                                            dcc.Graph(figure=plot_fmax(parafac_r.score,
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
                                            dcc.Graph(figure=plot_fmax(parafac_r.fmax,
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

        # variance explained
        if 'variance_explained' in validations:
            ve.append(parafac_r.variance_explained())

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

    if 'variance_explained' in validations:
        ve_table = pd.DataFrame({'Number of components': rank_list, 'Variance explained': ve})
        variance_explained_tabs.children.append(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    figure=px.line(
                                        x=ve_table['Number of components'],
                                        y=ve_table['Variance explained'],
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
                                dbc.Table.from_dataframe(ve_table,
                                                         bordered=True, hover=True,
                                                         )
                            ]
                        ),
                    ]
                ),
            ]),
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

    model_options = [{'label': '{r}-component'.format(r=r), 'value': r} for r in parafac_models.keys()]
    ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
        (eem_dataset_establishment.ref is not None) else None

    return (None, loadings_tabs, components_tabs, fmax_tabs, variance_explained_tabs, core_consistency_tabs,
            leverage_tabs, split_half_tabs, 'Build model', model_options, None, model_options, None, ref_options,
            None, parafac_models)


# -----------Update reference selection dropdown

@app.callback(
    [
        Output('parafac-establishment-corr-ref-selection', 'options'),
        Output('parafac-establishment-corr-ref-selection', 'value'),
    ],
    [
        Input('parafac-establishment-corr-model-selection', 'value'),
        State('parafac-models', 'data')
    ]
)
def update_parafac_reference_dropdown_by_selected_model(r, parafac_model):
    if all([r, parafac_model]):
        options = list(parafac_model[str(r)]['fitting_params'].keys())
        return options, None
    else:
        return [], None


# -----------Analyze correlations between score/Fmax and reference variables in model establishment

@app.callback(
    [
        Output('parafac-establishment-corr-graph', 'figure'),  # size, intervals?
        Output('parafac-establishment-corr-table', 'children'),
    ],
    [
        Input('parafac-establishment-corr-model-selection', 'value'),
        Input('parafac-establishment-corr-indicator-selection', 'value'),
        Input('parafac-establishment-corr-ref-selection', 'value'),
        State('parafac-models', 'data')
    ]
)
def on_parafac_establishment_correlations(r, indicator, ref_var, parafac_models):
    if all([r, indicator, ref_var, parafac_models]):
        ref_df = pd.DataFrame(parafac_models[str(r)]['ref'][1:], columns=parafac_models[str(r)]['ref'][0],
                              index=parafac_models[str(r)]['index'])
        fmax_df = pd.DataFrame(parafac_models[str(r)]['Fmax'][1:], columns=parafac_models[str(r)]['Fmax'][0],
                               index=parafac_models[str(r)]['index'])
        ref_df = pd.concat([ref_df, fmax_df], axis=1)
        var = ref_df[ref_var]
        parafac_var = pd.DataFrame(parafac_models[str(r)][indicator][1:], columns=parafac_models[str(r)][indicator][0],
                                   index=parafac_models[str(r)]['index'])
        fig = go.Figure()

        stats = parafac_models[str(r)]['fitting_params']

        for i, col in enumerate(parafac_var.columns):
            x = var
            y = parafac_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     marker=dict(color=colors[i % 10]), hoverinfo='text+x+y'))
            fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                     y=stats[ref_var][i][1] * np.array([x.min(), x.max()]) + stats[ref_var][i][2],
                                     mode='lines', name=f'{col}-Linear Regression Line',
                                     line=dict(dash='dash', color=colors[i % 10])))
        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text=indicator)

        tbl = pd.DataFrame(
            stats[ref_var],
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Fit a test EEM dataset using the established PARAFAC model components
@app.callback(
    [
        Output('parafac-eem-dataset-predict-message', 'children'),  # size, intervals?
        # Output('parafac-test-score', 'children'),
        Output('parafac-test-fmax', 'children'),
        Output('parafac-test-error', 'children'),
        Output('parafac-test-corr-ref-selection', 'options'),
        Output('parafac-test-corr-ref-selection', 'value'),
        Output('parafac-test-corr-indicator-selection', 'options'),
        Output('parafac-test-corr-indicator-selection', 'value'),
        Output('parafac-predict-spinner', 'children'),
        Output('parafac-test-results', 'data'),
    ],
    [
        Input('predict-parafac-model', 'n_clicks'),
        State('parafac-eem-dataset-predict-path-input', 'value'),
        State('parafac-test-index-kw-mandatory', 'value'),
        State('parafac-test-index-kw-optional', 'value'),
        State('parafac-test-model-selection', 'value'),
        State('parafac-models', 'data')
    ]
)
def on_parafac_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, model_r, parafac_models):
    if n_clicks is None:
        return (None, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    if path_predict is None:
        return (None, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    if not os.path.exists(path_predict):
        message = ('Error: No such file: ' + path_predict)
        return (message, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
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
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], index=eem_dataset_dict['index'],
                             columns=eem_dataset_dict['ref'][0]) if eem_dataset_dict['ref'] is not None else None,
        )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, copy=False)

    if eem_dataset_predict.ref is not None:
        valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
    else:
        valid_ref = None

    score_sample, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset_predict.eem_stack,
                                                                    np.array(
                                                                        [
                                                                            [
                                                                                [np.nan if x is None else x for x in
                                                                                 subsublist] for subsublist in sublist
                                                                            ]
                                                                            for sublist in
                                                                            parafac_models[str(model_r)][
                                                                                'components']
                                                                        ]
                                                                    ),
                                                                    fit_intercept=False)
    score_sample = pd.DataFrame(
        score_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i + 1) for i in range(model_r)]
    )
    fmax_sample = pd.DataFrame(
        fmax_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i + 1) for i in range(model_r)]
    )

    pred = {}
    if eem_dataset_predict.ref is not None:
        for ref_var in valid_ref:
            if ref_var in parafac_models[str(model_r)]['fitting_params'].keys():
                params = parafac_models[str(model_r)]['fitting_params'][ref_var]
                pred_var = fmax_sample.copy()
                for i, f_col in enumerate(fmax_sample):
                    pred_r = fmax_sample[f_col] - params[i][2]
                    pred_r = pred_r / params[i][1]
                    pred_var[f_col] = pred_r
                pred[ref_var] = [pred_var.columns.tolist()] + pred_var.values.tolist()
            else:
                pred[ref_var] = None

    score_tab = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(figure=plot_fmax(score_sample,
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
    ])

    fmax_tab = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(figure=plot_fmax(fmax_sample,
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
    ])

    error_tab = html.Div(children=[])

    if eem_dataset_predict.ref is not None:
        indicator_options = [
            # {'label': 'Score', 'value': 'Score'},
            {'label': 'Fmax', 'value': 'Fmax'},
            {'label': 'Prediction of reference', 'value': 'Prediction of reference'}
        ]
    else:
        indicator_options = [
            # {'label': 'Score', 'value': 'Score'},
            {'label': 'Fmax', 'value': 'Fmax'}
        ]

    ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

    test_results = {
        'Fmax': [fmax_sample.columns.tolist()] + fmax_sample.values.tolist(),
        'Score': [score_sample.columns.tolist()] + score_sample.values.tolist(),
        'Prediction of reference': pred,
        'ref': [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
        if eem_dataset_predict.ref is not None else None,
        'index': eem_dataset_predict.index
    }

    return (None, fmax_tab, error_tab, ref_options, None,
            indicator_options, None, 'predict', test_results)


# -----------Predict the corresponding reference variables for the test EEM datasets using the model fitted in the model
#            establishment step
@app.callback(
    [
        Output('parafac-test-pred-graph', 'figure'),  # size, intervals?
        Output('parafac-test-pred-table', 'children'),
    ],
    [
        Input('predict-parafac-model', 'n_clicks'),
        Input('parafac-test-pred-ref-selection', 'value'),
        Input('parafac-test-pred-model-selection', 'value'),
        State('parafac-test-results', 'data'),
    ]
)
def on_parafac_test_predict_reference(n_clicks, ref_var, pred_model, parafac_test_results):
    if all([ref_var, pred_model, parafac_test_results]):
        pred = parafac_test_results['Prediction of reference'][ref_var]
        pred = pd.DataFrame(pred[1:], columns=pred[0], index=parafac_test_results['index'])
        fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
        if parafac_test_results['ref'] is not None:
            ref = pd.DataFrame(parafac_test_results['ref'][1:], columns=parafac_test_results['ref'][0],
                               index=parafac_test_results['index'])
            true_value = ref[ref_var]
            nan_rows = true_value[true_value.isna()].index
            true_value = true_value.drop(nan_rows)
            pred_overlap = pred.drop(nan_rows)
            fig.add_trace(go.Scatter(x=true_value.index, y=true_value.values, mode='markers',
                                     marker=dict(color='black', size=5),
                                     name='True value'))
            rmse_results = []
            for f_col in pred.columns:
                rmse = sqrt(sum((true_value.values - pred_overlap[f_col]) ** 2) / len(true_value.values))
                rmse_results.append(rmse)
            tbl = html.Div(
                [
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.DataFrame([rmse_results], columns=pred.columns,
                                                              index=pd.Index(['RMSE'], name='Error metric')),
                                                 bordered=True, hover=True, index=True)
                    ),
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.concat([pred, ref[ref_var]], axis=1),
                                                 bordered=True, hover=True, index=True)
                    )
                ]
            )
        else:
            tbl = dbc.Table.from_dataframe(pred, bordered=True, hover=True, index=True)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Analyze correlations between Fmax and reference variables in model testing
@app.callback(
    [
        Output('parafac-test-corr-graph', 'figure'),  # size, intervals?
        Output('parafac-test-corr-table', 'children'),
    ],
    [
        Input('parafac-test-corr-indicator-selection', 'value'),
        Input('parafac-test-corr-ref-selection', 'value'),
        State('parafac-test-results', 'data'),
    ]
)
def on_parafac_test_correlations(indicator, ref_var, parafac_test_results):
    if all([indicator, ref_var, parafac_test_results]):
        ref_df = pd.DataFrame(parafac_test_results['ref'][1:], columns=parafac_test_results['ref'][0],
                              index=parafac_test_results['index'])
        var = ref_df[ref_var]
        if indicator != 'Prediction of reference':
            parafac_var = pd.DataFrame(parafac_test_results[indicator][1:], columns=parafac_test_results[indicator][0],
                                       index=parafac_test_results['index'])
        else:
            if parafac_test_results[indicator][ref_var] is not None:
                parafac_var = pd.DataFrame(parafac_test_results[indicator][ref_var][1:],
                                           columns=parafac_test_results[indicator][ref_var][0],
                                           index=parafac_test_results['index'])
            else:
                return go.Figure(), None
        fig = go.Figure()
        stats = []
        for i, col in enumerate(parafac_var.columns):
            x = var
            y = parafac_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            predictions = lm.predict(x_reshaped)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     hoverinfo='text+x+y', marker=dict(color=colors[i % 10])))
            fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name=f'{col} fit',
                                     line=dict(dash='dash', color=colors[i % 10])))
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([col, slope, intercept, r_squared, pearson_corr, pearson_p])

        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text="Prediction of " + ref_var if indicator == 'Prediction of reference' else indicator)

        tbl = pd.DataFrame(
            stats,
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


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
                                            {'label': 'Coordinate Descent solver', 'value': 'cd'},
                                            {'label': 'Multiplicative Update solver', 'value': 'mu'}
                                        ],
                                            value='cd', style={'width': '300px'}, id='nmf-solver'
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

#   -------------Layout of page #3

page_nmf = html.Div([
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


#   -------------Callbacks of page #3

#   -------------Establish NMF model

@app.callback(
    [
        Output('nmf-eem-dataset-establishment-message', 'children'),
        Output('nmf-components', 'children'),
        Output('nmf-fmax', 'children'),
        Output('nmf-residual', 'children'),
        Output('nmf-split-half', 'children'),
        Output('nmf-spinner', 'children'),
        Output('nmf-establishment-corr-model-selection', 'options'),
        Output('nmf-establishment-corr-model-selection', 'value'),
        # Output('nmf-establishment-corr-ref-selection', 'options'),
        # Output('nmf-establishment-corr-ref-selection', 'value'),
        Output('nmf-test-pred-ref-selection', 'options'),
        Output('nmf-test-pred-ref-selection', 'value'),
        Output('nmf-test-model-selection', 'options'),
        Output('nmf-test-model-selection', 'value'),
        Output('nmf-models', 'data'),
    ],
    [
        Input('build-nmf-model', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('nmf-eem-dataset-establishment-path-input', 'value'),
        State('nmf-establishment-index-kw-mandatory', 'value'),
        State('nmf-establishment-index-kw-optional', 'value'),
        State('nmf-rank', 'value'),
        State('nmf-solver', 'value'),
        State('nmf-init', 'value'),
        State('nmf-normalization-checkbox', 'value'),
        State('nmf-alpha-w', 'value'),
        State('nmf-alpha-h', 'value'),
        State('nmf-l1-ratio', 'value'),
        State('nmf-validations', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_nmf_model(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, rank, solver, init,
                       normalization, alpha_w, alpha_h, l1_ratio, validations, eem_dataset_dict):
    if n_clicks is None:
        return None, None, None, None, None, 'Build model', [], None, [], None, [], None, None
    if not path_establishment:
        if eem_dataset_dict is None:
            message = (
                'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                'section, or import an EEM dataset from file.')
            return message, None, None, None, None, 'Build model', [], None, [], None, [], None, None
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                             index=eem_dataset_dict['index'])
            if eem_dataset_dict['ref'] is not None else None,
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return message, None, None, None, None, 'Build model', [], None, [], None, [], None, None
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
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                 index=eem_dataset_dict['index'])
                if eem_dataset_dict['ref'] is not None else None,
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              copy=False)
    rank_list = num_string_to_list(rank)
    nmf_models = {}
    components_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    fmax_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    residual_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    split_half_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])

    if eem_dataset_establishment.ref is not None:
        valid_ref = eem_dataset_establishment.ref.columns[~eem_dataset_establishment.ref.isna().all()].tolist()
    else:
        valid_ref = None

    for r in rank_list:
        nmf_r = EEMNMF(
            n_components=r, solver=solver, init=init, normalization=normalization[0] if normalization else None,
            alpha_H=alpha_h, alpha_W=alpha_w, l1_ratio=l1_ratio
        )
        nmf_r.fit(eem_dataset_establishment)
        nmf_fit_params_r = {}
        if eem_dataset_establishment.ref is not None:
            for ref_var in valid_ref:
                x = eem_dataset_establishment.ref[ref_var]
                stats = []
                nmf_var = nmf_r.nnls_fmax
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                if x.shape[0] < 1:
                    continue
                for f_col in nmf_var.columns:
                    y = nmf_var[f_col]
                    y = y.drop(nan_rows)
                    x_reshaped = np.array(x).reshape(-1, 1)
                    lm = LinearRegression().fit(x_reshaped, y)
                    r_squared = lm.score(x_reshaped, y)
                    intercept = lm.intercept_
                    slope = lm.coef_[0]
                    pearson_corr, pearson_p = pearsonr(x, y)
                    stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                nmf_fit_params_r[ref_var] = stats
        for c_var in nmf_r.nnls_fmax.columns:
            x = nmf_r.nnls_fmax[c_var]
            parafac_var = nmf_r.nnls_fmax
            stats = []
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            if x.shape[0] < 1:
                continue
            for f_col in parafac_var.columns:
                y = parafac_var[f_col]
                y = y.drop(nan_rows)
                x_reshaped = np.array(x).reshape(-1, 1)
                lm = LinearRegression().fit(x_reshaped, y)
                r_squared = lm.score(x_reshaped, y)
                intercept = lm.intercept_
                slope = lm.coef_[0]
                pearson_corr, pearson_p = pearsonr(x, y)
                stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
            nmf_fit_params_r[c_var] = stats
        nmf_models[r] = {
            'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                           sublist in nmf_r.components.tolist()],
            'NNLS-Fmax': [nmf_r.nnls_fmax.columns.tolist()] + nmf_r.nnls_fmax.values.tolist(),
            'NMF-Fmax': [nmf_r.nmf_fmax.columns.tolist()] + nmf_r.nmf_fmax.values.tolist(),
            'index': eem_dataset_establishment.index,
            'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
            if eem_dataset_establishment.ref is not None else None,
            'fitting_params': nmf_fit_params_r
        }

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
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[3 * i]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
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
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    3 * i + 1]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
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
                                                            ex_range=eem_dataset_establishment.ex_range,
                                                            em_range=eem_dataset_establishment.em_range,
                                                            vmin=0 if np.min(
                                                                nmf_r.components[
                                                                    3 * i + 2]) > -1e-3 else None,
                                                            vmax=None,
                                                            auto_intensity_range=False,
                                                            plot_tool='plotly',
                                                            display=False,
                                                            figure_size=(5, 3.5),
                                                            axis_label_font_size=14,
                                                            cbar_font_size=12,
                                                            title_font_size=16,
                                                            title=f'C{3 * i + 3}',
                                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                                            else False,
                                                            rotate=True if 'rotate' in eem_graph_options else False,
                                                            ) if 3 * i + 3 <= r else go.Figure(),
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
                                            dcc.Graph(figure=plot_fmax(nmf_r.nnls_fmax,
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
                                            dbc.Table.from_dataframe(nmf_r.nnls_fmax,
                                                                     bordered=True, hover=True, index=True)
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=plot_fmax(nmf_r.nmf_fmax,
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
                                            dbc.Table.from_dataframe(nmf_r.nmf_fmax,
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

    model_options = [{'label': '{r}-component'.format(r=r), 'value': r} for r in nmf_models.keys()]
    ref_options = [{'label': var, 'value': var} for var in valid_ref] if (
            eem_dataset_establishment.ref is not None) else []

    return (None, components_tabs, fmax_tabs, residual_tabs, split_half_tabs, 'Build model', model_options, None,
            ref_options, None, model_options, None, nmf_models)


# -----------Update reference selection dropdown

@app.callback(
    [
        Output('nmf-establishment-corr-ref-selection', 'options'),
        Output('nmf-establishment-corr-ref-selection', 'value'),
    ],
    [
        Input('nmf-establishment-corr-model-selection', 'value'),
        State('nmf-models', 'data')
    ]
)
def update_nmf_reference_dropdown_by_selected_model(r, nmf_model):
    if all([r, nmf_model]):
        options = list(nmf_model[str(r)]['fitting_params'].keys())
        return options, None
    else:
        return [], None


# -----------Analyze correlations between score/Fmax and reference variables in model establishment

@app.callback(
    [
        Output('nmf-establishment-corr-graph', 'figure'),  # size, intervals?
        Output('nmf-establishment-corr-table', 'children'),
    ],
    [
        Input('nmf-establishment-corr-model-selection', 'value'),
        Input('nmf-establishment-corr-indicator-selection', 'value'),
        Input('nmf-establishment-corr-ref-selection', 'value'),
        State('nmf-models', 'data')
    ]
)
def on_nmf_establishment_correlations(r, indicator, ref_var, nmf_models):
    if all([r, indicator, ref_var, nmf_models]):
        ref_df = pd.DataFrame(nmf_models[str(r)]['ref'][1:], columns=nmf_models[str(r)]['ref'][0],
                              index=nmf_models[str(r)]['index'])
        nnls_fmax_df = pd.DataFrame(nmf_models[str(r)]['NNLS-Fmax'][1:], columns=nmf_models[str(r)]['NNLS-Fmax'][0],
                                    index=nmf_models[str(r)]['index'])
        nmf_fmax_df = pd.DataFrame(nmf_models[str(r)]['NMF-Fmax'][1:], columns=nmf_models[str(r)]['NMF-Fmax'][0],
                                   index=nmf_models[str(r)]['index'])
        ref_df = pd.concat([ref_df, nnls_fmax_df, nmf_fmax_df], axis=1)
        var = ref_df[ref_var]
        nmf_var = pd.DataFrame(nmf_models[str(r)][indicator][1:], columns=nmf_models[str(r)][indicator][0],
                               index=nmf_models[str(r)]['index'])
        fig = go.Figure()

        stats = nmf_models[str(r)]['fitting_params']

        for i, col in enumerate(nmf_var.columns):
            x = var
            y = nmf_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     marker=dict(color=colors[i % 10]), hoverinfo='text+x+y'))
            fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                     y=stats[ref_var][i][1] * np.array([x.min(), x.max()]) + stats[ref_var][i][2],
                                     mode='lines', name=f'{col}-Linear Regression Line',
                                     line=dict(dash='dash', color=colors[i % 10])))
        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text=indicator)

        tbl = pd.DataFrame(
            stats[ref_var],
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Fit a test EEM dataset using the established nmf model components
@app.callback(
    [
        Output('nmf-eem-dataset-predict-message', 'children'),  # size, intervals?
        Output('nmf-test-fmax', 'children'),
        Output('nmf-test-error', 'children'),
        Output('nmf-test-corr-ref-selection', 'options'),
        Output('nmf-test-corr-ref-selection', 'value'),
        Output('nmf-test-corr-indicator-selection', 'options'),
        Output('nmf-test-corr-indicator-selection', 'value'),
        Output('nmf-predict-spinner', 'children'),
        Output('nmf-test-results', 'data'),
    ],
    [
        Input('predict-nmf-model', 'n_clicks'),
        State('nmf-eem-dataset-predict-path-input', 'value'),
        State('nmf-test-index-kw-mandatory', 'value'),
        State('nmf-test-index-kw-optional', 'value'),
        State('nmf-test-model-selection', 'value'),
        State('nmf-models', 'data')
    ]
)
def on_nmf_prediction(n_clicks, path_predict, kw_mandatory, kw_optional, model_r, nmf_models):
    if n_clicks is None:
        return (None, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    if path_predict is None:
        return (None, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
    if not os.path.exists(path_predict):
        message = ('Error: No such file: ' + path_predict)
        return (message, None, None, [], None,
                [{'label': 'Fmax', 'value': 'Fmax'}], None, 'predict', None)
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
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], index=eem_dataset_dict['index'],
                             columns=eem_dataset_dict['ref'][0]) if eem_dataset_dict['ref'] is not None else None,
        )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_predict.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional, copy=False)

    if eem_dataset_predict.ref is not None:
        valid_ref = eem_dataset_predict.ref.columns[~eem_dataset_predict.ref.isna().all()].tolist()
    else:
        valid_ref = None

    _, fmax_sample, eem_stack_pred = eems_fit_components(eem_dataset_predict.eem_stack,
                                                         np.array(
                                                             [
                                                                 [
                                                                     [np.nan if x is None else x for x in
                                                                      subsublist] for subsublist in sublist
                                                                 ]
                                                                 for sublist in
                                                                 nmf_models[str(model_r)][
                                                                     'components']
                                                             ]
                                                         ),
                                                         fit_intercept=False)
    fmax_sample = pd.DataFrame(
        fmax_sample, index=eem_dataset_predict.index, columns=['component {i}'.format(i=i + 1) for i in range(model_r)]
    )

    pred = {}
    if eem_dataset_predict.ref is not None:
        for ref_var in valid_ref:
            if ref_var in nmf_models[str(model_r)]['fitting_params'].keys():
                params = nmf_models[str(model_r)]['fitting_params'][ref_var]
                pred_var = fmax_sample.copy()
                for i, f_col in enumerate(fmax_sample):
                    pred_r = fmax_sample[f_col] - params[i][2]
                    pred_r = pred_r / params[i][1]
                    pred_var[f_col] = pred_r
                pred[ref_var] = [pred_var.columns.tolist()] + pred_var.values.tolist()
            else:
                pred[ref_var] = None

    fmax_tab = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(figure=plot_fmax(fmax_sample,
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
    ])

    error_tab = html.Div(children=[])

    if eem_dataset_predict.ref is not None:
        indicator_options = [{'label': 'Fmax', 'value': 'Fmax'},
                             {'label': 'Prediction of reference', 'value': 'Prediction of reference'}]
    else:
        indicator_options = [{'label': 'Fmax', 'value': 'Fmax'}]

    ref_options = [{'label': var, 'value': var} for var in eem_dataset_predict.ref.columns]

    test_results = {
        'Fmax': [fmax_sample.columns.tolist()] + fmax_sample.values.tolist(),
        'Prediction of reference': pred,
        'ref': [eem_dataset_predict.ref.columns.tolist()] + eem_dataset_predict.ref.values.tolist()
        if eem_dataset_predict.ref is not None else None,
        'index': eem_dataset_predict.index
    }

    return (None, fmax_tab, error_tab, ref_options, None,
            indicator_options, None, 'predict', test_results)


# -----------Predict the corresponding reference variables for the test EEM datasets using the model fitted in the model
#            establishment step
@app.callback(
    [
        Output('nmf-test-pred-graph', 'figure'),  # size, intervals?
        Output('nmf-test-pred-table', 'children'),
    ],
    [
        Input('predict-parafac-model', 'n_clicks'),
        Input('nmf-test-pred-ref-selection', 'value'),
        Input('nmf-test-pred-model-selection', 'value'),
        State('nmf-test-results', 'data'),
    ]
)
def on_nmf_predict_reference_test(n_clicks, ref_var, pred_model, nmf_test_results):
    if all([ref_var, pred_model, nmf_test_results]):
        pred = nmf_test_results['Prediction of reference'][ref_var]
        pred = pd.DataFrame(pred[1:], columns=pred[0], index=nmf_test_results['index'])
        fig = plot_fmax(pred, display=False, yaxis_title=ref_var)
        if nmf_test_results['ref'] is not None:
            ref = pd.DataFrame(nmf_test_results['ref'][1:], columns=nmf_test_results['ref'][0],
                               index=nmf_test_results['index'])
            true_value = ref[ref_var]
            nan_rows = true_value[true_value.isna()].index
            true_value = true_value.drop(nan_rows)
            pred_overlap = pred.drop(nan_rows)
            fig.add_trace(go.Scatter(x=true_value.index, y=true_value.values, mode='markers',
                                     marker=dict(color='black', size=5), name='True value'))
            rmse_results = []
            for f_col in pred.columns:
                rmse = sqrt(sum((true_value.values - pred_overlap[f_col]) ** 2) / len(true_value.values))
                rmse_results.append(rmse)
            tbl = html.Div(
                [
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.DataFrame([rmse_results], columns=pred.columns,
                                                              index=pd.Index(['RMSE'], name='Error metric')),
                                                 bordered=True, hover=True, index=True)
                    ),
                    dbc.Row(
                        dbc.Table.from_dataframe(pd.concat([pred, ref[ref_var]], axis=1),
                                                 bordered=True, hover=True, index=True)
                    )
                ]
            )
        else:
            tbl = dbc.Table.from_dataframe(pred, bordered=True, hover=True, index=True)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Analyze correlations between score/Fmax and reference variables in model testing
@app.callback(
    [
        Output('nmf-test-corr-graph', 'figure'),  # size, intervals?
        Output('nmf-test-corr-table', 'children'),
    ],
    [
        Input('nmf-test-corr-indicator-selection', 'value'),
        Input('nmf-test-corr-ref-selection', 'value'),
        State('nmf-test-results', 'data'),
    ]
)
def on_nmf_test_correlations(indicator, ref_var, nmf_test_results):
    if all([indicator, ref_var, nmf_test_results]):
        ref_df = pd.DataFrame(nmf_test_results['ref'][1:], columns=nmf_test_results['ref'][0],
                              index=nmf_test_results['index'])
        var = ref_df[ref_var]
        if indicator != 'Prediction of reference':
            nmf_var = pd.DataFrame(nmf_test_results[indicator][1:], columns=nmf_test_results[indicator][0],
                                   index=nmf_test_results['index'])
        else:
            if nmf_test_results[indicator][ref_var] is not None:
                nmf_var = pd.DataFrame(nmf_test_results[indicator][ref_var][1:],
                                       columns=nmf_test_results[indicator][ref_var][0],
                                       index=nmf_test_results['index'])
            else:
                return go.Figure(), None

        fig = go.Figure()
        stats = []
        for i, col in enumerate(nmf_var.columns):
            x = var
            y = nmf_var[col]
            nan_rows = x[x.isna()].index
            x = x.drop(nan_rows)
            y = y.drop(nan_rows)
            if x.shape[0] < 1:
                return go.Figure(), None
            x_reshaped = np.array(x).reshape(-1, 1)
            lm = LinearRegression().fit(x_reshaped, y)
            predictions = lm.predict(x_reshaped)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=col, text=[i for i in x.index],
                                     hoverinfo='text+x+y', marker=dict(color=colors[i % 10])))
            fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name=f'{col} fit',
                                     line=dict(dash='dash', color=colors[i % 10])))
            r_squared = lm.score(x_reshaped, y)
            intercept = lm.intercept_
            slope = lm.coef_[0]
            pearson_corr, pearson_p = pearsonr(x, y)
            stats.append([col, slope, intercept, r_squared, pearson_corr, pearson_p])

        fig.update_xaxes(title_text=ref_var)
        fig.update_yaxes(title_text="Prediction of " + ref_var if indicator == 'Prediction of reference' else indicator)

        tbl = pd.DataFrame(
            stats,
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


# -----------Page #4: K-method--------------

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
                                dcc.Input(id='kmethod-eem-dataset-establishment-path-input', type='text', value=None,
                                          placeholder='Please enter the eem dataset path (.json and .pkl are supported).'
                                                      ' If empty, the model built in "eem pre-processing" '
                                                      'would be used',
                                          style={'width': '97%', 'height': '30px'}, debounce=True),
                                justify="center"
                            ),
                            dbc.Row(
                                dbc.Checklist(options=[{'label': html.Span("Read clustering output from the file",
                                                                           style={"font-size": 15,
                                                                                  "padding-left": 10}),
                                                        'value': True}],
                                              id='kmethod-cluster-from-file-checkbox', switch=True, value=[True]
                                              ),
                            ),
                            dbc.Row([
                                dbc.Col(
                                    html.Div([],
                                             id='kmethod-eem-dataset-establishment-message', style={'width': '80vw'}),
                                    width={"size": 12, "offset": 0}
                                )
                            ]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Index mandatory keywords"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='kmethod-establishment-index-kw-mandatory', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Index optional keywords"), width={'size': 1, 'offset': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='kmethod-establishment-index-kw-optional', type='text',
                                                  placeholder='',
                                                  style={'width': '100%', 'height': '30px'}, debounce=True, value=''),
                                        width={"offset": 0, "size": 2}
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Label("Base model"), width={'size': 1, 'offset': 0}
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(options=[
                                            {'label': 'PARAFAC', 'value': 'parafac'},
                                            {'label': 'NMF', 'value': 'nmf'}
                                        ],
                                            value=None, style={'width': '300px'}, id='kmethod-base-model'
                                        ),
                                        width={'size': 2}
                                    ),
                                    dbc.Col(
                                        html.Div([],
                                                 id='kmethod-base-model-message',
                                                 style={'width': '80vw'}),
                                        width={"size": 8, "offset": 1}
                                    )
                                ]
                            ),

                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("Num. components"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-rank', type='number',
                                              placeholder='Please enter only one number',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 1},
                                ),
                                dbc.Col(
                                    dbc.Label("Number of initial splits"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-num-init-splits', type='number',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 1},
                                ),
                                dbc.Col(
                                    dbc.Label("Number of base clustering runs"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-num-base-clusterings', type='number',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0),
                                    width={'size': 1},
                                ),
                                dbc.Col(
                                    dbc.Label("Maximum iterations for one time base clustering"),
                                    width={'size': 2}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-num-iterations', type='number',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=10),
                                    width={'size': 1},
                                ),
                            ]
                            ),

                            dbc.Row([

                                dbc.Col(
                                    dbc.Label("Convergence tolerance"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-convergence-tolerance', type='number',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0.01),
                                    width={'size': 1},
                                ),
                                dbc.Col(
                                    dbc.Label("Elimination"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-elimination', type='text',
                                              style={'width': '100px', 'height': '30px'}, debounce=True,
                                              value='default'),
                                    width={'size': 1},
                                ),
                                dbc.Col(
                                    dbc.Label("Subsampling portion"),
                                    width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Input(id='kmethod-subsampling-portion', type='number',
                                              style={'width': '100px', 'height': '30px'}, debounce=True, value=0.8),
                                    width={'size': 1},
                                ),
                            ]
                            ),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='kmethod-step1-spinner')],
                                               id='build-kmethod-consensus', className='col-2')
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
                                    dbc.Col(
                                        dbc.Label("Number of final clusters"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='kmethod-num-final-clusters', type='text',
                                                  placeholder='Multiple values possible, e.g., 3, 4',
                                                  style={'width': '250px', 'height': '30px'}, debounce=True),
                                        width={"offset": 0, "size": 3}
                                    ),
                                    dbc.Col(
                                        dbc.Label("Consensus conversion factor"), width={'size': 1}
                                    ),
                                    dbc.Col(
                                        dcc.Input(id='kmethod-consensus-conversion', type='number', value=1,
                                                  style={'width': '250px', 'height': '30px'}, debounce=True),
                                        width={"offset": 0, "size": 1}
                                    ),
                                ]
                            ),

                            dbc.Row([
                                dbc.Col(
                                    dbc.Label("Validations"), width={'size': 1}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[{'label': 'silhouette score', 'value': 'silhouette_score'},
                                                 {'label': 'reconstruction error reduction', 'value': 'RER'}],
                                        multi=True, id='kmethod-validations', value=['silhouette_score', 'RER']),
                                    width={'size': 4}
                                ),
                            ]),

                            dbc.Row(
                                dbc.Col(
                                    dbc.Button([dbc.Spinner(size="sm", id='kmethod-step2-spinner')],
                                               id='build-kmethod-clustering', className='col-2')
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

page_kmethod = html.Div([
    dbc.Stack(
        [
            dbc.Row(
                card_kmethod_param1
            ),
            dbc.Row(
                dcc.Tabs(
                    id='kmethod-results1',
                    children=[
                        dcc.Tab(label='Consensus matrix', id='kmethod-consensus-matrix'),
                        dcc.Tab(label='Label history', id='kmethod-label-history'),
                        dcc.Tab(label='Error history', id='kmethod-error-history'),
                    ],
                    vertical=True
                )
            ),
            dbc.Row(
                card_kmethod_param2
            ),
            dbc.Row(
                dcc.Tabs(
                    id='kmethod-results2',
                    children=[
                        dcc.Tab(label='Dendrogram', id='kmethod-dendrogram'),
                        dcc.Tab(label='Sorted consensus matrix', id='kmethod-sorted-consensus-matrix'),
                        dcc.Tab(label='Silhouette score', id='kmethod-silhouette-score'),
                        dcc.Tab(label='Reconstruction error reduction', id='kmethod-reconstruction-error-reduction'),
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
                                                                dbc.Label("Select K-method model"),
                                                                width={'size': 1, 'offset': 0}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id='kmethod-establishment-components-model-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                            dbc.Col(
                                                                dbc.Label("Select cluster"),
                                                                width={'size': 1, 'offset': 1}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id='kmethod-establishment-components-cluster-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        # dcc.Graph(id='kmethod-establishment-components-graph',
                                                        #           # config={'responsive': 'auto'},
                                                        #           style={'width': '45vw', 'height': '60vh'}
                                                        #           ),
                                                        html.Div(
                                                            children=[], id='kmethod-establishment-components-graph',
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
                            label='Components', id='kmethod-components'
                        ),
                        dcc.Tab(label='Fmax', id='kmethod-fmax'),
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
                                                                dbc.Label("Select K-method model"),
                                                                width={'size': 1, 'offset': 0}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id='kmethod-establishment-corr-model-selection'
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
                                                                    id='kmethod-establishment-corr-indicator-selection'
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
                                                                    id='kmethod-establishment-corr-ref-selection'
                                                                ),
                                                                width={'size': 2}
                                                            ),
                                                        ]
                                                    ),

                                                    dbc.Row([
                                                        dcc.Graph(id='kmethod-establishment-corr-graph',
                                                                  # config={'responsive': 'auto'},
                                                                  style={'width': '45vw', 'height': '60vh'}
                                                                  ),
                                                    ]),

                                                    dbc.Row(
                                                        html.Div([],
                                                                 id='kmethod-establishment-corr-table')
                                                    ),

                                                ],
                                                gap=2, style={"margin": "20px"}
                                            )
                                        )
                                    ],
                                    style={'width': '90vw'},
                                )
                            ],
                            label='Correlations', id='kmethod-establishment-corr'
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
                                                        dcc.Input(id='kmethod-eem-dataset-predict-path-input',
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
                                                                     id='kmethod-eem-dataset-predict-message',
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
                                                                dcc.Input(id='kmethod-test-index-kw-mandatory',
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
                                                                dcc.Input(id='kmethod-test-index-kw-optional',
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
                                                                    options=[], id='kmethod-test-model-selection'
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    [dbc.Spinner(size="sm",
                                                                                 id='kmethod-predict-spinner')],
                                                                    id='predict-kmethod-model', className='col-2')
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
                                                        id='kmethod-test-fmax'
                                                    ),
                                                    dcc.Tab(
                                                        label='Reconstruction error',
                                                        children=[],
                                                        id='kmethod-test-error'
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
                                                                                    id='kmethod-test-pred-ref-selection'
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
                                                                                    id='kmethod-test-pred-model'
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
                                                                                    id='kmethod-test-pred-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    config={'autosizable': False},
                                                                                    style={'width': 1700, 'height': 800}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='kmethod-test-pred-table'
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
                                                                                    id='kmethod-test-corr-indicator-selection'
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
                                                                                    id='kmethod-test-corr-ref-selection'
                                                                                ),
                                                                                width={'size': 2}
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Row([
                                                                                dcc.Graph(
                                                                                    id='kmethod-test-corr-graph',
                                                                                    # config={'responsive': 'auto'},
                                                                                    style={'width': '700',
                                                                                           'height': '900'}
                                                                                ),
                                                                            ]),

                                                                            dbc.Row(
                                                                                html.Div(
                                                                                    children=[],
                                                                                    id='kmethod-test-corr-table'
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
                            label='Predict', id='kmethod-predict'
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
                                                            dbc.Col(
                                                                dbc.Label("Select K-method model"),
                                                                width={'size': 1, 'offset': 0}
                                                            ),
                                                            dbc.Col(
                                                                dcc.Dropdown(
                                                                    options=[],
                                                                    id='kmethod-cluster-export-model-selection'
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
                            label='Export', id='kmethod-cluster-export'
                        ),
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

#       ---------------Step 1: Calculate consensus

@app.callback(
    [
        Output('kmethod-base-model-message', 'children')
    ],
    [
        Input('kmethod-base-model', 'value')
    ]
)
def on_kmethod_base_clustering_message(base_clustering):
    if base_clustering == 'parafac':
        message = ['Parameters "initialization", "non negativity" and "total fluorescence normalization" '
                   'are set in tab "PARAFAC".']
    elif base_clustering == 'nmf':
        message = ['Parameters "Initialization", "solver", "normalization" and "alpha_w", "alpha_h", "l1_ratio" are set in tab "NMF".']
    else:
        message = [None]
    return message


@app.callback(
    [
        Output('kmethod-eem-dataset-establishment-message', 'children'),
        Output('kmethod-consensus-matrix', 'children'),
        Output('kmethod-error-history', 'children'),
        Output('kmethod-step1-spinner', 'children'),
        Output('kmethod-consensus-matrix-data', 'data'),
        Output('kmethod-eem-dataset-establish', 'data'),
        Output('kmethod-base-clustering-parameters', 'data')
    ],
    [
        Input('build-kmethod-consensus', 'n_clicks'),
        State('eem-graph-options', 'value'),
        State('kmethod-eem-dataset-establishment-path-input', 'value'),
        State('kmethod-establishment-index-kw-mandatory', 'value'),
        State('kmethod-establishment-index-kw-optional', 'value'),
        State('kmethod-cluster-from-file-checkbox', 'value'),
        State('kmethod-rank', 'value'),
        State('kmethod-base-model', 'value'),
        State('kmethod-num-init-splits', 'value'),
        State('kmethod-num-base-clusterings', 'value'),
        State('kmethod-num-iterations', 'value'),
        State('kmethod-convergence-tolerance', 'value'),
        State('kmethod-elimination', 'value'),
        State('kmethod-subsampling-portion', 'value'),
        State('parafac-init-method', 'value'),
        State('parafac-nn-checkbox', 'value'),
        State('parafac-tf-checkbox', 'value'),
        State('nmf-solver', 'value'),
        State('nmf-init', 'value'),
        State('nmf-normalization-checkbox', 'value'),
        State('nmf-alpha-w', 'value'),
        State('nmf-alpha-h', 'value'),
        State('nmf-l1-ratio', 'value'),
        State('eem-dataset', 'data')
    ]
)
def on_build_consensus(n_clicks, eem_graph_options, path_establishment, kw_mandatory, kw_optional, cluster_from_file,
                       rank, base_clustering, n_init_splits, n_base_clusterings, n_iterations, tol, elimination,
                       subsampling_portion,
                       parafac_init, parafac_nn, parafac_tf,
                       nmf_solver, nmf_init, nmf_normalization, nmf_alpha_w, nmf_alpha_h, nmf_l1_ratio,
                       eem_dataset_dict):
    if n_clicks is None:
        return None, None, None, 'Calculate consensus', None, None, None
    if not path_establishment:
        if eem_dataset_dict is None:
            message = (
                'Error: No built EEM dataset detected. Please build an EEM dataset first in "EEM pre-processing" '
                'section, or import an EEM dataset from file.')
            return message, None, None, 'Calculate consensus', None, None, None
        eem_dataset_establishment = EEMDataset(
            eem_stack=np.array([[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                                in eem_dataset_dict['eem_stack']]),
            ex_range=np.array(eem_dataset_dict['ex_range']),
            em_range=np.array(eem_dataset_dict['em_range']),
            index=eem_dataset_dict['index'],
            ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                             index=eem_dataset_dict['index']) if eem_dataset_dict['ref'] is not None else None,
            cluster=eem_dataset_dict['cluster']
        )
    else:
        if not os.path.exists(path_establishment):
            message = ('Error: No such file or directory: ' + path_establishment)
            return message, None, None, 'Calculate consensus', None, None, None
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
                ref=pd.DataFrame(eem_dataset_dict['ref'][1:], columns=eem_dataset_dict['ref'][0],
                                 index=eem_dataset_dict['index']) if eem_dataset_dict['ref'] is not None else None,
                cluster=eem_dataset_dict['cluster']
            )
    kw_mandatory = str_string_to_list(kw_mandatory) if kw_mandatory else None
    kw_optional = str_string_to_list(kw_optional) if kw_optional else None
    eem_dataset_establishment.filter_by_index(mandatory_keywords=kw_mandatory, optional_keywords=kw_optional,
                                              copy=False)

    eem_dataset_establishment_json_dict = {
        'eem_stack': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for sublist in
                      eem_dataset_establishment.eem_stack.tolist()],
        'ex_range': eem_dataset_establishment.ex_range.tolist(),
        'em_range': eem_dataset_establishment.em_range.tolist(),
        'index': eem_dataset_establishment.index,
        'ref': [eem_dataset_establishment.ref.columns.tolist()] + eem_dataset_establishment.ref.values.tolist()
        if eem_dataset_establishment.ref is not None else None,
        'cluster': None,
    }

    if base_clustering == 'parafac':
        base_clustering_parameters = {
            'n_components': rank, 'init': parafac_init,
            'non_negativity': True if 'non_negative' in parafac_nn else False,
            'tf_normalization': True if 'tf_normalization' in parafac_tf else False,
            'sort_em': True
        }
        base_model = PARAFAC(**base_clustering_parameters)
    elif base_clustering == 'nmf':
        base_clustering_parameters = {
            'n_components': rank, 'solver': nmf_solver, 'init': nmf_init,
            'normalization': nmf_normalization[0],
            'alpha_H': nmf_alpha_h, 'alpha_W': nmf_alpha_w, 'l1_ratio': nmf_l1_ratio
        }
        base_model = EEMNMF(**base_clustering_parameters)

    kmethod = KMethod(base_model=base_model, n_initial_splits=n_init_splits, max_iter=n_iterations, tol=tol,
                      elimination=elimination, error_calculation="quenching_coefficient", kw_unquenched="B1C1", kw_quenched="B1C2")
    consensus_matrix, _, error_history = kmethod.calculate_consensus(eem_dataset_establishment, n_base_clusterings,
                                                                     subsampling_portion)
    consensus_matrix_tabs = dbc.Card(children=[])
    error_history_tabs = dbc.Card(children=[])

    fig_consensus_matrix = go.Figure(
        data=go.Heatmap(
            z=consensus_matrix,
            x=eem_dataset_establishment.index if eem_dataset_establishment.index
            else [i for i in range(consensus_matrix.shape[1])],
            y=eem_dataset_establishment.index if eem_dataset_establishment.index
            else [i for i in range(consensus_matrix.shape[0])],
            colorscale='reds',
            hoverongaps=False,
            hovertemplate='sample1: %{y}<br>sample2: %{x}<br>consensus coefficient: %{z}<extra></extra>'
        )
    )
    fig_consensus_matrix.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    consensus_matrix_tabs.children.append(
        html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=fig_consensus_matrix,
                                config={'autosizable': False},
                                style={'width': '40vw',
                                       'height': '70vh'}
                            )
                        ]
                    )
                ]
            ),
        ]),
    )

    error_means = [df.mean(axis=0) for df in error_history]
    error_means = pd.concat(error_means, axis=1).T
    error_means.columns = [f'iteration {i + 1}' for i in range(error_means.shape[1])]
    error_means.index = [f'base clustering {i + 1}' for i in range(len(error_history))]
    error_means_melted = error_means.melt(var_name='Column', value_name='Value')

    fig_error = px.box(error_means_melted, x='Column', y='Value', points='all')
    fig_error.update_layout(
        xaxis_title='Iterations',
        yaxis_title='Error'
    )

    error_history_tabs.children.append(
        html.Div([
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=fig_error
                            )
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Table.from_dataframe(error_means,
                                                     bordered=True, hover=True, index=True,

                                                     )
                        ]
                    )
                ]
            )
        ])
    )

    return (None, consensus_matrix_tabs, error_history_tabs, 'Calculate consensus', consensus_matrix.tolist(),
            eem_dataset_establishment_json_dict, base_clustering_parameters)


#   -----------------Step 2: Hierarchical clustering

@app.callback(
    [
        Output('kmethod-dendrogram', 'children'),
        Output('kmethod-sorted-consensus-matrix', 'children'),
        Output('kmethod-silhouette-score', 'children'),
        Output('kmethod-reconstruction-error-reduction', 'children'),
        Output('kmethod-fmax', 'children'),
        Output('kmethod-step2-spinner', 'children'),
        Output('kmethod-establishment-components-model-selection', 'options'),
        Output('kmethod-establishment-components-model-selection', 'value'),
        # Output('kmethod-establishment-components-cluster-selection', 'options'),
        # Output('kmethod-establishment-components-cluster-selection', 'value'),
        Output('kmethod-establishment-corr-model-selection', 'options'),
        Output('kmethod-establishment-corr-model-selection', 'value'),
        Output('kmethod-establishment-corr-indicator-selection', 'options'),
        Output('kmethod-establishment-corr-indicator-selection', 'value'),
        # Output('kmethod-establishment-corr-ref-selection', 'options'),
        # Output('kmethod-establishment-corr-ref-selection', 'value'),
        Output('kmethod-test-model-selection', 'options'),
        Output('kmethod-test-model-selection', 'value'),
        Output('kmethod-test-pred-ref-selection', 'options'),
        Output('kmethod-test-pred-ref-selection', 'value'),
        Output('kmethod-models', 'data')
    ],
    [
        Input('build-kmethod-clustering', 'n_clicks'),
        State('kmethod-base-model', 'value'),
        State('kmethod-num-final-clusters', 'value'),
        State('kmethod-consensus-conversion', 'value'),
        State('kmethod-validations', 'value'),
        State('kmethod-consensus-matrix-data', 'data'),
        State('kmethod-eem-dataset-establish', 'data'),
        State('kmethod-base-clustering-parameters', 'data')
    ]
)
def on_hierarchical_clustering(n_clicks, base_clustering, n_final_clusters, conversion, validations,
                               consensus_matrix, eem_dataset_establish_dict, base_clustering_parameters):
    if n_clicks is None:
        return None, None, None, None, None, 'Clustering', [], None, [], None, [], None, [], None, [], None, None
    else:
        if eem_dataset_establish_dict is None:
            return None, None, None, None, None, 'Clustering', [], None, [], None, [], None, [], None, [], None, None
        else:
            eem_dataset_establish = EEMDataset(
                eem_stack=np.array(
                    [[[np.nan if x is None else x for x in subsublist] for subsublist in sublist] for sublist
                     in eem_dataset_establish_dict['eem_stack']]),
                ex_range=np.array(eem_dataset_establish_dict['ex_range']),
                em_range=np.array(eem_dataset_establish_dict['em_range']),
                index=eem_dataset_establish_dict['index'],
                ref=pd.DataFrame(eem_dataset_establish_dict['ref'][1:],
                                 columns=eem_dataset_establish_dict['ref'][0],
                                 index=eem_dataset_establish_dict['index'])
                if eem_dataset_establish_dict['ref'] is not None else None,
                cluster=eem_dataset_establish_dict['cluster']
            )

    n_clusters_list = num_string_to_list(n_final_clusters)
    kmethod_fit_params = {}

    if eem_dataset_establish.ref is not None:
        valid_ref = eem_dataset_establish.ref.columns[~eem_dataset_establish.ref.isna().all()].tolist()
    else:
        valid_ref = []

    if base_clustering == 'parafac':
        base_model = PARAFAC(**base_clustering_parameters)
        unified_model = PARAFAC(**base_clustering_parameters)
    elif base_clustering == 'nmf':
        base_model = EEMNMF(**base_clustering_parameters)
        unified_model = EEMNMF(**base_clustering_parameters)

    unified_model.fit(eem_dataset_establish)

    dendrogram_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    sorted_consensus_matrix_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    silhouette_score_tabs = dbc.Card([])
    reconstruction_error_reduction_tabs = dbc.Card(
        [dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    fmax_establishment_tabs = dbc.Card([dbc.Tabs(children=[], persistence=True, persistence_type='session')])
    slt = []
    kmethod_models = {}

    for k in n_clusters_list:
        kmethod = KMethod(base_model=base_model, n_initial_splits=None)
        kmethod.consensus_matrix = np.array(consensus_matrix)
        try:
            kmethod.hierarchical_clustering(eem_dataset_establish, k, conversion)
        except ValueError:
            continue
        cluster_specific_models = kmethod.cluster_specific_models
        eem_clusters = kmethod.eem_clusters
        cluster_labels_combined = []
        for sub_dataset in eem_clusters.values():
            cluster_labels_combined += sub_dataset.cluster

        fmax_combined = pd.concat([model.fmax for model in cluster_specific_models.values()], axis=0)
        fmax_combined_sorted = fmax_combined.sort_index()
        cluster_labels_combined_sorted = [x for _, x in sorted(zip(fmax_combined.index, cluster_labels_combined))]
        fig_fmax = plot_fmax(table=fmax_combined_sorted, display=False, labels=cluster_labels_combined_sorted)
        fmax_combined_sorted['Cluster'] = cluster_labels_combined_sorted

        kmethod_fit_params_k = {}

        if base_clustering == 'parafac':
            component_names = unified_model.fmax.columns.tolist()
        elif base_clustering == 'nmf':
            component_names = unified_model.nnls_fmax.columns.tolist()
        for ref_var in valid_ref + component_names:
            kmethod_fit_params_k[ref_var] = []

        if eem_dataset_establish.ref is not None:
            for i in range(k):
                cluster_specific_model = kmethod.cluster_specific_models[i + 1]
                eem_cluster = kmethod.eem_clusters[i + 1]
                valid_ref_cluster = eem_cluster.ref.columns[~eem_cluster.ref.isna().all()].tolist()
                for ref_var in valid_ref_cluster:
                    stats = []
                    x = eem_cluster.ref[ref_var]
                    if base_clustering == 'parafac':
                        model_var = copy.copy(cluster_specific_model.fmax)
                    elif base_clustering == 'nmf':
                        model_var = copy.copy(cluster_specific_model.nnls_fmax)
                    model_var.columns = [f'Cluster {i + 1}-' + col for col in model_var.columns]
                    nan_rows = x[x.isna()].index
                    x = x.drop(nan_rows)
                    if x.shape[0] <= 1:
                        continue
                    for f_col in model_var.columns:
                        y = model_var[f_col]
                        y = y.drop(nan_rows)
                        x_reshaped = np.array(x).reshape(-1, 1)
                        lm = LinearRegression().fit(x_reshaped, y)
                        r_squared = lm.score(x_reshaped, y)
                        intercept = lm.intercept_
                        slope = lm.coef_[0]
                        pearson_corr, pearson_p = pearsonr(x, y)
                        stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                    kmethod_fit_params_k[ref_var] = kmethod_fit_params_k[ref_var] + stats

                for c_var in component_names:
                    if base_clustering == 'parafac':
                        x = cluster_specific_model.fmax[c_var]
                        model_var = cluster_specific_model.fmax
                    elif base_clustering == 'nmf':
                        x = cluster_specific_model.nnls_fmax[c_var]
                        model_var = cluster_specific_model.nnls_fmax
                    stats = []
                    nan_rows = x[x.isna()].index
                    x = x.drop(nan_rows)
                    if x.shape[0] <= 1:
                        continue
                    for f_col in model_var.columns:
                        y = model_var[f_col]
                        y = y.drop(nan_rows)
                        x_reshaped = np.array(x).reshape(-1, 1)
                        lm = LinearRegression().fit(x_reshaped, y)
                        r_squared = lm.score(x_reshaped, y)
                        intercept = lm.intercept_
                        slope = lm.coef_[0]
                        pearson_corr, pearson_p = pearsonr(x, y)
                        stats.append([f_col, slope, intercept, r_squared, pearson_corr, pearson_p])
                    kmethod_fit_params_k[c_var] = kmethod_fit_params_k[c_var] + stats

        fig_dendrogram = plot_dendrogram(kmethod.linkage_matrix, kmethod.threshold_r,
                                         eem_dataset_establish_dict['index'])
        dendrogram_tabs.children[0].children.append(
            dcc.Tab(label=f'{k}-cluster',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(
                                                figure=fig_dendrogram,
                                                config={'autosizable': False},
                                                style={'width': '50vw',
                                                       'height': '70vh'}
                                            )
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ],
                    style={'padding': '0', 'line-width': '100%'},
                    selected_style={'padding': '0', 'line-width': '100%'}
                    )
        )

        fig_sorted_consensus_matrix = go.Figure(
            data=go.Heatmap(
                z=kmethod.consensus_matrix_sorted,
                x=kmethod.index_sorted if eem_dataset_establish.index
                else [i for i in range(consensus_matrix.shape[1])],
                y=kmethod.index_sorted if eem_dataset_establish.index
                else [i for i in range(consensus_matrix.shape[0])],
                colorscale='reds',
                hoverongaps=False,
                hovertemplate='sample1: %{y}<br>sample2: %{x}<br>consensus coefficient: %{z}<extra></extra>'
            )
        )
        for j in range(max(kmethod.labels)):
            idx = np.where(np.sort(kmethod.labels) == j + 1)[0]
            fig_sorted_consensus_matrix.add_shape(type="rect", x0=min(idx), y0=min(idx),
                                                  x1=min(idx) + len(idx), y1=min(idx) + len(idx),
                                                  line=dict(color="black", width=3),
                                                  )
        fig_sorted_consensus_matrix.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

        sorted_consensus_matrix_tabs.children[0].children.append(
            dcc.Tab(
                label=f'{k}-cluster',
                children=[
                    html.Div([
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            figure=fig_sorted_consensus_matrix,
                                            config={'autosizable': False},
                                            style={'width': '50vw',
                                                   'height': '70vh'}
                                        )
                                    ]
                                )
                            ]
                        ),
                    ]),
                ],
                style={'padding': '0', 'line-width': '100%'},
                selected_style={'padding': '0', 'line-width': '100%'}
            )
        )

        if 'silhouette_score' in validations:
            slt.append(kmethod.silhouette_score)

        if 'RER' in validations:
            rmse_combined = pd.concat([model.sample_rmse() for model in cluster_specific_models.values()], axis=0)
            rmse_combined_sorted = rmse_combined.sort_index()
            rmse_combined_sorted = pd.concat([rmse_combined_sorted, unified_model.sample_rmse()], axis=1)
            rmse_combined_sorted.columns = ['RMSE with cluster-specific models', 'RMSE with unified model']
            rmse_combined_sorted['RMSE reduction (%)'] = (
                    100 * (1 - (rmse_combined_sorted.iloc[:, 0] / rmse_combined_sorted.iloc[:, 1])))
            fig_rer = plot_reconstruction_error(
                rmse_combined_sorted, display=False, bar_col_name='RMSE reduction (%)',
                labels=cluster_labels_combined_sorted
            )
            rmse_combined_sorted['Cluster'] = cluster_labels_combined_sorted

            reconstruction_error_reduction_tabs.children[0].children.append(
                dcc.Tab(label=f'{k}-cluster',
                        children=[
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(figure=fig_rer,
                                                          style={'width': '80vw',
                                                                 'height': '70vh'}
                                                          )
                                            ]
                                        )
                                    ]
                                ),

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Table.from_dataframe(rmse_combined_sorted,
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

        fmax_establishment_tabs.children[0].children.append(
            dcc.Tab(label=f'{k}-cluster',
                    children=[
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dcc.Graph(figure=fig_fmax,
                                                      style={'width': '80vw',
                                                             'height': '70vh'}
                                                      )
                                        ]
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Table.from_dataframe(fmax_combined_sorted,
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

        kmethod_models_n = {}
        for i in range(k):
            cluster_specific_model = kmethod.cluster_specific_models[i + 1]
            eem_cluster = kmethod.eem_clusters[i + 1]
            if base_clustering == 'parafac':
                kmethod_models_n[i+1] = {
                    'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                                   sublist in cluster_specific_model.components.tolist()],
                    'Fmax': [
                                cluster_specific_model.fmax.columns.tolist()] + cluster_specific_model.fmax.values.tolist(),
                    'index': eem_cluster.index,
                    'ref': [eem_cluster.ref.columns.tolist()] + eem_cluster.ref.values.tolist()
                    if eem_cluster.ref is not None else None,
                    'fitting_params': kmethod_fit_params_k
                }
            elif base_clustering == 'nmf':
                kmethod_models_n[i+1] = {
                    'components': [[[None if np.isnan(x) else x for x in subsublist] for subsublist in sublist] for
                                   sublist in cluster_specific_model.components.tolist()],
                    'NNLS_Fmax': [
                                     cluster_specific_model.nnls_fmax.columns.tolist()] + cluster_specific_model.nnls_fmax.values.tolist(),
                    'nmf_Fmax': [
                                    cluster_specific_model.nmf_fmax.columns.tolist()] + cluster_specific_model.nmf_fmax.values.tolist(),
                    'index': eem_cluster.index,
                    'ref': [eem_cluster.ref.columns.tolist()] + eem_cluster.ref.values.tolist()
                    if eem_cluster.ref is not None else None,
                    'fitting_params': kmethod_fit_params_k
                }

        kmethod_models[k] = kmethod_models_n

    if 'silhouette_score' in validations:
        slt_table = pd.DataFrame({'Number of clusters': list(kmethod_models.keys()), 'Silhouette score': slt})
        silhouette_score_tabs.children.append(
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    figure=px.line(
                                        x=slt_table['Number of clusters'],
                                        y=slt_table['Silhouette score'],
                                        markers=True,
                                        labels={'x': 'Number of cluster', 'y': 'Silhouette score'},
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
                                dbc.Table.from_dataframe(slt_table,
                                                         bordered=True, hover=True,
                                                         )
                            ]
                        ),
                    ]
                ),
            ]),
        )

    model_options = [{'label': '{r}-cluster'.format(r=r), 'value': r} for r in kmethod_models.keys()]
    ref_options = [{'label': var, 'value': var} for var in valid_ref] if \
        (eem_dataset_establish.ref is not None) else None

    if base_clustering == 'parafac':
        indicator_options = ['Fmax']
    elif base_clustering == 'nmf':
        indicator_options = ['NNLS-Fmax', 'NMF-Fmax']

    return (dendrogram_tabs, sorted_consensus_matrix_tabs, silhouette_score_tabs, reconstruction_error_reduction_tabs,
            fmax_establishment_tabs, 'Clustering', model_options, None, model_options, None, indicator_options, None,
            model_options, None, ref_options, None, kmethod_models)


# ---------Update cluster dropdown in components section-------------
@app.callback(
    [
        Output('kmethod-establishment-components-cluster-selection', 'options'),
        Output('kmethod-establishment-components-cluster-selection', 'value'),
    ],
    [
        Input('kmethod-establishment-components-model-selection', 'value')
    ]
)
def on_update_kmethod_component_cluster_list(k):
    if k is not None:
        options = [{'label': f'Cluster {n + 1}', 'value': n + 1} for n in range(int(k))]
        return options, None
    else:
        return [], None


# ---------Update reference dropdown in correlation section-----------
@app.callback(
    [
        Output('kmethod-establishment-corr-ref-selection', 'options'),
        Output('kmethod-establishment-corr-ref-selection', 'value'),
    ],
    [
        Input('kmethod-establishment-corr-model-selection', 'value'),
        State('kmethod-models', 'data')
    ]
)
def update_reference_dropdown_by_selected_model(k, model):
    if all([k, model]):
        options = []
        for c in list(model[str(k)].values()):
            options += list(c['fitting_params'].keys())
        options = list(set(options))
        options = sorted(options)
        return options, None
    else:
        return [], None


# ---------Plot components-----------
@app.callback(
    [
        Output('kmethod-establishment-components-graph', 'children')
    ],
    [
        Input('kmethod-establishment-components-model-selection', 'value'),
        Input('kmethod-establishment-components-cluster-selection', 'value'),
        State('eem-graph-options', 'value'),
        State('kmethod-models', 'data'),
        State('kmethod-eem-dataset-establish', 'data'),
    ],
)
def on_plot_kmethod_components(k, cluster_i, eem_graph_options, kmethod_models, eem_dataset_establish):
    if all([k, cluster_i, kmethod_models, eem_dataset_establish]):
        ex_range = np.array(eem_dataset_establish['ex_range'])
        em_range = np.array(eem_dataset_establish['em_range'])
        cluster_model = kmethod_models[str(k)][str(cluster_i)]
        r = len(cluster_model['components'])
        n_rows = (r - 1) // 3 + 1
        graphs = [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            figure=plot_eem(np.array(cluster_model['components'][3 * i]),
                                            ex_range=ex_range,
                                            em_range=em_range,
                                            vmin=0 if np.min(
                                                cluster_model['components'][3 * i]) > -1e-3 else None,
                                            vmax=None,
                                            auto_intensity_range=False,
                                            plot_tool='plotly',
                                            display=False,
                                            figure_size=(5, 3.5),
                                            axis_label_font_size=14,
                                            cbar_font_size=12,
                                            title_font_size=16,
                                            title=f'C{3 * i + 1}',
                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                            else False,
                                            rotate=True if 'rotate' in eem_graph_options else False,
                                            ) if 3 * i + 1 <= r else go.Figure(
                                layout={'width': 400, 'height': 300}),
                            # style={'width': '30vw'}
                        ),
                        width={'size': 4},
                    ),

                    dbc.Col(
                        dcc.Graph(
                            figure=plot_eem(np.array(cluster_model['components'][3 * i + 1]),
                                            ex_range=ex_range,
                                            em_range=em_range,
                                            vmin=0 if np.min(
                                                cluster_model['components'][
                                                    3 * i + 1]) > -1e-3 else None,
                                            vmax=None,
                                            auto_intensity_range=False,
                                            plot_tool='plotly',
                                            display=False,
                                            figure_size=(5, 3.5),
                                            axis_label_font_size=14,
                                            cbar_font_size=12,
                                            title_font_size=16,
                                            title=f'C{3 * i + 2}',
                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                            else False,
                                            rotate=True if 'rotate' in eem_graph_options else False,
                                            ) if 3 * i + 2 <= r else go.Figure(
                                layout={'width': 400, 'height': 300}),
                            # style={'width': '30vw'}
                        ),
                        width={'size': 4},
                    ),

                    dbc.Col(
                        dcc.Graph(
                            figure=plot_eem(np.array(cluster_model['components'][3 * i + 2]),
                                            ex_range=ex_range,
                                            em_range=em_range,
                                            vmin=0 if np.min(
                                                cluster_model['components'][
                                                    3 * i + 2]) > -1e-3 else None,
                                            vmax=None,
                                            auto_intensity_range=False,
                                            plot_tool='plotly',
                                            display=False,
                                            figure_size=(5, 3.5),
                                            axis_label_font_size=14,
                                            cbar_font_size=12,
                                            title_font_size=16,
                                            title=f'C{3 * i + 3}',
                                            fix_aspect_ratio=True if 'aspect_one' in eem_graph_options
                                            else False,
                                            rotate=True if 'rotate' in eem_graph_options else False,
                                            ) if 3 * i + 3 <= r else go.Figure(),
                            # style={'width': '30vw'}
                        ),
                        width={'size': 4},
                    ),
                ], style={'width': '90vw'}
            ) for i in range(n_rows)
        ]
        return [graphs]
    else:
        return [None]


# -----------Analyze correlations between Fmax and reference variables in model establishment

@app.callback(
    [
        Output('kmethod-establishment-corr-graph', 'figure'),
        Output('kmethod-establishment-corr-table', 'children'),
    ],
    [
        Input('kmethod-establishment-corr-model-selection', 'value'),
        Input('kmethod-establishment-corr-indicator-selection', 'value'),
        Input('kmethod-establishment-corr-ref-selection', 'value'),
        State('kmethod-models', 'data')
    ]
)
def on_kmethod_establishment_correlations(k, indicator, ref_var, kmethod_models):
    if all([k, indicator, ref_var, kmethod_models]):
        fig = go.Figure()
        for n in range(1, k + 1):
            cluster_specific_model = kmethod_models[str(k)][str(n)]
            r = len(cluster_specific_model['components'])
            ref_df = pd.DataFrame(cluster_specific_model['ref'][1:], columns=cluster_specific_model['ref'][0],
                                  index=cluster_specific_model['index'])
            if 'NNLS_Fmax' in list(cluster_specific_model.keys()):
                nnls_fmax_df = pd.DataFrame(cluster_specific_model['NNLS-Fmax'][1:],
                                            columns=cluster_specific_model['NNLS-Fmax'][0],
                                            index=cluster_specific_model['index'])
                nmf_fmax_df = pd.DataFrame(cluster_specific_model['NMF-Fmax'][1:],
                                           columns=cluster_specific_model['NMF-Fmax'][0],
                                           index=cluster_specific_model['index'])
                fmax_df = pd.concat([nnls_fmax_df, nmf_fmax_df], axis=1)
            elif 'Fmax' in list(cluster_specific_model.keys()):
                fmax_df = pd.DataFrame(cluster_specific_model['Fmax'][1:],
                                       columns=cluster_specific_model['Fmax'][0],
                                       index=cluster_specific_model['index'])

            ref_df = pd.concat([ref_df, fmax_df], axis=1)
            reference_variable = ref_df[ref_var]
            fluorescence_indicators = pd.DataFrame(cluster_specific_model[indicator][1:],
                                                   columns=cluster_specific_model[indicator][0],
                                                   index=cluster_specific_model['index'])

            stats_k = cluster_specific_model['fitting_params']

            for i, col in enumerate(fluorescence_indicators.columns):
                x = reference_variable
                y = fluorescence_indicators[col]
                nan_rows = x[x.isna()].index
                x = x.drop(nan_rows)
                y = y.drop(nan_rows)
                if x.shape[0] < 1:
                    return go.Figure(), None
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                         name=f'Cluster {n}-{col}',
                                         text=[i for i in x.index],
                                         marker=dict(color=colors[i % len(colors)],
                                                     symbol=marker_shapes[n % len(marker_shapes)]),
                                         hoverinfo='text+x+y'))
                fig.add_trace(go.Scatter(x=np.array([x.min(), x.max()]),
                                         y=stats_k[ref_var][r*n-r+i][1] * np.array([x.min(), x.max()]) + stats_k[ref_var][r*n-r+i][2],
                                         mode='lines', name=f'Cluster {n}-{col}-Linear Regression Line',
                                         line=dict(dash='dash', color=colors[i % len(colors)])))
            fig.update_xaxes(title_text=ref_var)
            fig.update_yaxes(title_text=indicator)


        fig.update_layout(legend=dict(y=-0.3, x=0.5, xanchor='center', yanchor='top'))

        tbl = pd.DataFrame(
            stats_k[ref_var],
            columns=['Variable', 'slope', 'intercept', 'R²', 'Pearson Correlation', 'Pearson p-value']
        )
        tbl = dbc.Table.from_dataframe(tbl, bordered=True, hover=True, index=False)
        return fig, tbl
    else:
        return go.Figure(), None


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
                dcc.Tab(label='Homepage', id='homepage', children=html.P(homepage)),
                dcc.Tab(label='EEM pre-processing', id='eem-pre-processing', children=html.P(page_eem_processing)),
                dcc.Tab(label='Peak picking', id='eem-peak-picking', children=html.P(page_peak_picking)),
                dcc.Tab(label='Regional integration', id='eem-regional-integration', children=html.P(page_regional_integration)),
                dcc.Tab(label='PARAFAC', id='parafac', children=html.P(page_parafac)),
                dcc.Tab(label='NMF', id='nmf', children=html.P(page_nmf)),
                dcc.Tab(label='K-method', id='kmethod', children=html.P(page_kmethod)),
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
        dcc.Store(id='pp-model'),
        dcc.Store(id='pp-test-results'),
        dcc.Store(id='ri-model'),
        dcc.Store(id='ri-test-results'),
        dcc.Store(id='parafac-models'),
        dcc.Store(id='parafac-test-results'),
        dcc.Store(id='kmethod-consensus-matrix-data'),
        dcc.Store(id='kmethod-eem-dataset-establish'),
        dcc.Store(id='kmethod-base-clustering-parameters'),
        dcc.Store(id='kmethod-models'),
        dcc.Store(id='kmethod-test-results'),
        dcc.Store(id='nmf-models'),
        dcc.Store(id='nmf-test-results'),
        content])


app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=True)

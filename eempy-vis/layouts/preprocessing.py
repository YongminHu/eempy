from __future__ import annotations

from dash import html, dcc
import dash_bootstrap_components as dbc


# -----------Page #1: EEM pre-processing--------------
# First setup the dbc cards, then the general layout in the end.

#   -------------Setting up the dbc cards

#       -----------dbc card for Selecting working folder and choosing EEM for previewing
help_icon_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "justifyContent": "center",
        "width": "18px",
        "height": "18px",
        "borderRadius": "50%",
        "backgroundColor": "#1e66ff",  # blue
        "color": "white",
        "fontSize": "12px",
        "fontWeight": "700",
        "lineHeight": "18px",
        "cursor": "help",
        "marginLeft": "6px",
        "userSelect": "none",
    }


card_selecting_files = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Select files", className="fw-bold"),
            dbc.Stack(
                [
                    dbc.Row(
                        dcc.Input(id='folder-path-input', type='text',
                                  placeholder='Please enter the data folder path...',
                                  style={'width': '97%', 'height': '30px'}, debounce=True),
                        justify="center"
                    ),

                    html.Div(
                        [
                        html.H6("Data format", className="fw-bold"),
                        html.Span("?", id="help-data-format", style=help_icon_style),
                        dbc.Popover(
                            [dbc.PopoverHeader("Help"), dbc.PopoverBody("Reserve for help message")],
                            target="help-qm",
                            trigger="legacy",
                            placement="right",
                        ),
                        ],
                        style={"display": "flex"}
                    ),

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

                    html.H6("File filtering keywords", className="fw-bold"),

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

                    html.H6("Index extraction from filenames", className="fw-bold"),

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

# @app.callback(Output("help-pop", "is_open"), Input("help-qm", "n_clicks"), State("help-pop", "is_open"))
# def toggle_pop(n, is_open):
#     if n:
#         return not is_open
#     return is_open

#       -----------dbc card for EEM display

card_eem_display = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Data preview", className="fw-bold"),
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
                    html.H5("Ex/Em/intensity ranges", className="fw-bold"),
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
                                    html.H5("Raman scattering unit (RSU) normalization from blank",
                                            className="fw-bold"),
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
                                    html.H5("Inner filter effect correction", className="fw-bold"),
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
                                    html.H5("Raman scattering removal", className="fw-bold"),
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
                                    html.H5("Rayleigh scattering removal", className="fw-bold"),
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
                                    html.H5("Gaussian Smoothing", className="fw-bold"),
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
                                    html.H5("Median filter", className="fw-bold"),
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
            html.H5("Build EEM dataset", className="fw-bold"),
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
            html.H5("Export Processed EEM Dataset", className="fw-bold"),
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

layout = html.Div([
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


import os
import dash
import datetime
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from eempy.plot import plot_eem
from eempy.read_data import read_eem
from eempy.eem_processing import eem_rayleigh_masking, eem_raman_masking
from matplotlib import pyplot as plt

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            html.Div(
                [
                    dbc.Row(
                        dcc.Input(id='folder-path-input', type='text',
                                  placeholder='Please enter the data folder path...',
                                  style={'width': '97%', 'height': '30px'}, debounce=True),
                        justify="center"
                    ),

                    html.Hr(),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("EEM file format"),
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='eem-data-format',
                                         options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                         value='aqualog', placeholder='Select EEM data file format',
                                         style={'width': '100%'}),
                        ),
                        dbc.Col(
                            dbc.Label("Absorbance file format"),
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='abs-data-format',
                                         options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                         value='aqualog', placeholder='Select EEM data file format',
                                         style={'width': '100%'}),
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Overarching file searching keyword"),
                        ),
                        dbc.Col(
                            dcc.Input(id='eem-file-keyword', type='text', placeholder='',
                                      style={'width': '160px', 'height': '20px'}, debounce=True, value='PEM'),
                        ),
                        dbc.Col(
                            dbc.Label("Sample EEM file searching keyword"),
                        ),
                        dbc.Col(
                            dcc.Input(id='eem-file-keyword', type='text', placeholder='',
                                      style={'width': '160px', 'height': '20px'}, debounce=True, value='PEM'),
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("Absorbance file searching keyword"),
                        ),
                        dbc.Col(
                            dcc.Input(id='abs-file-keyword', type='text', placeholder='',
                                      style={'width': '160px', 'height': '20px'}, debounce=True, value='ABS'),
                        ),
                        dbc.Col(
                            dbc.Label("Blank EEM file searching keyword"),
                        ),
                        dbc.Col(
                            dcc.Input(id='blank-file-keyword', type='text', placeholder='',
                                      style={'width': '160px', 'height': '20px'}, debounce=True, value='BEM'),
                        ),
                    ]),

                    html.Hr(),

                    dbc.Row([

                        dbc.Col(
                            dbc.Label("Index starting position"),
                        ),
                        dbc.Col(
                            dcc.Input(id='index-pos-left', type='number', placeholder='',
                                      style={'width': '160px', 'height': '20px'}, debounce=True),
                        ),
                        dbc.Col(
                            dbc.Label("Index end position"),
                        ),
                        dbc.Col(
                            dcc.Input(id='index-pos-right', type='text', placeholder='',
                                      style={'width': '160px', 'height': '20px'}, debounce=True),
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
                                      style={'width': '160px', 'height': '20px'}, debounce=True),
                        )
                    ]),
                ]
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
                                dcc.Dropdown(id='filename-dropdown', options=[], placeholder='Please select a file...'),
                            ),
                        ],
                        justify='end'
                    ),

                    dcc.Graph(id='eem-graph'),
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
            html.H5("Ex/Em/intensity ranges"),
            html.H6("Excitation wavelength", className="card-title"),
            html.Div(
                [
                    dbc.Row([
                        dbc.Col(
                            dcc.Input(id='excitation-wavelength-min', type='number', placeholder='min',
                                      style={'width': '120%', 'height':'20px'}, debounce=True),
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
                                      style={'width': '120%', 'height': '20px'}, debounce=True),
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
                                      style={'width': '120%', 'height':'20px'}, debounce=True),
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
                                      style={'width': '120%', 'height': '20px'}, debounce=True),
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
                                      style={'width': '120%', 'height': '20px'}, debounce=True),
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
                                      style={'width': '120%', 'height': '20px'}, debounce=True),
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

        ]
    ),
    className="w-100",
)

#       -----------dbc card for IFE correction

card_ife = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Inner filter effect", className="card-title"),
            html.Div(
                [
                    dbc.Row([
                        dbc.Checklist(id='ife-button',
                                      options=[{'label': html.Span("Inner filter effect correction",
                                                                   style={"font-size": 15, "padding-left": 10}),
                                                'value': "ife"}],
                                      switch=True),
                    ]),
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

#       -----------dbc card for automatic Raman scattering unit normalization

card_su = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    dbc.Row([
                        dbc.Col(
                            [
                                html.H5("Raman scattering unit (RSU) normalization from blank"),
                            ],
                            width={'size': 10}),
                        dbc.Col([
                            dbc.Checklist(id='su-butthn', options=[{'label': ' ', 'value': "su"}], switch=True,
                                          style={'transform': 'scale(1.5)'})
                        ], width={'size': 2})
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dbc.Label("RS Ex lower bound"),
                            width={'size': 6}
                        ),
                        dbc.Col(
                            dcc.Input(id='su-excitation-lower-bound', type='number', placeholder='max',
                                      style={'width': '120%', 'height': '20px'}, debounce=True, value=349),
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
                                      style={'width': '120%', 'height': '20px'}, debounce=True, value=351),
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
                                          style={'width': '120%', 'height': '20px'}, debounce=True, value=5)
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
                                          style={'width': '100%', 'height': '20px'}, debounce=True, value=0),
                                width={'size':6}
                            )
                        ),
                    ])

                ]
            ),
        ]
    ),
    className="w-100"
)

#       -----------dbc card for Rayleigh scattering interpolation

card_rayleigh = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Rayleigh scattering", className="card-title"),
            html.Div(
                [
                    dbc.Row([
                        dcc.Checklist(id='rayleigh-button',
                                      options=[{'label': 'Rayleigh scattering interpolation', 'value': "rayleigh"}]),
                    ]),
                    dbc.Row([
                        html.H6("First order")
                    ]),
                    dbc.Row([
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

                        dbc.Col(
                            dbc.Label("width")
                        ),
                        dbc.Col(
                            dcc.Input(id='rayleigh-o1-width',
                                      type='number', placeholder='max',
                                      style={'width': '100px', 'height': '20px'}, debounce=True, value=20)
                        ),
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
                    dbc.Row([
                        html.H6("Second order")
                    ]),
                    dbc.Row([
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

                        dbc.Col(
                            dbc.Label("width")
                        ),
                        dbc.Col(
                            dcc.Input(id='rayleigh-o2-width',
                                      type='number', placeholder='max',
                                      style={'width': '100px', 'height': '20px'}, debounce=True, value=20)
                        ),
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
            html.H5("Raman scattering", className="card-title"),
            html.Div(
                [
                    dbc.Row([
                        dcc.Checklist(id='raman-button',
                                      options=[{'label': 'Raman scattering interpolation', 'value': "raman"}]),
                    ]),
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

                        dbc.Col(
                            dbc.Label("width")
                        ),
                        dbc.Col(
                            dcc.Input(id='raman-width',
                                      type='number', placeholder='max',
                                      style={'width': '100px', 'height': '20px'}, debounce=True, value=15)
                        ),
                        dbc.Col(
                            dbc.Label("dimension")
                        ),
                        dbc.Col(
                            dcc.Dropdown(id='raman-dimension',
                                         options=[{'label': '1d-ex', 'value': '1d-ex'},
                                                  {'label': '1d-em', 'value': '1d-em'},
                                                  {'label': '2d', 'value': '2d'}],
                                         value='zero', placeholder='', style={'width': '100%'})
                        )
                    ]),
                ]
            ),
        ]
    ),
    className="w-100"
)

#       -----------dbc card for Gaussian smoothing

card_smoothing = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Gaussian smoothing", className="card-title"),
            html.Div(
                [
                    dbc.Row([
                        dcc.Checklist(id='gaussian-smoothing-button',
                                      options=[{'label': 'Gaussian smoothing', 'value': "gaussian"}]),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Label("sigma"),
                        ),
                        dbc.Col(
                            dcc.Input(id='gaussian-sigma',
                                      type='number', placeholder='max',
                                      style={'width': '100px', 'height': '20px'}, debounce=True, value=1),
                        ),
                        dbc.Col(
                            dbc.Label("truncate")
                        ),
                        dbc.Col(
                            dcc.Input(id='gaussian-truncate',
                                      type='number', placeholder='max',
                                      style={'width': '100px', 'height': '20px'}, debounce=True, value=3),
                        )
                    ])
                ]
            )
        ]
    ),
    className="w-100"
)

#       -----------dbc card for downloading pre-processed EEM

card_eem_downloading = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Download pre-processed EEM", className="card-title"),
            html.Div([
                dbc.Row([
                    dbc.Label("output format"),
                    dcc.Dropdown(
                        id="eem-downloading-format",
                        options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                        value='aqualog', placeholder='Select EEM data file format',
                        style={'width': '100%'})
                ]),
                dbc.Button("Download", id='eem-download', className='col-4')
            ])
        ]
    ),
    className="w-100"
)

#       -----------dbc card for building EEM dataset

card_built_eem_dataset = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Build EEM dataset", className="card-title"),
            html.Div([
                dbc.Row([
                    dcc.Checklist(
                        id="align-exem",
                        options=[{'label': 'Align Ex/Em (select if the Ex/Em intervals are different between EEMs)',
                                  'value': 'align'}])
                ]),
                dbc.Button("Build", id='eem-download', className='col-4')
            ])
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
#
#
# @app.callback(
#     [Output('filename-dropdown', 'options'),
#      Output('filename-dropdown', 'value')],
#     [Input('folder-path-input', 'value'),
#      Input('filename-keyword', 'value')]
# )
# def update_filenames(folder_path, kw):
#     # Get list of filenames in the specified folder
#     if folder_path:
#         try:
#             filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#             if kw:
#                 filenames = [file for file in filenames if kw in file]
#             options = [{'label': filename, 'value': filename} for filename in filenames]
#             return options, None
#         except FileNotFoundError:
#             return ['Folder not found - please check the folder path'], None
#     else:
#         return [], None
#
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
# @app.callback(
#     Output('eem-graph', 'figure'),
#     [Input('folder-path-input', 'value'),
#      Input('filename-dropdown', 'value'),
#      Input('data-format', 'value'),
#      Input('intensity-scale-min', 'value'),
#      Input('intensity-scale-max', 'value'),
#      Input('rayleigh-scattering', 'value'),
#      Input('raman-interpolation-button', 'value'),
#      Input('raman-interpolation-method', 'value'),
#      Input('raman-interpolation-axis', 'value'),
#      Input('raman-interpolation-width', 'value')]
# )
# def update_eem_plot(file_path, filename, data_format, vmin, vmax, rayleigh_button, raman_button, raman_method,
#                     raman_axis, raman_width):
#     intensity = np.zeros([100, 100])
#     ex_range = np.arange(100)
#     em_range = np.array(100)
#     try:
#         full_path = os.path.join(file_path, filename)
#     except:
#         pass
#     try:
#         intensity, ex_range, em_range, _ = read_eem(full_path, index_pos=None, data_format=data_format)
#         if rayleigh_button:
#             intensity, _, _ = eem_rayleigh_masking(intensity, ex_range, em_range)
#         if raman_button:
#             intensity, _ = eem_raman_masking(intensity, ex_range, em_range, interpolation_method=raman_method,
#                                              interpolation_axis=raman_axis, width=raman_width)
#     except:
#         pass
#     fig = plot_eem(intensity, ex_range, em_range, vmin=vmin, vmax=vmax, plot_tool='plotly', display=False,
#                    auto_intensity_range=False, cmap='jet')
#     return fig


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
                dbc.NavLink("K-PARAFAC", href="/k-parafacs", active="exact"),
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

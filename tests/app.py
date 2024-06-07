import os
import dash
import datetime
import numpy as np
from io import BytesIO
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

# -----------Page #1: EEM pre-processing--------------

#   -------------Setting up the dbc cards

#       -----------dbc card for Selecting working folder and choosing EEM for previewing

card_selecting_files = dbc.Card([
    html.Div(
        [
            dbc.Row(
                dcc.Input(id='folder-path-input', type='text', placeholder='Enter folder path...',
                          style={'width': '1000px', 'height': '20px'}, debounce=True),
            ),

            dbc.Row(
                dbc.Col(
                    dbc.Label("EEM file format"),
                ),
                dbc.Col(
                    dcc.Dropdown(id='eem-data-format',
                                 options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                 value='aqualog', placeholder='Select EEM data file format', style={'width': '30%'}),
                ),
                dbc.Col(
                    dbc.Label("Absorbance file format"),
                ),
                dbc.Col(
                    dcc.Dropdown(id='abs-data-format',
                                 options=[{'label': 'Horiba Aqualog .dat file', 'value': 'aqualog'}],
                                 value='aqualog', placeholder='Select EEM data file format', style={'width': '30%'}),
                ),
            ),

            dbc.Row(
                dbc.Col(
                    dbc.Label("Overarching file keyword"),
                ),
                dbc.Col(
                    dcc.Input(id='eem-file-keyword', type='text', placeholder='',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value='PEM'),
                ),
                dbc.Col(
                    dbc.Label("Sample EEM file keyword"),
                ),
                dbc.Col(
                    dcc.Input(id='eem-file-keyword', type='text', placeholder='',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value='PEM'),
                ),
            ),

            dbc.Row(
                dbc.Col(
                    dbc.Label("Absorbance file keyword"),
                ),
                dbc.Col(
                    dcc.Input(id='abs-file-keyword', type='text', placeholder='',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value='ABS'),
                ),
                dbc.Col(
                    dbc.Label("Blank EEM file keyword"),
                ),
                dbc.Col(
                    dcc.Input(id='blank-file-keyword', type='text', placeholder='',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value='BEM'),
                ),
            ),

            dbc.Row(

                dbc.Col(
                    dbc.Label("Index starting position"),
                ),
                dbc.Col(
                    dcc.Input(id='index-pos-left', type='number', placeholder='',
                              style={'width': '100px', 'height': '20px'}, debounce=True),
                ),
                dbc.Col(
                    dbc.Label("Index end position"),
                ),
                dbc.Col(
                    dcc.Input(id='blank-file-keyword', type='text', placeholder='',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value='BEM'),
                ),
            ),

            dbc.Row(
                dbc.Col(
                    dcc.Checklist(options=['Read index as timestamp'], id='timestamp-checkbox'),
                ),
                dbc.Col(
                    dbc.Label("timestamp format")
                ),
                dbc.Col(
                    dcc.Input(id='timestamp-format', type='text', placeholder='Enter timestamp format',
                              style={'width': '200px', 'height': '20px'}, debounce=True),
                )
            ),
        ]
    ),
])

#       -----------dbc card for EEM display

card_eem_display = dbc.Card([
    html.Div(
        [
            dbc.Row(
                dbc.Col(
                    dbc.Label("EEM to preview")
                ),
                dbc.Col(
                    dcc.Dropdown(id='filename-dropdown', options=[], placeholder='Select a filename...')
                ),
            ),

            dcc.Graph(id='eem-graph'),
        ]
    ),
])

#       -----------dbc card for Selecting excitation/emission/intensity display ranges

card_range_selection = dbc.Card([
    html.Div(
        [
            dbc.Row([
                dbc.Col(
                    dbc.Label("Excitation wavelength")
                ),
                dbc.Col(
                    dcc.Input(id='excitation-wavelength-min', type='number', placeholder='min',
                              style={'width': '100px', 'height': '20px'}, debounce=True)
                ),
                dbc.Col(
                    dbc.Label("nm - "), width='auto'
                ),
                dbc.Col(
                    dcc.Input(id='excitation-wavelength-max', type='number', placeholder='max',
                              style={'width': '100px', 'height': '20px'}, debounce=True)
                ),
                dbc.Col(
                    dbc.Label("nm"), width='auto'
                )
            ]),

            dbc.Row([
                dbc.Col(
                    dbc.Label("Emission wavelength")
                ),
                dbc.Col(
                    dcc.Input(id='emission-wavelength-min', type='number', placeholder='min',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                ),
                dbc.Col(
                    dbc.Label("nm -"), width='auto'
                ),
                dbc.Col(
                    dcc.Input(id='emission-wavelength-max', type='number', placeholder='max',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                ),
                dbc.Col(
                    dbc.Label("nm"), width='auto'
                )
            ]),

            dbc.Row([
                dbc.Col(
                    dbc.Label("Fluorescence intensity")
                ),
                dbc.Col(
                    dcc.Input(id='fluorescence-intensity-min', type='number', placeholder='min',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                ),
                dbc.Col(
                    dbc.Label("a.u. -"), width='auto'
                ),
                dbc.Col(
                    dcc.Input(id='fluorescence-intensity-max', type='number', placeholder='max',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                ),
                dbc.Col(
                    dbc.Label("a.u. -"), width='auto'
                )
            ])
        ]
    ),
])

#       -----------dbc card for IFE correction

card_ife = dbc.Card([
    html.Div(
        [
            dbc.Row(
                dcc.Checklist(id='ife-button',
                              options=[{'label': 'Inner filter effect correction', 'value': True}]),
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Label("method"),
                ),
                dbc.Col(
                    dcc.Dropdown(id='ife-methods', options=[{'label': 'default', 'value': 'default'}],
                                 value='default', placeholder='Select IFE correction method', style={'width': '30%'})
                )
            ),

        ]
    ),
])

#       -----------dbc card for automatic scattering unit normalization

card_su = dbc.Card([
    html.Div(
        [
            dbc.Row(
                dcc.Checklist(id='su-button',
                              options=[{'label': 'Inner filter effect correction', 'value': True}]),
            ),

            dbc.Row(
                dbc.Col(
                    dbc.Label("Ex lower bound"),
                ),
                dbc.Col(
                    dcc.Input(id='su-excitation-lower-bound', type='number', placeholder='max',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                ),
                dbc.Col(
                    dbc.Label("Ex upper bound"),
                ),
                dbc.Col(
                    dcc.Input(id='su-excitation-upper-bound', type='number', placeholder='max',
                              style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                ),
            ),

            dbc.Row(
                dbc.Col(
                    dbc.Label("Em width (wavenumber)")
                ),
                dbc.Col(
                    dbc.Col(
                        dcc.Input(id='su-emission-width', type='number', placeholder='max',
                                  style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                    )
                ),
                dbc.Col(
                    dbc.Label("1/cm")
                )
            ),

            dbc.Row(
                dbc.Col(
                    dbc.Label("Normalization factor")
                ),
                dbc.Col(
                    dbc.Col(
                        dcc.Input(id='su-normalization-factor', type='number', placeholder='max',
                                  style={'width': '100px', 'height': '20px'}, debounce=True, value=0)
                    )
                ),
            )

        ]
    ),
])

#       -----------dbc card for Rayleigh scattering interpolation


#       -----------dbc card for Raman scattering interpolation


#       -----------dbc card for EEM smoothing


#       -----------dbc card for downloading pre-processed EEM


#       -----------dbc card for EEM dataset construction


#   -------------layout of the 1st page


eem_preprocessing_content = html.Div([

    # --------EEM display

    html.Div([
        dcc.Graph(id='eem-graph'),
    ]),

    # --------Select excitation/emission/intensity display ranges
    html.Div([
        # Widgets for colorbar scale
        dcc.Input(id='intensity-scale-min', type='number', placeholder='Enter colorbar minimum',
                  style={'width': '100px', 'height': '20px'}, debounce=True, value=0),

        dcc.Input(id='intensity-scale-max', type='number', placeholder='Enter colorbar maximum',
                  style={'width': '100px', 'height': '20px'}, debounce=True, value=1000),
    ]),

    # Settings for inner filter effect correction, dilution correction and Raman scattering unit calibration

    # Settings for Raman scattering interpolation

    # Settings for Rayleigh scattering interpolation

    # Settings for Gaussian smoothing

    # Widgets for Rayleigh scattering

    dcc.Checklist(options=[{'label': 'Remove Rayleigh scattering', 'value': True}], id='rayleigh-scattering'),

    # Widgets for Raman scattering

    dcc.Checklist(options=[{'label': 'Remove Raman scattering', 'value': True}], id='raman-interpolation-button'),

    dcc.Dropdown(id='raman-interpolation-method', options=['linear', 'cubic', 'nan'], placeholder='Method',
                 value='linear', style={'width': '30%'}),

    dcc.Dropdown(id='raman-interpolation-axis', options=[{'label': '1d - along Ex', 'value': 'ex'},
                                                         {'label': '1d - along Em', 'value': 'em'},
                                                         {'label': '2d', 'value': 'grid'}],
                 placeholder='Dimension', value='2d', style={'width': '30%'}),

    dcc.Input(id='raman-interpolation-width', type='number', placeholder='Width',
              style={'width': '100px', 'height': '20px'}, debounce=True, value=20),
])


@app.callback(
    [Output('filename-dropdown', 'options'),
     Output('filename-dropdown', 'value')],
    [Input('folder-path-input', 'value'),
     Input('filename-keyword', 'value')]
)
def update_filenames(folder_path, kw):
    # Get list of filenames in the specified folder
    if folder_path:
        try:
            filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if kw:
                filenames = [file for file in filenames if kw in file]
            options = [{'label': filename, 'value': filename} for filename in filenames]
            return options, None
        except FileNotFoundError:
            return ['Folder not found - please check the folder path'], None
    else:
        return [], None


@app.callback(
    Output('index-display', 'children'),
    [Input('filename-dropdown', 'value'),
     Input('index-pos-left', 'value'),
     Input('index-pos-right', 'value'),
     Input('timestamp-checkbox', 'value'),
     Input('timestamp-format', 'value')]
)
def update_index_display(filename, pos_left, pos_right, ts_option, ts_format):
    if pos_left and pos_right:
        index = filename[pos_left:pos_right]
        if ts_option:
            index = datetime.strptime(index, ts_format)
        return index
    return None


@app.callback(
    Output('eem-graph', 'figure'),
    [Input('folder-path-input', 'value'),
     Input('filename-dropdown', 'value'),
     Input('data-format', 'value'),
     Input('intensity-scale-min', 'value'),
     Input('intensity-scale-max', 'value'),
     Input('rayleigh-scattering', 'value'),
     Input('raman-interpolation-button', 'value'),
     Input('raman-interpolation-method', 'value'),
     Input('raman-interpolation-axis', 'value'),
     Input('raman-interpolation-width', 'value')]
)
def update_eem_plot(file_path, filename, data_format, vmin, vmax, rayleigh_button, raman_button, raman_method,
                    raman_axis, raman_width):
    intensity = np.zeros([100, 100])
    ex_range = np.arange(100)
    em_range = np.array(100)
    try:
        full_path = os.path.join(file_path, filename)
    except:
        pass
    try:
        intensity, ex_range, em_range, _ = read_eem(full_path, index_pos=None, data_format=data_format)
        if rayleigh_button:
            intensity, _, _ = eem_rayleigh_masking(intensity, ex_range, em_range)
        if raman_button:
            intensity, _ = eem_raman_masking(intensity, ex_range, em_range, interpolation_method=raman_method,
                                             interpolation_axis=raman_axis, width=raman_width)
    except:
        pass
    fig = plot_eem(intensity, ex_range, em_range, vmin=vmin, vmax=vmax, plot_tool='plotly', display=False,
                   auto_intensity_range=False, cmap='jet')
    return fig


# --------------Setup the sidebar-----------------

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
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("EEM pre-processing", href="/", active="exact"),
                dbc.NavLink("PARAFAC", href="/page-1", active="exact"),
                dbc.NavLink("K-PARAFAC", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
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
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
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

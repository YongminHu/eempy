"""
Root tabs container (what users see as the main navigation).

This is separated so the app factory can simply do:
    app.layout = serve_layout
where serve_layout includes stores + ``layouts.root.layout``.
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from .home import layout as home_layout
from .preprocessing import layout as eem_processing_layout
from .peak_picking import layout as peak_picking_layout
from .rfi import layout as regional_integration_layout
from .parafac import layout as parafac_layout
from .nmf import layout as nmf_layout
from .kmethod import layout as kmethod_layout


layout = html.Div(
    [
        html.H2("eempy-vis", className="display-5"),
        html.Hr(),
        html.P("An open-source, interactive toolkit for EEM analysis", className="lead"),
        dbc.Tabs(
            id="tabs-content",
            children=[
                dcc.Tab(label="Homepage", id="homepage", children=home_layout),
                dcc.Tab(label="EEM pre-processing", id="eem-pre-processing", children=eem_processing_layout),
                dcc.Tab(label="Peak picking", id="eem-peak-picking", children=peak_picking_layout),
                dcc.Tab(label="Regional integration", id="eem-regional-integration", children=regional_integration_layout),
                dcc.Tab(label="PARAFAC", id="parafac", children=parafac_layout),
                dcc.Tab(label="NMF", id="nmf", children=nmf_layout),
                dcc.Tab(label="K-method", id="kmethod", children=kmethod_layout),
            ],
        ),
    ],
)

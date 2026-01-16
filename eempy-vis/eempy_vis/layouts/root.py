"""
Root tabs container (what users see as the main navigation).

This is separated so the app factory can simply do:
    app.layout = serve_layout
where serve_layout includes stores + ``layouts.root.layout``.
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from .home_layout import layout as home_layout
from .preprocessing_layout import layout as eem_processing_layout
from .peak_picking_layout import layout as peak_picking_layout
from .rfi_layout import layout as regional_integration_layout
from .parafac_layout import layout as parafac_layout
from .nmf_layout import layout as nmf_layout
from .kmethod_layout import layout as kmethod_layout
from ..ids import IDS


def build_shell_layout():
    shell = html.Div(
        [
            html.H2("eempy-vis", className="display-5"),
            html.Hr(),
            html.P("An open-source, interactive toolkit for EEM analysis", className="lead"),
            dbc.Tabs(
                id="tabs-content",
                children=[
                    dcc.Tab(label="Homepage", id=IDS.HOMEPAGE, children=home_layout),
                    dcc.Tab(label="EEM pre-processing", id=IDS.EEM_PRE_PROCESSING, children=eem_processing_layout),
                    dcc.Tab(label="Peak picking", id=IDS.EEM_PEAK_PICKING, children=peak_picking_layout),
                    dcc.Tab(label="Regional integration", id=IDS.EEM_REGIONAL_INTEGRATION,
                            children=regional_integration_layout),
                    dcc.Tab(label="PARAFAC", id=IDS.PARAFAC, children=parafac_layout),
                    dcc.Tab(label="NMF", id=IDS.NMF, children=nmf_layout),
                    dcc.Tab(label="K-method", id=IDS.KMETHOD, children=kmethod_layout),
                ],
            ),
        ],
    )
    return html.Div([
        dcc.Store(id=IDS.PRE_PROCESSED_EEM),
        dcc.Store(id=IDS.EEM_DATASET),
        dcc.Store(id=IDS.PP_MODEL),
        dcc.Store(id=IDS.PP_TEST_RESULTS),
        dcc.Store(id=IDS.RI_MODEL),
        dcc.Store(id=IDS.RI_TEST_RESULTS),
        dcc.Store(id=IDS.PARAFAC_MODELS),
        dcc.Store(id=IDS.PARAFAC_TEST_RESULTS),
        dcc.Store(id=IDS.KMETHOD_CONSENSUS_MATRIX_DATA),
        dcc.Store(id=IDS.KMETHOD_EEM_DATASET_ESTABLISH),
        dcc.Store(id=IDS.KMETHOD_BASE_CLUSTERING_PARAMETERS),
        dcc.Store(id=IDS.KMETHOD_MODELS),
        dcc.Store(id=IDS.KMETHOD_TEST_RESULTS),
        dcc.Store(id=IDS.NMF_MODELS),
        dcc.Store(id=IDS.NMF_TEST_RESULTS),
        shell])

"""
Shared configuration for the Dash EEM decomposition app.

Keep UI constants and styling defaults here so they can be imported across layouts/callback modules.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.express as px

# Dash theme(s)
EXTERNAL_STYLESHEETS = [dbc.themes.BOOTSTRAP]

# Default qualitative color cycle used across plots
PLOTLY_COLORS = px.colors.qualitative.Plotly

# Default marker shapes used for multi-series scatter plots
MARKER_SHAPES = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left',
                 'triangle-right', 'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw']

# Reusable style for the small "?" help icon
HELP_ICON_STYLE = {'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center', 'width': '18px',
                   'height': '18px', 'borderRadius': '50%', 'backgroundColor': '#1e66ff', 'color': 'white',
                   'fontSize': '12px', 'fontWeight': '700', 'lineHeight': '18px', 'cursor': 'help',
                   'marginLeft': '6px', 'userSelect': 'none'}

CARD_STYLE = {
    "borderRadius": "10px",
}

SECTION_TITLE_STYLE = {
    "fontWeight": "700",
    "marginBottom": "8px",
}

ROW_FLEX_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
}

SMALL_TEXT_STYLE = {
    "fontSize": "12px",
}

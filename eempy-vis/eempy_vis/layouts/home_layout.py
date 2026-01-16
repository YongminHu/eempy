"""
Homepage layout.
"""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

# Reuse shared styles if you want to style help icons etc.

layout = html.Div([
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

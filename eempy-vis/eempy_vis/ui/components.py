"""
Reusable UI components.

These helpers return Dash component trees to reduce layout repetition.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from dash import dcc, html
import dash_bootstrap_components as dbc

from ..eempy_vis import HELP_ICON_STYLE, ROW_FLEX_STYLE


def help_icon(
    component_id: str,
    *,
    style: Optional[dict] = None,
    title: str = "?",
) -> html.Span:
    """
    Create a circular help icon.

    Parameters
    ----------
    component_id : str
        The Dash component id for the icon (used as Popover/Tooltip target).
    style : dict, optional
        Style overrides merged on top of ``HELP_ICON_STYLE``.
    title : str, default "?"
        Text displayed inside the icon.

    Returns
    -------
    icon : dash.html.Span
        Styled span element.
    """
    s = dict(HELP_ICON_STYLE)
    if style:
        s.update(style)
    return html.Span(title, id=component_id, style=s)


def help_popover(
    target_id: str,
    body: Union[str, Sequence[Any]],
    *,
    header: Optional[str] = "Help",
    placement: str = "right",
    trigger: Union[str, None] = "legacy",
    popover_id: Optional[str] = None,
    style: Optional[dict] = None,
) -> dbc.Popover:
    """
    Create a Bootstrap popover attached to ``target_id``.

    Parameters
    ----------
    target_id : str
        The id of the element that triggers the popover.
    body : str or sequence of components
        Popover content. Use a list to include rich content (images, links, etc.).
    header : str or None, default "Help"
        Popover header. Set to None to omit.
    placement : str, default "right"
        Where to place the popover relative to the target.
    trigger : str or None, default "legacy"
        Bootstrap trigger. ``"legacy"`` means click to toggle and click outside to dismiss.
        If you want to control ``is_open`` with a callback, set ``trigger=None`` and pass ``popover_id``.
    popover_id : str, optional
        Id for the Popover component (only needed if you control ``is_open``).
    style : dict, optional
        Style overrides (e.g., maxWidth).

    Returns
    -------
    popover : dbc.Popover
        Popover component.
    """
    children = []
    if header is not None:
        children.append(dbc.PopoverHeader(header))
    children.append(dbc.PopoverBody(body))
    return dbc.Popover(
        children,
        id=popover_id,
        target=target_id,
        trigger=trigger,
        placement=placement,
        style=style,
    )


# def help_tooltip(
#     target_id: str,
#     text: str,
#     *,
#     placement: str = "right",
#     tooltip_id: Optional[str] = None,
#     style: Optional[dict] = None,
# ) -> dbc.Tooltip:
#     """
#     Create a Bootstrap tooltip attached to ``target_id``.
#
#     Parameters
#     ----------
#     target_id : str
#         The id of the element that triggers the tooltip.
#     text : str
#         Tooltip text (keep short).
#     placement : str, default "right"
#         Tooltip placement.
#     tooltip_id : str, optional
#         Id for the Tooltip component.
#     style : dict, optional
#         Style overrides merged on top of ``DEFAULT_TOOLTIP_STYLE``.
#
#     Returns
#     -------
#     tooltip : dbc.Tooltip
#         Tooltip component.
#     """
#     s = dict(DEFAULT_TOOLTIP_STYLE)
#     if style:
#         s.update(style)
#     return dbc.Tooltip(text, target=target_id, placement=placement, id=tooltip_id, style=s)


def labeled_help(
    label: Union[str, Any],
    help_text: Union[str, Sequence[Any]],
    *,
    icon_id: str,
    popover_header: Optional[str] = "Help",
    placement: str = "right",
    trigger: Union[str, None] = "legacy",
    container_style: Optional[dict] = None,
) -> html.Div:
    """
    Convenience wrapper: label + help icon + popover.

    Parameters
    ----------
    label : str or component
        Label content.
    help_text : str or sequence of components
        Popover content.
    icon_id : str
        Id for the help icon (also the popover target).
    popover_header : str or None, default "Help"
        Popover header.
    placement : str, default "right"
        Popover placement.
    trigger : str or None, default "legacy"
        Popover trigger.
    container_style : dict, optional
        Style overrides for the surrounding row container.

    Returns
    -------
    row : dash.html.Div
        A row-like div containing the label, icon, and popover.
    """
    s = dict(ROW_FLEX_STYLE)
    if container_style:
        s.update(container_style)
    icon = help_icon(icon_id)
    pop = help_popover(icon_id, help_text, header=popover_header, placement=placement, trigger=trigger)
    return html.Div([label, icon, pop], style=s)


def graph_block(
    graph_id: str,
    *,
    figure: Optional[dict] = None,
    height: Optional[int] = None,
    style: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dcc.Graph:
    """
    Standard Graph wrapper.

    Parameters
    ----------
    graph_id : str
        Graph id.
    figure : dict, optional
        Plotly figure. If None, an empty figure is used.
    height : int, optional
        Height in pixels.
    style : dict, optional
        Additional style dict for the Graph container.
    config : dict, optional
        Plotly config passed to dcc.Graph.

    Returns
    -------
    graph : dash.dcc.Graph
        Graph component.
    """
    fig = figure or {"data": [], "layout": {}}
    s = style or {}
    if height is not None:
        s = dict(s)
        s["height"] = height
    return dcc.Graph(id=graph_id, figure=fig, style=s, config=config or {"displaylogo": False})


def image_in_popover(src: str, *, width: str = "100%", style: Optional[dict] = None) -> html.Img:
    """
    Helper to embed an image inside a PopoverBody.

    Parameters
    ----------
    src : str
        Image source (e.g., ``"/assets/example.png"``).
    width : str, default "100%"
        CSS width.
    style : dict, optional
        Style overrides.

    Returns
    -------
    img : dash.html.Img
        Image component.
    """
    s = {"width": width, "borderRadius": "6px"}
    if style:
        s.update(style)
    return html.Img(src=src, style=s)

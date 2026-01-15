"""
Shared imports and small utilities for callbacks.

Keep heavy imports here to avoid repeating them across callback modules.
"""

from __future__ import annotations

import copy
import json
import os.path
import pickle
from math import sqrt

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# EEMPy imports used throughout the app
from eempy.read_data import (
    get_filelist,
    read_eem,
    read_abs,
    read_eem_dataset,
    read_abs_dataset,
)
from eempy.eem_processing import (
    eem_cutting,
    eem_raman_normalization,
    eem_ife_correction,
    eem_median_filter,
    eem_raman_scattering_removal,
    eem_rayleigh_scattering_removal,
    eem_gaussian_filter,
    EEMDataset,
    PARAFAC,
    EEMNMF,
    SplitValidation,
    KMethod,
    eems_fit_components,
)
from eempy.plot import (
    plot_eem,
    plot_abs,
    plot_loadings,
    plot_fmax,
    plot_dendrogram,
    plot_reconstruction_error,
    plot_error,
)
from eempy.utils import (
    str_string_to_list,
    num_string_to_list,
)

# Shared visual defaults (pulled from eem_app.config)
from ..eem_app import PLOTLY_COLORS as colors, MARKER_SHAPES as marker_shapes

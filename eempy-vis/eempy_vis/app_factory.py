from dash import Dash

from .config import EXTERNAL_STYLESHEETS
from ..layouts import preprocessing, parafac, nmf, kmethod, peak_picking

def create_app():
    app = Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
    )

    # Register callbacks page-by-page
    preprocessing.register_callbacks(app)
    parafac.register_callbacks(app)
    nmf.register_callbacks(app)
    kmethod.register_callbacks(app)
    peak_picking.register_callbacks(app)
    # ...

    return app
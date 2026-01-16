from dash import Dash

from .config import EXTERNAL_STYLESHEETS
from .callbacks.register import register_all_callbacks
from .layouts.root import build_shell_layout

def create_app():
    app = Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
    )

    app.layout = build_shell_layout()

    register_all_callbacks(app)

    return app
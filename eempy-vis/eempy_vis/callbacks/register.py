from . import preprocessing_callbacks, peak_picking_callbacks, rfi_callbacks, parafac_callbacks, nmf_callbacks, kmethod_callbacks

def register_all_callbacks(app):
    preprocessing_callbacks.register_callbacks(app)
    peak_picking_callbacks.register_callbacks(app)
    rfi_callbacks.register_callbacks(app)
    parafac_callbacks.register_callbacks(app)
    nmf_callbacks.register_callbacks(app)
    kmethod_callbacks.register_callbacks(app)
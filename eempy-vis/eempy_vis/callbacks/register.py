from . import preprocessing, peak_picking, rfi, parafac, nmf, kmethod

def register_all_callbacks(app):
    preprocessing.register_callbacks(app)
    peak_picking.register_callbacks(app)
    rfi.register_callbacks(app)
    parafac.register_callbacks(app)
    nmf.register_callbacks(app)
    kmethod.register_callbacks(app)
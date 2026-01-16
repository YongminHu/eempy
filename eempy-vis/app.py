from eempy_vis.app_factory import create_app

app = create_app()
server = app.server  # for gunicorn etc.

if __name__ == "__main__":
    app.run_server(debug=True)
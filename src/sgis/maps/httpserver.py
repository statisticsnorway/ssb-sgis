import os
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer

from IPython.display import HTML
from IPython.display import display


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from `start_port`."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port  # Port is available
    raise RuntimeError("No available ports found in range.")


def run_html_server(contents: str | None = None, port: int = 3000) -> None:
    """Run a simple, temporary http web server for serving static HTML content."""
    port = find_available_port(port)

    if "JUPYTERHUB_SERVICE_PREFIX" in os.environ:
        # Create a link using the https://github.com/jupyterhub/jupyter-server-proxy
        display_address = os.environ["JUPYTERHUB_SERVICE_PREFIX"] + f"proxy/{port}/"
        stop_address = os.environ["JUPYTERHUB_SERVICE_PREFIX"] + f"proxy/{port}/stop"
        display_content = HTML(
            f"""
        <p>Click <a href='{display_address}'>here</a> to open in browser.</p>
        <p>Click <a href='{stop_address}'>here</a> to stop.</p>
        """
        )
    else:
        display_address = f"http://localhost:{port}"
        display_content = HTML(
            f"""
        <p>Click <a href='http://localhost:{port}'>here</a> to open in browser.</p>
        <p>Click <a href='http://localhost:{port}/stop'>here</a> to stop.<p>"
        """
        )

    class HTTPServerRequestHandler(BaseHTTPRequestHandler):
        """A handler of request for the server, hosting static content."""

        allow_reuse_address = True

        def do_GET(self):
            """Handle GET requests."""
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            if self.path == "/stop":
                self.wfile.write(bytes("The server is stopped", encoding="utf-8"))
                raise KeyboardInterrupt
            else:
                self.wfile.write(bytes(contents, encoding="utf-8"))

    with HTTPServer(("127.0.0.1", port), HTTPServerRequestHandler) as httpd:
        display(display_content)
        webbrowser.open(display_address)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from IPython.core.display import display, HTML


def run_html_server(contents: str = None, port: int = 3000):
    """
    Run a simple, temporary http web server for serving static HTML content.
    """
    display_content = f"Server started at http://localhost:{port}"
    if 'JUPYTERHUB_SERVICE_PREFIX' in os.environ:
        # Create a link using the https://github.com/jupyterhub/jupyter-server-proxy
        display_address = os.environ['JUPYTERHUB_SERVICE_PREFIX'] + "proxy/{}/".format(port)
        display_content = HTML(f"Click <a href='{display_address}'>here</a> to open in browser.")

    class HTTPServerRequestHandler(BaseHTTPRequestHandler):
        """
        An handler of request for the server, hosting static content.
        """
        def do_GET(self):
            """Handle GET requests"""
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes(contents, encoding='utf-8'))

    with HTTPServer(('127.0.0.1', port), HTTPServerRequestHandler) as httpd:
        display(display_content)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")

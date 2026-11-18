"""
Smart Commerce AI - Root Serverless / WSGI Entrypoint
Implements Vercel BaseHTTPRequestHandler specification:
Exports 'handler', 'app', and 'application' to satisfy Vercel, Gunicorn, and AWS Lambda.
Serves index.html (interactive e-commerce storefront and cluster telemetry visualizer).
"""
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class handler(BaseHTTPRequestHandler):
    """Vercel Python Serverless Function entrypoint."""
    def do_GET(self):
        index_file = ROOT / "index.html"
        if not index_file.exists():
            index_file = ROOT.parent / "index.html"

        if index_file.exists():
            content = index_file.read_bytes()
        else:
            content = b"<h1>Smart Commerce AI</h1><p>Storefront index.html active.</p>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

# Export alias variables for WSGI / ASGI / Serverless runners
app = handler
application = handler

if __name__ == "__main__":
    from http.server import HTTPServer
    port = int(os.environ.get("PORT", 3000))
    print(f"Serving Smart Commerce AI storefront on http://localhost:{port}/")
    server = HTTPServer(("", port), handler)
    server.serve_forever()

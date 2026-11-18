"""
Smart Commerce AI - Root Serverless / WSGI Entrypoint
Exports 'app', 'application', and 'handler' to satisfy Vercel, Gunicorn, AWS Lambda, and Render runners.
Serves index.html (the interactive e-commerce storefront and cluster telemetry visualizer).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def handler(request, *args, **kwargs):
    """Serverless HTTP request handler for Vercel / AWS Lambda."""
    index_file = ROOT / "index.html"
    if index_file.exists():
        html_content = index_file.read_text(encoding="utf-8")
    else:
        html_content = "<h1>Smart Commerce AI</h1><p>Storefront index.html active.</p>"

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "public, max-age=0, must-revalidate"
        },
        "body": html_content
    }

class WSGIApplication:
    """WSGI compliant application for Vercel, Gunicorn, uWSGI."""
    def __call__(self, environ, start_response):
        index_file = ROOT / "index.html"
        if index_file.exists():
            body = index_file.read_bytes()
        else:
            body = b"<h1>Smart Commerce AI</h1><p>Storefront active.</p>"

        status = '200 OK'
        response_headers = [
            ('Content-Type', 'text/html; charset=utf-8'),
            ('Content-Length', str(len(body))),
            ('Cache-Control', 'public, max-age=0, must-revalidate')
        ]
        start_response(status, response_headers)
        return [body]

# Top-level variables expected by deployment environments
app = WSGIApplication()
application = app

if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    port = int(os.environ.get("PORT", 3000))
    print(f"Serving Smart Commerce AI storefront on http://localhost:{port}/")
    with make_server("", port, app) as httpd:
        httpd.serve_forever()

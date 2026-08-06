"""Local dev server for the arena.

Serves ``public/`` and proxies ``/api/*`` to the Flask functions
(run them first: ``python api/game.py`` and ``python api/learn.py``).

    python3 scripts/dev_server.py   # http://localhost:8080
"""
from __future__ import annotations

import http.server
import os
import sys
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "public")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
GAME_URL = "http://127.0.0.1:5001"
LEARN_URL = "http://127.0.0.1:5002"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PUBLIC, **kw)

    def _proxy(self, method: str):
        path = self.path
        target = GAME_URL if path.startswith("/api/game") else LEARN_URL
        url = target + path
        body = None
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            body = self.rfile.read(length)
        req = urllib.request.Request(url, data=body, method=method)
        for h in ("Content-Type",):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() in ("content-type", "cache-control"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(500)
            body = str(e).encode()
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self._proxy("GET")
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self._proxy("POST")
        return super().do_POST()

    def log_message(self, *a):  # silence
        pass


if __name__ == "__main__":
    print(f"arena dev server: http://localhost:{PORT}  (public={PUBLIC})")
    http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

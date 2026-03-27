"""
Local experiment browser server.
Serves results/ as static files and provides a delete API endpoint.

Usage (from project root):
    python server.py [port]
"""
import json
import shutil
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT      = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT      = Path(__file__).parent.resolve()
RESULTS   = ROOT / "results"


class Handler(SimpleHTTPRequestHandler):
    # Serve files from results/
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RESULTS), **kwargs)

    def end_headers(self):
        # Prevent caching of JSON data files
        if self.path and self.path.split('?')[0].endswith('.json'):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        # Quieter logs — only show non-200 responses
        code = args[1] if len(args) > 1 else "?"
        if str(code) not in ("200", "304"):
            super().log_message(fmt, *args)

    def do_OPTIONS(self):
        self._cors(200)

    def do_POST(self):
        if self.path == "/api/delete":
            self._handle_delete()
        elif self.path == "/api/open-folder":
            self._handle_open_folder()
        else:
            self.send_error(404)

    def _handle_open_folder(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            path   = Path(body.get("path", "")).resolve()
        except Exception:
            return self._json(400, {"ok": False, "error": "Bad request body"})

        # Security: must be inside the project root
        try:
            path.relative_to(ROOT)
        except ValueError:
            return self._json(403, {"ok": False, "error": "Path outside project directory"})

        if not path.exists():
            return self._json(404, {"ok": False, "error": "Path does not exist"})

        try:
            subprocess.Popen(["explorer.exe", str(path)])
            return self._json(200, {"ok": True})
        except Exception as exc:
            return self._json(500, {"ok": False, "error": str(exc)})

    def _handle_delete(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            exp_path = Path(body.get("exp_path", "")).resolve()
        except Exception:
            return self._json(400, {"ok": False, "error": "Bad request body"})

        # ── Security: must be inside RESULTS and must be a train_* folder ──
        try:
            exp_path.relative_to(RESULTS)
        except ValueError:
            return self._json(403, {"ok": False, "error": "Path outside results directory"})

        if not exp_path.name.startswith("train_"):
            return self._json(403, {"ok": False, "error": "Only train_* folders can be deleted"})

        if not exp_path.is_dir():
            return self._json(404, {"ok": False, "error": "Directory not found"})

        # ── Delete ──
        try:
            shutil.rmtree(exp_path)
        except Exception as exc:
            return self._json(500, {"ok": False, "error": f"Delete failed: {exc}"})

        return self._json(200, {"ok": True, "deleted": str(exp_path)})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _cors(self, code):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    print(f"Serving results/ at http://localhost:{PORT}/experiment_browser.html")
    print(f"Delete API: POST http://localhost:{PORT}/api/delete")
    print("Press Ctrl+C to stop.\n")
    HTTPServer(("", PORT), Handler).serve_forever()

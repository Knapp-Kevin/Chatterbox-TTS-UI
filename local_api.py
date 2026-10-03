"""Local HTTP API: lets other programs on this PC use the app's models and voices.

Only listens on 127.0.0.1. Uses the standard library HTTP server, so it adds no
dependencies. The app supplies a backend object with these methods (each may
raise ApiError):

  health() -> dict             models() -> list         voices() -> list
  synthesize(request: dict) -> dict with "path", "mime", plus details

Endpoints
  GET  /v1/health        what's loaded, and whether a request is running
  GET  /v1/models        the model list (what "model" can name)
  GET  /v1/voices        the voice library and the loaded model's built-in voices
  POST /v1/audio/speech  OpenAI-compatible: {"model", "input", "voice",
                         "response_format", "speed", "instructions"} -> audio bytes
  POST /v1/speech        native: {"text", "model", "voice", "format", "subtitles",
                         "speed", "language", "style", "name"} -> JSON with the saved
                         files (chatterbox_outputs/api/) and details
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8765
MAX_BODY = 4 * 1024 * 1024
OPENAI_FORMATS = {"wav": "WAV", "flac": "FLAC", "mp3": "MP3"}


class ApiError(Exception):
    def __init__(self, status, message, kind="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


class LocalApiServer:
    def __init__(self, backend, port=DEFAULT_PORT, token="", log=print):
        self.backend = backend
        self.port = int(port)
        self.token = token.strip()
        self.log = log
        self.httpd = None
        self.thread = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        server = self

        class Handler(_Handler):
            api = server

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="local-api", daemon=True)
        self.thread.start()
        self.log(f"Local API listening on {self.url}")

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
            self.log("Local API stopped.")


class _Handler(BaseHTTPRequestHandler):
    api = None  # set per server
    server_version = "LocalTTS/1.0"

    # --- plumbing ---

    def log_message(self, fmt, *args):
        pass  # requests are logged by _route with their outcome

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, error):
        self._send_json(error.status, {"error": {"message": error.message, "type": error.kind}})

    def _send_file(self, path, mime):
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        token = self.api.token
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {token}" or self.headers.get("X-API-Key", "") == token

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "Request body is too large (4 MB max).")
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "The body must be JSON.")
        if not isinstance(data, dict):
            raise ApiError(400, "The body must be a JSON object.")
        return data

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        status_note = "ok"
        try:
            if not self._authorized():
                raise ApiError(401, "Missing or wrong API token (Authorization: Bearer <token>).",
                               "authentication_error")
            backend = self.api.backend
            if method == "GET" and path in ("/", "/v1", "/v1/health"):
                self._send_json(200, backend.health())
            elif method == "GET" and path == "/v1/models":
                self._send_json(200, {"object": "list", "data": backend.models()})
            elif method == "GET" and path == "/v1/voices":
                self._send_json(200, {"object": "list", "data": backend.voices()})
            elif method == "POST" and path == "/v1/audio/speech":
                self._openai_speech(self._read_json())
            elif method == "POST" and path == "/v1/speech":
                self._native_speech(self._read_json())
            else:
                raise ApiError(404, f"No endpoint {method} {path}. See GET /v1/health.", "not_found")
        except ApiError as error:
            status_note = f"{error.status} {error.message}"
            self._send_error(error)
        except Exception as exc:  # never take the app down
            status_note = f"500 {exc}"
            self._send_error(ApiError(500, f"Internal error: {exc}", "server_error"))
        if path not in ("/v1/health",):
            self.api.log(f"API {method} {path}: {status_note}")

    # --- speech ---

    def _openai_speech(self, data):
        text = str(data.get("input") or "").strip()
        if not text:
            raise ApiError(400, "\"input\" is required.")
        response_format = str(data.get("response_format") or "mp3").lower()
        if response_format not in OPENAI_FORMATS:
            raise ApiError(400, f"response_format {response_format!r} isn't supported here; "
                                f"use one of {', '.join(OPENAI_FORMATS)}.")
        result = self.api.backend.synthesize({
            "text": text, "model": data.get("model"), "voice": data.get("voice"),
            "format": OPENAI_FORMATS[response_format], "speed": data.get("speed"),
            "style": data.get("instructions"), "save": False, "subtitles": None,
        })
        try:
            self._send_file(result["path"], result["mime"])
        finally:
            _cleanup(result)

    def _native_speech(self, data):
        text = str(data.get("text") or data.get("input") or "").strip()
        if not text:
            raise ApiError(400, "\"text\" is required.")
        result = self.api.backend.synthesize({
            "text": text, "model": data.get("model"), "voice": data.get("voice"),
            "format": str(data.get("format") or "WAV").upper(), "speed": data.get("speed"),
            "language": data.get("language"), "style": data.get("style"),
            "subtitles": data.get("subtitles"), "save": True, "name": data.get("name"),
        })
        self._send_json(200, {key: value for key, value in result.items() if key not in ("mime", "temporary")})


def _cleanup(result):
    if result.get("temporary"):
        for key in ("path", "subtitles"):
            path = result.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

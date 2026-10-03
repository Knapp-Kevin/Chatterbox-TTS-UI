"""Google Docs import.

Two ways in:
  * A link to a doc shared as "Anyone with the link can view": downloaded as .docx
    from Google's export URL, no sign-in needed.
  * Sign in with Google (OAuth 2.0 for desktop apps, with PKCE): list and search
    your own docs through the Drive API, read-only.

Google requires every app to use its own OAuth client, so signing in needs a
client ID made once in Google Cloud Console (see the README). Everything here
uses the standard library; tokens are kept on this PC in google_auth.json.
"""

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

AUTH_FILENAME = "google_auth.json"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
DRIVE_URL = "https://www.googleapis.com/drive/v3/files"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
SIGN_IN_TIMEOUT = 300  # seconds to finish signing in in the browser


class GoogleDocsError(Exception):
    pass


def doc_id_from_url(text):
    """The document id from a Google Docs link (or a bare id)."""
    text = (text or "").strip()
    match = re.search(r"/document/(?:u/\d+/)?d/([a-zA-Z0-9_-]{20,})", text)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]{20,})", text)
    if match:
        return match.group(1)
    return text if re.fullmatch(r"[a-zA-Z0-9_-]{25,}", text) else None


def _request(url, data=None, headers=None, method=None, timeout=60):
    body = urllib.parse.urlencode(data).encode() if isinstance(data, dict) else data
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()
    except urllib.error.URLError as exc:
        raise GoogleDocsError(f"Couldn't reach Google: {exc.reason}")


def download_shared(doc_id):
    """(.docx bytes, title) for a doc shared with "Anyone with the link"."""
    status, headers, body = _request(
        f"https://docs.google.com/document/d/{doc_id}/export?format=docx")
    content_type = headers.get("Content-Type", "") if headers else ""
    if status == 200 and "wordprocessingml" in content_type:
        return body, _title_from_disposition(headers.get("Content-Disposition", ""))
    if status == 404:
        raise GoogleDocsError("No Google Doc at that link. Check that it's a Docs link, not Sheets or Slides.")
    raise GoogleDocsError("That doc isn't shared publicly. Share it as “Anyone with the link can view”, "
                          "or sign in with Google to open your own docs.")


def _title_from_disposition(disposition):
    match = re.search(r"filename\*=UTF-8''([^;]+)", disposition)
    if match:
        name = urllib.parse.unquote(match.group(1))
    else:
        match = re.search(r'filename="([^"]+)"', disposition)
        name = match.group(1) if match else "Google Doc"
    return os.path.splitext(name)[0]


class GoogleAccount:
    """OAuth client settings and tokens, stored in google_auth.json."""

    def __init__(self, app_dir):
        self.path = os.path.join(app_dir, AUTH_FILENAME)
        self.data = {}
        self.lock = threading.Lock()
        try:
            with open(self.path, encoding="utf-8") as handle:
                self.data = json.load(handle)
        except (OSError, ValueError):
            self.data = {}

    def _save(self):
        temp = self.path + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
        os.replace(temp, self.path)

    # --- client setup ---

    @property
    def configured(self):
        return bool(self.data.get("client_id") and self.data.get("client_secret"))

    @property
    def signed_in(self):
        return bool(self.data.get("refresh_token"))

    @property
    def email(self):
        return self.data.get("email", "")

    def set_client(self, client_id, client_secret):
        client_id, client_secret = client_id.strip(), client_secret.strip()
        if not client_id.endswith(".apps.googleusercontent.com") or not client_secret:
            raise GoogleDocsError("That doesn't look like a Google OAuth client (the ID ends in "
                                  ".apps.googleusercontent.com, and there's a secret).")
        if client_id != self.data.get("client_id"):
            self.data = {}  # a different client can't use the old tokens
        self.data.update(client_id=client_id, client_secret=client_secret)
        self._save()

    def load_client_file(self, path):
        """Read a client_secret_*.json downloaded from Google Cloud Console."""
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        client = payload.get("installed") or payload.get("web") or {}
        if "web" in payload and "installed" not in payload:
            raise GoogleDocsError("This is a Web application client. Create a “Desktop app” "
                                  "client instead.")
        self.set_client(client.get("client_id", ""), client.get("client_secret", ""))

    # --- signing in ---

    def sign_in(self, cancelled=lambda: False):
        """Open the browser to Google's sign-in page and wait for the result."""
        if not self.configured:
            raise GoogleDocsError("Set up a Google client first.")
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        result = {}

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if query.get("state", [""])[0] != state:
                    self.send_response(400)
                    self.end_headers()
                    return
                result.update(code=query.get("code", [""])[0], error=query.get("error", [""])[0])
                message = ("Signed in. You can close this tab and go back to the app."
                           if result["code"] else "Sign-in was cancelled. You can close this tab.")
                body = f"<html><body style='font-family:sans-serif;margin:3em'><h2>{message}</h2></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode())

        server = HTTPServer(("127.0.0.1", 0), Callback)
        server.timeout = 0.5
        redirect = f"http://127.0.0.1:{server.server_port}"
        url = AUTH_URL + "?" + urllib.parse.urlencode({
            "client_id": self.data["client_id"], "redirect_uri": redirect, "response_type": "code",
            "scope": SCOPE + " email", "code_challenge": challenge, "code_challenge_method": "S256",
            "state": state, "access_type": "offline", "prompt": "consent"})
        webbrowser.open(url)
        deadline = time.monotonic() + SIGN_IN_TIMEOUT
        try:
            while not result and time.monotonic() < deadline and not cancelled():
                server.handle_request()
        finally:
            server.server_close()
        if not result:
            raise GoogleDocsError("Sign-in wasn't finished in the browser.")
        if not result.get("code"):
            raise GoogleDocsError(f"Google didn't sign you in ({result.get('error') or 'cancelled'}).")
        status, _headers, body = _request(TOKEN_URL, {
            "code": result["code"], "client_id": self.data["client_id"],
            "client_secret": self.data["client_secret"], "redirect_uri": redirect,
            "grant_type": "authorization_code", "code_verifier": verifier})
        tokens = json.loads(body or b"{}")
        if status != 200 or "access_token" not in tokens:
            raise GoogleDocsError(f"Google refused the sign-in: {tokens.get('error_description') or tokens.get('error') or status}")
        with self.lock:
            self.data.update(access_token=tokens["access_token"],
                             expires_at=time.time() + int(tokens.get("expires_in", 3600)) - 60,
                             refresh_token=tokens.get("refresh_token", self.data.get("refresh_token", "")))
            self.data["email"] = self._email_from_id_token(tokens.get("id_token", ""))
            self._save()

    @staticmethod
    def _email_from_id_token(id_token):
        try:
            payload = id_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            return json.loads(base64.urlsafe_b64decode(payload)).get("email", "")
        except Exception:
            return ""

    def access_token(self):
        with self.lock:
            if not self.signed_in:
                raise GoogleDocsError("Not signed in to Google.")
            if self.data.get("access_token") and time.time() < self.data.get("expires_at", 0):
                return self.data["access_token"]
            status, _headers, body = _request(TOKEN_URL, {
                "client_id": self.data["client_id"], "client_secret": self.data["client_secret"],
                "refresh_token": self.data["refresh_token"], "grant_type": "refresh_token"})
            tokens = json.loads(body or b"{}")
            if status != 200 or "access_token" not in tokens:
                self.data.pop("refresh_token", None)
                self._save()
                raise GoogleDocsError("Your Google sign-in has expired. Sign in again.")
            self.data.update(access_token=tokens["access_token"],
                             expires_at=time.time() + int(tokens.get("expires_in", 3600)) - 60)
            self._save()
            return self.data["access_token"]

    def sign_out(self):
        token = self.data.get("refresh_token") or self.data.get("access_token")
        if token:
            try:
                _request(REVOKE_URL, {"token": token}, timeout=15)
            except GoogleDocsError:
                pass  # offline: forget the tokens anyway
        for key in ("access_token", "refresh_token", "expires_at", "email"):
            self.data.pop(key, None)
        self._save()

    # --- Drive ---

    def _drive(self, url):
        status, _headers, body = _request(url, headers={"Authorization": f"Bearer {self.access_token()}"})
        if status == 200:
            return body
        try:
            message = json.loads(body).get("error", {}).get("message", "")
        except ValueError:
            message = ""
        if status == 403 and "has not been used" in message:
            raise GoogleDocsError("The Google Drive API isn't turned on for your Google Cloud project. "
                                  "Enable it in Cloud Console (APIs & Services → Library → Google Drive API).")
        if status == 404:
            raise GoogleDocsError("That doc wasn't found, or your Google account can't open it.")
        raise GoogleDocsError(f"Google Drive returned an error ({status}): {message}")

    def list_documents(self, search="", limit=50):
        """[(id, name, modified)] of your Google Docs, newest first."""
        query = f"mimeType='{GOOGLE_DOC_MIME}' and trashed=false"
        if search.strip():
            query += " and name contains '" + search.strip().replace("\\", "\\\\").replace("'", "\\'") + "'"
        params = urllib.parse.urlencode({"q": query, "orderBy": "modifiedTime desc", "pageSize": limit,
                                         "fields": "files(id,name,modifiedTime)"})
        files = json.loads(self._drive(f"{DRIVE_URL}?{params}")).get("files", [])
        return [(item["id"], item["name"], item.get("modifiedTime", "")) for item in files]

    def export_docx(self, doc_id):
        """(.docx bytes, title) of a doc your account can open."""
        meta = json.loads(self._drive(f"{DRIVE_URL}/{doc_id}?fields=name,mimeType"))
        if meta.get("mimeType") != GOOGLE_DOC_MIME:
            raise GoogleDocsError("That file isn't a Google Doc.")
        body = self._drive(f"{DRIVE_URL}/{doc_id}/export?" + urllib.parse.urlencode({"mimeType": DOCX_MIME}))
        return body, meta.get("name", "Google Doc")

"""Import a Google Doc by link or from the signed-in account."""

import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from this_voice_thing.integrations import google_docs
from this_voice_thing.ui.threads import TaskThread


class GoogleDocsDialog(QDialog):
    """Open a Google Doc: a shared link, or your own docs after signing in."""

    def __init__(self, account, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open from Google Docs")
        self.setMinimumSize(620, 480)
        self.account = account
        self.result_docx = None  # (bytes, title) once a doc is chosen
        self.threads = []
        self.cancel_sign_in = False
        layout = QVBoxLayout(self)

        link_title = QLabel("Shared link")
        link_title.setObjectName("CardTitle")
        layout.addWidget(link_title)
        link_hint = QLabel("For docs shared as \u201cAnyone with the link can view\u201d. No sign-in needed.")
        link_hint.setObjectName("Muted")
        layout.addWidget(link_hint)
        link_row = QHBoxLayout()
        self.link_input = QLineEdit()
        self.link_input.setPlaceholderText("https://docs.google.com/document/d/...")
        self.link_input.returnPressed.connect(self.open_link)
        link_row.addWidget(self.link_input, 1)
        self.link_button = QPushButton("Open link")
        self.link_button.clicked.connect(self.open_link)
        link_row.addWidget(self.link_button)
        layout.addLayout(link_row)

        layout.addSpacing(8)
        account_row = QHBoxLayout()
        mine_title = QLabel("Your Google Docs")
        mine_title.setObjectName("CardTitle")
        account_row.addWidget(mine_title)
        account_row.addStretch(1)
        self.account_label = QLabel()
        self.account_label.setObjectName("Muted")
        account_row.addWidget(self.account_label)
        self.setup_button = self._link("Set up...", self.set_up_client)
        self.setup_button.setToolTip("Load the OAuth client (client_secret_....json) from Google Cloud Console.")
        account_row.addWidget(self.setup_button)
        self.sign_in_button = QPushButton("Sign in with Google")
        self.sign_in_button.clicked.connect(self.sign_in)
        account_row.addWidget(self.sign_in_button)
        self.sign_out_button = self._link("Sign out", self.sign_out)
        account_row.addWidget(self.sign_out_button)
        layout.addLayout(account_row)
        self.setup_hint = QLabel(
            "Signing in needs a free Google Cloud OAuth client (Desktop app) with the Google Drive API "
            "enabled; the README walks through it in about 5 minutes. Access is read-only, and your "
            "sign-in stays on this PC.")
        self.setup_hint.setObjectName("Note")
        self.setup_hint.setWordWrap(True)
        layout.addWidget(self.setup_hint)
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search your docs by name")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.refresh_docs)
        search_row.addWidget(self.search_input, 1)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.refresh_docs)
        search_row.addWidget(self.search_button)
        layout.addLayout(search_row)
        self.docs_list = QListWidget()
        self.docs_list.itemDoubleClicked.connect(lambda _item: self.open_selected())
        layout.addWidget(self.docs_list, 1)
        self.status_label = QLabel()
        self.status_label.setObjectName("Muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.open_button = QPushButton("Open selected")
        self.open_button.setProperty("accent", True)
        self.open_button.clicked.connect(self.open_selected)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.update_account()
        if self.account.signed_in:
            QTimer.singleShot(0, self.refresh_docs)

    def _link(self, text, slot):
        button = QPushButton(text)
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        return button

    def run(self, function, on_done, status):
        self.status_label.setText(status)
        self.set_busy(True)
        thread = TaskThread(function, self)

        def finished(result, error):
            self.set_busy(False)
            on_done(result, error)

        thread.done.connect(finished)
        self.threads.append(thread)
        thread.start()

    def set_busy(self, busy):
        for widget in (self.link_button, self.search_button, self.open_button, self.setup_button):
            widget.setEnabled(not busy)
        self.sign_in_button.setEnabled(not busy and self.account.configured)

    def update_account(self):
        signed_in = self.account.signed_in
        self.account_label.setText(f"Signed in as {self.account.email}" if signed_in and self.account.email
                                   else "Signed in" if signed_in else "")
        self.sign_in_button.setVisible(not signed_in)
        self.sign_in_button.setEnabled(self.account.configured)
        self.sign_in_button.setToolTip("" if self.account.configured else "Set up a Google client first.")
        self.sign_out_button.setVisible(signed_in)
        self.setup_button.setVisible(not signed_in)
        self.setup_hint.setVisible(not self.account.configured)
        for widget in (self.search_input, self.search_button, self.docs_list):
            widget.setEnabled(signed_in)
        if not signed_in:
            self.docs_list.clear()

    def set_up_client(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Google OAuth client", os.path.expanduser("~/Downloads"), "Client file (*.json)")
        if not path:
            return
        try:
            self.account.load_client_file(path)
        except (google_docs.GoogleDocsError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Google", f"Couldn't use that file:\n{exc}")
            return
        self.status_label.setText("Client loaded. Click Sign in with Google.")
        self.update_account()

    def sign_in(self):
        self.cancel_sign_in = False

        def done(_result, error):
            self.update_account()
            if error:
                self.status_label.setText(error)
            else:
                self.refresh_docs()

        self.run(lambda: self.account.sign_in(lambda: self.cancel_sign_in), done,
                 "Finish signing in in your browser\u2026 (this window waits up to 5 minutes)")

    def sign_out(self):
        self.account.sign_out()
        self.update_account()
        self.status_label.setText("Signed out. The app's access to your Google account was revoked.")

    def refresh_docs(self):
        search = self.search_input.text()

        def done(docs, error):
            self.docs_list.clear()
            if error:
                self.status_label.setText(error)
                self.update_account()
                return
            for doc_id, name, modified in docs:
                item = QListWidgetItem(f"{name}    {modified[:10]}")
                item.setData(Qt.ItemDataRole.UserRole, doc_id)
                self.docs_list.addItem(item)
            self.status_label.setText(f"{len(docs)} doc{'s' if len(docs) != 1 else ''}, newest first."
                                      if docs else "No docs found.")

        self.run(lambda: self.account.list_documents(search), done, "Loading your docs\u2026")

    def finish(self, result, error):
        if error:
            self.status_label.setText(error)
            return
        self.result_docx = result
        self.accept()

    def open_link(self):
        doc_id = google_docs.doc_id_from_url(self.link_input.text())
        if not doc_id:
            self.status_label.setText("Paste a Google Docs link (docs.google.com/document/d/...).")
            return

        def fetch():
            try:
                return google_docs.download_shared(doc_id)
            except google_docs.GoogleDocsError:
                if self.account.signed_in:  # not public, but maybe your account can open it
                    return self.account.export_docx(doc_id)
                raise

        self.run(fetch, self.finish, "Downloading\u2026")

    def open_selected(self):
        item = self.docs_list.currentItem()
        if item is None:
            if self.link_input.text().strip():
                self.open_link()
            return
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        self.run(lambda: self.account.export_docx(doc_id), self.finish, "Downloading\u2026")

    def reject(self):
        self.cancel_sign_in = True
        super().reject()

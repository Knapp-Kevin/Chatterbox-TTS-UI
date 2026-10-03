"""Opening documents and Google Docs into the Generate page."""

import os
import tempfile

from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QMessageBox

from this_voice_thing.core import documents
from this_voice_thing.ui.dialogs.google_docs import GoogleDocsDialog
from this_voice_thing.ui.widgets import dialog_accepted


class Documents:
    """Opening documents into the text box. Mixed into ChatterboxApp."""

    # --- Documents ---

    def open_document(self):
        start_dir = self.app_settings.get("last_document_dir") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", start_dir, documents.DOCUMENT_FILTER)
        if not path:
            return
        try:
            text = documents.load_document(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could Not Open Document",
                                f"{os.path.basename(path)} could not be read:\n{exc}")
            return
        if not text:
            QMessageBox.warning(self, "Empty Document",
                                f"No readable text was found in {os.path.basename(path)}.")
            return
        self.app_settings["last_document_dir"] = os.path.dirname(path)
        self.show_document(text, os.path.basename(path), documents.safe_file_stem(path))

    def show_document(self, text, label, file_stem):
        self.text_input.setPlainText(text)
        self.current_document_name = file_stem
        self.document_label.setText(label)
        self.update_text_stats()
        self.set_status_message(f"Status: Loaded {label}. Try Preview before generating.")

    def open_google_doc(self):
        dialog = GoogleDocsDialog(self.google_account, self)
        if not dialog_accepted(dialog.exec()) or not dialog.result_docx:
            return
        data, title = dialog.result_docx
        path = os.path.join(tempfile.gettempdir(), "google_doc_import.docx")
        try:
            with open(path, "wb") as handle:
                handle.write(data)
            text = documents.load_document(path)
        except Exception as exc:
            QMessageBox.warning(self, "Google Docs", f"\u201c{title}\u201d couldn't be read:\n{exc}")
            return
        finally:
            if os.path.exists(path):
                os.remove(path)
        if not text:
            QMessageBox.warning(self, "Google Docs", f"No readable text was found in \u201c{title}\u201d.")
            return
        self.show_document(text, f"{title} (Google Docs)", documents.safe_file_stem(title + ".docx"))

    def on_text_changed(self):
        if not self.text_input.toPlainText().strip():
            self.current_document_name = None
            self.document_label.clear()
        self.text_stats_timer.start()

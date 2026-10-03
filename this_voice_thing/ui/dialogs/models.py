"""Find models on Hugging Face, and add or edit a model entry."""

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from this_voice_thing.core import model_registry
from this_voice_thing.engines import qwen as qwen_engine
from this_voice_thing.ui import theme as ui_theme
from this_voice_thing.ui.common import DUAL_MODE_BACKENDS, languages_for_backend, QWEN_BACKEND
from this_voice_thing.ui.widgets import dialog_accepted


# --- Hugging Face model search ---


class FindModelsDialog(QDialog):
    """Search Hugging Face for repos this app can load and pick one to add."""

    ENGINE_FILTERS = (("All engines", "all"), ("Chatterbox", "chatterbox"), ("Qwen3-TTS", "qwen3"),
                      ("Kokoro", "kokoro"), ("VoxCPM", "voxcpm"), ("OmniVoice", "omnivoice"),
                      ("VibeVoice", "vibevoice"))

    def __init__(self, token, existing_repos, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find models on Hugging Face")
        self.setMinimumSize(640, 460)
        self.token = token
        self.existing_repos = existing_repos
        self.selected_repo = None

        layout = QVBoxLayout(self)
        intro = QLabel("Only models this app can load are listed. Pick one to add it; nothing "
                       "downloads until you load it.")
        intro.setObjectName("Muted")
        layout.addWidget(intro)
        search_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Search by name or language, e.g. norwegian, arabic, 0.6B")
        self.query_input.returnPressed.connect(self.run_search)
        search_row.addWidget(self.query_input, 1)
        self.engine_combo = QComboBox()
        for label, key in self.ENGINE_FILTERS:
            self.engine_combo.addItem(label, key)
        self.engine_combo.currentIndexChanged.connect(lambda _i: self.run_search())
        search_row.addWidget(self.engine_combo)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.run_search)
        search_row.addWidget(search_button)
        layout.addLayout(search_row)

        self.results_list = QListWidget()
        self.results_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.results_list.itemDoubleClicked.connect(lambda _item: self.use_selected())
        self.results_list.currentRowChanged.connect(lambda _row: self._update_buttons())
        layout.addWidget(self.results_list, 1)
        self.status_label = QLabel()
        self.status_label.setObjectName("Muted")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.open_page_button = QPushButton("Open on Hugging Face")
        self.open_page_button.clicked.connect(self.open_selected_page)
        buttons.addWidget(self.open_page_button)
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.use_button = QPushButton("Add selected...")
        self.use_button.setProperty("accent", True)
        self.use_button.clicked.connect(self.use_selected)
        buttons.addWidget(self.use_button)
        layout.addLayout(buttons)
        self._update_buttons()
        QTimer.singleShot(0, self.run_search)

    def run_search(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = model_registry.search_models(
                self.query_input.text(), self.engine_combo.currentData(), self.token)
            error = None
        except Exception as exc:
            results, error = [], str(exc)
        finally:
            QApplication.restoreOverrideCursor()
        self.results_list.clear()
        for result in results:
            details = [result.summary]
            if result.languages:
                shown = ", ".join(result.languages[:6]) + (" ..." if len(result.languages) > 6 else "")
                details.append(shown)
            details.append(f"{result.downloads:,} downloads")
            if result.updated:
                details.append(f"updated {result.updated}")
            if result.gated:
                details.append("gated: token and accepted terms needed")
            added = "   (already in your list)" if result.repo_id in self.existing_repos else ""
            item = QListWidgetItem(f"{result.repo_id}{added}\n      " + " · ".join(details))
            item.setData(Qt.ItemDataRole.UserRole, result.repo_id)
            item.setToolTip(f"{result.repo_id}\n{' · '.join(details)}")
            self.results_list.addItem(item)
        if error:
            self.status_label.setText(error)
        elif not results:
            self.status_label.setText("No loadable models found. Try another word or engine.")
        else:
            noun = "model" if len(results) == 1 else "models"
            self.status_label.setText(f"{len(results)} loadable {noun}, most downloaded first.")
        self._update_buttons()

    def _selected(self):
        item = self.results_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _update_buttons(self):
        has = bool(self._selected())
        self.use_button.setEnabled(has)
        self.open_page_button.setEnabled(has)

    def open_selected_page(self):
        repo = self._selected()
        if repo:
            QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{repo}"))

    def use_selected(self):
        self.selected_repo = self._selected()
        if self.selected_repo:
            self.accept()


# --- Model entry editor ---


class ModelEntryDialog(QDialog):
    """Add or edit one models.json entry, with a live Hugging Face check."""

    def __init__(self, entry, other_entries, token, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit model" if entry else "Add model")
        self.setMinimumWidth(560)
        self.other_entries = other_entries
        self.token = token
        self.original = dict(entry or {})
        self.checked = {}
        entry = dict(entry or {"backend": "multilingual", "multilingual_t3_model": "v3",
                               "enabled": True, "language_id": "en"})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        repo_row = QHBoxLayout()
        self.repo_input = QLineEdit(entry.get("repo_id", ""))
        self.repo_input.setPlaceholderText("owner/model-name")
        self.repo_input.textEdited.connect(self._repo_edited)
        repo_row.addWidget(self.repo_input, 1)
        self.check_button = QPushButton("Check")
        self.check_button.setToolTip("Look the repo up on Hugging Face and detect its engine.")
        self.check_button.clicked.connect(self.check_repo)
        repo_row.addWidget(self.check_button)
        find_button = QPushButton("Find...")
        find_button.setToolTip("Search Hugging Face for models this app can load.")
        find_button.clicked.connect(self.find_repo)
        repo_row.addWidget(find_button)
        form.addRow("Hugging Face repo", repo_row)
        self.check_label = QLabel("Enter a repo and click Check to confirm it can be loaded.")
        self.check_label.setObjectName("Muted")
        self.check_label.setWordWrap(True)
        form.addRow("", self.check_label)

        self.name_input = QLineEdit(entry.get("label", ""))
        self.name_input.setPlaceholderText("Shown in the model switcher")
        form.addRow("Name", self.name_input)

        self.engine_combo = QComboBox()
        for engine in model_registry.ENGINES.values():
            self.engine_combo.addItem(engine.label, engine.key)
            self.engine_combo.setItemData(
                self.engine_combo.count() - 1, engine.description, Qt.ItemDataRole.ToolTipRole)
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(entry.get("backend"))))
        self.engine_combo.currentIndexChanged.connect(self._engine_changed)
        form.addRow("Engine", self.engine_combo)

        self.weights_combo = QComboBox()
        for short in model_registry.WEIGHT_VERSIONS:
            self.weights_combo.addItem(short.upper(), short)
        current_weights = str(entry.get("multilingual_t3_model") or "v3")
        for short, filename in model_registry.WEIGHT_VERSIONS.items():
            if current_weights in (short, filename):
                self.weights_combo.setCurrentIndex(self.weights_combo.findData(short))
        self.weights_combo.setToolTip("Which multilingual weights to load. V3 is the newest.")
        self.weights_label = QLabel("Weights")
        form.addRow(self.weights_label, self.weights_combo)

        self.variant_combo = QComboBox()
        for key, label in model_registry.QWEN_VARIANTS.items():
            self.variant_combo.addItem(label, key)
            self.variant_combo.setItemData(self.variant_combo.count() - 1,
                                           qwen_engine.VARIANTS[key], Qt.ItemDataRole.ToolTipRole)
        variant_index = self.variant_combo.findData(entry.get("qwen_variant"))
        self.variant_combo.setCurrentIndex(max(0, variant_index))
        self.variant_label = QLabel("Qwen variant")
        form.addRow(self.variant_label, self.variant_combo)

        self.mode_combo = QComboBox()
        for key, label in model_registry.VOICE_MODES.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(model_registry.entry_mode(entry))))
        self.mode_combo.setToolTip("This model does both; add one entry for each to switch between them.")
        self.mode_label = QLabel("Use for")
        form.addRow(self.mode_label, self.mode_combo)

        self.language_combo = QComboBox()
        self.language_combo.setToolTip("Language selected by default when this model is loaded.")
        form.addRow("Default language", self.language_combo)
        self._preferred_language = entry.get("language_id", "en")

        self.test_text_input = QLineEdit(entry.get("test_text", ""))
        self.test_text_input.setPlaceholderText("Optional sentence used by 'Sample text'")
        form.addRow("Sample sentence", self.test_text_input)
        self.notes_input = QLineEdit(entry.get("notes", ""))
        self.notes_input.setPlaceholderText("Optional")
        form.addRow("Notes", self.notes_input)
        self.enabled_checkbox = QCheckBox("Show in the model switcher")
        self.enabled_checkbox.setChecked(bool(entry.get("enabled", True)))
        form.addRow("", self.enabled_checkbox)
        layout.addLayout(form)

        self.error_label = QLabel()
        ui_theme.set_tone(self.error_label, "error")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._engine_changed()
        if entry.get("repo_id") and not entry.get("label"):
            QTimer.singleShot(0, self.check_repo)

    def find_repo(self):
        existing = {e.get("repo_id") for e in self.other_entries}
        finder = FindModelsDialog(self.token, existing, self)
        if dialog_accepted(finder.exec()) and finder.selected_repo:
            self.repo_input.setText(finder.selected_repo)
            self.name_input.clear()
            self.check_repo()

    def _repo_edited(self, text):
        self.check_label.setText("Click Check to confirm this repo can be loaded.")
        ui_theme.set_tone(self.check_label, "")

    def _engine_changed(self, *_args):
        engine = model_registry.ENGINES[self.engine_combo.currentData()]
        self.weights_combo.setVisible(engine.uses_weights_version)
        self.weights_label.setVisible(engine.uses_weights_version)
        self.variant_combo.setVisible(engine.key == QWEN_BACKEND)
        self.variant_label.setVisible(engine.key == QWEN_BACKEND)
        self.mode_combo.setVisible(engine.key in DUAL_MODE_BACKENDS)
        self.mode_label.setVisible(engine.key in DUAL_MODE_BACKENDS)
        current = self.language_combo.currentData() or self._preferred_language
        self.language_combo.clear()
        for language_id, name in languages_for_backend(engine.key).items():
            self.language_combo.addItem(f"{name} [{language_id}]", language_id)
        index = self.language_combo.findData(current)
        self.language_combo.setCurrentIndex(index if index >= 0 else max(0, self.language_combo.findData("en")))

    def check_repo(self):
        repo_id = self.repo_input.text().strip()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = model_registry.check_repo(repo_id, self.token)
        finally:
            QApplication.restoreOverrideCursor()
        self.check_label.setText(result.message)
        ui_theme.set_tone(self.check_label, "success" if result.ok else "error")
        self.adjustSize()
        if result.ok:
            self.checked = {"download_bytes": result.download_bytes}
            if result.license:
                self.checked["license"] = result.license
            self.engine_combo.setCurrentIndex(self.engine_combo.findData(result.detected_backend))
            if result.qwen_variant:
                self.variant_combo.setCurrentIndex(self.variant_combo.findData(result.qwen_variant))
            if result.weight_versions and self.weights_combo.currentData() not in result.weight_versions:
                self.weights_combo.setCurrentIndex(self.weights_combo.findData(result.weight_versions[0]))
            if not self.name_input.text().strip():
                self.name_input.setText(repo_id.split("/")[-1].replace("-", " ").replace("_", " "))
        return result

    def result_entry(self):
        engine = self.engine_combo.currentData()
        entry = dict(self.original)
        entry.update({
            "repo_id": self.repo_input.text().strip(),
            "label": self.name_input.text().strip(),
            "backend": engine,
            "language_id": self.language_combo.currentData() or "en",
            "test_text": self.test_text_input.text().strip(),
            "notes": self.notes_input.text().strip(),
            "enabled": self.enabled_checkbox.isChecked(),
            "experimental": entry.get("experimental", False),
            "multilingual_t3_model": self.weights_combo.currentData() if engine == "multilingual" else "",
            "qwen_variant": self.variant_combo.currentData() if engine == QWEN_BACKEND else "",
            "mode": self.mode_combo.currentData() if engine in DUAL_MODE_BACKENDS else "",
            "test_texts": entry.get("test_texts", {}),
        })
        if self.repo_input.text().strip() == self.original.get("repo_id") or self.checked:
            entry.update(self.checked)
        else:
            entry.pop("license", None)  # a different repo: don't keep the old one's license
            entry.pop("download_bytes", None)
        return entry

    def _save(self):
        entry = self.result_entry()
        problem = None
        if not model_registry.is_valid_repo_id(entry["repo_id"]):
            problem = "Enter the Hugging Face repo as owner/name."
        elif not entry["label"]:
            problem = "Give the model a name."
        elif any(other.get("label") == entry["label"] for other in self.other_entries):
            problem = "Another model already uses this name."
        if problem:
            self.error_label.setText(problem)
            self.error_label.setVisible(True)
            self.adjustSize()
            return
        self.accept()

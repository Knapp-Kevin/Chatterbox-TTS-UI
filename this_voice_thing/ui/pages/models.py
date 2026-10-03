"""The Model page: model tiles, discovery, engine installs and model loading."""

import gc
import math
import os

import torch
from PySide6.QtCore import QUrl
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtGui import QFont
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QMenu
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QTabBar
from PySide6.QtWidgets import QWidget

from this_voice_thing.core import model_registry
from this_voice_thing.ui import tiles as model_tiles
from this_voice_thing.ui import theme as ui_theme
from this_voice_thing.ui.common import BACKEND_MULTILINGUAL
from this_voice_thing.ui.common import CHATTERBOX_AVAILABLE
from this_voice_thing.ui.common import DEFAULT_MODEL_REPO
from this_voice_thing.ui.common import DEFAULT_MULTILINGUAL_T3_MODEL
from this_voice_thing.ui.common import DUAL_MODE_BACKENDS
from this_voice_thing.ui.common import DUAL_MODE_TYPES
from this_voice_thing.ui.common import ENGINE_INSTALL_NOTES
from this_voice_thing.ui.common import ENGINE_MODULES
from this_voice_thing.ui.common import MODEL_CONFIG_FILENAME
from this_voice_thing.ui.common import WORKER_MODEL_TYPES
from this_voice_thing.ui.common import load_models_config
from this_voice_thing.ui.dialogs.models import ModelEntryDialog
from this_voice_thing.ui.threads import EngineInstallThread
from this_voice_thing.ui.threads import ModelLoaderThread
from this_voice_thing.ui.widgets import dialog_accepted


class ModelPage:
    """The Model page: model tiles, discovery, engine installs and model loading. Mixed into ChatterboxApp."""

    def _build_model_page(self):
        model_page, model_layout = self._make_page(
            "Model", "Models download once, then load from the local cache.")
        models_card, models_layout = self._make_card()
        tabs_row = QHBoxLayout()
        tabs_row.setSpacing(6)
        self.capability_tabs = QTabBar()
        self.capability_tabs.setObjectName("CapabilityTabs")
        self.capability_tabs.setDrawBase(False)
        self.capability_tabs.setExpanding(False)
        self.capability_tabs.setUsesScrollButtons(False)
        self.capability_tabs.setCursor(Qt.CursorShape.PointingHandCursor)
        for key, (title, description) in model_registry.CAPABILITIES.items():
            index = self.capability_tabs.addTab(model_registry.CAPABILITY_TABS[key])
            self.capability_tabs.setTabData(index, key)
            self.capability_tabs.setTabToolTip(index, f"{title}: {description}")
        self.capability_tabs.currentChanged.connect(lambda _index: self.render_model_tiles())
        tabs_row.addWidget(self.capability_tabs)
        tabs_row.addStretch(1)
        add_model_button = self._link(QPushButton("+ Add repo..."))
        add_model_button.setToolTip("Add a Hugging Face repo you already know, e.g. owner/model-name.")
        add_model_button.clicked.connect(self.add_model)
        tabs_row.addWidget(add_model_button)
        models_layout.addLayout(tabs_row)
        self.capability_note = QLabel()
        self.capability_note.setObjectName("Muted")
        self.capability_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        models_layout.addWidget(self.capability_note)

        your_label = QLabel("YOUR MODELS")
        your_label.setObjectName("SectionLabel")
        your_label.setToolTip("Click a model to load it. Right-click or \u22ef for edit, hide and remove.")
        models_layout.addWidget(your_label)
        self.your_tiles = model_tiles.TileArea(min_rows=1)
        models_layout.addWidget(self.your_tiles, 3)

        discover_row = QHBoxLayout()
        discover_label = QLabel("DISCOVER ON HUGGING FACE")
        discover_label.setObjectName("SectionLabel")
        discover_row.addWidget(discover_label)
        discover_row.addStretch(1)
        self.discover_input = QLineEdit()
        self.discover_input.setPlaceholderText("Search, e.g. norwegian, arabic, 0.6B")
        self.discover_input.setClearButtonEnabled(True)
        self.discover_input.setFixedWidth(210)
        self.discover_input.returnPressed.connect(self.start_discover)
        discover_row.addWidget(self.discover_input)
        discover_button = QPushButton("Search")
        discover_button.clicked.connect(self.start_discover)
        discover_row.addWidget(discover_button)
        models_layout.addLayout(discover_row)
        self.discover_tiles = model_tiles.TileArea(min_rows=1)
        models_layout.addWidget(self.discover_tiles, 2)
        self.discover_status = QLabel()
        self.discover_status.setObjectName("Muted")
        self.discover_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        models_layout.addWidget(self.discover_status)
        model_layout.addWidget(models_card, 1)

        hf_card, hf_layout = self._make_card("Hugging Face access")
        hf_row = QHBoxLayout()
        self.hf_token_input = QLineEdit(str(self.app_settings.get("hf_token", "")))
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setPlaceholderText("hf_... (optional)")
        self.hf_token_input.returnPressed.connect(self.save_hf_token)
        hf_row.addWidget(self.hf_token_input, 1)
        save_token_button = QPushButton("Save")
        save_token_button.clicked.connect(self.save_hf_token)
        hf_row.addWidget(save_token_button)
        test_token_button = QPushButton("Test")
        test_token_button.clicked.connect(self.test_hf_token)
        hf_row.addWidget(test_token_button)
        hf_layout.addLayout(hf_row)
        self.hf_token_status = QLabel()
        self.hf_token_status.setObjectName("Muted")
        self.hf_token_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hf_layout.addWidget(self.hf_token_status)
        model_layout.addWidget(hf_card)

        tuning_card, tuning_layout = self._make_card()
        tuning_header = QHBoxLayout()
        self.tuning_toggle = self._link(QPushButton())
        self.tuning_toggle.clicked.connect(lambda: self.set_tuning_expanded(self.tuning_panel.isHidden()))
        tuning_header.addWidget(self.tuning_toggle)
        self.tuning_summary_label = QLabel()
        self.tuning_summary_label.setObjectName("Muted")
        tuning_header.addWidget(self.tuning_summary_label)
        tuning_header.addStretch(1)
        tuning_layout.addLayout(tuning_header)
        self.tuning_panel = QWidget()
        tuning_grid = QGridLayout(self.tuning_panel)
        tuning_grid.setContentsMargins(0, 0, 0, 0)
        tuning_grid.setHorizontalSpacing(14)

        def tuning_spin(minimum, maximum, step, value):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.valueChanged.connect(self.on_tuning_changed)
            return spin

        self.repetition_spin = tuning_spin(0.5, 3.0, 0.05, self.repetition_penalty)
        self.min_p_spin = tuning_spin(0.0, 1.0, 0.01, self.min_p)
        self.top_p_spin = tuning_spin(0.0, 1.0, 0.01, self.top_p)
        for column, (title, spin, tip) in enumerate((
                ("Repetition control", self.repetition_spin,
                 "Discourages repeated words and stutters. Default 1.20; raise slightly if "
                 "phrases repeat. [repetition_penalty]"),
                ("Unlikely-sound filter", self.min_p_spin,
                 "Skips very unlikely sounds. Default 0.05; higher is steadier but flatter. [min_p]"),
                ("Top-p", self.top_p_spin,
                 "Limits choices to the most likely sounds. 1.00 means off. [top_p]"))):
            caption = QLabel(title)
            caption.setToolTip(tip)
            spin.setToolTip(tip)
            row, col = divmod(column, 2)
            tuning_grid.addWidget(caption, row, col * 2)
            tuning_grid.addWidget(spin, row, col * 2 + 1)
        tuning_grid.setColumnStretch(1, 1)
        tuning_grid.setColumnStretch(3, 1)
        tuning_reset = QPushButton("Reset")
        tuning_reset.clicked.connect(self.reset_tuning)
        tuning_grid.addWidget(tuning_reset, 1, 3, Qt.AlignmentFlag.AlignRight)
        tuning_layout.addWidget(self.tuning_panel)
        model_layout.addWidget(tuning_card)
        self.pages.addWidget(model_page)

    def apply_hf_token_setting(self):
        hf_token = str(self.app_settings.get("hf_token", "")).strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        else:
            os.environ.pop("HF_TOKEN", None)

    def start_default_model_load(self):
        if not CHATTERBOX_AVAILABLE:
            return
        self.set_status_message(
            "Status: Starting default model load. First run may download model files and can take several minutes. Watch Activity Log for progress."
        )
        self.load_model()

    def on_model_repo_changed(self, _index):
        entry = self.get_selected_model_entry()
        self.selected_model_repo = entry["repo_id"]
        self.refresh_model_repo_tooltip()
        self.refresh_language_options()
        if self.entry_key(entry) != self.loaded_entry_key():
            self.load_model(entry)

    # --- Model page ---

    @staticmethod
    def entry_key(entry):
        if entry.get("backend") in DUAL_MODE_BACKENDS:
            return (entry.get("repo_id"), entry.get("backend"), model_registry.entry_mode(entry))
        return (entry.get("repo_id"), entry.get("backend"), entry.get("multilingual_t3_model") or "")

    def loaded_entry_key(self):
        backend = self.current_model_backend
        weights = self.current_multilingual_t3_model if backend == BACKEND_MULTILINGUAL else ""
        if backend in DUAL_MODE_BACKENDS:
            weights = getattr(self, "current_mode", "clone")
        return (self.current_model_repo, backend, weights or "")

    def refresh_models_page(self, select_entry=None):
        if not hasattr(self, "capability_tabs"):
            return
        self.model_cache_sizes = model_registry.cached_repo_sizes()
        if select_entry is not None:
            self.show_capability(model_registry.capability_for(select_entry))
        self.render_model_tiles()

    def show_capability(self, capability):
        for index in range(self.capability_tabs.count()):
            if self.capability_tabs.tabData(index) == capability:
                self.capability_tabs.setCurrentIndex(index)  # renders via currentChanged

    def current_capability(self):
        return self.capability_tabs.tabData(self.capability_tabs.currentIndex())

    def is_active_entry(self, entry):
        return self.model is not None and self.entry_key(entry) == self.loaded_entry_key()

    def model_busy(self):
        return getattr(self, "model_is_loading", False) or self.is_generating or getattr(self, "api_busy", False)

    def render_model_tiles(self):
        if not hasattr(self, "capability_tabs"):
            return
        if getattr(self, "model_cache_sizes", None) is None:
            self.model_cache_sizes = model_registry.cached_repo_sizes()
        groups = {capability: members for capability, _title, members
                  in model_registry.group_by_capability(self.model_entries)}
        for index in range(self.capability_tabs.count()):
            capability = self.capability_tabs.tabData(index)
            title = model_registry.CAPABILITY_TABS[capability]
            count = len(groups.get(capability, []))
            self.capability_tabs.setTabText(index, f"{title}  {count}" if count else title)
        capability = self.current_capability()
        self.capability_note.setText(model_registry.CAPABILITIES[capability][1])
        self.your_tiles.set_tiles(
            [self.model_tile(entry) for entry in groups.get(capability, [])],
            "None yet. Add one from Discover below.")
        self.render_discover_tiles()

    def engine_installed(self, entry):
        module = ENGINE_MODULES.get(entry.get("backend"))
        return module is None or module.is_installed()

    def gpu_memory(self):
        """(name, GB) of the first CUDA GPU, or None."""
        if not hasattr(self, "_gpu_memory"):
            self._gpu_memory = None
            try:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    self._gpu_memory = (props.name, props.total_memory / 1024 ** 3)
            except Exception:
                pass
        return self._gpu_memory

    def hardware_line(self, backend, repo_id):
        """(kind, text, tooltip) comparing a model's GPU memory needs with this PC."""
        needs = model_registry.hardware_needs(backend, repo_id)
        wanted = f"{needs.min_gb:g} GB" if needs.good_gb == needs.min_gb else \
            f"{needs.min_gb:g}\u2013{needs.good_gb:g} GB"
        gpu = self.gpu_memory()
        detail = (f"Needs about {needs.min_gb:g} GB of GPU memory"
                  + ("" if needs.good_gb == needs.min_gb else f", {needs.good_gb:g} GB for full speed")
                  + f". {needs.note}")
        if gpu is None:
            if needs.cpu_ok:
                return "tight", "No GPU found: runs on the CPU", detail + "\nNo NVIDIA GPU was found."
            return "short", f"Needs an NVIDIA GPU ({needs.min_gb:g} GB+)", detail
        name, memory = gpu
        detail += f"\nYour GPU: {name}, {memory:.0f} GB."
        if memory + 0.5 >= needs.good_gb:
            return "good", f"\u2713 GPU {wanted} \u00b7 yours {memory:.0f} GB", detail
        if memory + 0.5 >= needs.min_gb:
            return "tight", f"! GPU {wanted} \u00b7 yours {memory:.0f} GB", \
                detail + "\nIt runs, but slower than on a bigger GPU."
        fallback = " (CPU, slow)" if needs.cpu_ok else ""
        return "short", f"\u2717 Needs {needs.min_gb:g} GB GPU{fallback} \u00b7 yours {memory:.0f} GB", detail

    def typical_speed_text(self, entry):
        """Estimated time for 1,000 characters, split the way generation would split them."""
        count = max(1, math.ceil(1000 / (self.max_section_chars_for(entry) * 0.85)))
        seconds, measured = self.estimate_seconds(entry, [1000 // count] * count)
        amount = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"
        return f"\u2248 {amount} per 1,000 characters", measured

    def model_tile(self, entry):
        engine = model_registry.engine_for(entry)
        active = self.is_active_entry(entry)
        repo = entry["repo_id"]
        subtitle = model_registry.engine_label(entry)
        if engine.uses_weights_version:
            subtitle += f" {model_registry.weights_file(entry).split('_')[-1].split('.')[0].upper()}"
        subtitle += f" \u00b7 {engine.languages_summary}"
        badges = []
        if active:
            badges.append(("active", "Loaded", "This model is loaded and ready to generate."))
        elif not self.engine_installed(entry):
            badges.append(("status", "Needs engine", "Click to install this engine (it runs in its own environment)."))
        elif model_registry.is_downloaded(entry):
            size = model_registry.format_size(self.model_cache_sizes.get(repo, 0))
            badges.append(("status", f"Ready \u00b7 {size}", "Downloaded; loads from the local cache."))
        elif entry.get("download_bytes"):
            size = model_registry.format_size(entry["download_bytes"])
            badges.append(("status", f"Download {size}", "Downloads the first time you load it."))
        else:
            badges.append(("status", "Not downloaded", "Downloads the first time you load it."))
        badges.append(model_tiles.license_badge(model_registry.license_of(entry)))
        speed, measured = self.typical_speed_text(entry)
        if not entry.get("enabled", True):
            speed = f"Hidden · {speed}"
        tooltip = "\n".join(line for line in (
            f"{entry['label']}  ({repo})",
            engine.description,
            entry.get("notes", ""),
            "" if entry.get("enabled", True) else "Hidden from the model switcher on the Generate page.",
            f"Speed {'measured from your runs' if measured else 'estimated until you generate with it'}.",
            "" if active else "Click to load. Right-click for more.") if line)
        needs = self.hardware_line(entry.get("backend"), repo)
        tooltip += "\n" + needs[2]
        tile = model_tiles.ModelTile(entry["label"], subtitle, badges, speed, tooltip,
                                     active=active, with_menu=True, needs=needs)
        tile.clicked.connect(lambda e=entry: self.load_entry(e))
        tile.menu_requested.connect(lambda pos, e=entry: self.show_model_menu(e, pos))
        return tile

    def show_model_menu(self, entry, pos):
        menu = QMenu(self)
        active = self.is_active_entry(entry)
        hidden = not entry.get("enabled", True)
        is_default = entry["repo_id"] == DEFAULT_MODEL_REPO and entry.get("backend") == BACKEND_MULTILINGUAL \
            and sum(1 for e in self.model_entries if self.entry_key(e) == self.entry_key(entry)) == 1
        visible = sum(1 for e in self.model_entries if e.get("enabled", True))
        actions = (
            ("Loaded" if active else "Load", lambda: self.load_entry(entry),
             not active and not self.model_busy()),
            None,
            ("Edit...", lambda: self.edit_model(entry), True),
            ("Duplicate...", lambda: self.duplicate_model(entry), True),
            ("Show in model switcher" if hidden else "Hide from model switcher",
             lambda: self.toggle_model_hidden(entry), hidden or visible > 1),
            ("Open on Hugging Face",
             lambda: QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{entry['repo_id']}")), True),
            None,
            ("Remove (official fallback)" if is_default else "Remove (unload it first)" if active
             else "Remove...", lambda: self.remove_model(entry), not is_default and not active),
        )
        for item in actions:
            if item is None:
                menu.addSeparator()
                continue
            text, callback, enabled = item
            action = menu.addAction(text)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False, callback=callback: callback())
        menu.exec(pos)

    def update_model_details(self):
        self.render_model_tiles()

    # --- Discover ---

    def start_discover(self):
        if getattr(self, "discover_thread", None) is not None:
            self.discover_rerun = True  # search again with the newest text when this one ends
            return
        self.discover_rerun = False
        self.discover_status.setText("Searching Hugging Face...")
        self.discover_thread = model_tiles.DiscoverThread(
            self.discover_input.text(), self.app_settings.get("hf_token"), self)
        self.discover_thread.found.connect(self.on_discover_results)
        self.discover_thread.start()

    def on_discover_results(self, results, error):
        self.discover_thread.wait()
        self.discover_thread = None
        self.discover_results = results
        self.discover_error = error
        if getattr(self, "discover_rerun", False):
            self.start_discover()
            return
        self.render_discover_tiles()

    def render_discover_tiles(self):
        results = getattr(self, "discover_results", None)
        if results is None:
            if getattr(self, "discover_thread", None) is None:
                self.discover_status.setText("Open this page with an internet connection to see more models.")
            return
        existing = {entry.get("repo_id") for entry in self.model_entries}
        capability = self.current_capability()
        title = model_registry.CAPABILITIES[capability][0].lower()
        matching = [r for r in results if capability in r.capabilities and r.repo_id not in existing]
        self.discover_tiles.set_tiles([self.discover_tile(result) for result in matching[:30]])
        query = self.discover_input.text().strip()
        if self.discover_error:
            self.discover_status.setText(self.discover_error)
        elif not matching:
            self.discover_status.setText(
                f"No other {title} models match \u201c{query}\u201d." if query else
                f"No other {title} models found. Try a search, e.g. a language.")
        else:
            noun = "model" if len(matching) == 1 else "models"
            scope = f" matching \u201c{query}\u201d" if query else ""
            self.discover_status.setText(
                f"{len(matching)} {noun}{scope}, most downloaded first. Click one to add it.")
            self.discover_status.setToolTip("Only models this app can load are shown. Nothing "
                                            "downloads until you load a model.")

    def discover_tile(self, result):
        owner, _sep, name = result.repo_id.partition("/")
        badges = [model_tiles.license_badge(result.license)]
        if result.gated:
            badges.append(("status", "Gated", "Accept the terms on huggingface.co and save a token below."))
        if result.languages:
            count = len(result.languages)
            shown = f"{count} languages" if count > 1 else f"Language: {result.languages[0]}"
            badges.append(("status", shown, "Languages: " + ", ".join(result.languages)))
        detail = f"{model_tiles.compact_count(result.downloads)} downloads \u00b7 {result.likes} likes"
        if result.updated:
            detail += f" \u00b7 {result.updated}"
        tooltip = f"{result.repo_id}\n{result.summary}\nClick to add it to your models."
        needs = self.hardware_line(result.backend, result.repo_id)
        tooltip += "\n" + needs[2]
        tile = model_tiles.ModelTile(name, f"{owner} \u00b7 {result.summary}", badges, detail, tooltip,
                                     needs=needs)
        tile.clicked.connect(lambda r=result: self.add_from_discover(r))
        tile.menu_requested.connect(lambda _pos, r=result: QDesktopServices.openUrl(
            QUrl(f"https://huggingface.co/{r.repo_id}")))
        return tile

    def add_from_discover(self, result):
        seed = {"repo_id": result.repo_id}
        if result.backend in DUAL_MODE_BACKENDS:
            seed["mode"] = "design" if self.current_capability() == "design" else "clone"
        if result.license:
            seed["license"] = result.license
        new_entry = self._edit_entry_dialog(seed)
        if new_entry:
            self.persist_model_entries(self.model_entries + [new_entry], new_entry)

    def persist_model_entries(self, entries, select_entry=None):
        try:
            model_registry.save_models_config(self.model_config_path, entries)
        except OSError as exc:
            QMessageBox.warning(self, "Could Not Save", f"{MODEL_CONFIG_FILENAME} could not be written:\n{exc}")
            return False
        self.model_entries = load_models_config(self.model_config_path)
        self.refresh_model_repo_options()
        self.refresh_model_repo_tooltip()
        self.refresh_models_page(select_entry)
        self.set_status_message(f"Status: Saved model list to {MODEL_CONFIG_FILENAME}.")
        return True

    def _edit_entry_dialog(self, entry, replacing=None):
        others = [e for e in self.model_entries if e is not replacing]
        dialog = ModelEntryDialog(entry, others, self.app_settings.get("hf_token"), self)
        if not dialog_accepted(dialog.exec()):
            return None
        return dialog.result_entry()

    def add_model(self):
        new_entry = self._edit_entry_dialog(None)
        if new_entry:
            self.persist_model_entries(self.model_entries + [new_entry], new_entry)

    def edit_model(self, entry):
        updated = self._edit_entry_dialog(entry, replacing=entry)
        if updated:
            entries = [updated if e is entry else e for e in self.model_entries]
            self.persist_model_entries(entries, updated)

    def duplicate_model(self, entry):
        copy = dict(entry)
        base, n = f"{entry['label']} copy", 2
        copy["label"] = base
        while any(e.get("label") == copy["label"] for e in self.model_entries):
            copy["label"] = f"{base} {n}"
            n += 1
        updated = self._edit_entry_dialog(copy)
        if updated:
            self.persist_model_entries(self.model_entries + [updated], updated)

    def toggle_model_hidden(self, entry):
        updated = dict(entry, enabled=not entry.get("enabled", True))
        self.persist_model_entries([updated if e is entry else e for e in self.model_entries], updated)

    def remove_model(self, entry):
        answer = QMessageBox.question(
            self, "Remove Model",
            f"Remove '{entry['label']}' from the model list?\n\nDownloaded files stay in the "
            "Hugging Face cache, so adding it back later won't download again.")
        if answer == QMessageBox.StandardButton.Yes:
            self.persist_model_entries([e for e in self.model_entries if e is not entry])

    def load_entry(self, entry):
        if self.is_active_entry(entry):
            return
        if self.model_busy():
            self.set_status_message("Status: Wait for the current load or generation to finish.")
            return
        if not self.engine_installed(entry):
            self.install_engine(entry)
            return
        index = self.model_repo_combo.findText(entry["label"])
        if index >= 0 and index != self.model_repo_combo.currentIndex():
            self.model_repo_combo.setCurrentIndex(index)  # loads via the switcher
        elif self.entry_key(entry) != self.loaded_entry_key() or self.model is None:
            self.load_model(entry)

    def install_engine(self, entry):
        backend = entry.get("backend")
        name, note = ENGINE_INSTALL_NOTES[backend]
        if getattr(self, "engine_install_thread", None) is not None and self.engine_install_thread.isRunning():
            self.set_status_message("Status: An engine is still installing. Progress is on the Log page.")
            return
        answer = QMessageBox.question(self, f"Install {name} Engine", f"{note}\n\nInstall now?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.set_status_message(f"Status: Installing the {name} engine. Progress is on the Log page.")
        self.pending_install_entry = entry
        self.engine_install_thread = EngineInstallThread(ENGINE_MODULES[backend])
        self.engine_install_thread.finished_with.connect(
            lambda error, name=name: self.on_engine_install_finished(name, error))
        self.engine_install_thread.start()

    def on_engine_install_finished(self, name, error):
        if error:
            self.set_status_message(f"Status: {name} engine install failed. See the Log page.")
            QMessageBox.warning(self, f"{name} Engine", f"The install failed:\n{error}")
        else:
            self.set_status_message(f"Status: {name} engine installed. Loading the model...")
            self.refresh_models_page()
            if self.pending_install_entry is not None:
                self.load_entry(self.pending_install_entry)
        self.render_model_tiles()

    def save_hf_token(self):
        token = self.hf_token_input.text().strip()
        if token:
            self.app_settings["hf_token"] = token
        else:
            self.app_settings.pop("hf_token", None)
        self.apply_hf_token_setting()
        self.save_app_settings()
        self.update_hf_token_status("Token saved." if token else "Token removed.")

    def test_hf_token(self):
        token = self.hf_token_input.text().strip()
        if not token:
            self.update_hf_token_status("Enter a token to test.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, message = model_registry.whoami(token)
        finally:
            QApplication.restoreOverrideCursor()
        self.update_hf_token_status(message, ok)

    def update_hf_token_status(self, message=None, ok=None):
        saved = bool(self.app_settings.get("hf_token"))
        base = ("A token is saved and used for downloads." if saved else
                "Optional: only needed for gated or private repos.")
        self.hf_token_status.setText(f"{message}  {base}" if message else base)
        ui_theme.set_tone(self.hf_token_status, "success" if ok else "error" if ok is False else "")
        self.hf_token_status.setToolTip(
            "Public models need no token. A token unlocks gated or private repos and higher "
            "download limits. It is stored only in app_settings.json on this computer (ignored "
            "by git), never in models.json.")

    def get_selected_model_repo(self):
        return self.get_selected_model_entry()["repo_id"]

    def get_visible_model_entries(self):
        enabled_entries = [
            entry for entry in self.model_entries
            if entry.get("enabled", True)
        ]
        return enabled_entries or self.model_entries[:1]

    def refresh_model_repo_options(self):
        selected_label = None
        if hasattr(self, "model_repo_combo") and self.model_repo_combo.count() > 0:
            selected_label = self.model_repo_combo.currentText()

        visible_entries = self.get_visible_model_entries()
        self.model_repo_combo.blockSignals(True)
        self.model_repo_combo.clear()
        header_font = QFont(self.model_repo_combo.font())
        header_font.setBold(True)
        for _capability, title, members in model_registry.group_by_capability(visible_entries):
            self.model_repo_combo.addItem(title.upper(), None)
            header = self.model_repo_combo.model().item(self.model_repo_combo.count() - 1)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setFont(header_font)
            header.setForeground(QApplication.palette().color(QPalette.ColorRole.Link))
            for entry in members:
                self.model_repo_combo.addItem(entry["label"], visible_entries.index(entry))
                self.model_repo_combo.setItemData(
                    self.model_repo_combo.count() - 1,
                    f"{entry['repo_id']} \u00b7 {model_registry.engine_label(entry)}",
                    Qt.ItemDataRole.ToolTipRole)

        selected_index = -1
        if selected_label:
            selected_index = self.model_repo_combo.findText(selected_label)
            if selected_index >= 0 and self.model_repo_combo.itemData(selected_index) is None:
                selected_index = -1
        if selected_index < 0:
            for index, entry in enumerate(visible_entries):
                if self.entry_key(entry) == self.loaded_entry_key():
                    selected_index = self.model_repo_combo.findData(index)
                    break

        if selected_index < 0:
            for index, entry in enumerate(visible_entries):
                if entry["repo_id"] == DEFAULT_MODEL_REPO and entry.get("backend") == BACKEND_MULTILINGUAL:
                    selected_index = self.model_repo_combo.findData(index)
                    break

        if selected_index < 0 and visible_entries:
            selected_index = self.model_repo_combo.findData(0)

        if selected_index >= 0:
            self.model_repo_combo.setCurrentIndex(selected_index)
        self.model_repo_combo.blockSignals(False)

    def get_selected_model_entry(self):
        visible_entries = self.get_visible_model_entries()
        selected_index = self.model_repo_combo.currentData()
        if isinstance(selected_index, int) and 0 <= selected_index < len(visible_entries):
            return visible_entries[selected_index]
        return visible_entries[0]

    def refresh_model_repo_tooltip(self):
        selected_entry = self.get_selected_model_entry()
        tooltip = (f"{selected_entry['repo_id']} \u00b7 "
                   f"{model_registry.engine_for(selected_entry).label}\n"
                   "Switching loads the model. Manage models on the Model page.")
        if selected_entry.get("notes"):
            tooltip += f"\n\n{selected_entry['notes']}"
        self.model_repo_combo.setToolTip(tooltip)

    def set_model_loading_state(self, is_loading):
        if is_loading:
            self.model_load_progress.setEnabled(True)
            self.model_load_progress.setRange(0, 0)
        else:
            self.model_load_progress.setRange(0, 1)
            self.model_load_progress.setValue(0)
            self.model_load_progress.setEnabled(False)
        self.model_is_loading = is_loading
        self.model_repo_combo.setEnabled(not is_loading)
        self.use_preset_button.setEnabled(not is_loading)
        self.update_model_details()
        if is_loading:
            self.language_combo.setEnabled(False)
        else:
            self.refresh_language_options()

    def release_model(self):
        """Free the current model (and stop a Qwen worker) before loading another."""
        old_model, self.model = self.model, None
        if isinstance(old_model, WORKER_MODEL_TYPES):
            old_model.close()
        del old_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, "qwen_row"):
            self.update_engine_controls()

    def load_model(self, selected_entry=None):
        if not CHATTERBOX_AVAILABLE:
            QMessageBox.critical(
                self, "Error", "ChatterboxTTS library not installed.")
            return
        if getattr(self, "model_is_loading", False):
            return
        selected_entry = selected_entry or self.get_selected_model_entry()
        selected_repo = selected_entry["repo_id"]
        selected_backend = selected_entry.get("backend", BACKEND_MULTILINGUAL)
        if (selected_backend in DUAL_MODE_BACKENDS and isinstance(self.model, DUAL_MODE_TYPES)
                and self.model.backend == selected_backend and self.model.repo_id == selected_repo):
            self.switch_voice_mode(selected_entry)
            return
        selected_multilingual_t3_model = selected_entry.get(
            "multilingual_t3_model",
            DEFAULT_MULTILINGUAL_T3_MODEL,
        )
        self.current_model_repo = selected_repo
        self.current_model_backend = selected_backend
        self.current_multilingual_t3_model = selected_multilingual_t3_model
        self.set_status_message(
            f"Status: Loading {selected_entry['label']}. A model that isn't downloaded yet "
            "is fetched first; progress appears on the Log page."
        )
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.release_model()
        self.set_model_loading_state(True)
        self.current_mode = model_registry.entry_mode(selected_entry)
        self.model_loader_thread = ModelLoaderThread(
            selected_repo,
            selected_backend,
            selected_multilingual_t3_model,
        )
        self.model_loader_thread.mode = self.current_mode
        self.cuda_runtime_issue = self.model_loader_thread.cuda_probe_error
        self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loader_thread.error_occurred.connect(
            self.on_model_load_error)
        self.model_loader_thread.start()

    def switch_voice_mode(self, entry):
        """Switch a loaded dual-mode model between cloning and design without reloading."""
        self.current_mode = model_registry.entry_mode(entry)
        self.model.set_mode(self.current_mode)
        self.set_status_message(f"Status: Switched to {entry['label']} (same model, no reload). Ready.")
        self.update_engine_controls()
        self.refresh_language_options()
        self.update_text_stats()
        self.render_model_tiles()
        self.after_voice_model_ready()

    def after_voice_model_ready(self):
        if self.pending_voice is not None:
            self.apply_pending_voice()
        clip_voice = getattr(self, "pending_clip_voice", None)
        if clip_voice is not None:
            self.pending_clip_voice = None
            if self.voice_is_active(clip_voice):
                self.make_voice_clip(clip_voice)
        self.render_voice_tiles()

    def on_model_loaded(self, model_instance, device_used):
        self.model_is_warm = False
        self.model = model_instance
        self.device_used = device_used
        status_message = (
            f"Status: Model loaded from {self.current_model_repo} "
            f"using {self.current_model_backend} on {self.device_used}. Ready."
        )
        if self.system_has_nvidia_gpu and self.device_used == "cpu":
            status_message += " NVIDIA GPU detected, but PyTorch CUDA is unavailable."
        self.set_status_message(status_message)
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.set_model_loading_state(False)
        self.update_text_stats()
        self.refresh_models_page()
        self.update_engine_controls()
        self.after_voice_model_ready()
        if self.system_has_nvidia_gpu and self.device_used == "cpu":
            details = self.cuda_runtime_issue or (
                "This Python environment is using a CPU-only PyTorch build."
            )
            QMessageBox.warning(
                self,
                "CUDA Not Active",
                "An NVIDIA GPU was detected, but this Python environment is using "
                "a PyTorch configuration that cannot run on the detected GPU.\n\n"
                f"Details: {details}\n\n"
                "Re-run the installer to repair the PyTorch installation for CUDA.",
            )

    def on_model_load_error(self, error_msg):
        self.model = None
        self.set_status_message(f"Status: Model load failed. {error_msg}")
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.set_model_loading_state(False)
        self.refresh_models_page()
        QMessageBox.critical(self, "Model Load Error", error_msg)

    def on_tuning_changed(self, *_args):
        self.repetition_penalty = self.repetition_spin.value()
        self.min_p = self.min_p_spin.value()
        self.top_p = self.top_p_spin.value()
        self.app_settings["sampling"] = {
            "repetition_penalty": self.repetition_penalty, "min_p": self.min_p, "top_p": self.top_p}
        defaults = (abs(self.repetition_penalty - 1.2) < 1e-9 and abs(self.min_p - 0.05) < 1e-9
                    and abs(self.top_p - 1.0) < 1e-9)
        self.tuning_summary_label.setText("defaults" if defaults else
                                          f"repetition {self.repetition_penalty:.2f}, "
                                          f"min-p {self.min_p:.2f}, top-p {self.top_p:.2f}")

    def reset_tuning(self):
        self.repetition_spin.setValue(1.2)
        self.min_p_spin.setValue(0.05)
        self.top_p_spin.setValue(1.0)

    def set_tuning_expanded(self, expanded):
        self.tuning_panel.setVisible(expanded)
        arrow = "\u25be" if expanded else "\u25b8"
        self.tuning_toggle.setText(f"{arrow} Fine-tuning")
        self.app_settings["tuning_expanded"] = expanded
        self.on_tuning_changed()
        if self.isVisible():
            self.update_minimum_size()

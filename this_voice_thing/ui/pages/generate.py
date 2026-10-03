"""The Generate page: the text, delivery and player cards, and running a generation."""

import os
import time

import numpy as np
from PySide6.QtCore import QTime
from PySide6.QtCore import QUrl
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QGridLayout
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QMenu
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QTextEdit
from PySide6.QtWidgets import QWidget

from this_voice_thing.core import documents
from this_voice_thing.core import model_registry
from this_voice_thing.core import voice_library
from this_voice_thing.ui import theme as ui_theme
from this_voice_thing.ui.common import BACKEND_MULTILINGUAL
from this_voice_thing.ui.common import DEFAULT_LANGUAGE_TEST_TEXTS
from this_voice_thing.ui.common import DEFAULT_PREVIEW_CHARS
from this_voice_thing.ui.common import DUAL_MODE_BACKENDS
from this_voice_thing.ui.common import KOKORO_BACKEND
from this_voice_thing.ui.common import PREVIEW_LENGTHS
from this_voice_thing.ui.common import QWEN_BACKEND
from this_voice_thing.ui.common import VIBEVOICE_BACKEND
from this_voice_thing.ui.common import languages_for_backend
from this_voice_thing.ui.common import preview_cut
from this_voice_thing.ui.dialogs.voices import VoiceDetailsDialog
from this_voice_thing.ui.threads import AudioGeneratorThread
from this_voice_thing.ui.widgets import ElidingChip
from this_voice_thing.ui.widgets import SliderWithValue
from this_voice_thing.ui.widgets import dialog_accepted


class GeneratePage:
    """The Generate page: cards and the generation run. Mixed into ChatterboxApp."""

    """The Generate page: text, documents, estimates, generation and finishing. Mixed into ChatterboxApp."""

    def _build_generate_page(self):
        generate_page, generate_layout = self._make_page(
            "Generate", "Write your text, choose the delivery, then generate.")

        voice_row = QHBoxLayout()
        voice_row.setContentsMargins(ui_theme.SHADOW, 0, ui_theme.SHADOW, 2)
        voice_row.addWidget(QLabel("Voice"))
        self.voice_chip = ElidingChip("Default voice")
        self.voice_chip.setObjectName("VoiceChip")
        self.voice_chip.setTextFormat(Qt.TextFormat.PlainText)
        voice_row.addWidget(self.voice_chip)
        change_voice_button = self._link(QPushButton("Change..."))
        change_voice_button.clicked.connect(
            lambda: self.sidebar.setCurrentRow(self.PAGE_VOICE))
        voice_row.addWidget(change_voice_button)
        voice_row.addStretch(1)
        voice_row.addWidget(QLabel("Model"))
        self.model_repo_combo = QComboBox()
        self.model_repo_combo.setMinimumWidth(190)
        self.model_repo_combo.currentIndexChanged.connect(self.on_model_repo_changed)
        voice_row.addWidget(self.model_repo_combo)
        generate_layout.addLayout(voice_row)

        generate_layout.addWidget(self._build_text_card(), 3)
        self._build_engine_rows()
        generate_layout.addWidget(self._build_delivery_card())
        generate_layout.addWidget(self._build_player_card(), 2)
        self.pages.addWidget(generate_page)

    def _build_text_card(self):
        text_card, text_card_layout = self._make_card()
        text_header = QHBoxLayout()
        text_title = QLabel("Text")
        text_title.setObjectName("CardTitle")
        text_header.addWidget(text_title)
        self.document_label = QLabel()
        self.document_label.setObjectName("Muted")
        self.document_label.setTextFormat(Qt.TextFormat.PlainText)
        self.document_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text_header.addWidget(self.document_label, 1)
        text_card_layout.addLayout(text_header)
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Enter text to synthesize, or open a document. Long text is split where a reader "
            "would pause and stitched back together."
        )
        self.text_input.setMinimumHeight(90)
        self.text_input.setAcceptRichText(False)
        self.text_input.textChanged.connect(self.on_text_changed)
        # Fill the leftover height instead of forcing the page to scroll.
        self.text_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        text_card_layout.addWidget(self.text_input, 1)

        text_status_row = QHBoxLayout()
        text_status_row.setSpacing(10)
        self.estimate_button = self._link(QPushButton())
        self.estimate_button.setToolTip(
            "Estimated generation time with the active model. Click to compare every model.")
        self.estimate_button.clicked.connect(self.show_estimate_menu)
        self.estimate_button.setVisible(False)
        text_status_row.addWidget(self.estimate_button)
        self.text_stats_label = QLabel()
        self.text_stats_label.setObjectName("Muted")
        self.text_stats_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_stats_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.text_stats_label.setToolTip(
            "Updates as you edit. Times are learned per model from the sections you generate.")
        text_status_row.addWidget(self.text_stats_label, 1)
        self.activity_label = QLabel()
        self.activity_label.setTextFormat(Qt.TextFormat.PlainText)
        self.activity_label.setToolTip("Section being generated / total sections, and time left.")
        text_status_row.addWidget(self.activity_label)
        self.generation_progress = QProgressBar()
        self.generation_progress.setTextVisible(False)
        self.generation_progress.setFixedHeight(8)
        self.generation_progress.setFixedWidth(100)
        self.generation_progress.setVisible(False)
        text_status_row.addWidget(self.generation_progress)
        self.keep_take_button = self._link(QPushButton("Keep this take"))
        self.keep_take_button.setToolTip(
            "Lock the take number used by the preview so the full render matches it.")
        self.keep_take_button.clicked.connect(self.keep_preview_take)
        self.keep_take_button.setVisible(False)
        text_status_row.addWidget(self.keep_take_button)
        self.keep_voice_button = self._link(QPushButton("Keep this voice"))
        self.keep_voice_button.setToolTip(
            "Save the designed voice you just heard to the voice library and lock it in, so the full "
            "render (and later ones) use exactly this voice instead of designing a new one.")
        self.keep_voice_button.clicked.connect(self.keep_designed_voice)
        self.keep_voice_button.setVisible(False)
        text_status_row.addWidget(self.keep_voice_button)
        status_row_widget = QWidget()
        status_row_widget.setLayout(text_status_row)
        text_status_row.setContentsMargins(0, 0, 0, 0)
        status_row_widget.setFixedHeight(QPushButton("X").sizeHint().height())
        text_card_layout.addWidget(status_row_widget)

        generate_actions_layout = QHBoxLayout()
        self.open_document_button = QPushButton("Open...")
        self.open_document_button.setToolTip("Load a .txt, .md or .docx file, or a Google Doc, to read aloud.")
        open_menu = QMenu(self.open_document_button)
        open_menu.addAction("From this computer...").triggered.connect(lambda _checked=False: self.open_document())
        open_menu.addAction("From Google Docs...").triggered.connect(lambda _checked=False: self.open_google_doc())
        self.open_document_button.setMenu(open_menu)
        self.open_menu = open_menu
        generate_actions_layout.addWidget(self.open_document_button)
        self.use_preset_button = QPushButton("Sample")
        self.use_preset_button.setToolTip("Fill in a short test sentence for the selected language.")
        self.use_preset_button.clicked.connect(self.apply_selected_text_preset)
        generate_actions_layout.addWidget(self.use_preset_button)
        generate_actions_layout.addStretch()
        self.preview_length_combo = QComboBox()
        for label, characters in PREVIEW_LENGTHS:
            self.preview_length_combo.addItem(label, characters)
        saved_length = self.preview_length_combo.findData(self.app_settings.get("preview_chars", DEFAULT_PREVIEW_CHARS))
        if saved_length < 0:  # a length from an older version
            saved_length = self.preview_length_combo.findData(DEFAULT_PREVIEW_CHARS)
        self.preview_length_combo.setCurrentIndex(saved_length)
        self.preview_length_combo.setToolTip(
            "About how long Preview reads: the opening sentence or two of the text (headings "
            "skipped). Select text to preview exactly that instead.")
        self.preview_length_combo.currentIndexChanged.connect(
            lambda _index: self.app_settings.update(preview_chars=self.preview_length_combo.currentData()))
        self.preview_length_combo.setFixedWidth(84)
        generate_actions_layout.addWidget(self.preview_length_combo)
        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip(
            "Generate a short sample with the current settings before rendering everything: "
            "the selected text, or the opening of the text (length chosen on the left).")
        self.preview_button.clicked.connect(lambda: self.start_generation(preview=True))
        self.preview_button.setEnabled(False)
        generate_actions_layout.addWidget(self.preview_button)
        self.generate_button = self._accent(QPushButton("Generate Audio"))
        self.generate_button.clicked.connect(self.handle_generate_stop_toggle)
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumWidth(140)
        generate_actions_layout.addWidget(self.generate_button)
        text_card_layout.addLayout(generate_actions_layout)
        return text_card

    def _build_delivery_card(self):
        delivery_card, delivery_layout = self._make_card("Delivery")
        delivery_layout.addWidget(self.qwen_row)
        params_layout = QGridLayout()
        params_layout.setHorizontalSpacing(14)
        params_layout.setVerticalSpacing(8)
        params_layout.setColumnStretch(1, 1)
        params_layout.setColumnStretch(3, 1)

        def add_control(row, column, title, widget, tooltip):
            label = QLabel(title)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            params_layout.addWidget(label, row, column)
            params_layout.addWidget(widget, row, column + 1)
            return label

        self.exaggeration_slider = self._create_slider(0.25, 2.0, 0.05, 0.5)
        self.exaggeration_label = add_control(0, 0, "Expressiveness", self.exaggeration_slider,
                    "How animated and emotional the delivery sounds. 0.5 is neutral; "
                    "higher is more dramatic (and often a bit faster). [exaggeration]")
        self.cfg_slider = self._create_slider(0.2, 1.0, 0.05, 0.5)
        self.cfg_label = add_control(0, 2, "Pacing", self.cfg_slider,
                    "Lower gives slower, more deliberate speech; higher is brisker and "
                    "follows the reference voice's style more closely. Try 0.3 for "
                    "expressive or fast-talking voices. [cfg_weight]")
        self.temp_slider = self._create_slider(0.05, 5.0, 0.05, 0.8)
        add_control(1, 0, "Variation", self.temp_slider,
                    "How different each take sounds. Higher is livelier but can become "
                    "unstable; lower is steadier and more predictable. [temperature]")
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1_000_000_000)
        self.seed_input.setValue(0)
        self.seed_input.setSpecialValueText("New take each time")
        self.seed_input.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.language_combo = QComboBox()
        self.language_combo.currentIndexChanged.connect(lambda _i: self.on_language_changed())
        add_control(2, 0, "Language", self.language_combo,
                    "Language of the text. The list depends on the selected model.")
        add_control(1, 2, "Take number", self.seed_input,
                    "Leave on 'New take each time' for a fresh result on every run. Enter a "
                    "number to reproduce the same take exactly; the number used is shown in "
                    "the file name. [seed]")
        params_layout.addWidget(self.qwen_watermark_checkbox, 2, 2, 1, 2)
        delivery_layout.addLayout(params_layout)
        delivery_hint = QLabel(
            "Tip: commas and ellipses add pauses; question marks lift the ending.")
        delivery_hint.setObjectName("Muted")
        delivery_hint.setToolTip("Advanced sampling options are under Model > Sampling.")
        delivery_layout.addWidget(delivery_hint)
        self._build_finishing_panel(delivery_layout)
        return delivery_card

    def handle_generate_stop_toggle(self):
        if not self.is_generating:
            self.start_generation(preview=False)
            return
        if hasattr(self, 'audio_generator_thread') and self.audio_generator_thread.isRunning():
            print("UI: Requesting stop for audio_generator_thread")
            self.audio_generator_thread.stop()
            self.generate_button.setText("Stopping...")
            self.generate_button.setEnabled(False)
            self.set_status_message(
                "Status: Stopping after the current section. Finished sections will be kept.")
        else:
            print("UI: Stop requested, but no active generation thread found. Resetting UI.")
            self.on_generation_thread_finished()

    def preview_text(self):
        """(text, character budget): a selection is previewed whole; otherwise a short
        excerpt from the start of the text, about the length picked next to Preview."""
        selected = self.text_input.textCursor().selectedText().replace("\u2029", "\n").strip()
        if selected:
            return selected, None
        return documents.excerpt(self.text_input.toPlainText(), self.preview_length_combo.currentData()), None

    def start_generation(self, preview=False):
        if self.is_generating:
            return
        if self.api_busy:
            self.set_status_message("Status: Busy with a request from the local API; try again in a moment.")
            return
        if self.model is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load the model first.")
            return
        preview_budget = None
        if preview:
            text, preview_budget = self.preview_text()
        else:
            text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to synthesize.")
            return
        qwen_problem = self.prepare_qwen_generation()
        if qwen_problem:
            QMessageBox.information(self, "Voice", qwen_problem)
            return

        self.is_generating = True
        self.generation_is_preview = preview
        self.generation_char_count = len(text)
        plan_entry = self.loaded_entry() or self.get_selected_model_entry()
        lengths = self.section_lengths(text, plan_entry)
        if preview:
            lengths = lengths[:preview_cut(lengths, preview_budget)]
        self.generation_plan = self.batch_plan(plan_entry, lengths)
        self.generation_estimate = sum(cost for _f, _l, cost in self.generation_plan)
        self.progress_range = None
        self.progress_done_cost = 0.0
        self.progress_done_time = 0.0
        self.generation_started_at = time.monotonic()
        self.keep_take_button.setVisible(False)
        self.keep_voice_button.setVisible(False)
        self.generate_button.setText("Stop")
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(False)
        self.open_document_button.setEnabled(False)
        self.model_repo_combo.setEnabled(False)
        self.generation_progress.setValue(0)
        self.generation_progress.setVisible(True)
        self.activity_label.setText("Previewing..." if preview else "Starting...")

        self.generation_start_time = QTime.currentTime()
        self.generation_timer.start(1000)
        self.update_generation_time_display()

        self.audio_generator_thread = AudioGeneratorThread(
            self.model, text,
            self.ref_audio_path_label.toolTip(),
            self.exaggeration_slider.get_value(),
            self.temp_slider.get_value(),
            self.cfg_slider.get_value(),
            self.seed_input.value(),
            self.output_directory,
            language_id=self.language_combo.currentData() or "en",
            repetition_penalty=self.repetition_penalty,
            min_p=self.min_p,
            top_p=self.top_p,
            finishing=self.current_finishing_settings(),
            output_name=self.current_document_name,
            preview=preview,
        )
        self.audio_generator_thread.pronunciations = self.pronunciations
        self.audio_generator_thread.preview_chars = preview_budget
        self.audio_generator_thread.generation_complete.connect(self.on_generation_complete)
        self.audio_generator_thread.error_occurred.connect(self.on_generation_error)
        self.audio_generator_thread.chunk_generated.connect(self.on_chunk_generated_progress)
        self.audio_generator_thread.section_timed.connect(self.on_section_timed)
        self.audio_generator_thread.finished.connect(self.on_generation_thread_finished)
        self.audio_generator_thread.start()

    def keep_designed_voice(self):
        """Save the voice a design model just made (the preview's first section) to the
        library, and lock it in for every render."""
        model = self.active_qwen_model()
        anchor = getattr(model, "_anchor", None)
        if model is None or model.mode != "voice_design" or not anchor or not os.path.exists(anchor[0]):
            self.keep_voice_button.setVisible(False)
            return
        entry = self.loaded_entry()
        description = self.qwen_instruct_input.text().strip()
        voice = voice_library.Voice(
            name="Designed voice", kind="design", backend=entry["backend"], repo_id=entry["repo_id"],
            mode=model_registry.entry_mode(entry) if entry["backend"] in DUAL_MODE_BACKENDS else "",
            description=description, language=self.language_combo.currentData() or "")
        dialog = VoiceDetailsDialog("Keep this voice", voice, self.voice_library, parent=self)
        if not dialog_accepted(dialog.exec()):
            return
        dialog.apply()
        path = self.voice_library.new_clip_path(voice.name)
        import soundfile
        wav, sr = soundfile.read(anchor[0], dtype="float32")
        soundfile.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        voice_library.write_transcript(path, anchor[1])
        voice.clip = self.voice_library.to_stored(path)
        self.voice_library.add(voice)
        model.locked_anchor = (path, anchor[1])
        self.locked_description = description
        self.locked_voice_name = voice.name
        self.active_voice_id = voice.id
        self.keep_voice_button.setVisible(False)
        self.refresh_voice_chip()
        self.render_voice_tiles()
        self.set_status_message(f"Status: Kept {voice.name}. Every section now uses this voice; "
                                "it's in the voice library too.")

    def keep_preview_take(self):
        if self.last_preview_seed:
            self.seed_input.setValue(self.last_preview_seed)
            self.keep_take_button.setVisible(False)
            self.activity_label.setText(f"Take {self.last_preview_seed} locked")

    def _create_slider(self, min_val, max_val, step_val, default_val, value_format="{:.2f}"):
        return SliderWithValue(min_val, max_val, step_val, default_val, value_format)

    def refresh_language_options(self):
        selected_entry = self.get_selected_model_entry()
        supported_languages = languages_for_backend(
            selected_entry.get("backend", BACKEND_MULTILINGUAL)
        )
        preferred_language = selected_entry.get("language_id", "en")

        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        for language_id, language_name in supported_languages.items():
            self.language_combo.addItem(f"{language_name} [{language_id}]", language_id)

        preferred_index = self.language_combo.findData(preferred_language)
        if preferred_index < 0:
            preferred_index = self.language_combo.findData("en")
        if preferred_index < 0 and self.language_combo.count() > 0:
            preferred_index = 0
        if preferred_index >= 0:
            self.language_combo.setCurrentIndex(preferred_index)

        is_multilingual = selected_entry.get("backend") in (BACKEND_MULTILINGUAL, QWEN_BACKEND, KOKORO_BACKEND,
                                                            VIBEVOICE_BACKEND, *DUAL_MODE_BACKENDS)
        self.language_combo.setEnabled(is_multilingual)
        self.language_combo.setToolTip(
            "Language used by the multilingual Chatterbox backend."
            if is_multilingual else
            "Legacy English backend only."
        )
        self.language_combo.blockSignals(False)

    def apply_selected_text_preset(self):
        selected_entry = self.get_selected_model_entry()
        selected_language = self.language_combo.currentData() or selected_entry.get("language_id", "en")
        preset_text = selected_entry.get("test_texts", {}).get(selected_language)
        if not preset_text and selected_entry.get("backend") == BACKEND_MULTILINGUAL:
            preset_text = DEFAULT_LANGUAGE_TEST_TEXTS.get(
                selected_language,
                DEFAULT_LANGUAGE_TEST_TEXTS["en"],
            )
        if not preset_text:
            preset_text = selected_entry.get("test_text")
        if preset_text:
            self.current_document_name = None
            self.document_label.clear()
            self.text_input.setPlainText(preset_text)

    def on_generation_thread_finished(self):
        print("UI: audio_generator_thread.finished signal received.")

        # Stop the timer regardless of how the thread finished
        if self.generation_timer.isActive():
            print("UI: Stopping generation timer.")
            self.generation_timer.stop()

        # Reset UI elements
        self.is_generating = False
        self.generate_button.setText("Generate Audio")
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.open_document_button.setEnabled(True)
        self.model_repo_combo.setEnabled(not getattr(self, "model_is_loading", False))
        self.update_model_details()
        self.generation_progress.setVisible(False)
        if not self.keep_take_button.isVisible() and not self.keep_voice_button.isVisible():
            self.activity_label.clear()
        self.update_text_stats()

        # Final status update based on how the thread might have ended,
        # if not already set by on_generation_complete or on_generation_error.
        # This ensures "Stopping..." doesn't linger.
        current_status = self.status_bar.text()
        if "stopping generation..." in current_status.lower() or \
           "stop requested." in current_status.lower():
            self.set_status_message("Status: Generation stopped by user.")
        elif not any(marker in current_status.lower() for marker in (
                "full audio generated", "failed", "stopped by user",
                "preview ready", "stopped. saved")):
            # If no specific completion or error message was set, default to Ready
            self.set_status_message("Status: Ready.")

    def update_generation_time_display(self):
        self.refresh_progress_activity()
        if self.generation_start_time and self.is_generating:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            # Only update if not showing chunk progress, to avoid flicker
            # and if the button still says "Stop Generation" (i.e. not "Stopping...")
            if "chunk" not in self.status_bar.text().lower() and \
               self.generate_button.text() == "Stop Generation":
                self.set_status_message(
                    f"Status: Generating... (Elapsed: {self.format_time(elapsed_ms)})")
        elif not self.is_generating and self.generation_timer.isActive():
            # This is a failsafe, should be stopped by on_generation_thread_finished
            print(
                "UI: Generation timer stopped by failsafe in update_generation_time_display.")
            self.generation_timer.stop()

    def update_generation_time(self):
        if self.generation_start_time and self.is_generating:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            seconds = int((elapsed_ms / 1000) % 60)
            minutes = int((elapsed_ms / (1000 * 60)) % 60)
            self.set_status_message(
                f"Status: Generating audio... {minutes:02}:{seconds:02}")

    def on_chunk_generated_progress(self, first, last, total):
        if not self.is_generating:
            return
        done = first - 1
        elapsed = time.monotonic() - self.generation_started_at
        # Everything before this batch has just finished: calibrate against the plan.
        self.progress_done_cost = sum(cost for _f, end, cost in getattr(self, "generation_plan", [])
                                      if end <= done)
        self.progress_done_time = elapsed
        self.progress_range = (first, last, total)
        self.generation_progress.setMaximum(total)
        self.generation_progress.setValue(done)
        span = f"{first}" if first == last else f"{first}\u2013{last}"
        self.set_status_message(f"Status: Generating section {span} of {total}...")
        self.refresh_progress_activity()

    def refresh_progress_activity(self):
        """Activity text with a countdown; called on progress and every second."""
        if not self.is_generating or not getattr(self, "progress_range", None):
            return
        first, last, total = self.progress_range
        elapsed = time.monotonic() - self.generation_started_at
        estimate = getattr(self, "generation_estimate", 0.0)
        if self.progress_done_cost > 0:
            projected = self.progress_done_time / self.progress_done_cost * estimate
        else:
            projected = estimate
        remaining = max(projected - elapsed, 0.0)
        span = f"{first}" if first == last else f"{first}\u2013{last}"
        activity = f"{span}/{total}"
        if self.generation_is_preview:
            activity = f"Preview {activity}"
        if estimate > 0:
            activity += f" \u00b7 {self.format_clock(remaining)} left" if remaining >= 1 else " \u00b7 finishing"
        self.activity_label.setText(activity)

    def on_generation_complete(self, output_path, sample_rate):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        total_generation_time_str = ""
        if self.generation_start_time:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            total_generation_time_str = f" (Total time: {self.format_time(elapsed_ms)})"

        thread = self.audio_generator_thread
        self.generation_progress.setValue(self.generation_progress.maximum())
        if thread.preview:
            self.last_preview_seed = thread.actual_seed_used
            self.activity_label.setText(f"Preview take {self.last_preview_seed}")
            self.keep_take_button.setVisible(self.seed_input.value() == 0)
            designed = self.active_qwen_model()
            self.keep_voice_button.setVisible(
                designed is not None and designed.mode == "voice_design"
                and getattr(designed, "_anchor", None) is not None and not getattr(designed, "locked_anchor", None))
            self.set_status_message(f"Status: Preview ready{total_generation_time_str}.")
        elif thread.partial_info:
            done, total = thread.partial_info
            self.set_status_message(
                f"Status: Stopped. Saved {done} of {total} sections: {os.path.basename(output_path)}")
        else:
            captions = f" + {os.path.basename(thread.subtitle_path)}" if thread.subtitle_path else ""
            self.set_status_message(
                f"Status: Full audio generated: {os.path.basename(output_path)}{captions}{total_generation_time_str}")

        self.current_audio_file = output_path
        # ... (rest of the method same as your working version)
        self.current_file_label.setText(
            f"Last generated: {os.path.basename(output_path)}")
        self.media_player.setSource(QUrl.fromLocalFile(output_path))
        self.play_pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.playhead_slider.setEnabled(True)
        self.update_output_log()
        if self.autoplay_checkbox.isChecked() or thread.preview:
            self.media_player.play()

    def on_generation_error(self, error_msg):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        is_user_stop = "stopped by user" in error_msg.lower()

        final_status_msg = f"Status: {'Generation stopped by user.' if is_user_stop else 'Generation failed.'}"
        if not is_user_stop and error_msg:
            first_line_error = error_msg.splitlines()[0]
            if len(first_line_error) > 70:
                # Adjusted length
                first_line_error = first_line_error[:67] + "..."
            final_status_msg += f" ({first_line_error})"

        self.set_status_message(final_status_msg)

        if not is_user_stop:
            QMessageBox.critical(self, "Generation Error", error_msg)
        else:
            print(f"User stop confirmed by error signal: {error_msg}")

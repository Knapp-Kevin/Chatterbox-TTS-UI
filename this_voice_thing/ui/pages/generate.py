"""The Generate page: text, documents, estimates, generation and finishing."""

import os
import tempfile
import time

import numpy as np
import torch
from PySide6.QtCore import Qt, QTime, QUrl
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from this_voice_thing.core import audio_effects, documents, model_registry, voice_library
from this_voice_thing.engines import omnivoice as omnivoice_engine, vibevoice as vibevoice_engine
from this_voice_thing.ui import theme as ui_theme
from this_voice_thing.ui.common import (
    BACKEND_MULTILINGUAL,
    BATCH_COST_SLOPE,
    DEFAULT_LANGUAGE_TEST_TEXTS,
    DEFAULT_PREVIEW_CHARS,
    DEFAULT_SECONDS_PER_CHAR,
    DUAL_MODE_BACKENDS,
    ENGINE_MODULES,
    KOKORO_BACKEND,
    languages_for_backend,
    LOSSLESS_FORMATS,
    MAX_TEXT_INPUT_LENGTH,
    preview_cut,
    PREVIEW_LENGTHS,
    QWEN_BACKEND,
    VIBEVOICE_BACKEND,
)
from this_voice_thing.ui.dialogs.google_docs import GoogleDocsDialog
from this_voice_thing.ui.dialogs.voices import VoiceDetailsDialog
from this_voice_thing.ui.threads import AudioGeneratorThread
from this_voice_thing.ui.widgets import dialog_accepted, ElidingChip, SliderWithValue


class GeneratePage:
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
        generate_layout.addWidget(text_card, 3)

        # Qwen controls share Delivery's first row with the Chatterbox-only sliders,
        # so switching engines never changes the window's minimum height.
        self.qwen_row = QWidget()
        qwen_row_layout = QHBoxLayout(self.qwen_row)
        qwen_row_layout.setContentsMargins(0, 0, 0, 0)
        qwen_row_layout.setSpacing(10)
        qwen_settings = self.app_settings.get("qwen", {})
        self.qwen_speaker_combo = QComboBox()
        # Kokoro's voice names are long ("Heart (US English, female)"); the list opens wide
        # anyway, so the box itself stays compact instead of widening the window.
        self.qwen_speaker_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.qwen_speaker_combo.setMinimumContentsLength(14)
        self.qwen_speaker_combo.view().setMinimumWidth(260)
        self.qwen_speaker_combo.setToolTip("Built-in Qwen speaker.")
        self.qwen_speaker_combo.currentIndexChanged.connect(lambda _i: self.refresh_voice_chip())
        self.qwen_speaker_label = QLabel("Speaker")
        qwen_row_layout.addWidget(self.qwen_speaker_label)
        qwen_row_layout.addWidget(self.qwen_speaker_combo)
        self.qwen_instruct_input = QLineEdit()
        self.qwen_instruct_label = QLabel("Style")
        qwen_row_layout.addWidget(self.qwen_instruct_label)
        qwen_row_layout.addWidget(self.qwen_instruct_input, 1)
        self.design_attributes_button = self._link(QPushButton("Attributes\u2026"))
        self.design_attributes_button.setToolTip("Pick the voice's gender, age, pitch, accent and more.")
        attributes_menu = QMenu(self)
        # Keep Python references: PySide can otherwise free submenus made by addMenu(title).
        self.design_attribute_menus = [attributes_menu]
        for group, items in omnivoice_engine.DESIGN_ATTRIBUTES.items():
            submenu = QMenu(group, attributes_menu)
            attributes_menu.addMenu(submenu)
            self.design_attribute_menus.append(submenu)
            for item in items:
                action = submenu.addAction(item)
                action.triggered.connect(lambda _checked=False, item=item: self.qwen_instruct_input.setText(
                    omnivoice_engine.set_attribute(self.qwen_instruct_input.text(), item)))
        attributes_menu.addSeparator()
        attributes_menu.addAction("Clear").triggered.connect(lambda: self.qwen_instruct_input.clear())
        self.design_attributes_button.setMenu(attributes_menu)
        self.design_attributes_button.setVisible(False)
        qwen_row_layout.addWidget(self.design_attributes_button)
        self.qwen_transcript_label = QLabel("Clip transcript")
        self.qwen_transcript_input = QLineEdit()
        self.qwen_transcript_input.setPlaceholderText(
            "What is said in the reference clip (optional, improves likeness)")
        self.qwen_transcript_input.setToolTip(
            "With a transcript, Qwen and VoxCPM clone more closely. It must match what is said in "
            "the clip: a wrong transcript can garble VoxCPM's output. Recordings made with "
            "Record... fill this in with the passage you read; edit it if you said something different.")
        self.qwen_transcript_input.editingFinished.connect(self.save_reference_transcript)
        qwen_row_layout.addWidget(self.qwen_transcript_label)
        qwen_row_layout.addWidget(self.qwen_transcript_input, 1)
        self.cast_label = QLabel()
        self.cast_label.setObjectName("Muted")
        self.cast_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.cast_button = QPushButton("Cast\u2026")
        self.cast_button.setToolTip("Choose a voice for each speaker in the script.")
        self.cast_button.clicked.connect(self.edit_cast)
        self.cast_title = QLabel("Speakers")
        for widget in (self.cast_title, self.cast_label, self.cast_button):
            widget.setVisible(False)
        qwen_row_layout.addWidget(self.cast_title)
        qwen_row_layout.addWidget(self.cast_label, 1)
        qwen_row_layout.addWidget(self.cast_button)
        self.qwen_watermark_checkbox = QCheckBox("Add AI watermark")
        self.qwen_watermark_checkbox.setChecked(bool(qwen_settings.get("watermark", True)))
        self.qwen_watermark_checkbox.setToolTip(
            "Qwen and Kokoro don't watermark their audio. When ticked, the same inaudible Perth watermark "
            "Chatterbox uses is added, so output from every engine is marked the same way.")
        self.qwen_settings = qwen_settings
        self.kokoro_settings = self.app_settings.get("kokoro", {})
        self.voxcpm_settings = self.app_settings.get("voxcpm", {})
        self.omnivoice_settings = self.app_settings.get("omnivoice", {})
        self.vibevoice_settings = self.app_settings.get("vibevoice", {})
        self.qwen_row.setVisible(False)
        self.qwen_watermark_checkbox.setVisible(False)

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

        finishing_header = QHBoxLayout()
        self.finishing_toggle = self._link(QPushButton())
        self.finishing_toggle.setToolTip("Adjustments applied to the audio after it is generated.")
        self.finishing_toggle.clicked.connect(
            lambda: self.set_finishing_expanded(self.finishing_panel.isHidden()))
        finishing_header.addWidget(self.finishing_toggle)
        self.finishing_summary_label = QLabel()
        self.finishing_summary_label.setObjectName("Muted")
        finishing_header.addWidget(self.finishing_summary_label)
        finishing_header.addStretch(1)
        delivery_layout.addLayout(finishing_header)

        self.finishing_panel = QWidget()
        finishing_grid = QGridLayout(self.finishing_panel)
        finishing_grid.setContentsMargins(0, 0, 0, 0)
        finishing_grid.setHorizontalSpacing(14)
        finishing_grid.setColumnStretch(1, 1)
        finishing_grid.setColumnStretch(3, 1)

        def add_finishing(row, column, title, widget, tooltip):
            label = QLabel(title)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            finishing_grid.addWidget(label, row, column)
            finishing_grid.addWidget(widget, row, column + 1)

        self.pause_slider = self._create_slider(
            *audio_effects.PAUSE_RANGE, 0.1, 0.6, "{:.1f} s")
        add_finishing(0, 0, "Paragraph pause", self.pause_slider,
                      "Silence between paragraphs (headings get a little more). Pauses between "
                      "sentences and inside long sentences are kept short and even automatically.")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(LOSSLESS_FORMATS)
        add_finishing(0, 2, "Save as", self.output_format_combo,
                      "WAV is uncompressed; FLAC is lossless and about half the size. "
                      "MP3 is on the Advanced page.")
        finishing_checks = QHBoxLayout()
        self.even_volume_checkbox = QCheckBox("Even out volume")
        self.even_volume_checkbox.setToolTip(
            "Bring every result to a consistent, comfortable loudness without clipping.")
        self.trim_silence_checkbox = QCheckBox("Trim silence")
        self.trim_silence_checkbox.setToolTip(
            "Remove dead air before the first word and after the last.")
        finishing_checks.setSpacing(18)
        self.subtitles_checkbox = QCheckBox("Save subtitles")
        self.subtitles_checkbox.setToolTip(
            "Also save captions timed to the audio, next to it (.srt, or .vtt: the format is on the "
            "Advanced page). Timing comes from the generated sections and the pauses in them.")
        finishing_checks.addWidget(self.even_volume_checkbox)
        finishing_checks.addWidget(self.trim_silence_checkbox)
        finishing_checks.addWidget(self.subtitles_checkbox)
        finishing_checks.addStretch(1)
        reset_finishing_button = QPushButton("Reset")
        reset_finishing_button.setToolTip(
            "Restore the default finishing settings (Advanced effects are kept).")
        reset_finishing_button.clicked.connect(self.reset_finishing)
        finishing_checks.addWidget(reset_finishing_button)
        finishing_grid.addLayout(finishing_checks, 1, 0, 1, 4)
        delivery_layout.addWidget(self.finishing_panel)
        self.finishing_toggle.setToolTip(
            "Adjustments applied to the audio after it is generated. Speed, pitch and MP3 "
            "are on the Advanced page.")

        self.pause_slider.slider.valueChanged.connect(self.update_finishing_summary)
        self.output_format_combo.currentTextChanged.connect(self.update_finishing_summary)
        self.even_volume_checkbox.toggled.connect(self.update_finishing_summary)
        self.trim_silence_checkbox.toggled.connect(self.update_finishing_summary)
        self.subtitles_checkbox.toggled.connect(self.update_finishing_summary)
        generate_layout.addWidget(delivery_card)

        player_card, player_layout = self._make_card()
        player_header = QHBoxLayout()
        player_title = QLabel("Player")
        player_title.setObjectName("CardTitle")
        player_header.addWidget(player_title)
        player_header.addSpacing(12)
        self.autoplay_checkbox = QCheckBox("Auto-play results")
        self.autoplay_checkbox.setChecked(True)
        player_header.addWidget(self.autoplay_checkbox)
        player_header.addStretch(1)
        self.current_file_label = QLabel("Currently playing: None")
        self.current_file_label.setObjectName("Muted")
        player_header.addWidget(self.current_file_label)
        player_layout.addLayout(player_header)
        player_controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setMinimumWidth(80)
        player_controls_layout.addWidget(self.play_pause_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumWidth(80)
        player_controls_layout.addWidget(self.stop_button)
        self.current_time_label = QLabel("00:00")
        self.playhead_slider = QSlider(Qt.Orientation.Horizontal)
        self.playhead_slider.sliderPressed.connect(self.slider_pressed)
        self.playhead_slider.sliderMoved.connect(self.seek_audio_on_move)
        self.playhead_slider.sliderReleased.connect(self.slider_released)
        self.playhead_slider.setEnabled(False)
        self.duration_label = QLabel("00:00")
        player_controls_layout.addSpacing(8)
        player_controls_layout.addWidget(self.current_time_label)
        player_controls_layout.addWidget(self.playhead_slider, 1)
        player_controls_layout.addWidget(self.duration_label)
        player_layout.addLayout(player_controls_layout)
        history_label = QLabel("Generated files (double-click to play)")
        history_label.setObjectName("Muted")
        player_layout.addWidget(history_label)
        self.output_log_listwidget = QListWidget()
        self.output_log_listwidget.itemDoubleClicked.connect(
            self.play_selected_from_log)
        self.output_log_listwidget.setMinimumHeight(70)
        self.output_log_listwidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        player_layout.addWidget(self.output_log_listwidget, 1)
        generate_layout.addWidget(player_card, 2)
        self.pages.addWidget(generate_page)

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

    def speed_device(self):
        if self.model is not None:
            return self.device_used
        return "cuda" if torch.cuda.is_available() else "cpu"

    def speed_key(self, entry):
        variant = entry.get("qwen_variant") or entry.get("multilingual_t3_model") or ""
        if self.batch_size_for(entry) > 1:
            variant += f"|batch{self.batch_size_for(entry)}"
        return f"{self.speed_device()}|{entry.get('repo_id')}|{entry.get('backend')}|{variant}"

    def seconds_per_char_for(self, entry):
        """(seconds per character, measured?) for an entry on the current device."""
        device = self.speed_device()
        measured = self.app_settings.get("speed_by_model", {}).get(self.speed_key(entry))
        if measured:
            return measured, True
        engine = entry.get("backend") if entry.get("backend") in ENGINE_MODULES else "chatterbox"
        if engine == "chatterbox":
            legacy = self.app_settings.get("seconds_per_char", {}).get(device)  # older single rate
            if legacy:
                return legacy, False
        return DEFAULT_SECONDS_PER_CHAR.get((engine, device), 0.35), False

    def loaded_entry(self):
        return next((e for e in self.model_entries if self.entry_key(e) == self.loaded_entry_key()), None)

    def batch_size_for(self, entry):
        module = ENGINE_MODULES.get(entry.get("backend"))
        if module is not None and hasattr(module, "BATCH_SIZE") and self.speed_device() == "cuda":
            return module.BATCH_SIZE
        return 1

    def batch_plan(self, entry, lengths):
        """[(first, last, estimated seconds)] for generating sections of these lengths."""
        rate, _measured = self.seconds_per_char_for(entry)
        size = self.batch_size_for(entry)
        budget = ENGINE_MODULES[entry["backend"]].BATCH_CHAR_BUDGET if size > 1 else None
        plan = []
        for start, end in documents.plan_batches(lengths, size, budget):
            batch = lengths[start:end]
            if size > 1:
                cost = max(batch) * (1 + BATCH_COST_SLOPE * len(batch)) * rate
            else:
                cost = sum(batch) * rate
            plan.append((start + 1, start + len(batch), cost))
        return plan

    def estimate_seconds(self, entry, lengths):
        _rate, measured = self.seconds_per_char_for(entry)
        return sum(cost for _first, _last, cost in self.batch_plan(entry, lengths)), measured

    @staticmethod
    def max_section_chars_for(entry):
        if entry.get("backend") in ENGINE_MODULES:
            return ENGINE_MODULES[entry["backend"]].MAX_SECTION_CHARS
        return MAX_TEXT_INPUT_LENGTH

    def split_text(self, text, entry):
        """Section texts the way the entry's engine will generate them."""
        if entry.get("backend") == VIBEVOICE_BACKEND:
            return [section.text for section in
                    documents.plan_script_sections(text, vibevoice_engine.MAX_SECTION_CHARS)]
        return documents.split_into_sections(text, self.max_section_chars_for(entry))

    def section_lengths(self, text, entry):
        return [len(section) for section in self.split_text(text, entry)]

    def model_estimates(self, text):
        """[(entry, seconds, measured, active)] for every model in the switcher, fastest first."""
        rows = []
        active_key = self.loaded_entry_key() if self.model is not None else None
        lengths_by_size = {}
        for entry in self.get_visible_model_entries():
            size = (self.max_section_chars_for(entry), entry.get("backend") == VIBEVOICE_BACKEND)
            if size not in lengths_by_size:
                lengths_by_size[size] = self.section_lengths(text, entry)
            seconds, measured = self.estimate_seconds(entry, lengths_by_size[size])
            rows.append((entry, seconds, measured, self.entry_key(entry) == active_key))
        return sorted(rows, key=lambda row: row[1])

    def update_text_stats(self):
        # Always current, including while a preview or render runs; a running
        # render keeps using the text it started with.
        self.refresh_cast_label()
        text = self.text_input.toPlainText().strip()
        if not text:
            self.estimate_button.setVisible(False)
            self.text_stats_label.setText("Type or paste text, or open a document.")
            return
        entry = self.loaded_entry() or self.get_selected_model_entry()
        lengths = self.section_lengths(text, entry)
        sections = len(lengths)
        seconds, measured = self.estimate_seconds(entry, lengths)
        self.estimate_button.setText(f"About {self.format_duration(seconds)} \u25be")
        self.estimate_button.setVisible(True)
        respelled = self.pronunciations.count_in(text)
        self.text_stats_label.setText(
            f"{sections} section{'s' if sections != 1 else ''} \u00b7 {len(text):,} characters"
            + (f" \u00b7 {respelled} respelled" if respelled else ""))
        self.text_stats_label.setToolTip(
            "Words changed by the pronunciation dictionary (Advanced page)." if respelled else "")
        visible = self.get_visible_model_entries()
        for index in range(self.model_repo_combo.count()):
            position = self.model_repo_combo.itemData(index)
            if isinstance(position, int) and position < len(visible):
                item_seconds, item_measured = self.estimate_seconds(
                    visible[position], self.section_lengths(text, visible[position]))
                self.model_repo_combo.setItemData(
                    index,
                    f"About {self.format_duration(item_seconds)} for the current text"
                    f" ({'measured' if item_measured else 'estimate'})",
                    Qt.ItemDataRole.ToolTipRole)

    def show_estimate_menu(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            return
        menu = QMenu(self)
        header = menu.addAction(f"Time for this text ({len(text):,} characters), by model")
        header.setEnabled(False)
        menu.addSeparator()
        estimates = {self.entry_key(e) + (e["label"],): row
                     for row in self.model_estimates(text) for e in [row[0]]}
        for _capability, title, members in model_registry.group_by_capability(self.get_visible_model_entries()):
            menu.addSection(title)
            rows = sorted((estimates[self.entry_key(e) + (e["label"],)] for e in members), key=lambda r: r[1])
            for entry, seconds, measured, active in rows:
                self._add_estimate_action(menu, entry, seconds, measured, active)
        menu.addSeparator()
        note = menu.addAction("Estimates become measurements once a model has generated a few sections.")
        note.setEnabled(False)
        menu.exec(self.estimate_button.mapToGlobal(self.estimate_button.rect().bottomLeft()))

    def _add_estimate_action(self, menu, entry, seconds, measured, active):
            marker = "\u25cf " if active else "    "
            label = f"{marker}{entry['label']}  \u2014  about {self.format_duration(seconds)}"
            label += "" if measured else "  (estimate)"
            if not active:
                label += "  + load"
            action = menu.addAction(label)
            action.setEnabled(not active and not self.is_generating and not getattr(self, "model_is_loading", False))
            action.triggered.connect(lambda _checked=False, e=entry: self.switch_to_entry(e))

    def switch_to_entry(self, entry):
        index = self.model_repo_combo.findText(entry["label"])
        if index >= 0:
            self.model_repo_combo.setCurrentIndex(index)

    def on_section_timed(self, characters, count, seconds):
        # The first section after a load includes warm-up, so it isn't a fair sample.
        if not self.model_is_warm:
            self.model_is_warm = True
            return
        entry = self.loaded_entry()
        if entry is None or characters < 20:
            return
        weight = characters
        if self.batch_size_for(entry) > 1:
            weight = characters * (1 + BATCH_COST_SLOPE * count)
        measured = seconds / weight
        rates = self.app_settings.setdefault("speed_by_model", {})
        key = self.speed_key(entry)
        previous = rates.get(key)
        rates[key] = round(measured if previous is None else 0.7 * previous + 0.3 * measured, 5)

    @staticmethod
    def format_clock(seconds):
        minutes, seconds = divmod(int(round(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    @staticmethod
    def format_duration(seconds):
        seconds = int(round(seconds))
        if seconds < 60:
            return f"{max(seconds, 1)} s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes} min {seconds:02d} s" if minutes < 10 else f"{minutes} min"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} h {minutes:02d} min"

    def _create_slider(self, min_val, max_val, step_val, default_val, value_format="{:.2f}"):
        return SliderWithValue(min_val, max_val, step_val, default_val, value_format)

    # --- Finishing touches ---

    def current_finishing_settings(self):
        return audio_effects.FinishingSettings(
            speed=round(self.speed_slider.get_value(), 2),
            pitch_semitones=round(self.pitch_slider.get_value(), 1),
            paragraph_pause=round(self.pause_slider.get_value(), 1),
            even_volume=self.even_volume_checkbox.isChecked(),
            trim_silence=self.trim_silence_checkbox.isChecked(),
            output_format="MP3" if self.mp3_checkbox.isChecked() else self.output_format_combo.currentText(),
            save_subtitles=self.subtitles_checkbox.isChecked(),
            subtitle_format=self.subtitle_format_combo.currentText(),
        )

    def apply_finishing_settings(self, settings):
        self.speed_slider.set_value(settings.speed)
        self.pitch_slider.set_value(settings.pitch_semitones)
        self.pause_slider.set_value(settings.paragraph_pause)
        self.even_volume_checkbox.setChecked(settings.even_volume)
        self.trim_silence_checkbox.setChecked(settings.trim_silence)
        self.mp3_checkbox.setChecked(settings.output_format == "MP3")
        if settings.output_format in LOSSLESS_FORMATS:
            self.output_format_combo.setCurrentText(settings.output_format)
        self.subtitles_checkbox.setChecked(settings.save_subtitles)
        self.subtitle_format_combo.setCurrentText(settings.subtitle_format)
        self.update_finishing_summary()

    def reset_finishing(self):
        """Finishing touches only; Advanced effects are left as they are."""
        defaults = audio_effects.FinishingSettings()
        self.pause_slider.set_value(defaults.paragraph_pause)
        self.even_volume_checkbox.setChecked(defaults.even_volume)
        self.trim_silence_checkbox.setChecked(defaults.trim_silence)
        self.output_format_combo.setCurrentText(defaults.output_format)
        self.subtitles_checkbox.setChecked(defaults.save_subtitles)
        self.update_finishing_summary()

    def on_mp3_toggled(self, checked):
        self.output_format_combo.setEnabled(not checked)
        self.output_format_combo.setToolTip(
            "MP3 is selected on the Advanced page." if checked else
            "WAV is uncompressed; FLAC is lossless and about half the size. "
            "MP3 is on the Advanced page.")
        self.update_finishing_summary()

    def update_finishing_summary(self, *_args):
        if not hasattr(self, "subtitle_format_combo"):
            return  # Advanced page not built yet
        settings = self.current_finishing_settings()
        summary = settings.summary()
        advanced = (abs(settings.speed - 1.0) > 1e-6 or abs(settings.pitch_semitones) > 1e-6
                    or settings.output_format == "MP3")
        self.finishing_summary_label.setText(summary + ("  (effects on Advanced page)" if advanced else ""))

    def set_finishing_expanded(self, expanded):
        self.finishing_panel.setVisible(expanded)
        arrow = "\u25be" if expanded else "\u25b8"
        self.finishing_toggle.setText(f"{arrow} Finishing touches")
        self.finishing_summary_label.setVisible(not expanded)
        if self.isVisible():
            self.update_minimum_size()

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

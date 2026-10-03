"""The Voice page: reference clips, recording and the voice library."""

import datetime
import os
import wave

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabBar,
)

from this_voice_thing.core import model_registry, voice_library
from this_voice_thing.engines import kokoro as kokoro_engine
from this_voice_thing.ui import tiles as model_tiles
from this_voice_thing.ui.common import (
    DUAL_MODE_BACKENDS,
    KOKORO_BACKEND,
    MAX_RECORDING_SECONDS,
    MIN_RECORDING_SECONDS,
    REFERENCE_READING_SCRIPTS,
    SILENT_RECORDING_PEAK,
)
from this_voice_thing.ui.dialogs.recording import pcm_to_mono_float, RecordingDialog
from this_voice_thing.ui.dialogs.voices import VoiceDetailsDialog
from this_voice_thing.ui.threads import MakeClipThread
from this_voice_thing.ui.widgets import dialog_accepted


class VoicePage:
    """The Voice page: reference clips, recording and the voice library. Mixed into ChatterboxApp."""

    def _build_voice_page(self):
        voice_page, voice_layout = self._make_page(
            "Voice", "Your voice library: recordings, clips, presets and designed voices.")

        current_card, current_layout = self._make_card("Current voice")
        current_row = QHBoxLayout()
        self.ref_audio_path_label = QLabel("None selected.")
        self.ref_audio_path_label.setWordWrap(False)
        self.ref_audio_path_label.setTextFormat(Qt.TextFormat.PlainText)
        # Long file names are clipped (full path in the tooltip) instead of widening the window.
        self.ref_audio_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        current_row.addWidget(self.ref_audio_path_label, 1)
        self.preview_reference_button = QPushButton("Preview")
        self.preview_reference_button.clicked.connect(self.toggle_reference_preview)
        current_row.addWidget(self.preview_reference_button)
        self.clear_reference_button = QPushButton("Use default voice")
        self.clear_reference_button.clicked.connect(self.clear_reference_audio)
        current_row.addWidget(self.clear_reference_button)
        save_current_button = QPushButton("Save to library...")
        save_current_button.setToolTip("Save the voice you're using: a clip, a preset or a designed voice.")
        save_current_button.clicked.connect(self.save_current_voice)
        current_row.addWidget(save_current_button)
        current_layout.addLayout(current_row)
        voice_layout.addWidget(current_card)

        library_card, library_layout = self._make_card()
        library_header = QHBoxLayout()
        self.voice_filter_tabs = QTabBar()
        self.voice_filter_tabs.setObjectName("CapabilityTabs")
        self.voice_filter_tabs.setDrawBase(False)
        self.voice_filter_tabs.setExpanding(False)
        self.voice_filter_tabs.setUsesScrollButtons(False)
        self.voice_filter_tabs.setCursor(Qt.CursorShape.PointingHandCursor)
        for key, title in (("", "All"), ("clip", "Clips"), ("preset", "Presets"), ("design", "Designed")):
            self.voice_filter_tabs.setTabData(self.voice_filter_tabs.addTab(title), key)
        self.voice_filter_tabs.currentChanged.connect(lambda _index: self.render_voice_tiles())
        library_header.addWidget(self.voice_filter_tabs)
        library_header.addStretch(1)
        self.voice_search = QLineEdit()
        self.voice_search.setPlaceholderText("Search voices")
        self.voice_search.setToolTip("Matches names, tags and notes.")
        self.voice_search.setClearButtonEnabled(True)
        self.voice_search.setFixedWidth(170)
        self.voice_search.textChanged.connect(lambda _text: self.render_voice_tiles())
        library_header.addWidget(self.voice_search)
        library_layout.addLayout(library_header)
        library_hint = QLabel("Click a voice to use it. Clip voices work with every cloning model; "
                              "presets and designed voices load their model.")
        library_hint.setObjectName("Muted")
        library_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        library_layout.addWidget(library_hint)
        self.voice_tiles = model_tiles.TileArea(min_rows=1)
        library_layout.addWidget(self.voice_tiles, 1)
        library_actions = QHBoxLayout()
        self.mic_combo = QComboBox()
        self.mic_combo.setToolTip("Microphone used for Record...")
        self.mic_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mic_combo.setMinimumContentsLength(10)
        library_actions.addWidget(self.mic_combo, 1)
        self.record_button = self._accent(QPushButton("Record..."))
        self.record_button.setToolTip(
            "Record a new clip voice: read about 15 seconds in a quiet room; the first few seconds "
            f"matter most ({MIN_RECORDING_SECONDS}-{MAX_RECORDING_SECONDS} s).")
        self.record_button.clicked.connect(self.open_recording_dialog)
        library_actions.addWidget(self.record_button)
        self.media_devices.audioInputsChanged.connect(self.populate_microphones)
        self.populate_microphones()
        browse_ref_button = QPushButton("Add a file...")
        browse_ref_button.setToolTip("Add a .wav, .mp3 or .flac clip to the library and use it.")
        browse_ref_button.clicked.connect(self.browse_reference_audio)
        library_actions.addWidget(browse_ref_button)
        open_recordings_button = self._link(QPushButton("Open folder"))
        open_recordings_button.clicked.connect(self.open_recordings_folder)
        library_actions.addWidget(open_recordings_button)
        library_layout.addLayout(library_actions)
        voice_layout.addWidget(library_card, 1)
        self.pages.addWidget(voice_page)

    def browse_reference_audio(self):
        default_dir = self.last_reference_audio_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", default_dir, "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            voice = self.voice_library.add_clip(file_path)
            self.active_voice_id = voice.id
            self.set_reference_audio(file_path)
            self.last_reference_audio_dir = os.path.dirname(file_path)
            self.render_voice_tiles()
            self.set_status_message(f"Status: Added {voice.name} to the voice library and selected it.")

    @staticmethod
    def transcript_path(audio_path):
        return os.path.splitext(audio_path)[0] + ".txt"

    def load_reference_transcript(self, audio_path):
        text = ""
        if audio_path and os.path.exists(self.transcript_path(audio_path)):
            with open(self.transcript_path(audio_path), encoding="utf-8") as handle:
                text = handle.read().strip()
        self.qwen_transcript_input.setText(text)

    def save_reference_transcript(self):
        audio_path = self.ref_audio_path_label.toolTip()
        if not audio_path:
            return
        text = self.qwen_transcript_input.text().strip()
        path = self.transcript_path(audio_path)
        try:
            if text:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text + "\n")
            elif os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f"Could not save transcript for {os.path.basename(audio_path)}: {exc}")

    # --- Voice selection ---

    def set_reference_audio(self, path):
        if path:
            saved = self.voice_library.find_clip(path) if hasattr(self, "voice_library") else None
            name = saved.name if saved else os.path.basename(path)
            self.ref_audio_path_label.setText(name)
            self.ref_audio_path_label.setToolTip(path)
            self.voice_chip.setText(name)
            self.voice_chip.setToolTip(path)
        else:
            self.ref_audio_path_label.setText("Default voice (no reference clip)")
            self.ref_audio_path_label.setToolTip("")
            self.voice_chip.setText("Default voice")
            self.voice_chip.setToolTip("The model's built-in voice. Pick a reference clip on the Voice page to clone a voice.")
        self.preview_reference_button.setEnabled(bool(path))
        self.clear_reference_button.setEnabled(bool(path))
        if hasattr(self, "qwen_transcript_input"):
            self.load_reference_transcript(path)
            self.refresh_voice_chip()
        self.render_voice_tiles()

    def clear_reference_audio(self):
        self.stop_reference_preview()
        self.set_reference_audio(None)
        self.set_status_message("Status: Using the default voice.")

    def refresh_recordings_list(self):
        """Bring new recordings into the library and redraw it."""
        self.voice_library.import_recordings()
        self.render_voice_tiles()

    def _start_reference_preview(self, path, button):
        self.stop_reference_preview()
        self.preview_button_playing = button
        self.preview_player.setSource(QUrl.fromLocalFile(path))
        self.preview_player.play()
        button.setText("Stop preview")

    def stop_reference_preview(self):
        self.preview_player.stop()

    def _on_preview_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState and self.preview_button_playing:
            self.preview_reference_button.setText("Preview")
            self.preview_button_playing = None

    def toggle_reference_preview(self):
        if self.preview_button_playing is self.preview_reference_button:
            self.stop_reference_preview()
            return
        path = self.ref_audio_path_label.toolTip()
        if path:
            self._start_reference_preview(path, self.preview_reference_button)


    def open_recordings_folder(self):
        os.makedirs(self.recordings_directory, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.recordings_directory))

    # --- Voice library ---

    def voice_is_active(self, voice):
        if voice.kind == "clip":
            path = self.voice_library.clip_path(voice)
            return bool(path) and os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                os.path.abspath(self.ref_audio_path_label.toolTip() or "~none~"))
        model = self.active_qwen_model()
        return (voice.id == self.active_voice_id and model is not None
                and getattr(model, "backend", "") == voice.backend)

    def render_voice_tiles(self):
        if not hasattr(self, "voice_tiles"):
            return
        kind = self.voice_filter_tabs.tabData(self.voice_filter_tabs.currentIndex())
        query = self.voice_search.text().strip().lower()
        voices = [voice for voice in self.voice_library.voices
                  if (not kind or voice.kind == kind or (kind == "clip" and voice.has_clip))
                  and (not query or query in " ".join([voice.name, voice.notes, *voice.tags]).lower())]
        voices.sort(key=lambda voice: voice.created, reverse=True)
        counts = {key: sum(1 for voice in self.voice_library.voices
                           if not key or voice.kind == key or (key == "clip" and voice.has_clip))
                  for key in ("", "clip", "preset", "design")}
        for index in range(self.voice_filter_tabs.count()):
            key = self.voice_filter_tabs.tabData(index)
            title = {"": "All", "clip": "Clips", "preset": "Presets", "design": "Designed"}[key]
            self.voice_filter_tabs.setTabText(index, f"{title}  {counts[key]}" if counts[key] else title)
        empty = ("No voices match." if query or kind else
                 "No voices yet. Record one, add a file, or save the voice you're using.")
        self.voice_tiles.set_tiles([self.voice_tile(voice) for voice in voices], empty)

    def voice_tile(self, voice):
        library = self.voice_library
        active = self.voice_is_active(voice)
        engine = model_registry.ENGINES.get(voice.backend)
        path = library.clip_path(voice)
        if voice.kind == "clip":
            exists = os.path.exists(path)
            seconds = voice_library.clip_seconds(path) if exists else 0
            subtitle = f"Clip \u00b7 {seconds:.0f} s" if exists else "Clip \u00b7 file missing"
            transcript = voice_library.read_transcript(path) if exists else ""
            detail = f"\u201c{transcript}\u201d" if transcript else "No transcript"
        elif voice.kind == "preset":
            label = kokoro_engine.voice_label(voice.speaker) if voice.backend == KOKORO_BACKEND \
                else voice.speaker.replace("_", " ").title()
            subtitle = f"Preset \u00b7 {engine.label if engine else voice.backend} \u00b7 {label}"
            detail = voice.style or voice.notes or "Built-in voice"
        else:
            subtitle = f"Designed \u00b7 {engine.label if engine else voice.backend}"
            detail = voice.description
        badges = [("active", "In use", "This is the voice you're using.")] if active else []
        if voice.kind != "clip" and voice.has_clip:
            badges.append(("status", "Has clip", "Also usable as a clip voice by cloning models."))
        badges += [("status", tag, "Tag") for tag in voice.tags[:2]]
        tooltip = "\n".join(line for line in (voice.name, subtitle, detail if voice.kind != "clip" else "",
                                               voice.notes, path, "Click to use. Right-click for more.") if line)
        tile = model_tiles.ModelTile(voice.name, subtitle, badges, detail, tooltip, active=active, with_menu=True)
        tile.clicked.connect(lambda v=voice: self.use_voice(v))
        tile.menu_requested.connect(lambda pos, v=voice: self.show_voice_menu(v, pos))
        return tile

    def show_voice_menu(self, voice, pos):
        menu = QMenu(self)
        path = self.voice_library.clip_path(voice)
        playing = self.preview_button_playing is self.voice_tiles and getattr(self, "previewing_voice", None) is voice
        entries = [
            ("Use", lambda: self.use_voice(voice), True),
            ("Stop preview" if playing else "Preview clip", lambda: self.preview_voice(voice),
             bool(path) and os.path.exists(path)),
        ]
        if path and os.path.exists(path):
            has_text = bool(voice_library.read_transcript(path))
            entries.append(("Transcribe again" if has_text else "Transcribe clip",
                            lambda: self.transcribe_voice_clip(voice), True))
        if voice.kind != "clip":
            entries.append(("Use as a clip voice", lambda: self.use_voice(voice, as_clip=True),
                            bool(path) and os.path.exists(path)))
            entries.append(("Make clip..." if not voice.has_clip else "Remake clip...",
                            lambda: self.make_voice_clip(voice), not self.model_busy()))
        entries += [None, ("Edit...", lambda: self.edit_voice(voice), True),
                    ("Show file in folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path))),
                     bool(path) and os.path.exists(path)),
                    None, ("Remove from library...", lambda: self.remove_voice(voice), True)]
        for item in entries:
            if item is None:
                menu.addSeparator()
                continue
            text, callback, enabled = item
            action = menu.addAction(text)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False, callback=callback: callback())
        menu.exec(pos)

    def preview_voice(self, voice):
        if self.preview_button_playing is self.voice_tiles and getattr(self, "previewing_voice", None) is voice:
            self.stop_reference_preview()
            return
        self.stop_reference_preview()
        self.previewing_voice = voice
        self.preview_button_playing = self.voice_tiles
        self.preview_player.setSource(QUrl.fromLocalFile(self.voice_library.clip_path(voice)))
        self.preview_player.play()

    def entry_for_voice(self, voice):
        for entry in self.model_entries:
            if entry.get("backend") != voice.backend or entry.get("repo_id") != voice.repo_id:
                continue
            if voice.backend in DUAL_MODE_BACKENDS and model_registry.entry_mode(entry) != (voice.mode or "design"):
                continue
            return entry
        return None

    def use_voice(self, voice, as_clip=False):
        path = self.voice_library.clip_path(voice)
        if voice.kind == "clip" or as_clip:
            if not os.path.exists(path):
                QMessageBox.warning(self, "Voice", f"The clip for {voice.name} is missing:\n{path}")
                return
            self.set_reference_audio(path)
            self.active_voice_id = voice.id
            model = self.active_qwen_model()
            note = ""
            if model is not None and model.mode not in ("base",):
                note = " It's used by cloning models; the loaded model doesn't clone."
            self.set_status_message(f"Status: Voice set to {voice.name}.{note}")
            self.render_voice_tiles()
            return
        entry = self.entry_for_voice(voice)
        if entry is None:
            engine = model_registry.ENGINES.get(voice.backend)
            QMessageBox.information(
                self, "Voice", f"{voice.name} needs {engine.label if engine else voice.backend} "
                f"({voice.repo_id}), which isn't in your model list. Add it on the Model page first.")
            return
        if self.model_busy():
            self.set_status_message("Status: Wait for the current load or generation to finish.")
            return
        self.pending_voice = voice
        if self.is_active_entry(entry):
            self.apply_pending_voice()
        else:
            self.set_status_message(f"Status: Loading {entry['label']} for {voice.name}...")
            self.load_entry(entry)

    def apply_pending_voice(self):
        voice, self.pending_voice = self.pending_voice, None
        model = self.active_qwen_model()
        if voice is None or model is None or getattr(model, "backend", "") != voice.backend:
            return
        if voice.language:
            index = self.language_combo.findData(voice.language)
            if index >= 0:
                self.language_combo.setCurrentIndex(index)
        if voice.kind == "preset":
            if isinstance(model, kokoro_engine.KokoroModel):
                language = voice.language or kokoro_engine.voice_language(voice.speaker) or "en"
                self.kokoro_settings.setdefault("voice_by_language", {})[language] = voice.speaker
            else:
                self.qwen_settings.update(speaker=voice.speaker, style=voice.style)
        else:
            self.engine_settings(model)["description"] = voice.description
            path = self.voice_library.clip_path(voice)
            if hasattr(model, "locked_anchor"):
                model.locked_anchor = None
                self.locked_voice_name = None
            if voice.has_clip and os.path.exists(path) and hasattr(model, "locked_anchor"):
                model.locked_anchor = (path, voice_library.read_transcript(path))
                self.locked_description = voice.description
                self.locked_voice_name = voice.name
        self.update_engine_controls()
        if voice.kind == "preset":
            index = self.qwen_speaker_combo.findData(voice.speaker)
            if index >= 0:
                self.qwen_speaker_combo.setCurrentIndex(index)
        self.active_voice_id = voice.id
        self.set_status_message(f"Status: Voice set to {voice.name}.")
        self.refresh_voice_chip()
        self.render_voice_tiles()

    def current_voice_spec(self):
        """(Voice, error): the voice in use, as an unsaved library voice."""
        model = self.active_qwen_model()
        entry = self.loaded_entry()
        language = self.language_combo.currentData() or ""
        if model is not None and model.mode == "conversation":
            return None, ("A conversation uses a cast of voices. Save each speaker's voice as a clip "
                          "voice instead (record it, or add the file), then pick it in Cast\u2026.")
        if model is not None and entry is not None and model.mode in ("custom_voice", "preset"):
            speaker = self.qwen_speaker_combo.currentData() or ""
            style = self.qwen_instruct_input.text().strip() if model.mode == "custom_voice" else ""
            name = self.qwen_speaker_combo.currentText().split(" (")[0]
            return voice_library.Voice(name=name, kind="preset", backend=entry["backend"], repo_id=entry["repo_id"],
                                       speaker=speaker, style=style, language=language), None
        if model is not None and entry is not None and model.mode == "voice_design":
            description = self.qwen_instruct_input.text().strip()
            if not description:
                return None, "Describe the voice first (Voice description, in the Delivery card)."
            return voice_library.Voice(name="Designed voice", kind="design", backend=entry["backend"],
                                       repo_id=entry["repo_id"], mode=model_registry.entry_mode(entry)
                                       if entry["backend"] in DUAL_MODE_BACKENDS else "",
                                       description=description, language=language), None
        path = self.ref_audio_path_label.toolTip()
        if not path:
            return None, "You're using the model's default voice. Record a clip or add a file to save a voice."
        existing = self.voice_library.find_clip(path)
        if existing:
            return existing, None
        return voice_library.Voice(name=os.path.splitext(os.path.basename(path))[0], kind="clip",
                                   clip=self.voice_library.to_stored(path)), None

    def save_current_voice(self):
        voice, problem = self.current_voice_spec()
        if problem:
            QMessageBox.information(self, "Save Voice", problem)
            return
        if voice in self.voice_library.voices:
            self.edit_voice(voice)
            return
        transcript = None
        if voice.kind == "clip":
            transcript = voice_library.read_transcript(self.voice_library.clip_path(voice))
        dialog = VoiceDetailsDialog("Save voice", voice, self.voice_library, transcript,
                                    offer_clip=voice.kind != "clip", parent=self)
        if not dialog_accepted(dialog.exec()):
            return
        transcript = dialog.apply()
        self.voice_library.add(voice)
        if transcript is not None:
            voice_library.write_transcript(self.voice_library.clip_path(voice), transcript)
            self.load_reference_transcript(self.voice_library.clip_path(voice))
        self.active_voice_id = voice.id
        self.set_status_message(f"Status: Saved {voice.name} to the voice library.")
        self.render_voice_tiles()
        if dialog.make_clip():
            self.make_voice_clip(voice)

    def edit_voice(self, voice):
        path = self.voice_library.clip_path(voice)
        transcript = voice_library.read_transcript(path) if path and os.path.exists(path) else None
        dialog = VoiceDetailsDialog("Edit voice", voice, self.voice_library, transcript, parent=self)
        if not dialog_accepted(dialog.exec()):
            return
        transcript = dialog.apply()
        if transcript is not None:
            voice_library.write_transcript(path, transcript)
            if self.voice_is_active(voice):
                self.load_reference_transcript(path)
        self.voice_library.save()
        self.render_voice_tiles()

    def remove_voice(self, voice):
        path = self.voice_library.clip_path(voice)
        owned = bool(path) and os.path.abspath(path).startswith(os.path.abspath(self.voice_library.clips_dir))
        detail = ("Its clip, made by the library, is deleted too." if owned else
                  "The audio file stays where it is." if path else "")
        answer = QMessageBox.question(self, "Remove Voice", f"Remove {voice.name} from the library?\n\n{detail}")
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.preview_button_playing is self.voice_tiles:
            self.stop_reference_preview()
        if owned and self.voice_is_active(voice):
            self.set_reference_audio(None)
        self.voice_library.remove(voice)
        self.render_voice_tiles()

    def make_voice_clip(self, voice):
        """Generate a clip of a preset or designed voice reading a passage, so cloning
        models can use it. The voice has to be the one in use (it is loaded first)."""
        if not self.voice_is_active(voice):
            self.use_voice(voice)
            if not self.voice_is_active(voice):
                self.pending_clip_voice = voice  # made once the model has loaded
                return
        problem = self.prepare_qwen_generation()
        if problem:
            QMessageBox.information(self, "Make Clip", problem)
            return
        language = self.language_combo.currentData() or "en"
        passage = REFERENCE_READING_SCRIPTS[0]
        self.set_status_message(f"Status: Making a clip of {voice.name}...")
        self.generate_button.setEnabled(False)
        self.clip_thread = MakeClipThread(self.model, passage, language, self)
        self.clip_thread.finished_with.connect(
            lambda wav, sr, error, v=voice, text=passage: self.on_voice_clip_made(v, text, wav, sr, error))
        self.clip_thread.start()

    def on_voice_clip_made(self, voice, text, wav, sr, error):
        self.generate_button.setEnabled(self.model is not None)
        if error or wav is None:
            self.set_status_message("Status: Could not make the clip. See the Log page.")
            QMessageBox.warning(self, "Make Clip", f"Could not make the clip:\n{error}")
            return
        old = self.voice_library.clip_path(voice)
        path = self.voice_library.new_clip_path(voice.name)
        import soundfile
        soundfile.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        voice_library.write_transcript(path, text)
        voice.clip = self.voice_library.to_stored(path)
        self.voice_library.save()
        if old and old != path and old.startswith(os.path.abspath(self.voice_library.clips_dir)):
            for target in (old, voice_library.transcript_path(old)):
                if os.path.exists(target):
                    os.remove(target)
        self.set_status_message(f"Status: Made a {len(wav) / sr:.0f} s clip of {voice.name}. "
                                "Cloning models can use it now.")
        self.render_voice_tiles()

    # --- Reference audio recording ---

    def populate_microphones(self):
        previous = self.mic_combo.currentData()
        previous_id = previous.id() if previous is not None else None
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        default_id = QMediaDevices.defaultAudioInput().id()
        for device in QMediaDevices.audioInputs():
            label = device.description()
            if device.id() == default_id:
                label += " (default)"
            self.mic_combo.addItem(label, device)
        selected_index = 0
        for index in range(self.mic_combo.count()):
            device_id = self.mic_combo.itemData(index).id()
            if device_id == previous_id or (previous_id is None and device_id == default_id):
                selected_index = index
                break
        self.mic_combo.setCurrentIndex(selected_index)
        self.mic_combo.blockSignals(False)
        has_inputs = self.mic_combo.count() > 0
        if not has_inputs:
            self.mic_combo.addItem("No microphone found")
        self.mic_combo.setEnabled(has_inputs)
        self.record_button.setEnabled(has_inputs)

    def open_recording_dialog(self):
        device = self.mic_combo.currentData()
        if device is None or device.isNull():
            QMessageBox.warning(self, "No Microphone",
                                "No audio input device is available.")
            return
        dialog = RecordingDialog(device, self)
        if not dialog_accepted(dialog.exec()):
            self.set_status_message("Status: Recording cancelled.")
            return
        self.recording_format = dialog.audio_format
        self.recording_buffer = dialog.recorded_bytes
        self.recording_script = dialog.script_label.text()
        self._save_recording()
        self.recording_buffer = bytearray()

    def _save_recording(self):
        audio_format = self.recording_format
        mono = pcm_to_mono_float(bytes(self.recording_buffer), audio_format)
        duration = mono.size / audio_format.sampleRate()
        if duration < MIN_RECORDING_SECONDS:
            self.set_status_message("Status: Recording discarded (too short).")
            QMessageBox.warning(
                self, "Recording Too Short",
                f"The recording was {duration:.1f}s. Please record at least "
                f"{MIN_RECORDING_SECONDS}s; about 10-15s of clear speech works best.")
            return
        os.makedirs(self.recordings_directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.recordings_directory, f"reference_{timestamp}.wav")
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio_format.sampleRate())
            wav_file.writeframes(pcm16.tobytes())
        script = getattr(self, "recording_script", "")
        if script:
            with open(self.transcript_path(output_path), "w", encoding="utf-8") as handle:
                handle.write(script + "\n")
        self.voice_library.import_recordings()
        self.active_voice_id = getattr(self.voice_library.find_clip(output_path), "id", None)
        self.set_reference_audio(output_path)
        self.refresh_recordings_list()
        self.last_reference_audio_dir = self.recordings_directory
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        print(f"Saved reference recording ({duration:.1f}s, peak {peak:.3f}): {output_path}")
        self.set_status_message(
            f"Status: Recorded {duration:.1f}s reference clip and selected it.")
        if peak < SILENT_RECORDING_PEAK:
            QMessageBox.warning(
                self, "Recording Is Nearly Silent",
                "The clip was saved and selected, but almost no sound was captured.\n\n"
                "Check that the right microphone is selected, that it isn't muted, and that "
                "Windows allows desktop apps to use it (Settings > Privacy & security > Microphone).")

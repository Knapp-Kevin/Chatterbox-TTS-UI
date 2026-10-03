"""The Voice page: the current voice and its reference clip."""

import os

from PySide6.QtCore import QUrl
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy


class VoicePage:
    """The Voice page and the reference clip. Mixed into ChatterboxApp."""

    """The Voice page: reference clips, recording and the voice library. Mixed into ChatterboxApp."""

    def _build_voice_page(self):
        voice_page, voice_layout = self._make_page(
            "Voice", "Your voice library: recordings, clips, presets and designed voices.")

        voice_layout.addWidget(self._build_current_voice_card())
        voice_layout.addWidget(self._build_library_card(), 1)
        self.pages.addWidget(voice_page)

    def _build_current_voice_card(self):
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
        return current_card

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

"""The guided reference-clip recording window."""

import time
from collections import deque

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtMultimedia import QAudioFormat, QAudioSource
from PySide6.QtWidgets import QDialog, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from this_voice_thing.ui import theme as ui_theme
from this_voice_thing.ui.common import (
    MAX_RECORDING_SECONDS,
    MIN_RECORDING_SECONDS,
    RECORDING_SAMPLE_RATE,
    REFERENCE_READING_SCRIPTS,
)
from this_voice_thing.ui.widgets import dialog_accepted, LevelHistoryWidget


# --- Reference audio recording ---


def choose_recording_format(device):
    audio_format = QAudioFormat()
    audio_format.setSampleRate(RECORDING_SAMPLE_RATE)
    audio_format.setChannelCount(1)
    audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    if device.isFormatSupported(audio_format):
        return audio_format
    return device.preferredFormat()


def pcm_to_mono_float(data, audio_format):
    sample_format = audio_format.sampleFormat()
    dtypes = {
        QAudioFormat.SampleFormat.UInt8: (np.uint8, 128.0, 128.0),
        QAudioFormat.SampleFormat.Int16: (np.int16, 0.0, 32768.0),
        QAudioFormat.SampleFormat.Int32: (np.int32, 0.0, 2147483648.0),
        QAudioFormat.SampleFormat.Float: (np.float32, 0.0, 1.0),
    }
    if sample_format not in dtypes:
        raise ValueError(f"Unsupported microphone sample format: {sample_format}")
    dtype, offset, scale = dtypes[sample_format]
    channels = max(1, audio_format.channelCount())
    frame_size = np.dtype(dtype).itemsize * channels
    data = data[:len(data) - len(data) % frame_size]
    samples = np.frombuffer(data, dtype=dtype).astype(np.float32)
    samples = (samples - offset) / scale
    return samples.reshape(-1, channels).mean(axis=1)


class RecordingDialog(QDialog):
    """Modal recorder: countdown with live mic check, then timed capture."""

    COUNTDOWN_SECONDS = 3
    TICK_MS = 50

    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Reference Audio")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.device = device
        self.audio_format = choose_recording_format(device)
        self.recorded_bytes = bytearray()
        self.audio_source = None
        self.audio_io = None
        self.phase = "countdown"
        self.countdown_started = None
        self.recent_peaks = deque(maxlen=int(1500 / self.TICK_MS))

        layout = QVBoxLayout(self)
        # Let wrapped labels grow the dialog instead of being clipped.
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        mic_label = QLabel(f"Microphone: {device.description()}")
        mic_label.setTextFormat(Qt.TextFormat.PlainText)
        mic_label.setStyleSheet("color: gray;")
        layout.addWidget(mic_label)

        self.phase_label = QLabel("Get ready...")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setObjectName("RecordingPhase")
        layout.addWidget(self.phase_label)

        self.big_label = QLabel(str(self.COUNTDOWN_SECONDS))
        self.big_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.big_label.setObjectName("RecordingClock")
        layout.addWidget(self.big_label)

        self.hint_label = QLabel(
            f"Read the text below at your normal pace (about 15 seconds). Recording "
            f"stops automatically at {MAX_RECORDING_SECONDS} seconds.")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self.script_group = script_group = QGroupBox("Read this aloud")
        script_layout = QVBoxLayout(script_group)
        self.script_index = 0
        self.script_label = QLabel()
        self.script_label.setWordWrap(True)
        self.script_label.setTextFormat(Qt.TextFormat.PlainText)
        self.script_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.script_label.setObjectName("ReadAloud")
        script_layout.addWidget(self.script_label)
        script_footer = QHBoxLayout()
        script_note = QLabel("Read with natural expression. Any language works.")
        script_note.setStyleSheet("color: gray;")
        self.next_script_button = QPushButton("Different text")
        self.next_script_button.clicked.connect(self.show_next_script)
        script_footer.addWidget(script_note, 1)
        script_footer.addWidget(self.next_script_button)
        script_layout.addLayout(script_footer)
        layout.addWidget(script_group)
        self.show_script(0)

        self.level_widget = LevelHistoryWidget(parent=self)
        layout.addWidget(self.level_widget)

        self.level_status_label = QLabel("Checking microphone...")
        self.level_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.level_status_label)

        self.time_bar = QProgressBar()
        self.time_bar.setRange(0, MAX_RECORDING_SECONDS * 1000)
        self.time_bar.setValue(0)
        self.time_bar.setTextVisible(False)
        self.time_bar.setFixedHeight(8)
        layout.addWidget(self.time_bar)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.stop_button = QPushButton("Stop && Use")
        self.stop_button.setEnabled(False)
        self.stop_button.setDefault(True)
        self.stop_button.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        QTimer.singleShot(0, self._open_microphone)

    def show_script(self, index):
        self.script_index = index % len(REFERENCE_READING_SCRIPTS)
        self.script_label.setText(REFERENCE_READING_SCRIPTS[self.script_index])

    def show_next_script(self):
        self.show_script(self.script_index + 1)

    def _open_microphone(self):
        # Open during the countdown: Bluetooth headsets need a moment to switch
        # profiles, and the live meter doubles as a mic check.
        self.audio_source = QAudioSource(self.device, self.audio_format, self)
        self.audio_io = self.audio_source.start()
        error = self.audio_source.error()
        if self.audio_io is None or getattr(error, "name", "NoError") != "NoError":
            self._close_microphone()
            self.phase = "error"
            self.phase_label.setText("Microphone unavailable")
            self.big_label.setText("!")
            self.hint_label.setText(
                f"Could not open '{self.device.description()}' "
                f"({getattr(error, 'name', error)}). Check that it is connected and awake, "
                "and that Windows allows desktop apps to use the microphone "
                "(Settings > Privacy & security > Microphone).")
            self.level_status_label.setText("")
            self.script_group.setVisible(False)
            self.level_widget.setVisible(False)
            self.time_bar.setVisible(False)
            self.cancel_button.setText("Close")
            return
        print(
            f"Recording dialog opened '{self.device.description()}' "
            f"({self.audio_format.sampleRate()} Hz, {self.audio_format.channelCount()} ch).")
        self.countdown_started = time.monotonic()
        self.level_widget.set_active(False)
        self.timer.start(self.TICK_MS)

    def _close_microphone(self):
        self.timer.stop()
        if self.audio_source is not None:
            self.audio_source.stop()
            self.audio_source.deleteLater()
        self.audio_source = None
        self.audio_io = None

    def recorded_seconds(self):
        bytes_per_frame = self.audio_format.bytesPerFrame()
        if not bytes_per_frame:
            return 0.0
        return len(self.recorded_bytes) / bytes_per_frame / self.audio_format.sampleRate()

    def _tick(self):
        chunk = bytes(self.audio_io.readAll().data()) if self.audio_io is not None else b""
        peak = 0.0
        if chunk:
            samples = pcm_to_mono_float(chunk, self.audio_format)
            if samples.size:
                peak = float(np.max(np.abs(samples)))
        self.level_widget.push(peak)
        self.recent_peaks.append(peak)
        self._update_level_status()

        if self.phase == "countdown":
            remaining = self.COUNTDOWN_SECONDS - (time.monotonic() - self.countdown_started)
            if remaining > 0:
                self.big_label.setText(str(int(remaining) + 1))
                return
            # Countdown audio is a mic check only; capture starts now.
            self.phase = "recording"
            self.phase_label.setText("Recording")
            ui_theme.set_tone(self.phase_label, "error")  # red "Recording", like a tally light
            self.level_widget.set_active(True)
            self.next_script_button.setEnabled(False)
            return

        self.recorded_bytes.extend(chunk)
        elapsed = self.recorded_seconds()
        self.big_label.setText(
            f"{int(elapsed) // 60}:{int(elapsed) % 60:02d} / "
            f"{MAX_RECORDING_SECONDS // 60}:{MAX_RECORDING_SECONDS % 60:02d}")
        self.time_bar.setValue(int(min(elapsed, MAX_RECORDING_SECONDS) * 1000))
        if elapsed >= MIN_RECORDING_SECONDS:
            self.stop_button.setEnabled(True)
            self.hint_label.setText("Click Stop & Use when you're done.")
        else:
            self.hint_label.setText(
                f"Keep talking - at least {MIN_RECORDING_SECONDS} seconds are needed.")
        if elapsed >= MAX_RECORDING_SECONDS:
            self.accept()

    def _update_level_status(self):
        recent = max(self.recent_peaks) if self.recent_peaks else 0.0
        if recent >= 0.98:
            text, tone = "Too loud - clipping. Move back a little.", "error"
        elif recent < 0.02:
            text, tone = "Too quiet - speak up or check the microphone.", "warning"
        else:
            text, tone = "Good level", "success"
        self.level_status_label.setText(text)
        ui_theme.set_tone(self.level_status_label, tone)

    def done(self, result):
        if self.audio_io is not None and self.phase == "recording":
            self.recorded_bytes.extend(bytes(self.audio_io.readAll().data()))
        self._close_microphone()
        if not dialog_accepted(result):
            self.recorded_bytes = bytearray()
        super().done(result)

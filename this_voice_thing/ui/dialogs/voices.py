"""Voice details and the VibeVoice cast picker."""

import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from this_voice_thing.ui import theme as ui_theme


class VoiceDetailsDialog(QDialog):
    """Name, tags and notes for a library voice (and the transcript of a clip)."""

    def __init__(self, title, voice, library, transcript=None, offer_clip=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.voice = voice
        self.library = library
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.name_input = QLineEdit(voice.name)
        form.addRow("Name", self.name_input)
        self.tags_input = QLineEdit(", ".join(voice.tags))
        self.tags_input.setPlaceholderText("Optional, e.g. narrator, warm, project name")
        form.addRow("Tags", self.tags_input)
        self.notes_input = QLineEdit(voice.notes)
        self.notes_input.setPlaceholderText("Optional")
        form.addRow("Notes", self.notes_input)
        self.transcript_input = None
        if transcript is not None:
            self.transcript_input = QLineEdit(transcript)
            self.transcript_input.setPlaceholderText("What is said in the clip (needed by some cloning models)")
            form.addRow("Transcript", self.transcript_input)
        layout.addLayout(form)
        self.clip_checkbox = None
        if offer_clip:
            self.clip_checkbox = QCheckBox("Also make a clip of this voice reading a short passage")
            self.clip_checkbox.setChecked(True)
            self.clip_checkbox.setToolTip(
                "Generates about 15 seconds with this voice and keeps it with its exact transcript, "
                "so every cloning model (Chatterbox, Qwen, VoxCPM, OmniVoice, VibeVoice) can use "
                "the same voice.")
            layout.addWidget(self.clip_checkbox)
        self.error_label = QLabel()
        ui_theme.set_tone(self.error_label, "error")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self):
        name = self.name_input.text().strip()
        if not name:
            self.error_label.setText("Give the voice a name.")
        elif any(other.name.lower() == name.lower() for other in self.library.voices if other is not self.voice):
            self.error_label.setText("Another voice already has this name.")
        else:
            self.accept()
            return
        self.error_label.setVisible(True)

    def apply(self):
        self.voice.name = self.name_input.text().strip()
        self.voice.tags = [tag.strip() for tag in self.tags_input.text().split(",") if tag.strip()]
        self.voice.notes = self.notes_input.text().strip()
        return self.transcript_input.text().strip() if self.transcript_input is not None else None

    def make_clip(self):
        return bool(self.clip_checkbox and self.clip_checkbox.isChecked())


class CastDialog(QDialog):
    """Pick a voice for each speaker in a conversation script."""

    BROWSE = "__browse__"

    def __init__(self, speakers, cast, sample_paths, recordings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cast")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        intro = QLabel("Choose a voice for each speaker in the script. Sample voices come with "
                       "VibeVoice; clip voices from your library and any clip work too. Only clone "
                       "voices of people who have agreed to it.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.combos = {}
        for row, speaker in enumerate(speakers):
            grid.addWidget(QLabel(speaker), row, 0)
            combo = QComboBox()
            for name, path in sample_paths.items():
                combo.addItem(f"{name}  (sample)", path)
            if recordings:
                combo.insertSeparator(combo.count())
                for label, path in recordings:
                    combo.addItem(label, path)
            combo.insertSeparator(combo.count())
            combo.addItem("Other clip\u2026", self.BROWSE)
            current = cast.get(speaker)
            if current and combo.findData(current) < 0:
                combo.insertItem(combo.count() - 2, os.path.basename(current), current)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            combo.setProperty("previous", combo.currentIndex())
            combo.activated.connect(lambda _index, combo=combo: self._maybe_browse(combo))
            grid.addWidget(combo, row, 1)
            self.combos[speaker] = combo
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _maybe_browse(self, combo):
        if combo.currentData() != self.BROWSE:
            combo.setProperty("previous", combo.currentIndex())
            return
        path, _filter = QFileDialog.getOpenFileName(self, "Choose a voice clip", "",
                                                    "Audio Files (*.wav *.mp3 *.flac)")
        if path:
            combo.insertItem(combo.count() - 2, os.path.basename(path), path)
            combo.setCurrentIndex(combo.count() - 3)
            combo.setProperty("previous", combo.currentIndex())
        else:
            combo.setCurrentIndex(combo.property("previous") or 0)

    def result_cast(self):
        return {speaker: combo.currentData() for speaker, combo in self.combos.items()
                if combo.currentData() and combo.currentData() != self.BROWSE}

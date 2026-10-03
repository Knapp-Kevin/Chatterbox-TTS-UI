"""The pronunciation dictionary editor."""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from this_voice_thing.core import pronunciation


class PronunciationDialog(QDialog):
    """Edit the pronunciation dictionary: "write this" -> "say it as"."""

    COLUMNS = ("Write", "Say it as", "Whole word", "Match case", "On")

    def __init__(self, dictionary, speak, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pronunciation dictionary")
        self.setMinimumSize(640, 460)
        self.dictionary = dictionary
        self.speak = speak  # callable(text): plays text with the loaded model
        layout = QVBoxLayout(self)
        intro = QLabel("Respell words the way they should sound, e.g. Nguyen \u2192 Win, SQL \u2192 sequel, "
                       "Siobhan \u2192 Shiv-awn. Works with every model; subtitles keep your spelling.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(lambda _item: self.update_try())
        for rule in dictionary.rules:
            self._add_row(rule)
        layout.addWidget(self.table, 1)
        row_actions = QHBoxLayout()
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_rule)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_rules)
        self.hear_button = QPushButton("Hear it")
        self.hear_button.setToolTip("Speak the selected respelling with the loaded model and voice.")
        self.hear_button.clicked.connect(self.hear_selected)
        for button in (add_button, remove_button, self.hear_button):
            row_actions.addWidget(button)
        row_actions.addStretch(1)
        import_button = self._link_button("Import\u2026", self.import_rules)
        export_button = self._link_button("Export\u2026", self.export_rules)
        row_actions.addWidget(import_button)
        row_actions.addWidget(export_button)
        layout.addLayout(row_actions)
        try_row = QHBoxLayout()
        try_row.addWidget(QLabel("Try"))
        self.try_input = QLineEdit()
        self.try_input.setPlaceholderText("Type a sentence to see (and hear) how it will be read")
        self.try_input.textChanged.connect(lambda _text: self.update_try())
        try_row.addWidget(self.try_input, 1)
        hear_try = QPushButton("Hear")
        hear_try.clicked.connect(lambda: self.speak(self.try_result()) if self.try_input.text().strip() else None)
        try_row.addWidget(hear_try)
        layout.addLayout(try_row)
        self.try_output = QLabel()
        self.try_output.setObjectName("Muted")
        self.try_output.setWordWrap(True)
        layout.addWidget(self.try_output)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _link_button(self, text, slot):
        button = QPushButton(text)
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        return button

    def _add_row(self, rule):
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(rule.word))
        self.table.setItem(row, 1, QTableWidgetItem(rule.say))
        for column, value in ((2, rule.whole_word), (3, rule.match_case), (4, rule.enabled)):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
            self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        return row

    def rules(self):
        rules = []
        for row in range(self.table.rowCount()):
            text = lambda column: (self.table.item(row, column).text() if self.table.item(row, column) else "").strip()
            checked = lambda column: self.table.item(row, column).checkState() == Qt.CheckState.Checked
            if text(0):
                rules.append(pronunciation.Rule(text(0), text(1), checked(2), checked(3), checked(4)))
        return rules

    def add_rule(self):
        row = self._add_row(pronunciation.Rule("", ""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def remove_rules(self):
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
        self.update_try()

    def hear_selected(self):
        row = self.table.currentRow()
        if row >= 0 and self.table.item(row, 1) and self.table.item(row, 1).text().strip():
            self.speak(self.table.item(row, 1).text().strip())

    def try_result(self):
        preview = pronunciation.Dictionary.__new__(pronunciation.Dictionary)
        preview.rules, preview.enabled, preview._pattern, preview._key = self.rules(), True, None, None
        return preview.apply(self.try_input.text())[0]

    def update_try(self):
        text = self.try_input.text().strip()
        self.try_output.setText(f"Read as: {self.try_result()}" if text else "")

    def import_rules(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import pronunciations", "", "Word lists (*.json *.txt *.csv *.tsv);;All files (*.*)")
        if not path:
            return
        try:
            imported = pronunciation.import_rules(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import", f"Couldn't read {os.path.basename(path)}:\n{exc}")
            return
        existing = {rule.word.lower() for rule in self.rules()}
        added = [rule for rule in imported if rule.word.lower() not in existing]
        for rule in added:
            self._add_row(rule)
        self.update_try()
        QMessageBox.information(self, "Import", f"Added {len(added)} of {len(imported)} words "
                                f"({len(imported) - len(added)} were already in the dictionary).")

    def export_rules(self):
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export pronunciations", "pronunciations.txt", "Text list (*.txt);;JSON (*.json)")
        if path:
            pronunciation.export_rules(path, self.rules())

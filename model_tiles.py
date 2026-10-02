"""Model tiles for the Model page: a card per model, laid out in a wrapping grid.

Tiles are plain widgets styled by ui_theme (QFrame#Tile, QLabel#Badge). The
Discover search runs in a background thread so the page never blocks on the
network.
"""

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLayout, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

import model_registry

TILE_WIDTH = 220
TILE_SPACING = 10


class FlowLayout(QLayout):
    """A grid of equal-width items: as many columns as fit, widened to fill each row."""

    def __init__(self, parent=None, spacing=TILE_SPACING):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect, apply):
        if not self._items:
            return 0
        width = max(rect.width(), TILE_WIDTH)
        columns = max(1, (width + self._spacing) // (TILE_WIDTH + self._spacing))
        item_width = (width - (columns - 1) * self._spacing) // columns
        row_height = max(item.sizeHint().height() for item in self._items)
        for index, item in enumerate(self._items):
            row, column = divmod(index, columns)
            if apply:
                item.setGeometry(QRect(rect.x() + column * (item_width + self._spacing),
                                       rect.y() + row * (row_height + self._spacing),
                                       item_width, row_height))
        rows = (len(self._items) + columns - 1) // columns
        return rows * row_height + (rows - 1) * self._spacing

    def clear(self):
        while self._items:
            item = self._items.pop()
            widget = item.widget()
            if widget:
                widget.hide()  # gone now, not just when the event loop gets to deleteLater
                widget.setParent(None)
                widget.deleteLater()


class TileArea(QScrollArea):
    """A vertically scrolling grid of tiles, with a message when it is empty."""

    def __init__(self, min_rows=1, parent=None):
        super().__init__(parent)
        self.setObjectName("TileArea")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body = QWidget()
        outer = QVBoxLayout(body)
        # Room for the hover border so tiles never touch the clip edge.
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        self.empty_label = QLabel()
        self.empty_label.setObjectName("Muted")
        self.empty_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        outer.addWidget(self.empty_label)
        self.flow = FlowLayout()
        outer.addLayout(self.flow)
        outer.addStretch(1)
        self.setWidget(body)
        self._min_rows = min_rows

    def minimumSizeHint(self):
        return QSize(TILE_WIDTH + 2, ModelTile.HEIGHT * self._min_rows + 4)

    def set_tiles(self, tiles, empty_message=""):
        self.flow.clear()
        for tile in tiles:
            self.flow.addWidget(tile)
        self.empty_label.setText(empty_message)
        self.empty_label.setVisible(not tiles and bool(empty_message))
        self.widget().updateGeometry()


class ModelTile(QFrame):
    """One model: name, engine and languages, badges, and a detail line."""

    HEIGHT = 130
    clicked = Signal()
    menu_requested = Signal(QPoint)

    def __init__(self, title, subtitle, badges, detail, tooltip="", active=False,
                 with_menu=False, needs=None, parent=None):
        """needs: (kind, text, tooltip) for the hardware line; kind is good/tight/short."""
        super().__init__(parent)
        self.setObjectName("Tile")
        self.setProperty("active", active)
        self.setMinimumWidth(TILE_WIDTH)
        self.setFixedHeight(self.HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 6, 9)
        layout.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(4)
        self.title_label = _elided_label(title, "TileTitle")
        top.addWidget(self.title_label, 1)
        if with_menu:
            menu_button = QPushButton("⋯")
            menu_button.setObjectName("TileMenu")
            menu_button.setFlat(True)
            menu_button.setCursor(Qt.CursorShape.PointingHandCursor)
            menu_button.setToolTip("More actions")
            menu_button.setFixedSize(26, 22)
            menu_button.clicked.connect(
                lambda: self.menu_requested.emit(menu_button.mapToGlobal(QPoint(0, menu_button.height()))))
            top.addWidget(menu_button)
        else:
            top.addSpacing(6)
        layout.addLayout(top)
        layout.addWidget(_elided_label(subtitle, "Muted"))
        badge_row = QHBoxLayout()
        badge_row.setSpacing(5)
        for kind, text, tip in badges:
            badge = QLabel(text)
            badge.setObjectName("Badge")
            badge.setProperty("kind", kind)
            badge.setToolTip(tip)
            badge_row.addWidget(badge)
        badge_row.addStretch(1)
        layout.addLayout(badge_row)
        layout.addStretch(1)
        if needs:
            kind, text, tip = needs
            needs_label = _elided_label(text, "TileNeeds")
            needs_label.setProperty("kind", kind)
            needs_label.setToolTip(tip)
            layout.addWidget(needs_label)
        layout.addWidget(_elided_label(detail, "Muted"))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        self.menu_requested.emit(event.globalPos())


class _ElidedLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.PlainText)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, self.width()))


def _elided_label(text, object_name):
    label = _ElidedLabel(text)
    label.setObjectName(object_name)
    label.setText(text)
    return label


def license_badge(license_id):
    kind, text = model_registry.license_badge(license_id)
    tips = {
        "permissive": f"License: {license_id}. Commercial use allowed (check the model card).",
        "noncommercial": f"License: {license_id}. Not for commercial use.",
        "unknown": f"License: {license_id or 'not stated'}. Read the model card before using output commercially.",
    }
    return kind, text, tips[kind]


def compact_count(number):
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}k"
    return str(number)


class DiscoverThread(QThread):
    """Searches Hugging Face for loadable models in the background."""

    found = Signal(list, str)  # results, error

    def __init__(self, query, token, parent=None):
        super().__init__(parent)
        self.query = query
        self.token = token

    def run(self):
        try:
            self.found.emit(model_registry.search_models(self.query, "all", self.token, limit=150), "")
        except Exception as exc:
            self.found.emit([], str(exc))

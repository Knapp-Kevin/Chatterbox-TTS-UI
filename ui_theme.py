"""Light/dark theme with an amber accent, following the Windows app mode.

Depth and texture are painted by a few small widgets (cards with soft
shadows, a grained page background, a gradient sidebar) instead of
QGraphicsDropShadowEffect, which renders whole subtrees offscreen and makes
text fields and lists slow and glitchy.
"""

import os

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPalette, QPen, QPixmap)
from PySide6.QtWidgets import QApplication, QFrame, QWidget

LIGHT = {
    "window": "#f5f3f0",
    "surface": "#ffffff",
    "surface_alt": "#faf8f5",
    "sidebar": "#ece8e2",
    "border": "#ddd7cf",
    "border_strong": "#cbc3b8",
    "text": "#1f1c18",
    "muted": "#6f6860",
    "accent": "#e8891c",
    "accent_hover": "#f29a35",
    "accent_pressed": "#cf7710",
    "accent_text": "#1f1406",
    "accent_soft": "#fbe7cf",
    "accent_top": "#f4a141",
    "accent_bottom": "#e2801a",
    "accent_edge": "#c46c10",
    "disabled": "#b3aca3",
    # depth and texture
    "page_top": "#f8f6f2",
    "page_bottom": "#eeeae3",
    "sidebar_top": "#eeeae4",
    "sidebar_bottom": "#e2ddd4",
    "surface_top": "#ffffff",
    "surface_bottom": "#fbf9f6",
    "button_top": "#ffffff",
    "button_bottom": "#f3efe9",
    "shadow": "#5a4a35",
    "shadow_strength": 0.16,
    "edge_highlight": "#ffffff",
    "grain_color": "#3a2f22",
    "grain_alpha": 10,
    # license badges
    "good_bg": "#e2f1df", "good_text": "#2d6a28",
    "bad_bg": "#f9e0dd", "bad_text": "#8f2a21",
}

DARK = {
    "window": "#1c1d20",
    "surface": "#25272b",
    "surface_alt": "#2b2d32",
    "sidebar": "#18191b",
    "border": "#3a3c42",
    "border_strong": "#4a4d55",
    "text": "#ebe8e3",
    "muted": "#9c978f",
    "accent": "#f2a03d",
    "accent_hover": "#f7b25e",
    "accent_pressed": "#d98a26",
    "accent_text": "#1f1406",
    "accent_soft": "#3d2f1d",
    "accent_top": "#f7b55f",
    "accent_bottom": "#e8922c",
    "accent_edge": "#b8701c",
    "disabled": "#64615c",
    "page_top": "#222327",
    "page_bottom": "#18191c",
    "sidebar_top": "#1a1b1e",
    "sidebar_bottom": "#121315",
    "surface_top": "#2b2d32",
    "surface_bottom": "#24262a",
    "button_top": "#34363c",
    "button_bottom": "#2a2c31",
    "shadow": "#000000",
    "shadow_strength": 0.55,
    "edge_highlight": "#ffffff14",
    "grain_color": "#ffffff",
    "grain_alpha": 7,
    "good_bg": "#203522", "good_text": "#93d28c",
    "bad_bg": "#3c2321", "bad_text": "#f09d93",
}

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SHADOW = 7          # pixels around each card reserved for its painted shadow
CARD_RADIUS = 10
_current = LIGHT
_grain_cache = {}


def _asset(name):
    return os.path.join(ASSETS_DIR, name).replace("\\", "/")


LIGHT["chevron"] = _asset("chevron_down_light.svg")
DARK["chevron"] = _asset("chevron_down_dark.svg")
LIGHT["check"] = DARK["check"] = _asset("check.svg")
LOGO = _asset("logo.svg")


def current():
    return _current


def _color(value):
    color = QColor(value)
    if isinstance(value, str) and len(value) == 9:  # #rrggbbaa
        color = QColor(value[:7])
        color.setAlpha(int(value[7:], 16))
    return color


def _grain(colors):
    """A small tile of random, very faint specks for a matte-paper texture."""
    key = (colors["grain_color"], colors["grain_alpha"])
    if key not in _grain_cache:
        size = 160
        rng = np.random.default_rng(7)
        alpha = (rng.random((size, size)) ** 3 * colors["grain_alpha"] * 2).astype(np.uint8)
        base = _color(colors["grain_color"])
        argb = np.zeros((size, size, 4), dtype=np.uint8)
        argb[..., 0], argb[..., 1], argb[..., 2] = base.blue(), base.green(), base.red()
        argb[..., 3] = alpha
        image = QImage(argb.data, size, size, size * 4, QImage.Format.Format_ARGB32).copy()
        _grain_cache[key] = QPixmap.fromImage(image)
    return _grain_cache[key]


def _vertical(rect, top, bottom):
    gradient = QLinearGradient(QPointF(rect.left(), rect.top()), QPointF(rect.left(), rect.bottom()))
    gradient.setColorAt(0, _color(top))
    gradient.setColorAt(1, _color(bottom))
    return gradient


class CardFrame(QFrame):
    """A raised card: soft shadow, gentle top-to-bottom gradient, top highlight."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setContentsMargins(SHADOW, SHADOW - 3, SHADOW, SHADOW + 3)

    def body_rect(self):
        return QRectF(self.rect()).adjusted(SHADOW, SHADOW - 3, -SHADOW, -(SHADOW + 3))

    def paintEvent(self, _event):
        c = _current
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = self.body_rect()
        shadow = _color(c["shadow"])
        painter.setPen(Qt.PenStyle.NoPen)
        for spread in range(SHADOW, 0, -1):
            layer = QColor(shadow)
            layer.setAlphaF(c["shadow_strength"] / SHADOW * (1 - spread / (SHADOW + 1)))
            painter.setBrush(layer)
            painter.drawRoundedRect(body.adjusted(-spread, -spread + 3, spread, spread + 3),
                                    CARD_RADIUS + spread, CARD_RADIUS + spread)
        painter.setBrush(_vertical(body, c["surface_top"], c["surface_bottom"]))
        painter.setPen(QPen(_color(c["border"]), 1))
        painter.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), CARD_RADIUS, CARD_RADIUS)
        painter.setPen(QPen(_color(c["edge_highlight"]), 1))
        painter.drawLine(QPointF(body.left() + CARD_RADIUS, body.top() + 1.5),
                         QPointF(body.right() - CARD_RADIUS, body.top() + 1.5))
        painter.end()


class TexturedArea(QWidget):
    """Page background: soft gradient, fine grain, and the sidebar's cast shadow."""

    def paintEvent(self, _event):
        c = _current
        painter = QPainter(self)
        rect = QRectF(self.rect())
        painter.fillRect(rect, _vertical(rect, c["page_top"], c["page_bottom"]))
        painter.drawTiledPixmap(self.rect(), _grain(c))
        edge = QLinearGradient(QPointF(0, 0), QPointF(12, 0))
        shade = _color(c["shadow"])
        shade.setAlphaF(c["shadow_strength"] * 0.45)
        edge.setColorAt(0, shade)
        shade.setAlphaF(0)
        edge.setColorAt(1, shade)
        painter.fillRect(QRectF(0, 0, 12, rect.height()), edge)
        painter.end()


class SidebarPanel(QWidget):
    """Sidebar: darker gradient with grain and a crisp right edge."""

    def paintEvent(self, _event):
        c = _current
        painter = QPainter(self)
        rect = QRectF(self.rect())
        painter.fillRect(rect, _vertical(rect, c["sidebar_top"], c["sidebar_bottom"]))
        painter.drawTiledPixmap(self.rect(), _grain(c))
        painter.setPen(QPen(_color(c["border"]), 1))
        painter.drawLine(QPointF(rect.right() - 0.5, 0), QPointF(rect.right() - 0.5, rect.bottom()))
        painter.end()


def _is_dark(app):
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def _palette(c):
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: c["window"],
        QPalette.ColorRole.WindowText: c["text"],
        QPalette.ColorRole.Base: c["surface"],
        QPalette.ColorRole.AlternateBase: c["surface_alt"],
        QPalette.ColorRole.Text: c["text"],
        QPalette.ColorRole.Button: c["surface"],
        QPalette.ColorRole.ButtonText: c["text"],
        QPalette.ColorRole.ToolTipBase: c["surface"],
        QPalette.ColorRole.ToolTipText: c["text"],
        QPalette.ColorRole.PlaceholderText: c["muted"],
        QPalette.ColorRole.Highlight: c["accent"],
        QPalette.ColorRole.HighlightedText: c["accent_text"],
        QPalette.ColorRole.Link: c["accent"],
        QPalette.ColorRole.Mid: c["disabled"],
    }
    for role, value in roles.items():
        p.setColor(role, QColor(value))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["disabled"]))
    return p


def _gradient(top, bottom):
    return f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bottom})"


def _stylesheet(c):
    button = _gradient(c["button_top"], c["button_bottom"])
    accent = _gradient(c["accent_top"], c["accent_bottom"])
    accent_hover = _gradient(c["accent_hover"], c["accent_top"])
    chip = _gradient(c["accent_soft"], c["accent_soft"])
    return f"""
    QWidget {{ font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 10pt; }}
    QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}

    QListWidget#Sidebar {{
        background: transparent; border: none; padding: 6px 8px; outline: 0; font-size: 11pt;
    }}
    QListWidget#Sidebar::item {{
        padding: 10px 12px; margin: 2px 0; border-radius: 8px; color: {c['muted']};
    }}
    QListWidget#Sidebar::item:hover {{ background: {c['surface_alt']}; color: {c['text']}; }}
    QListWidget#Sidebar::item:selected {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['accent_soft']}, stop:1 transparent);
        color: {c['text']}; border-left: 3px solid {c['accent']}; font-weight: 600;
    }}
    QLabel#AppTitle {{
        font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 14pt; font-weight: 700;
        color: {c['text']};
    }}

    QLabel#PageTitle {{ font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 17pt; font-weight: 600; }}
    QLabel#PageSubtitle, QLabel#Muted {{ color: {c['muted']}; }}
    QFrame#Card {{ background: transparent; border: none; }}
    QLabel#CardTitle {{ font-weight: 700; font-size: 10.5pt; }}
    QLabel#RecordingPhase {{ font-size: 12pt; font-weight: 700; }}
    QLabel#RecordingClock {{ font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 28pt; font-weight: 700; }}
    QLabel#ReadAloud {{ font-size: 12.5pt; }}
    QPlainTextEdit#LogView {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 9.5pt; }}
    QLabel#Note {{
        background: {c['accent_soft']}; border-left: 3px solid {c['accent']};
        border-radius: 6px; padding: 8px 10px; color: {c['text']};
    }}
    QLabel#VoiceChip {{
        background: {chip}; border: 1px solid {c['accent_soft']}; border-bottom-color: {c['border_strong']};
        border-radius: 12px; padding: 4px 12px; font-weight: 600;
    }}

    QFrame#Tile {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {c['surface_top']}, stop:1 {c['surface_bottom']});
        border: 1px solid {c['border']}; border-bottom-color: {c['border_strong']}; border-radius: 10px;
    }}
    QFrame#Tile:hover {{ border-color: {c['accent']}; }}
    QFrame#Tile[active="true"] {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {c['accent_soft']}, stop:1 {c['surface_bottom']});
        border: 2px solid {c['accent']};
    }}
    QLabel#TileTitle {{ font-weight: 700; }}
    QPushButton#TileMenu {{
        background: transparent; border: none; border-radius: 5px; padding: 0; font-size: 12pt; color: {c['muted']};
    }}
    QPushButton#TileMenu:hover {{ background: {c['accent_soft']}; color: {c['text']}; }}
    QLabel#Badge {{
        border-radius: 8px; padding: 1px 7px; font-size: 8.5pt; font-weight: 600;
        background: {c['surface_alt']}; color: {c['muted']}; border: 1px solid {c['border']};
    }}
    QLabel#Badge[kind="permissive"] {{ background: {c['good_bg']}; color: {c['good_text']}; border-color: {c['good_bg']}; }}
    QLabel#Badge[kind="noncommercial"] {{ background: {c['bad_bg']}; color: {c['bad_text']}; border-color: {c['bad_bg']}; }}
    QLabel#Badge[kind="unknown"] {{ background: {c['accent_soft']}; color: {c['text']}; border-color: {c['accent_soft']}; }}
    QLabel#Badge[kind="active"] {{ background: {accent}; color: {c['accent_text']}; border-color: {c['accent_bottom']}; }}
    QTabBar#CapabilityTabs {{ font-size: 10.5pt; }}
    QTabBar#CapabilityTabs::tab {{
        background: transparent; border: none; border-bottom: 2px solid transparent;
        padding: 6px 10px 7px 10px; margin-right: 2px; color: {c['muted']};
    }}
    QTabBar#CapabilityTabs::tab:hover {{ color: {c['text']}; }}
    QTabBar#CapabilityTabs::tab:selected {{ color: {c['text']}; border-bottom-color: {c['accent']}; font-weight: 600; }}
    QScrollArea#TileArea QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollArea#TileArea QScrollBar::handle:vertical {{ background: {c['border_strong']}; border-radius: 4px; min-height: 28px; }}
    QScrollArea#TileArea QScrollBar::handle:vertical:hover {{ background: {c['accent']}; }}
    QScrollArea#TileArea QScrollBar::add-line, QScrollArea#TileArea QScrollBar::sub-line,
    QScrollArea#TileArea QScrollBar::add-page, QScrollArea#TileArea QScrollBar::sub-page {{ height: 0; background: none; }}
    QLabel#SectionLabel {{ color: {c['muted']}; font-size: 8.5pt; font-weight: 700; letter-spacing: 1px; }}

    QPushButton {{
        background: {button}; border: 1px solid {c['border']}; border-bottom-color: {c['border_strong']};
        border-radius: 7px; padding: 6px 14px;
    }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['accent_soft']}; border-bottom-color: {c['border']}; }}
    QPushButton:disabled {{ color: {c['disabled']}; border-color: {c['border']}; background: {c['button_bottom']}; }}
    QPushButton[accent="true"] {{
        background: {accent}; color: {c['accent_text']}; border: 1px solid {c['accent_bottom']};
        border-bottom-color: {c['accent_edge']}; font-weight: 700; padding: 9px 22px;
    }}
    QPushButton[accent="true"]:hover {{ background: {accent_hover}; }}
    QPushButton[accent="true"]:pressed {{ background: {c['accent_pressed']}; border-bottom-color: {c['accent_pressed']}; }}
    QPushButton[accent="true"]:disabled {{ background: {c['border']}; color: {c['disabled']}; border-color: {c['border']}; }}
    QPushButton[flat="true"] {{ background: transparent; border: none; color: {c['accent']}; padding: 4px 6px; }}
    QPushButton[flat="true"]:hover {{ text-decoration: underline; }}
    QPushButton[flat="true"]:disabled {{ color: {c['disabled']}; }}

    QTextEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QLineEdit, QDoubleSpinBox {{
        background: {c['surface_alt']}; border: 1px solid {c['border']}; border-top-color: {c['border_strong']};
        border-radius: 7px; padding: 4px 6px;
        selection-background-color: {c['accent']}; selection-color: {c['accent_text']};
    }}
    QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{
        border-color: {c['accent']};
    }}
    QComboBox {{ background: {button}; border-top-color: {c['border']}; border-bottom-color: {c['border_strong']}; }}
    QListWidget::item {{ padding: 4px; border-radius: 5px; }}
    QListWidget::item:selected {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['accent_soft']}, stop:1 {c['surface_alt']});
        color: {c['text']};
    }}

    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['accent_bottom']}, stop:1 {c['accent_top']});
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {c['surface_top']}; border: 3px solid {c['accent']};
        width: 10px; height: 10px; margin: -6px 0; border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{ border-color: {c['accent_hover']}; }}
    QSlider::handle:horizontal:disabled {{ border-color: {c['disabled']}; }}
    QSlider::sub-page:horizontal:disabled {{ background: {c['disabled']}; }}

    QProgressBar {{ background: {c['border']}; border: none; border-radius: 4px; }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['accent_bottom']}, stop:1 {c['accent_top']});
        border-radius: 4px;
    }}

    QCheckBox::indicator {{
        width: 16px; height: 16px; border-radius: 4px; border: 1px solid {c['border']};
        border-top-color: {c['border_strong']}; background: {c['surface_alt']};
    }}
    QCheckBox::indicator:checked {{ background: {accent}; border-color: {c['accent_bottom']}; image: url({c['check']}); }}
    QCheckBox {{ spacing: 8px; }}

    QGroupBox {{ border: 1px solid {c['border']}; border-radius: 8px; margin-top: 10px; padding-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['muted']}; }}
    QStatusBar {{ background: {c['sidebar_bottom']}; color: {c['muted']}; border-top: 1px solid {c['border']}; }}
    QStatusBar QLabel {{ padding-left: 8px; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url({c['chevron']}); width: 12px; height: 8px; margin-right: 8px; }}
    QComboBox QAbstractItemView {{
        background: {c['surface']}; border: 1px solid {c['border']};
        selection-background-color: {c['accent_soft']}; selection-color: {c['text']};
    }}
    QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; padding: 4px; }}
    QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
    QMenu::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
    QMenu::item:disabled {{ color: {c['muted']}; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QSplitter::handle {{ background: transparent; }}
    """


def use(app, colors):
    """Apply one colour set now (also used by tests to force light or dark)."""
    global _current
    _current = colors
    app.setPalette(_palette(colors))
    app.setStyleSheet(_stylesheet(colors))
    for widget in QApplication.allWidgets():
        QWidget.update(widget)  # item views overload update(index)


def apply_theme(app):
    """Apply the palette and stylesheet, and follow later light/dark switches."""
    app.setStyle("Fusion")

    def refresh(*_args):
        use(app, DARK if _is_dark(app) else LIGHT)

    refresh()
    try:
        app.styleHints().colorSchemeChanged.connect(refresh)
    except AttributeError:
        pass

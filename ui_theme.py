"""Light/dark theme with an amber accent, following the Windows app mode."""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

LIGHT = {
    "window": "#f5f3f0",
    "surface": "#ffffff",
    "surface_alt": "#faf8f5",
    "sidebar": "#ece8e2",
    "border": "#ddd7cf",
    "text": "#1f1c18",
    "muted": "#6f6860",
    "accent": "#e8891c",
    "accent_hover": "#f29a35",
    "accent_pressed": "#cf7710",
    "accent_text": "#1f1406",
    "accent_soft": "#fbe7cf",
    "disabled": "#b3aca3",
}

DARK = {
    "window": "#1c1d20",
    "surface": "#25272b",
    "surface_alt": "#2b2d32",
    "sidebar": "#18191b",
    "border": "#3a3c42",
    "text": "#ebe8e3",
    "muted": "#9c978f",
    "accent": "#f2a03d",
    "accent_hover": "#f7b25e",
    "accent_pressed": "#d98a26",
    "accent_text": "#1f1406",
    "accent_soft": "#3d2f1d",
    "disabled": "#64615c",
}


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _asset(name):
    return os.path.join(ASSETS_DIR, name).replace("\\", "/")


LIGHT["chevron"] = _asset("chevron_down_light.svg")
DARK["chevron"] = _asset("chevron_down_dark.svg")


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


def _stylesheet(c):
    return f"""
    QWidget {{ font-size: 10pt; }}
    QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}

    QWidget#SidebarPanel {{ background: {c['sidebar']}; }}
    QListWidget#Sidebar {{
        background: {c['sidebar']}; border: none; padding: 12px 8px; outline: 0;
        font-size: 11pt;
    }}
    QListWidget#Sidebar::item {{
        padding: 10px 12px; margin: 2px 0; border-radius: 8px; color: {c['muted']};
    }}
    QListWidget#Sidebar::item:hover {{ background: {c['surface_alt']}; color: {c['text']}; }}
    QListWidget#Sidebar::item:selected {{
        background: {c['accent_soft']}; color: {c['text']};
        border-left: 3px solid {c['accent']}; font-weight: 600;
    }}
    QLabel#AppTitle {{ font-size: 13pt; font-weight: 700; color: {c['accent']}; padding: 4px 12px 10px 12px; }}

    QLabel#PageTitle {{ font-size: 16pt; font-weight: 700; }}
    QLabel#PageSubtitle, QLabel#Muted {{ color: {c['muted']}; }}
    QFrame#Card {{
        background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 10px;
    }}
    QLabel#CardTitle {{ font-weight: 700; font-size: 10.5pt; }}
    QLabel#VoiceChip {{
        background: {c['accent_soft']}; border-radius: 12px; padding: 4px 12px; font-weight: 600;
    }}

    QPushButton {{
        background: {c['surface_alt']}; border: 1px solid {c['border']}; border-radius: 7px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton:pressed {{ background: {c['accent_soft']}; }}
    QPushButton:disabled {{ color: {c['disabled']}; border-color: {c['border']}; }}
    QPushButton[accent="true"] {{
        background: {c['accent']}; color: {c['accent_text']}; border: none; font-weight: 700;
        padding: 9px 22px;
    }}
    QPushButton[accent="true"]:hover {{ background: {c['accent_hover']}; }}
    QPushButton[accent="true"]:pressed {{ background: {c['accent_pressed']}; }}
    QPushButton[accent="true"]:disabled {{ background: {c['border']}; color: {c['disabled']}; }}
    QPushButton[flat="true"] {{ background: transparent; border: none; color: {c['accent']}; padding: 4px 6px; }}
    QPushButton[flat="true"]:hover {{ text-decoration: underline; }}

    QTextEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QLineEdit, QDoubleSpinBox {{
        background: {c['surface_alt']}; border: 1px solid {c['border']}; border-radius: 7px;
        padding: 4px 6px; selection-background-color: {c['accent']}; selection-color: {c['accent_text']};
    }}
    QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{
        border-color: {c['accent']};
    }}
    QListWidget::item {{ padding: 4px; border-radius: 5px; }}
    QListWidget::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}

    QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {c['accent']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    }}
    QSlider::handle:horizontal:disabled, QSlider::sub-page:horizontal:disabled {{ background: {c['disabled']}; }}

    QProgressBar {{ background: {c['border']}; border: none; border-radius: 4px; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 4px; }}

    QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {c['border']}; background: {c['surface_alt']}; }}
    QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}

    QGroupBox {{ border: 1px solid {c['border']}; border-radius: 8px; margin-top: 10px; padding-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['muted']}; }}
    QStatusBar {{ background: {c['sidebar']}; color: {c['muted']}; }}
    QStatusBar QLabel {{ padding-left: 8px; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url({c['chevron']}); width: 12px; height: 8px; margin-right: 8px; }}
    QComboBox QAbstractItemView {{
        background: {c['surface']}; border: 1px solid {c['border']};
        selection-background-color: {c['accent_soft']}; selection-color: {c['text']};
    }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QSplitter::handle {{ background: transparent; }}
    """


def apply_theme(app):
    """Apply the palette and stylesheet, and follow later light/dark switches."""
    app.setStyle("Fusion")

    def refresh(*_args):
        colors = DARK if _is_dark(app) else LIGHT
        app.setPalette(_palette(colors))
        app.setStyleSheet(_stylesheet(colors))

    refresh()
    try:
        app.styleHints().colorSchemeChanged.connect(refresh)
    except AttributeError:
        pass

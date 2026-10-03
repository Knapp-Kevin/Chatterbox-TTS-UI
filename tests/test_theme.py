"""Theme token tests: roles, contrast, colour meaning, and no stray hard-coded colours.

Run from the project folder:  .venv\\Scripts\\python.exe -m unittest discover tests
"""

import colorsys
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from this_voice_thing.ui import theme as ui_theme  # noqa: E402

THEMES = {"dark": ui_theme.DARK, "light": ui_theme.LIGHT}
AA_TEXT = 4.5      # WCAG AA, normal text
AA_COMPONENT = 3.0  # WCAG AA, focus rings and other UI components
OLD_AMBER_ACCENTS = ("#e8891c", "#f29a35", "#cf7710", "#f4a141", "#e2801a", "#c46c10",
                     "#f2a03d", "#f7b25e", "#d98a26", "#f7b55f", "#e8922c", "#b8701c",
                     "#fbe7cf", "#3d2f1d")


def hue(hex_color):
    red, green, blue = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(red, green, blue)[0] * 360


class TokenRoles(unittest.TestCase):
    def test_both_themes_define_every_semantic_role(self):
        for name, theme in THEMES.items():
            missing = [role for role in ui_theme.SEMANTIC_ROLES if role not in theme]
            self.assertEqual(missing, [], f"{name} theme lacks {missing}")

    def test_themes_have_the_same_keys(self):
        self.assertEqual(set(ui_theme.DARK), set(ui_theme.LIGHT))

    def test_stylesheet_builds_for_both_themes(self):
        for theme in THEMES.values():
            sheet = ui_theme._stylesheet(theme)
            self.assertIn("QPushButton", sheet)


class Contrast(unittest.TestCase):
    def check(self, foreground, background, minimum):
        for name, theme in THEMES.items():
            ratio = ui_theme.contrast_ratio(theme[foreground], theme[background])
            self.assertGreaterEqual(ratio, minimum,
                                    f"{name}: {foreground} on {background} is {ratio:.2f}:1")

    def test_body_text(self):
        for background in ("window", "surface", "surface_alt", "page_top", "page_bottom", "selected_bg"):
            self.check("text", background, AA_TEXT)

    def test_muted_text(self):
        for background in ("window", "surface", "surface_alt", "sidebar_top"):
            self.check("muted", background, AA_TEXT)

    def test_accent_as_link_text(self):
        for background in ("window", "surface", "surface_alt"):
            self.check("accent", background, AA_TEXT)

    def test_text_on_accent_fills(self):
        for fill in ("accent", "accent_hover", "accent_pressed"):
            self.check("accent_text", fill, AA_TEXT)

    def test_status_badges(self):
        for tone in ("success", "warning", "error"):
            self.check(f"{tone}_text", f"{tone}_bg", AA_TEXT)
            self.check(f"{tone}_text", "surface", AA_TEXT)  # tile hardware lines, status labels

    def test_focus_ring_is_visible(self):
        for background in ("surface", "surface_alt", "window"):
            self.check("focus", background, AA_COMPONENT)


class ColourMeaning(unittest.TestCase):
    def test_warnings_stay_amber(self):
        for name, theme in THEMES.items():
            self.assertTrue(30 <= hue(theme["warning_text"]) <= 55, f"{name} warning isn't amber")

    def test_errors_stay_red(self):
        for name, theme in THEMES.items():
            h = hue(theme["error_text"])
            self.assertTrue(h >= 330 or h <= 15, f"{name} error isn't red ({h:.0f})")

    def test_success_stays_green(self):
        for name, theme in THEMES.items():
            self.assertTrue(100 <= hue(theme["success_text"]) <= 170, f"{name} success isn't green")

    def test_primary_accent_is_cyan_not_amber(self):
        for name, theme in THEMES.items():
            self.assertTrue(180 <= hue(theme["accent"]) <= 200, f"{name} accent isn't cyan")

    def test_brand_family(self):
        for name, theme in THEMES.items():
            self.assertTrue(215 <= hue(theme["brand_blue"]) <= 235, f"{name} brand blue")
            self.assertTrue(250 <= hue(theme["brand_secondary"]) <= 285, f"{name} purple")
            self.assertTrue(315 <= hue(theme["brand_tertiary"]) <= 340, f"{name} pink")

    def test_semantic_colours_differ_from_the_accent(self):
        for name, theme in THEMES.items():
            accent = theme["accent"]
            for role in ("warning_text", "error_text", "success_text"):
                self.assertNotEqual(theme[role], accent, f"{name}: {role} equals the accent")

    def test_dark_theme_is_deep_navy(self):
        background = ui_theme.DARK["window"]
        self.assertLess(ui_theme.relative_luminance(background), 0.02)
        self.assertTrue(215 <= hue(background) <= 235)


class NoStrayColours(unittest.TestCase):
    def test_no_old_amber_accent_in_the_stylesheet(self):
        for theme in THEMES.values():
            sheet = ui_theme._stylesheet(theme).lower()
            for amber in OLD_AMBER_ACCENTS:
                self.assertNotIn(amber, sheet)

    def test_widgets_use_tokens_not_hex_colours(self):
        package = os.path.join(ROOT, "this_voice_thing")
        widget_files = [os.path.join(folder, name) for folder, _dirs, names in os.walk(package)
                        for name in names if name.endswith(".py") and not name.startswith("theme")]
        for filename in widget_files:
            with open(filename, encoding="utf-8") as handle:
                found = re.findall(r"[\"']#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?[\"']", handle.read())
            self.assertEqual(found, [], f"hard-coded colours in {filename}: {found}")

    def test_assets_referenced_by_the_theme_exist(self):
        for theme in THEMES.values():
            for key in ("chevron", "check"):
                self.assertTrue(os.path.exists(theme[key]), theme[key])
        self.assertTrue(os.path.exists(ui_theme.FALLBACK_ICON))


class AppIcon(unittest.TestCase):
    def test_brand_icon_preferred_when_present_else_fallback(self):
        from PySide6.QtGui import QImage
        usable = os.path.exists(ui_theme.BRAND_ICON) and not QImage(ui_theme.BRAND_ICON).isNull()
        expected = ui_theme.BRAND_ICON if usable else ui_theme.FALLBACK_ICON
        self.assertEqual(ui_theme.app_icon_path(), expected)
        self.assertFalse(os.path.isabs(os.path.relpath(ui_theme.BRAND_ICON, ROOT)))  # inside the repo

    def test_branding_assets_decode(self):
        """Catch truncated or corrupt brand images before they ship."""
        from PySide6.QtGui import QImage
        folder = os.path.join(ROOT, "assets", "branding")
        if not os.path.isdir(folder):
            self.skipTest("no branding assets yet")
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                self.assertFalse(QImage(os.path.join(folder, name)).isNull(), f"{name} doesn't decode")

    def test_product_name(self):
        self.assertEqual(ui_theme.APP_NAME, "This Voice Thing")


if __name__ == "__main__":
    unittest.main()

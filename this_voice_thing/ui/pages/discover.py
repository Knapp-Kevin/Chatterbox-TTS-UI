"""Discover on Hugging Face: search for loadable models and add them."""

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from this_voice_thing.core import model_registry
from this_voice_thing.ui import tiles as model_tiles
from this_voice_thing.ui.common import DUAL_MODE_BACKENDS


class Discover:
    """Hugging Face discovery on the Model page. Mixed into ChatterboxApp."""

    # --- Discover ---

    def start_discover(self):
        if getattr(self, "discover_thread", None) is not None:
            self.discover_rerun = True  # search again with the newest text when this one ends
            return
        self.discover_rerun = False
        self.discover_status.setText("Searching Hugging Face...")
        self.discover_thread = model_tiles.DiscoverThread(
            self.discover_input.text(), self.app_settings.get("hf_token"), self)
        self.discover_thread.found.connect(self.on_discover_results)
        self.discover_thread.start()

    def on_discover_results(self, results, error):
        self.discover_thread.wait()
        self.discover_thread = None
        self.discover_results = results
        self.discover_error = error
        if getattr(self, "discover_rerun", False):
            self.start_discover()
            return
        self.render_discover_tiles()

    def render_discover_tiles(self):
        results = getattr(self, "discover_results", None)
        if results is None:
            if getattr(self, "discover_thread", None) is None:
                self.discover_status.setText("Open this page with an internet connection to see more models.")
            return
        existing = {entry.get("repo_id") for entry in self.model_entries}
        capability = self.current_capability()
        title = model_registry.CAPABILITIES[capability][0].lower()
        matching = [r for r in results if capability in r.capabilities and r.repo_id not in existing]
        self.discover_tiles.set_tiles([self.discover_tile(result) for result in matching[:30]])
        query = self.discover_input.text().strip()
        if self.discover_error:
            self.discover_status.setText(self.discover_error)
        elif not matching:
            self.discover_status.setText(
                f"No other {title} models match \u201c{query}\u201d." if query else
                f"No other {title} models found. Try a search, e.g. a language.")
        else:
            noun = "model" if len(matching) == 1 else "models"
            scope = f" matching \u201c{query}\u201d" if query else ""
            self.discover_status.setText(
                f"{len(matching)} {noun}{scope}, most downloaded first. Click one to add it.")
            self.discover_status.setToolTip("Only models this app can load are shown. Nothing "
                                            "downloads until you load a model.")

    def discover_tile(self, result):
        owner, _sep, name = result.repo_id.partition("/")
        badges = [model_tiles.license_badge(result.license)]
        if result.gated:
            badges.append(("status", "Gated", "Accept the terms on huggingface.co and save a token below."))
        if result.languages:
            count = len(result.languages)
            shown = f"{count} languages" if count > 1 else f"Language: {result.languages[0]}"
            badges.append(("status", shown, "Languages: " + ", ".join(result.languages)))
        detail = f"{model_tiles.compact_count(result.downloads)} downloads \u00b7 {result.likes} likes"
        if result.updated:
            detail += f" \u00b7 {result.updated}"
        tooltip = f"{result.repo_id}\n{result.summary}\nClick to add it to your models."
        needs = self.hardware_line(result.backend, result.repo_id)
        tooltip += "\n" + needs[2]
        tile = model_tiles.ModelTile(name, f"{owner} \u00b7 {result.summary}", badges, detail, tooltip,
                                     needs=needs)
        tile.clicked.connect(lambda r=result: self.add_from_discover(r))
        tile.menu_requested.connect(lambda _pos, r=result: QDesktopServices.openUrl(
            QUrl(f"https://huggingface.co/{r.repo_id}")))
        return tile

    def add_from_discover(self, result):
        seed = {"repo_id": result.repo_id}
        if result.backend in DUAL_MODE_BACKENDS:
            seed["mode"] = "design" if self.current_capability() == "design" else "clone"
        if result.license:
            seed["license"] = result.license
        new_entry = self._edit_entry_dialog(seed)
        if new_entry:
            self.persist_model_entries(self.model_entries + [new_entry], new_entry)

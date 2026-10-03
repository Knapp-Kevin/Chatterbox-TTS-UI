"""Time estimates: per-model speed, sectioning and batch plans, and the text stats line."""

import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu

from this_voice_thing.core import documents
from this_voice_thing.core import model_registry
from this_voice_thing.engines import vibevoice as vibevoice_engine
from this_voice_thing.ui.common import BATCH_COST_SLOPE
from this_voice_thing.ui.common import DEFAULT_SECONDS_PER_CHAR
from this_voice_thing.ui.common import ENGINE_MODULES
from this_voice_thing.ui.common import MAX_TEXT_INPUT_LENGTH
from this_voice_thing.ui.common import VIBEVOICE_BACKEND


class Estimates:
    """Time estimates and text stats. Mixed into ChatterboxApp."""

    def speed_device(self):
        if self.model is not None:
            return self.device_used
        return "cuda" if torch.cuda.is_available() else "cpu"

    def speed_key(self, entry):
        variant = entry.get("qwen_variant") or entry.get("multilingual_t3_model") or ""
        if self.batch_size_for(entry) > 1:
            variant += f"|batch{self.batch_size_for(entry)}"
        return f"{self.speed_device()}|{entry.get('repo_id')}|{entry.get('backend')}|{variant}"

    def seconds_per_char_for(self, entry):
        """(seconds per character, measured?) for an entry on the current device."""
        device = self.speed_device()
        measured = self.app_settings.get("speed_by_model", {}).get(self.speed_key(entry))
        if measured:
            return measured, True
        engine = entry.get("backend") if entry.get("backend") in ENGINE_MODULES else "chatterbox"
        if engine == "chatterbox":
            legacy = self.app_settings.get("seconds_per_char", {}).get(device)  # older single rate
            if legacy:
                return legacy, False
        return DEFAULT_SECONDS_PER_CHAR.get((engine, device), 0.35), False

    def loaded_entry(self):
        return next((e for e in self.model_entries if self.entry_key(e) == self.loaded_entry_key()), None)

    def batch_size_for(self, entry):
        module = ENGINE_MODULES.get(entry.get("backend"))
        if module is not None and hasattr(module, "BATCH_SIZE") and self.speed_device() == "cuda":
            return module.BATCH_SIZE
        return 1

    def batch_plan(self, entry, lengths):
        """[(first, last, estimated seconds)] for generating sections of these lengths."""
        rate, _measured = self.seconds_per_char_for(entry)
        size = self.batch_size_for(entry)
        budget = ENGINE_MODULES[entry["backend"]].BATCH_CHAR_BUDGET if size > 1 else None
        plan = []
        for start, end in documents.plan_batches(lengths, size, budget):
            batch = lengths[start:end]
            if size > 1:
                cost = max(batch) * (1 + BATCH_COST_SLOPE * len(batch)) * rate
            else:
                cost = sum(batch) * rate
            plan.append((start + 1, start + len(batch), cost))
        return plan

    def estimate_seconds(self, entry, lengths):
        _rate, measured = self.seconds_per_char_for(entry)
        return sum(cost for _first, _last, cost in self.batch_plan(entry, lengths)), measured

    @staticmethod
    def max_section_chars_for(entry):
        if entry.get("backend") in ENGINE_MODULES:
            return ENGINE_MODULES[entry["backend"]].MAX_SECTION_CHARS
        return MAX_TEXT_INPUT_LENGTH

    def split_text(self, text, entry):
        """Section texts the way the entry's engine will generate them."""
        if entry.get("backend") == VIBEVOICE_BACKEND:
            return [section.text for section in
                    documents.plan_script_sections(text, vibevoice_engine.MAX_SECTION_CHARS)]
        return documents.split_into_sections(text, self.max_section_chars_for(entry))

    def section_lengths(self, text, entry):
        return [len(section) for section in self.split_text(text, entry)]

    def model_estimates(self, text):
        """[(entry, seconds, measured, active)] for every model in the switcher, fastest first."""
        rows = []
        active_key = self.loaded_entry_key() if self.model is not None else None
        lengths_by_size = {}
        for entry in self.get_visible_model_entries():
            size = (self.max_section_chars_for(entry), entry.get("backend") == VIBEVOICE_BACKEND)
            if size not in lengths_by_size:
                lengths_by_size[size] = self.section_lengths(text, entry)
            seconds, measured = self.estimate_seconds(entry, lengths_by_size[size])
            rows.append((entry, seconds, measured, self.entry_key(entry) == active_key))
        return sorted(rows, key=lambda row: row[1])

    def update_text_stats(self):
        # Always current, including while a preview or render runs; a running
        # render keeps using the text it started with.
        self.refresh_cast_label()
        text = self.text_input.toPlainText().strip()
        if not text:
            self.estimate_button.setVisible(False)
            self.text_stats_label.setText("Type or paste text, or open a document.")
            return
        entry = self.loaded_entry() or self.get_selected_model_entry()
        lengths = self.section_lengths(text, entry)
        sections = len(lengths)
        seconds, measured = self.estimate_seconds(entry, lengths)
        self.estimate_button.setText(f"About {self.format_duration(seconds)} \u25be")
        self.estimate_button.setVisible(True)
        respelled = self.pronunciations.count_in(text)
        self.text_stats_label.setText(
            f"{sections} section{'s' if sections != 1 else ''} \u00b7 {len(text):,} characters"
            + (f" \u00b7 {respelled} respelled" if respelled else ""))
        self.text_stats_label.setToolTip(
            "Words changed by the pronunciation dictionary (Advanced page)." if respelled else "")
        visible = self.get_visible_model_entries()
        for index in range(self.model_repo_combo.count()):
            position = self.model_repo_combo.itemData(index)
            if isinstance(position, int) and position < len(visible):
                item_seconds, item_measured = self.estimate_seconds(
                    visible[position], self.section_lengths(text, visible[position]))
                self.model_repo_combo.setItemData(
                    index,
                    f"About {self.format_duration(item_seconds)} for the current text"
                    f" ({'measured' if item_measured else 'estimate'})",
                    Qt.ItemDataRole.ToolTipRole)

    def show_estimate_menu(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            return
        menu = QMenu(self)
        header = menu.addAction(f"Time for this text ({len(text):,} characters), by model")
        header.setEnabled(False)
        menu.addSeparator()
        estimates = {self.entry_key(e) + (e["label"],): row
                     for row in self.model_estimates(text) for e in [row[0]]}
        for _capability, title, members in model_registry.group_by_capability(self.get_visible_model_entries()):
            menu.addSection(title)
            rows = sorted((estimates[self.entry_key(e) + (e["label"],)] for e in members), key=lambda r: r[1])
            for entry, seconds, measured, active in rows:
                self._add_estimate_action(menu, entry, seconds, measured, active)
        menu.addSeparator()
        note = menu.addAction("Estimates become measurements once a model has generated a few sections.")
        note.setEnabled(False)
        menu.exec(self.estimate_button.mapToGlobal(self.estimate_button.rect().bottomLeft()))

    def _add_estimate_action(self, menu, entry, seconds, measured, active):
            marker = "\u25cf " if active else "    "
            label = f"{marker}{entry['label']}  \u2014  about {self.format_duration(seconds)}"
            label += "" if measured else "  (estimate)"
            if not active:
                label += "  + load"
            action = menu.addAction(label)
            action.setEnabled(not active and not self.is_generating and not getattr(self, "model_is_loading", False))
            action.triggered.connect(lambda _checked=False, e=entry: self.switch_to_entry(e))

    def switch_to_entry(self, entry):
        index = self.model_repo_combo.findText(entry["label"])
        if index >= 0:
            self.model_repo_combo.setCurrentIndex(index)

    def on_section_timed(self, characters, count, seconds):
        # The first section after a load includes warm-up, so it isn't a fair sample.
        if not self.model_is_warm:
            self.model_is_warm = True
            return
        entry = self.loaded_entry()
        if entry is None or characters < 20:
            return
        weight = characters
        if self.batch_size_for(entry) > 1:
            weight = characters * (1 + BATCH_COST_SLOPE * count)
        measured = seconds / weight
        rates = self.app_settings.setdefault("speed_by_model", {})
        key = self.speed_key(entry)
        previous = rates.get(key)
        rates[key] = round(measured if previous is None else 0.7 * previous + 0.3 * measured, 5)

    @staticmethod
    def format_clock(seconds):
        minutes, seconds = divmod(int(round(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    @staticmethod
    def format_duration(seconds):
        seconds = int(round(seconds))
        if seconds < 60:
            return f"{max(seconds, 1)} s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes} min {seconds:02d} s" if minutes < 10 else f"{minutes} min"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} h {minutes:02d} min"

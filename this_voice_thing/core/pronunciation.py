"""Pronunciation dictionary: respell words before they're spoken.

Each rule says "write X, say it as Y". Respelling works with every engine, so
one dictionary covers them all. Rules are applied to the text sent to the model
only: subtitles keep the original spelling, and in conversation scripts the
speaker names are left alone.
"""

import json
import os
import re
from dataclasses import asdict, dataclass

from this_voice_thing.core import documents

FILENAME = "pronunciations.json"


@dataclass
class Rule:
    word: str
    say: str
    whole_word: bool = True
    match_case: bool = False
    enabled: bool = True


class Dictionary:
    def __init__(self, path):
        self.path = path
        self.rules = []
        self.enabled = True
        self._pattern = None
        self._key = None
        self.load()

    # --- storage ---

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            data = {}
        self.enabled = bool(data.get("enabled", True))
        self.rules = rules_from(data.get("rules", []))
        self._pattern = None

    def save(self):
        payload = {"enabled": self.enabled, "rules": [asdict(rule) for rule in self.rules]}
        temp = self.path + ".tmp"
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp, self.path)
        self._pattern = None

    def set_rules(self, rules):
        self.rules = [rule for rule in rules if rule.word.strip()]
        self._pattern = None

    # --- applying ---

    def active_rules(self):
        return [rule for rule in self.rules if rule.enabled and rule.word.strip()] if self.enabled else []

    def _compiled(self):
        key = (self.enabled, tuple((rule.word, rule.say, rule.whole_word, rule.match_case, rule.enabled)
                                   for rule in self.rules))
        if self._pattern is None or self._key != key:
            self._key = key
            parts = []
            # Longest first, so "New York City" wins over "New York".
            for index, rule in sorted(enumerate(self.active_rules()), key=lambda item: -len(item[1].word)):
                body = re.escape(rule.word.strip())
                if rule.whole_word:
                    body = rf"(?<!\w){body}(?!\w)"
                if not rule.match_case:
                    body = f"(?i:{body})"
                parts.append(f"(?P<r{index}>{body})")
            self._pattern = re.compile("|".join(parts)) if parts else False
        return self._pattern

    def apply(self, text):
        """(respelled text, number of replacements)."""
        pattern = self._compiled()
        if not pattern or not text:
            return text, 0
        rules = self.active_rules()
        count = 0

        def replace(match):
            nonlocal count
            count += 1
            rule = rules[int(match.lastgroup[1:])]
            found, say = match.group(0), rule.say
            # Keep a capital where the word had one ("Nguyen" -> "Win"), but not for an
            # all-caps acronym ("GIF" -> "jif").
            capitalized = found[:1].isupper() and (len(found) == 1 or not found.isupper())
            if not rule.match_case and capitalized and say[:1].islower():
                say = say[:1].upper() + say[1:]
            return say

        return pattern.sub(replace, text), count

    def apply_section(self, text):
        """Like apply, but a conversation section ("Name: line" lines) keeps its names."""
        lines = text.splitlines()
        if len(lines) > 1 or (lines and documents.SPEAKER_LINE.match(lines[0])):
            matches = [documents.SPEAKER_LINE.match(line) for line in lines]
            if all(matches):
                out, total = [], 0
                for match in matches:
                    spoken, count = self.apply(match.group("text"))
                    out.append(f"{match.group('name')}: {spoken}")
                    total += count
                return "\n".join(out), total
        return self.apply(text)

    def count_in(self, text):
        return self.apply(text)[1]


def rules_from(items):
    rules = []
    for item in items or []:
        if isinstance(item, dict) and str(item.get("word", "")).strip():
            rules.append(Rule(word=str(item["word"]).strip(), say=str(item.get("say", "")).strip(),
                              whole_word=bool(item.get("whole_word", True)),
                              match_case=bool(item.get("match_case", False)),
                              enabled=bool(item.get("enabled", True))))
    return rules


def import_rules(path):
    """Rules from a .json export, or a text/CSV file with one "word, say it as" per line
    (also "word = say it as" or tab-separated)."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return rules_from(data.get("rules", data) if isinstance(data, dict) else data)
    rules = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for separator in ("\t", "=", ","):
                if separator in line:
                    word, say = (part.strip().strip('"') for part in line.split(separator, 1))
                    if word and say:
                        rules.append(Rule(word=word, say=say))
                    break
    return rules


def export_rules(path, rules):
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"rules": [asdict(rule) for rule in rules]}, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# word, say it as\n")
        for rule in rules:
            handle.write(f"{rule.word}, {rule.say}\n")

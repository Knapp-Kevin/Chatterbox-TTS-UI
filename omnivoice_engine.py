"""Main-app side of the OmniVoice engine (k2-fsa/OmniVoice).

License: the code is Apache-2.0 but the pre-trained weights are CC BY-NC 4.0
(non-commercial) because of their training data, so the app badges it red.

OmniVoice is a small, fast model (Qwen3-0.6B based) for voice cloning and
voice design in 600+ languages. Voice design takes a fixed vocabulary of
attributes (gender, age, pitch, whisper, accent) rather than free text.
It runs in engines/omnivoice/.venv as a worker; OmniVoiceModel exposes the same
interface as a loaded Chatterbox model. One loaded model serves both the
cloning and the design entries.
"""

import os
import tempfile

import numpy as np
import soundfile as sf
import torch

import engine_worker

NAME = "omnivoice"
PYTHON = engine_worker.venv_python(NAME)
WORKER = os.path.join(engine_worker.engine_dir(NAME), "omnivoice_worker.py")
# Pinning torch keeps the CUDA build installed first; plain PyPI would swap in a CPU build.
PACKAGES = [["omnivoice", "torch==2.8.0", "torchaudio==2.8.0"]]
MAX_SECTION_CHARS = 400
# Measured on an RTX 5070 Ti: one sentence ~1.2 s; a batch of 8 sentences ~4 s at a
# 3.4 GB peak (16 took twice as long, so no gain past 8).
BATCH_SIZE = 8
BATCH_CHAR_BUDGET = 3200

# Voice design vocabulary (from omnivoice's _resolve_instruct). One choice per group.
DESIGN_ATTRIBUTES = {
    "Gender": ["male", "female"],
    "Age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "Pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "Style": ["whisper"],
    "Accent (English)": ["american accent", "australian accent", "british accent", "canadian accent",
                         "chinese accent", "indian accent", "japanese accent", "korean accent",
                         "portuguese accent", "russian accent"],
    "Dialect (Chinese)": ["河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话",
                          "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"],
}
GROUP_OF = {item: group for group, items in DESIGN_ATTRIBUTES.items() for item in items}

# The Language box: OmniVoice covers 600+ languages; these are the common ones, and
# "auto" lets the model work it out for anything else.
LANGUAGE_LABELS = {
    "auto": "Detect automatically",
    "ar": "Arabic", "bn": "Bengali", "bg": "Bulgarian", "ca": "Catalan", "zh": "Chinese",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "nl": "Dutch", "en": "English",
    "et": "Estonian", "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic",
    "id": "Indonesian", "ga": "Irish", "it": "Italian", "ja": "Japanese", "kk": "Kazakh",
    "km": "Khmer", "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian", "ms": "Malay",
    "ml": "Malayalam", "mr": "Marathi", "no": "Norwegian", "fa": "Persian", "pl": "Polish",
    "pt": "Portuguese", "pa": "Punjabi", "ro": "Romanian", "ru": "Russian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sw": "Swahili", "sv": "Swedish",
    "tl": "Tagalog", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "cy": "Welsh",
    "yo": "Yoruba", "zu": "Zulu",
}


def is_installed():
    return os.path.exists(PYTHON) and os.path.exists(WORKER)


def install(log=print):
    engine_worker.install_env(NAME, PACKAGES, log)


def split_description(text):
    return [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]


def check_description(text):
    """None if every attribute is one OmniVoice knows, else a message saying what's wrong."""
    unknown = [part for part in split_description(text) if part.lower() not in GROUP_OF
               and part not in GROUP_OF]
    if not unknown:
        return None
    return ("OmniVoice designs voices from a fixed set of attributes, so it doesn't understand: "
            + ", ".join(f"“{part}”" for part in unknown)
            + ".\n\nUse Attributes… next to the description, or combine: "
            "male/female, child to elderly, very low to very high pitch, whisper, and an accent.")


def set_attribute(text, item):
    """Add an attribute, replacing any other choice from the same group."""
    group = GROUP_OF[item]
    kept = [part for part in split_description(text) if GROUP_OF.get(part.lower(), GROUP_OF.get(part)) != group]
    return ", ".join(kept + [item])


class OmniVoiceModel:
    """Chatterbox-compatible facade over an OmniVoice worker."""

    backend = NAME

    def __init__(self, repo_id, worker, mode):
        self.repo_id = repo_id
        self.worker = worker
        self.sr = 24000
        self.device = worker.device
        self.supported_languages = dict(LANGUAGE_LABELS)
        self.speakers = []
        self.speaker = None
        self.batch_size = BATCH_SIZE if self.device == "cuda" else 1
        self.max_section_chars = MAX_SECTION_CHARS
        self.batch_char_budget = BATCH_CHAR_BUDGET
        self.set_mode(mode)
        # Set by the UI before each generation.
        self.instruct = ""   # design attributes
        self.ref_text = ""   # transcript of the reference clip (required for cloning)
        self.watermark = True
        self._watermark = engine_worker.PerthWatermark()
        self._temp_dir = tempfile.mkdtemp(prefix="omnivoice_tts_")
        self._count = 0
        self._anchor = None
        self.locked_anchor = None  # (clip, transcript) of a kept designed voice

    def set_mode(self, mode):
        self.voice_mode = "design" if mode == "design" else "clone"
        self.mode = "voice_design" if self.voice_mode == "design" else "base"

    def begin_run(self):
        """A designed voice differs on every call, so the first section of a run becomes
        the reference for the rest and the whole document keeps one voice. A kept voice
        is used from the start."""
        self._anchor = self.locked_anchor

    def to(self, _device):
        return self

    def generate(self, text, **kwargs):
        return self.generate_batch([text], **kwargs)[0]

    def _request(self, texts, language_id, **voice):
        paths = []
        for _text in texts:
            self._count += 1
            paths.append(os.path.join(self._temp_dir, f"section_{self._count}.wav"))
        reply = self.worker.request(
            cmd="generate", texts=list(texts), out_paths=paths,
            language=None if language_id in (None, "", "auto") else language_id,
            seed=int(torch.initial_seed() % 2**31), **voice)
        return reply["paths"]

    def generate_batch(self, texts, audio_prompt_path=None, language_id=None, **_ignored):
        texts = list(texts)
        paths = []
        if self.voice_mode == "design":
            if self._anchor is None:
                first = self._request(texts[:1], language_id, instruct=self.instruct.strip())
                self._anchor = (first[0], texts[0])
                paths += first
                texts = texts[1:]
            if texts:
                paths += self._request(texts, language_id, ref_audio=self._anchor[0], ref_text=self._anchor[1])
        else:
            paths = self._request(texts, language_id, ref_audio=audio_prompt_path, ref_text=self.ref_text.strip())
        results = []
        for path in paths:
            wav, sr = sf.read(path, dtype="float32")
            self.sr = sr
            if self.watermark:
                wav = self._watermark.apply(wav, sr, "OmniVoice")
            results.append(torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0))
        return results

    def close(self):
        self.worker.close()


def load_omnivoice_model(repo_id, mode="clone", log=print):
    if not is_installed():
        raise RuntimeError("The OmniVoice engine isn't installed. Click its tile on the Model page to install it.")
    worker = engine_worker.WorkerProcess(
        PYTHON, WORKER, "omnivoice", log,
        skip=lambda line: "warn" in line.lower() or "it/s]" in line or "s/it]" in line)
    try:
        worker.request(cmd="load", model_id=repo_id)
    except Exception:
        worker.close()
        raise
    return OmniVoiceModel(repo_id, worker, mode)

"""Main-app side of the VibeVoice engine (Microsoft VibeVoice, transformers port).

VibeVoice turns a script with up to four speakers into one natural conversation,
each speaker cloned from their own clip. The license is MIT, but Microsoft's
model card limits it to research use, recommends against commercial use, and
rules out impersonation without recorded consent; the app badges it amber
("Research use").

It runs in engines/vibevoice/.venv (transformers 5.17+ and diffusers) as a
worker. VibeVoiceModel exposes the same interface as a loaded Chatterbox model;
each section it receives is a block of whole "Name: text" turns.
"""

import os
import tempfile

import numpy as np
import soundfile as sf
import torch

from this_voice_thing.core import documents
from this_voice_thing.engines import worker as engine_worker

NAME = "vibevoice"
PYTHON = engine_worker.venv_python(NAME)
WORKER = os.path.join(engine_worker.engine_dir(NAME), "vibevoice_worker.py")
# Pinning torch keeps the CUDA build installed first; plain PyPI would swap in a CPU build.
PACKAGES = [["transformers>=5.17", "diffusers==0.35.2", "accelerate", "soundfile", "librosa",
             "torch==2.8.0", "torchaudio==2.8.0"]]
# A conversation flows best in long pieces; GPU memory grows with length (6.2 GB peak for
# ~320 characters on the 1.5B model), so sections are a couple of minutes of speech.
MAX_SECTION_CHARS = 1500
MAX_SPEAKERS = 4

# Demo voices from Microsoft's VibeVoice repo (MIT), mirrored on the Hub. The two
# recorded over background music are left out.
SAMPLE_VOICE_REPO = "bezzam/vibevoice_samples"
SAMPLE_VOICES = {
    "Alice (US English, woman)": "voices/en-Alice_woman.wav",
    "Frank (US English, man)": "voices/en-Frank_man.wav",
    "Maya (US English, woman)": "voices/en-Maya_woman.wav",
    "Carter (US English, man)": "voices/en-Carter_man.wav",
    "Samuel (Indian English, man)": "voices/in-Samuel_man.wav",
    "Xinran (Chinese, woman)": "voices/zh-Xinran_woman.wav",
    "Bowen (Chinese, man)": "voices/zh-Bowen_man.wav",
}
# Order used to give speakers without a chosen voice distinct ones.
AUTO_CAST_ORDER = ["Alice (US English, woman)", "Frank (US English, man)", "Maya (US English, woman)",
                   "Carter (US English, man)", "Samuel (Indian English, man)"]

LANGUAGE_LABELS = {"en": "English", "zh": "Chinese"}
SCRIPT_HINT = ("Write a script, one speaker per line, up to 4 speakers:\n"
               "Linda: Welcome back to the show.\n"
               "Thomas: Thanks for having me.")


def is_installed():
    return os.path.exists(PYTHON) and os.path.exists(WORKER)


def install(log=print):
    engine_worker.install_env(NAME, PACKAGES, log)


def resolve_cast(speakers, chosen, sample_paths):
    """{speaker: clip path}: the chosen clip where it still exists, otherwise a sample
    voice not used by anyone else (so every speaker sounds different)."""
    cast = {}
    for speaker in speakers:
        path = chosen.get(speaker)
        if path and os.path.exists(path):
            cast[speaker] = path
    used = set(cast.values())
    spare = [sample_paths[name] for name in AUTO_CAST_ORDER if sample_paths.get(name) not in used]
    for speaker in speakers:
        if speaker not in cast and spare:
            cast[speaker] = spare.pop(0)
    return cast


def voice_name(path, sample_paths):
    for name, sample in sample_paths.items():
        if sample == path:
            return name.split(" (")[0]
    return os.path.splitext(os.path.basename(path))[0]


class VibeVoiceModel:
    """Chatterbox-compatible facade over a VibeVoice worker."""

    backend = NAME
    mode = "conversation"

    def __init__(self, repo_id, worker, sample_paths):
        self.repo_id = repo_id
        self.worker = worker
        self.sample_paths = sample_paths  # {display name: local wav}
        self.sr = 24000
        self.device = worker.device
        self.supported_languages = dict(LANGUAGE_LABELS)
        self.speakers = []
        self.speaker = None
        self.batch_size = 1
        self.max_section_chars = MAX_SECTION_CHARS
        # Set by the UI before each generation.
        self.cast = {}       # {speaker name: clip path}
        self.instruct = ""
        self.ref_text = ""
        self.watermark = True
        self._watermark = engine_worker.PerthWatermark()
        self._temp_dir = tempfile.mkdtemp(prefix="vibevoice_tts_")
        self._count = 0
        self._roles = {}

    def plan_sections(self, text):
        """Split only between whole turns (used by the generation thread)."""
        return documents.plan_script_sections(text, self.max_section_chars)

    def begin_run(self):
        self._roles = {}

    def to(self, _device):
        return self

    def generate(self, text, **kwargs):
        return self.generate_batch([text], **kwargs)[0]

    def generate_batch(self, texts, **_ignored):
        results = []
        for text in texts:
            turns, seen = [], set()
            for speaker, words in documents.parse_script(text):
                # Keep each speaker's role number the same across sections.
                role = self._roles.setdefault(speaker, str(len(self._roles) % MAX_SPEAKERS))
                audio = None
                if speaker not in seen:
                    seen.add(speaker)
                    audio = self.cast.get(speaker)
                turns.append({"role": role, "text": words, "audio": audio})
            self._count += 1
            path = os.path.join(self._temp_dir, f"section_{self._count}.wav")
            reply = self.worker.request(cmd="generate", turns=turns, out_path=path,
                                        seed=int(torch.initial_seed() % 2**31))
            wav, sr = sf.read(reply["path"], dtype="float32")
            self.sr = sr
            if self.watermark:
                wav = self._watermark.apply(wav, sr, "VibeVoice")
            results.append(torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0))
        return results

    def close(self):
        self.worker.close()


def load_vibevoice_model(repo_id, log=print):
    if not is_installed():
        raise RuntimeError("The VibeVoice engine isn't installed. Click its tile on the Model page to install it.")
    worker = engine_worker.WorkerProcess(
        PYTHON, WORKER, "vibevoice", log,
        skip=lambda line: "warn" in line.lower() or "it/s]" in line or "s/it]" in line)
    try:
        info = worker.request(cmd="load", model_id=repo_id, sample_repo=SAMPLE_VOICE_REPO,
                              samples=SAMPLE_VOICES)
    except Exception:
        worker.close()
        raise
    return VibeVoiceModel(repo_id, worker, info.get("samples") or {})

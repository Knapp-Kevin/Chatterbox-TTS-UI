"""Main-app side of the Qwen3-TTS engine.

Qwen3-TTS needs transformers 4.57.x while Chatterbox pins 5.2.0, so Qwen runs
in its own environment (engines/qwen/.venv) as a worker process. QwenModel
exposes the same interface as a loaded Chatterbox model (sr, device,
generate(...)), so sectioning, previews, progress and finishing work unchanged.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np
import soundfile as sf
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, "engines", "qwen")
PYTHON = os.path.join(ENGINE_DIR, ".venv", "Scripts" if os.name == "nt" else "bin",
                      "python.exe" if os.name == "nt" else "python")
WORKER = os.path.join(ENGINE_DIR, "qwen_worker.py")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

VARIANTS = {
    "custom_voice": "Preset voices with style instructions",
    "voice_design": "Design a new voice from a description",
    "base": "Clone a voice from a reference clip",
}
# App language codes -> Qwen language names (Qwen3-TTS covers 10 languages).
LANGUAGE_NAMES = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", "de": "german",
    "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian",
}
LANGUAGE_LABELS = {code: name.capitalize() for code, name in LANGUAGE_NAMES.items()}


def is_installed():
    return os.path.exists(PYTHON) and os.path.exists(WORKER)


def install(log=print):
    """Create engines/qwen/.venv with CUDA PyTorch and qwen-tts using uv."""
    import shutil
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv was not found on PATH; it is needed to install the Qwen engine.")
    os.makedirs(ENGINE_DIR, exist_ok=True)
    venv = os.path.join(ENGINE_DIR, ".venv")
    steps = [
        [uv, "venv", venv, "--python", "3.11"],
        [uv, "pip", "install", "--python", PYTHON, "torch==2.8.0", "torchaudio==2.8.0",
         "--index-url", TORCH_INDEX],
        [uv, "pip", "install", "--python", PYTHON, "qwen-tts"],
    ]
    for command in steps:
        log("Running: " + " ".join(command[1:4]) + " ...")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in process.stdout:
            log(line.rstrip())
        if process.wait() != 0:
            raise RuntimeError(f"Qwen engine install step failed: {' '.join(command[1:3])}")


class QwenWorker:
    """One long-lived worker process; requests are serialised with a lock."""

    def __init__(self, log=print):
        if not is_installed():
            raise RuntimeError("The Qwen engine is not installed.")
        self.log = log
        self.lock = threading.Lock()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.process = subprocess.Popen(
            [PYTHON, WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        ready = self._read_reply()
        if not ready.get("ok"):
            raise RuntimeError(ready.get("error", "Qwen worker failed to start."))
        self.device = "cuda" if ready.get("cuda") else "cpu"

    def _pump_stderr(self):
        for line in self.process.stderr:
            line = line.rstrip()
            if line and "SoX could not be found" not in line and not line.startswith(("*", " - - -")):
                self.log(f"[qwen] {line}")

    def _read_reply(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The Qwen worker stopped unexpectedly. See the Log page.")
        return json.loads(line)

    def request(self, **payload):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("The Qwen worker is not running.")
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            reply = self._read_reply()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Qwen request failed."))
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.request(cmd="shutdown")
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


class QwenModel:
    """Chatterbox-compatible facade over a Qwen worker."""

    backend = "qwen3"

    def __init__(self, repo_id, worker, mode, speakers, languages):
        self.repo_id = repo_id
        self.worker = worker
        self.mode = mode
        self.speakers = speakers
        self.languages = languages
        self.sr = 24000
        self.device = worker.device
        self.supported_languages = {code: LANGUAGE_LABELS[code] for code, name in LANGUAGE_NAMES.items()
                                    if not languages or name in languages}
        # Set by the UI before each generation.
        self.speaker = speakers[0] if speakers else None
        self.instruct = ""
        self.ref_text = ""
        self.watermark = True
        self._watermarker = None
        self._temp_dir = tempfile.mkdtemp(prefix="qwen_tts_")

    def to(self, _device):
        return self

    def generate(self, text, audio_prompt_path=None, exaggeration=0.5, temperature=0.8,
                 cfg_weight=0.5, language_id=None, repetition_penalty=1.2, min_p=0.05, top_p=1.0):
        out_path = os.path.join(self._temp_dir, "section.wav")
        reply = self.worker.request(
            cmd="generate", mode=self.mode, text=text,
            language=LANGUAGE_NAMES.get(language_id or "", None),
            speaker=self.speaker, instruct=self.instruct.strip(),
            ref_audio=audio_prompt_path, ref_text=self.ref_text.strip() or None,
            seed=int(torch.initial_seed() % 2**31), out_path=out_path,
            temperature=float(temperature), top_p=float(top_p),
            repetition_penalty=float(repetition_penalty))
        wav, sr = sf.read(reply["path"], dtype="float32")
        self.sr = sr
        if self.watermark:
            wav = self._apply_watermark(wav, sr)
        return torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)

    def _apply_watermark(self, wav, sr):
        try:
            if self._watermarker is None:
                import perth
                self._watermarker = perth.PerthImplicitWatermarker()
            return np.asarray(self._watermarker.apply_watermark(wav, sample_rate=sr), dtype=np.float32)
        except Exception as exc:
            print(f"Perth watermark could not be applied to Qwen audio: {exc}")
            return wav

    def close(self):
        self.worker.close()


def load_qwen_model(repo_id, log=print):
    worker = QwenWorker(log=log)
    try:
        info = worker.request(cmd="load", model_id=repo_id)
    except Exception:
        worker.close()
        raise
    return QwenModel(repo_id, worker, info.get("mode"), info.get("speakers") or [],
                     info.get("languages") or [])

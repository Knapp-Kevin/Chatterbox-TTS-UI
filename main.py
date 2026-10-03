# ruff: noqa: E402

import sys
import os
import io
import contextlib
import datetime
import random
import json
import base64
import numpy as np
import torch

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Determine the best available device
if torch.cuda.is_available():
    map_location = 'cuda'
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    map_location = 'mps'
else:
    map_location = 'cpu'

# Patch torch.load to default to the best device if map_location not explicitly set
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load
import torchaudio
import traceback


def configure_console_stream(stream):
    if stream is None or not hasattr(stream, "reconfigure"):
        return
    try:
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def safe_console_text(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


APP_LOG_SINK = None


class TeeStream:
    def __init__(self, wrapped_stream):
        self.wrapped_stream = wrapped_stream
        self.encoding = getattr(wrapped_stream, "encoding", "utf-8")

    def write(self, value):
        text = str(value)
        written = self.wrapped_stream.write(text)
        sink = APP_LOG_SINK
        if sink is not None and text:
            try:
                sink(text)
            except Exception:
                pass
        return written

    def flush(self):
        return self.wrapped_stream.flush()

    def isatty(self):
        return bool(getattr(self.wrapped_stream, "isatty", lambda: False)())

    def reconfigure(self, *args, **kwargs):
        reconfigure_fn = getattr(self.wrapped_stream, "reconfigure", None)
        if callable(reconfigure_fn):
            return reconfigure_fn(*args, **kwargs)
        return None


configure_console_stream(sys.stdout)
configure_console_stream(sys.stderr)
sys.stdout = TeeStream(sys.stdout)
sys.stderr = TeeStream(sys.stderr)

# --- NLTK Import and Setup (Same as your working version) ---
try:
    import nltk
    NLTK_RESOURCES_OK = True
    nltk_resources_to_check = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab"
    }
    for resource_id, resource_path in nltk_resources_to_check.items():
        try:
            nltk.data.find(resource_path)
            print(
                f"NLTK '{resource_id}' (path: {resource_path}) resource found.")
        except LookupError:
            print(
                f"NLTK '{resource_id}' (path: {resource_path}) resource not found. Attempting to download '{resource_id}'...")
            try:
                nltk.download(resource_id, quiet=False)
                nltk.data.find(resource_path)
                print(
                    f"NLTK '{resource_id}' resource downloaded and verified successfully.")
            except Exception as e_download:
                print(
                    f"ERROR: Failed to download or verify NLTK '{resource_id}' resource: {e_download}")
                print(f"  python -m nltk.downloader {resource_id}")
                NLTK_RESOURCES_OK = False
                break
        except Exception as e_other:
            print(
                f"ERROR: Unexpected error while checking NLTK '{resource_id}' resource: {e_other}")
            NLTK_RESOURCES_OK = False
            break
except ImportError:
    print("FATAL ERROR: NLTK library not found.")
    nltk = None
    NLTK_RESOURCES_OK = False
# --- END NLTK ---

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGridLayout, QLabel, QTextEdit, QPushButton, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QListWidget, QListWidgetItem, QGroupBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QPlainTextEdit, QSplitter, QLineEdit,
    QCheckBox, QComboBox, QProgressBar, QSizePolicy, QFrame, QStackedWidget, QLayout,
    QMenu, QTabBar, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
# QStandardPaths was in your full file, good.
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QTime, QObject
from PySide6.QtMultimedia import (
    QMediaPlayer, QAudioOutput, QAudioSource, QAudioFormat, QMediaDevices
)
from PySide6.QtGui import QDesktopServices, QPainter, QColor, QFont, QPalette, QIcon, QLinearGradient
import ui_theme
import audio_effects
import subtitles
import pronunciation
import local_api
import google_docs
import threading
import documents
import model_registry
import qwen_engine
import kokoro_engine
import voxcpm_engine
import omnivoice_engine
import vibevoice_engine
import model_tiles
import voice_library
import math
import gc
import time
import tempfile
from collections import deque
import wave

try:
    import model_backends as chatterbox_backends
    CHATTERBOX_AVAILABLE = True
except ImportError:
    chatterbox_backends = None
    CHATTERBOX_AVAILABLE = False
    print("WARNING: chatterbox-tts library not found.")

DEFAULT_MODEL_REPO = getattr(
    chatterbox_backends, "DEFAULT_MODEL_REPO", "ResembleAI/chatterbox"
)
BACKEND_LEGACY = getattr(chatterbox_backends, "BACKEND_LEGACY", "legacy")
BACKEND_MULTILINGUAL = getattr(
    chatterbox_backends, "BACKEND_MULTILINGUAL", "multilingual"
)
DEFAULT_MULTILINGUAL_T3_MODEL = getattr(
    chatterbox_backends, "DEFAULT_MULTILINGUAL_T3_MODEL", "v3"
)
UnsupportedChatterboxRepoError = getattr(
    chatterbox_backends, "UnsupportedChatterboxRepoError", RuntimeError
)
get_supported_languages_for_backend = getattr(
    chatterbox_backends,
    "get_supported_languages_for_backend",
    lambda _backend: {"en": "English"},
)
has_system_nvidia_gpu = getattr(
    chatterbox_backends,
    "has_system_nvidia_gpu",
    lambda: False,
)
load_chatterbox_model = getattr(
    chatterbox_backends,
    "load_chatterbox_model",
    lambda repo_id, backend, device, multilingual_t3_model=None: None,
)
probe_usable_torch_cuda = getattr(
    chatterbox_backends,
    "probe_usable_torch_cuda",
    lambda: (False, "torch.cuda probe unavailable."),
)
resolve_multilingual_t3_model = getattr(
    chatterbox_backends,
    "resolve_multilingual_t3_model",
    lambda multilingual_t3_model: str(
        multilingual_t3_model or DEFAULT_MULTILINGUAL_T3_MODEL
    ).strip(),
)

MAX_TEXT_INPUT_LENGTH = documents.MAX_SECTION_LENGTH
LOSSLESS_FORMATS = ["WAV", "FLAC"]
QWEN_BACKEND = "qwen3"
KOKORO_BACKEND = "kokoro"
VOXCPM_BACKEND = "voxcpm"
OMNIVOICE_BACKEND = "omnivoice"
VIBEVOICE_BACKEND = "vibevoice"
# Engines that run in their own environment (engines/<name>), installed on first use.
ENGINE_MODULES = {QWEN_BACKEND: qwen_engine, KOKORO_BACKEND: kokoro_engine, VOXCPM_BACKEND: voxcpm_engine,
                  OMNIVOICE_BACKEND: omnivoice_engine, VIBEVOICE_BACKEND: vibevoice_engine}
WORKER_MODEL_TYPES = (qwen_engine.QwenModel, kokoro_engine.KokoroModel, voxcpm_engine.VoxCPMModel,
                      omnivoice_engine.OmniVoiceModel, vibevoice_engine.VibeVoiceModel)
# One model that both clones and designs; its entries switch mode without reloading.
DUAL_MODE_BACKENDS = {VOXCPM_BACKEND, OMNIVOICE_BACKEND}
DUAL_MODE_TYPES = (voxcpm_engine.VoxCPMModel, omnivoice_engine.OmniVoiceModel)
ENGINE_INSTALL_NOTES = {
    QWEN_BACKEND: ("Qwen", "Qwen3-TTS runs in its own Python environment (engines/qwen) because it "
                           "needs different library versions than Chatterbox.\n\nInstalling downloads "
                           "about 3 GB of PyTorch and Qwen packages (less if PyTorch is already "
                           "cached). Model weights download the first time each Qwen model is loaded."),
    KOKORO_BACKEND: ("Kokoro", "Kokoro runs in its own Python environment (engines/kokoro) because its "
                               "text processing needs packages Chatterbox doesn't use.\n\nInstalling "
                               "downloads PyTorch (about 3 GB, or nothing if another engine already "
                               "cached it) plus about 300 MB of text-processing packages. The model "
                               "itself is about 340 MB."),
    VOXCPM_BACKEND: ("VoxCPM", "VoxCPM2 runs in its own Python environment (engines/voxcpm) because it "
                               "needs packages Chatterbox doesn't use.\n\nInstalling downloads "
                               "PyTorch (about 3 GB, or nothing if another engine already cached it) "
                               "plus about 1.5 GB of VoxCPM packages. The model itself is about 5 GB, "
                               "shared by the voice cloning and voice design entries."),
    OMNIVOICE_BACKEND: ("OmniVoice", "OmniVoice runs in its own Python environment (engines/omnivoice) "
                                     "because it needs packages Chatterbox doesn't use.\n\nInstalling "
                                     "downloads PyTorch (about 3 GB, or nothing if another engine already "
                                     "cached it) plus OmniVoice's packages. The model itself is about "
                                     "3 GB, shared by the voice cloning and voice design entries.\n\n"
                                     "Note: OmniVoice's weights are licensed CC BY-NC 4.0, for "
                                     "non-commercial use only."),
    VIBEVOICE_BACKEND: ("VibeVoice", "VibeVoice runs in its own Python environment (engines/vibevoice) "
                                     "because it needs a newer transformers than Chatterbox.\n\n"
                                     "Installing downloads PyTorch (about 3 GB, or nothing if another "
                                     "engine already cached it) plus about 1 GB of packages. The 1.5B "
                                     "model is about 5 GB.\n\nNote: VibeVoice is MIT-licensed, but "
                                     "Microsoft's model card limits it to research use and rules out "
                                     "cloning anyone's voice without their recorded consent."),
}


def languages_for_backend(backend):
    if backend == QWEN_BACKEND:
        return dict(qwen_engine.LANGUAGE_LABELS)
    if backend == KOKORO_BACKEND:
        return dict(kokoro_engine.LANGUAGE_LABELS)
    if backend == VOXCPM_BACKEND:
        return dict(voxcpm_engine.LANGUAGE_LABELS)
    if backend == OMNIVOICE_BACKEND:
        return dict(omnivoice_engine.LANGUAGE_LABELS)
    if backend == VIBEVOICE_BACKEND:
        return dict(vibevoice_engine.LANGUAGE_LABELS)
    return get_supported_languages_for_backend(backend)
MODEL_CONFIG_FILENAME = "models.json"
APP_SETTINGS_FILENAME = "app_settings.json"
REFERENCE_RECORDINGS_DIRNAME = "reference_recordings"
RECORDING_SAMPLE_RATE = 48000
MIN_RECORDING_SECONDS = 3
MAX_RECORDING_SECONDS = 30
# Preview lengths offered next to Preview: (label, characters). Speech runs at roughly
# 15 characters a second.
PREVIEW_LENGTHS = (("~10 s", 150), ("~20 s", 300), ("~30 s", 450), ("~1 min", 900))
DEFAULT_PREVIEW_CHARS = 300


def preview_cut(lengths, budget):
    """How many leading sections make up a preview of about budget characters (at least one)."""
    if not budget:
        return len(lengths)
    total = 0
    for count, length in enumerate(lengths, 1):
        total += length
        if total >= budget:
            return count
    return len(lengths)
# Batched engines: a batch costs about longest_section_chars * rate * (1 + slope * sections).
# Fitted on an RTX 5070 Ti (Qwen3 1.7B): 16 sections ~78 s, 2 sections ~28 s.
BATCH_COST_SLOPE = 0.07
# Initial generation-speed guesses (seconds per character), refined by measurement.
# Initial speed guesses (seconds of generation per character of text) per engine
# and device; replaced by measurements as each model is used.
DEFAULT_SECONDS_PER_CHAR = {
    ("chatterbox", "cuda"): 0.045, ("chatterbox", "cpu"): 0.35,
    ("qwen3", "cuda"): 0.13, ("qwen3", "cpu"): 2.0,  # Qwen on CUDA: see BATCH_COST_SLOPE
    # Kokoro: ~0.001 s/char warm on an RTX 5070 Ti, plus a little per section.
    ("kokoro", "cuda"): 0.002, ("kokoro", "cpu"): 0.02,
    # VoxCPM2: ~0.06-0.09 s/char on an RTX 5070 Ti (about real time).
    ("voxcpm", "cuda"): 0.07, ("voxcpm", "cpu"): 1.5,
    # OmniVoice batches: ~0.0035 s/char across a batch of 8 on an RTX 5070 Ti.
    ("omnivoice", "cuda"): 0.012, ("omnivoice", "cpu"): 0.5,
    # VibeVoice 1.5B: 0.075-0.11 s/char on an RTX 5070 Ti (slower for longer sections).
    ("vibevoice", "cuda"): 0.1, ("vibevoice", "cpu"): 2.0,
}
SILENT_RECORDING_PEAK = 0.01  # ~ -40 dBFS; quieter usually means a blocked/muted mic
# Read-aloud passages for reference recordings (~15 s each). Each one covers
# every English vowel, diphthong and consonant (including the rarer "zh",
# "th", "ng", "oy" sounds) and mixes a statement, question and exclamation to
# capture inflection. Chatterbox weighs the first 6-10 s most heavily, so the
# densest sentences come first.
REFERENCE_READING_SCRIPTS = [
    "Would you hand me the yellow measuring cup before the soup boils over? "
    "The quick thinker judged each chance, then sighed with real pleasure. "
    "Out by the oyster boats, a young fisherman sang about his long voyage home. "
    "I can't believe it, the whole village showed up early!",

    "Have you ever watched a thunderstorm roll across the open prairie? "
    "Judy's shiny beige jacket hung on a hook behind the kitchen door. "
    "Thousands of noisy geese flew south, honking loudly over the calm bay. "
    "Please, just breathe slowly and think about what you really want!",

    "My brother usually orders the cheese pizza with extra garlic and mushrooms. "
    "Why would anyone leave a shiny new toy out in the pouring rain? "
    "The judge thoughtfully weighed the evidence while the phone kept ringing. "
    "Wow, that's the most beautiful sunset I've seen all year!",
]
DEFAULT_LANGUAGE_TEST_TEXTS = {
    "ar": "مرحبا. هذا اختبار قصير للنموذج متعدد اللغات.",
    "da": "Hej. Dette er en kort test af den flersprogede model.",
    "de": "Hallo. Dies ist ein kurzer Test des mehrsprachigen Modells.",
    "el": "Γεια σας. Αυτή είναι μια σύντομη δοκιμή του πολύγλωσσου μοντέλου.",
    "en": "Hello. This is a short test of the multilingual model.",
    "es": "Hola. Esta es una prueba corta del modelo multilingue.",
    "fi": "Hei. Tama on lyhyt testi monikieliselle mallille.",
    "fr": "Bonjour. Ceci est un court test du modele multilingue.",
    "he": "שלום. זהו מבחן קצר למודל הרב לשוני.",
    "hi": "नमस्ते। यह बहुभाषी मॉडल की एक छोटी जांच है।",
    "it": "Ciao. Questo e un breve test del modello multilingue.",
    "ja": "こんにちは。これは多言語モデルの短いテストです。",
    "ko": "안녕하세요. 이것은 다국어 모델의 짧은 테스트입니다.",
    "ms": "Halo. Ini ialah ujian ringkas untuk model berbilang bahasa.",
    "nl": "Hallo. Dit is een korte test van het meertalige model.",
    "no": "Hei. Dette er en kort test av den flerspraklige modellen.",
    "pl": "Czesc. To jest krotki test modelu wielojezycznego.",
    "pt": "Ola. Este e um teste curto do modelo multilingue.",
    "ru": "Привет. Это короткая проверка многоязычной модели.",
    "sv": "Hej. Det har ar ett kort test av den flersprakiga modellen.",
    "sw": "Hujambo. Huu ni mtihani mfupi wa modeli ya lugha nyingi.",
    "tr": "Merhaba. Bu, cok dilli model icin kisa bir testtir.",
    "zh": "你好。这是多语言模型的简短测试。",
}
DEFAULT_MODELS_CONFIG = {
    "models": [
        {
            "repo_id": DEFAULT_MODEL_REPO,
            "label": "Official multilingual",
            "enabled": True,
            "experimental": False,
            "test_text": DEFAULT_LANGUAGE_TEST_TEXTS["en"],
            "test_texts": DEFAULT_LANGUAGE_TEST_TEXTS,
            "notes": "Recommended default model.",
            "backend": BACKEND_MULTILINGUAL,
            "language_id": "en",
            "multilingual_t3_model": DEFAULT_MULTILINGUAL_T3_MODEL,
        },
    ]
}


class ElidingChip(QLabel):
    """A label that shortens long text with an ellipsis instead of widening the window."""

    MAX_WIDTH = 180
    PADDING = 28  # the chip's left/right padding and border in the stylesheet

    def __init__(self, text=""):
        super().__init__()
        self.setMaximumWidth(self.MAX_WIDTH)
        self.setText(text)

    def setText(self, text):
        self.full_text = text
        super().setText(self.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight,
                                                      self.MAX_WIDTH - self.PADDING))

    def text(self):
        return self.full_text


class SliderWithValue(QWidget):
    def __init__(self, min_val, max_val, step_val, default_val, value_format="{:.2f}"):
        super().__init__()
        self._step_val = step_val
        self._value_format = value_format
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        # round(), not int(): e.g. int(0.25 / 0.05) truncates to 4.
        self.slider.setMinimum(round(min_val / step_val))
        self.slider.setMaximum(round(max_val / step_val))
        self.slider.setValue(round(default_val / step_val))
        self.slider.setSingleStep(1)
        self.value_label = QLabel(value_format.format(default_val))
        widest = max((value_format.format(v) for v in (min_val, max_val)), key=len)
        self.value_label.setMinimumWidth(self.value_label.fontMetrics().horizontalAdvance(widest) + 6)
        self.slider.valueChanged.connect(
            lambda val: self.value_label.setText(self._value_format.format(val * self._step_val))
        )
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)

    def get_value(self):
        return self.slider.value() * self._step_val

    def set_value(self, value):
        self.slider.setValue(round(value / self._step_val))


def read_models_config_payload(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"Model config '{config_path}' not found."
    except Exception as exc:
        return None, f"Failed to read model config '{config_path}': {exc}"


def load_models_config(config_path):
    payload, load_error = read_models_config_payload(config_path)
    if load_error or payload is None:
        if load_error:
            print(f"{load_error} Using built-in defaults.")
        else:
            print("Model config payload missing. Using built-in defaults.")
        payload = DEFAULT_MODELS_CONFIG

    raw_models = payload.get("models", [])
    normalized_models = []
    seen_model_keys = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("repo_id", "")).strip()
        backend = str(item.get("backend", BACKEND_MULTILINGUAL)).strip() or BACKEND_MULTILINGUAL
        label = str(item.get("label", repo_id)).strip() or repo_id
        multilingual_t3_model = (
            resolve_multilingual_t3_model(
                item.get("multilingual_t3_model", DEFAULT_MULTILINGUAL_T3_MODEL)
            )
            if backend == BACKEND_MULTILINGUAL else ""
        )
        model_key = (
            repo_id,
            backend,
            label,
            multilingual_t3_model or str(item.get("mode") or item.get("voxcpm_mode") or ""),
        )
        if not repo_id or model_key in seen_model_keys:
            continue
        raw_test_texts = item.get("test_texts", {})
        normalized_test_texts = {}
        if isinstance(raw_test_texts, dict):
            for language_id, preset_text in raw_test_texts.items():
                language_key = str(language_id).strip().lower()
                preset_value = str(preset_text).strip()
                if language_key and preset_value:
                    normalized_test_texts[language_key] = preset_value
        optional = {}
        if str(item.get("license", "")).strip():
            optional["license"] = str(item["license"]).strip().lower()
        if isinstance(item.get("download_bytes"), int) and item["download_bytes"] > 0:
            optional["download_bytes"] = item["download_bytes"]
        if backend in DUAL_MODE_BACKENDS:
            optional["mode"] = model_registry.entry_mode(item)
        normalized_models.append(
            {
                **optional,
                "repo_id": repo_id,
                "label": label,
                "enabled": bool(item.get("enabled", True)),
                "experimental": bool(item.get("experimental", repo_id != DEFAULT_MODEL_REPO)),
                "test_text": str(item.get("test_text", "")).strip(),
                "notes": str(item.get("notes", "")).strip(),
                "backend": backend,
                "language_id": str(item.get("language_id", "en")).strip().lower() or "en",
                "multilingual_t3_model": multilingual_t3_model,
                "qwen_variant": str(item.get("qwen_variant", "")).strip(),
                "test_texts": normalized_test_texts,
            }
        )
        seen_model_keys.add(model_key)

    if not any(
        entry["repo_id"] == DEFAULT_MODEL_REPO and
        entry.get("backend") == BACKEND_MULTILINGUAL
        for entry in normalized_models
    ):
        normalized_models.insert(
            0,
            {
                "repo_id": DEFAULT_MODEL_REPO,
                "label": "Official multilingual",
                "enabled": True,
                "experimental": False,
                "test_text": DEFAULT_MODELS_CONFIG["models"][0]["test_text"],
                "notes": "Recommended default model.",
                "backend": BACKEND_MULTILINGUAL,
                "language_id": "en",
                "multilingual_t3_model": DEFAULT_MULTILINGUAL_T3_MODEL,
                "test_texts": DEFAULT_LANGUAGE_TEST_TEXTS.copy(),
            },
        )

    return normalized_models


def read_json_payload(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, None
    except Exception as exc:
        return None, f"Failed to read '{config_path}': {exc}"


def write_json_payload(config_path, payload):
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# --- ModelLoaderThread (Same as your working version) ---


class ModelLoaderThread(QThread):
    model_loaded = Signal(object, str)
    error_occurred = Signal(str)

    def __init__(
        self,
        repo_id=DEFAULT_MODEL_REPO,
        backend=BACKEND_MULTILINGUAL,
        multilingual_t3_model=DEFAULT_MULTILINGUAL_T3_MODEL,
    ):
        super().__init__()
        self.cuda_probe_error = None
        if torch.cuda.is_available():
            cuda_ok, cuda_error = probe_usable_torch_cuda()
            if cuda_ok:
                self.device = "cuda"
            else:
                self.device = "cpu"
                self.cuda_probe_error = cuda_error
        else:
            self.device = "cpu"
        self.repo_id = repo_id
        self.backend = backend
        self.multilingual_t3_model = multilingual_t3_model
        self.mode = "clone"

    def run(self):
        try:
            if not CHATTERBOX_AVAILABLE:
                self.error_occurred.emit(
                    "ChatterboxTTS library is not installed.")
                return
            if self.cuda_probe_error:
                print(
                    "CUDA was detected but is not usable with the current "
                    f"PyTorch build. Falling back to CPU. Reason: {self.cuda_probe_error}"
                )
            print(
                f"Attempting to load Chatterbox model from repo '{self.repo_id}' "
                f"using backend '{self.backend}' on device: {self.device}..."
            )
            if self.backend == QWEN_BACKEND:
                if not qwen_engine.is_installed():
                    raise RuntimeError(
                        "The Qwen engine isn't installed. Use Load this model on the Model page "
                        "to install it.")
                model_instance = qwen_engine.load_qwen_model(self.repo_id, log=print)
            elif self.backend == KOKORO_BACKEND:
                model_instance = kokoro_engine.load_kokoro_model(self.repo_id, log=print)
            elif self.backend == VOXCPM_BACKEND:
                model_instance = voxcpm_engine.load_voxcpm_model(self.repo_id, self.mode, log=print)
            elif self.backend == OMNIVOICE_BACKEND:
                model_instance = omnivoice_engine.load_omnivoice_model(self.repo_id, self.mode, log=print)
            elif self.backend == VIBEVOICE_BACKEND:
                model_instance = vibevoice_engine.load_vibevoice_model(self.repo_id, log=print)
            else:
                model_instance = load_chatterbox_model(
                    self.repo_id,
                    self.backend,
                    self.device,
                    multilingual_t3_model=self.multilingual_t3_model,
                )
            if model_instance is None:
                raise RuntimeError("Model backend loader returned no model instance.")
            model_to = getattr(model_instance, "to", None)
            if callable(model_to):
                model_to(self.device)
            model_device = getattr(model_instance, "device", "N/A")
            print(
                f"Model loaded successfully from repo '{self.repo_id}' "
                f"using backend '{self.backend}'. Model device: "
                f"{model_device}"
            )
            self.model_loaded.emit(model_instance, self.device)
        except UnsupportedChatterboxRepoError as e:
            print(
                f"Unsupported model repo '{self.repo_id}': {e}"
            )
            self.error_occurred.emit(str(e))
        except Exception as e:
            tb_str = traceback.format_exc()
            print(
                f"Error loading model repo '{self.repo_id}': {e}\nTraceback:\n{tb_str}")
            self.error_occurred.emit(
                f"Failed to load model repo '{self.repo_id}': {str(e)}\nSee console for traceback.")

# --- AudioGeneratorThread (Same as your working version with stop flag) ---


class AudioGeneratorThread(QThread):
    generation_complete = Signal(str, int)
    error_occurred = Signal(str)
    chunk_generated = Signal(int, int, int)  # first, last, total
    section_timed = Signal(int, int, float)  # characters (longest in a batch), sections, seconds

    def __init__(
        self,
        model,
        text,
        audio_prompt_path,
        exaggeration,
        temperature,
        cfg_weight,
        seed,
        output_dir,
        language_id="en",
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        finishing=None,
        output_name=None,
        preview=False,
    ):
        super().__init__()
        self.finishing = finishing or audio_effects.FinishingSettings()
        self.output_name = output_name
        self.preview = preview
        self.partial_info = None
        self.subtitle_path = None
        self.pronunciations = None  # set by the app; respells what is spoken
        self.preview_chars = DEFAULT_PREVIEW_CHARS  # set by the app; None previews all the text
        self.model = model
        self.original_text = text
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.temperature = temperature
        self.cfg_weight = cfg_weight
        self.input_seed = seed
        self.output_dir = output_dir
        self.language_id = language_id
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.top_p = top_p
        self.actual_seed_used = seed
        self._is_stopped = False
        self.main_voice_prompt_path = None # For V0 / Default
        self.voice_prompt_1_path = None    # For V1
        self.voice_prompt_2_path = None    # For V2
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def stop(self):
        print("Stop requested for audio generation thread.")
        self._is_stopped = True

    def set_seed_internal(self, seed_val: int):
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)  # Corrected from manual_seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        print(f"Seed set to: {seed_val}")
        self.actual_seed_used = seed_val

    def run(self):
        try:
            if nltk is None or not NLTK_RESOURCES_OK:
                self.error_occurred.emit(
                    "NLTK or 'punkt' missing. Check setup.")
                return
            if self.model is None:
                self.error_occurred.emit("Model not loaded.")
                return
            if self._is_stopped:
                self.error_occurred.emit("Stopped by user before start.")
                return

            if self.input_seed == 0:
                r_seed = random.randint(1, 1_000_000)
                self.set_seed_internal(r_seed)
                print(
                    f"Input seed 0. Using random seed: {self.actual_seed_used}")
            else:
                self.set_seed_internal(self.input_seed)

            max_chars = getattr(self.model, "max_section_chars", MAX_TEXT_INPUT_LENGTH)
            planner = getattr(self.model, "plan_sections", None)
            planned = planner(self.original_text) if planner else documents.plan_sections(self.original_text, max_chars)
            if self.preview:
                planned = planned[:preview_cut([len(section.text) for section in planned], self.preview_chars)]
            final_chunks = [section.text for section in planned]
            self.section_boundaries = [section.boundary for section in planned]
            # What the model is given: the pronunciation dictionary applied. final_chunks
            # keeps the written text for subtitles.
            spoken_chunks = list(final_chunks)
            if self.pronunciations is not None:
                respelled = [self.pronunciations.apply_section(chunk) for chunk in final_chunks]
                spoken_chunks = [chunk for chunk, _count in respelled]
                replaced = sum(count for _chunk, count in respelled)
                if replaced:
                    print(f"Pronunciation dictionary: respelled {replaced} word(s).")
            print(f"Split into {len(planned)} sections (up to {max_chars} characters, "
                  "never across paragraphs).")

            if not final_chunks:
                self.error_occurred.emit(
                    "Input text empty or resulted in no chunks.")
                return

            total_chunks = len(final_chunks)
            print(f"Processed into {total_chunks} chunks.")
            # (Optional debug print for chunks can go here)

            all_audio_tensors = []
            sr = self.model.sr
            # Engines that support it (Qwen) generate several sections per call.
            batch_size = max(1, int(getattr(self.model, "batch_size", 1)))
            batches = documents.plan_batches(
                [len(text) for text in final_chunks], batch_size,
                getattr(self.model, "batch_char_budget", None))
            generate_kwargs = dict(
                audio_prompt_path=self.audio_prompt_path if self.audio_prompt_path else None,
                exaggeration=self.exaggeration,
                temperature=self.temperature,
                cfg_weight=self.cfg_weight,
                language_id=self.language_id,
                repetition_penalty=self.repetition_penalty,
                min_p=self.min_p,
                top_p=self.top_p,
            )
            for i, batch_end in batches:
                if self._is_stopped:
                    if all_audio_tensors and not self.preview:
                        # Keep the finished sections of a long render.
                        self.partial_info = (i, total_chunks)
                        print(f"Stopped at section {i + 1}/{total_chunks}; saving {i} finished sections.")
                        break
                    self.error_occurred.emit(
                        f"Generation stopped by user at chunk {i+1}/{total_chunks}.")
                    return
                batch = spoken_chunks[i:batch_end]
                current_chunk_num = i + 1
                last = i + len(batch)
                self.chunk_generated.emit(current_chunk_num, last, total_chunks)
                span = f"{current_chunk_num}" if len(batch) == 1 else f"{current_chunk_num}-{last}"
                print(f"\nGenerating section {span}/{total_chunks} (seed: {self.actual_seed_used}).")
                section_started = time.monotonic()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if len(batch) > 1:
                        wav_tensors = self.model.generate_batch(batch, **generate_kwargs)
                    else:
                        wav_tensors = [self.model.generate(batch[0], **generate_kwargs)]
                for wav_tensor_chunk in wav_tensors:
                    if wav_tensor_chunk.ndim == 1:
                        wav_tensor_chunk = wav_tensor_chunk.unsqueeze(0)
                    all_audio_tensors.append(wav_tensor_chunk.cpu())
                # A batch takes as long as its longest section, so time is measured against that.
                self.section_timed.emit(max(len(text) for text in batch) if len(batch) > 1 else len(batch[0]),
                                        len(batch), time.monotonic() - section_started)

            if self._is_stopped and self.partial_info is None and len(all_audio_tensors) < total_chunks:
                self.error_occurred.emit("Stopped before final concat.")
                return
            if not all_audio_tensors:
                self.error_occurred.emit("No audio data generated.")
                return

            print("\nConcatenating audio chunks...")
            finishing = self.finishing
            sections = [chunk.reshape(-1).float().numpy() for chunk in all_audio_tensors]
            spans, timing = [], {}
            joined_audio = audio_effects.join_sections(
                sections, sr, self.section_boundaries[:len(sections)], finishing.paragraph_pause, spans)
            print(f"Applying finishing touches: {finishing.summary()}")
            final_audio = audio_effects.apply_finishing(joined_audio, sr, finishing, timing=timing)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.preview:
                output_dir = os.path.join(self.output_dir, "previews")
                os.makedirs(output_dir, exist_ok=True)
                file_stem = f"preview_{timestamp}_seed{self.actual_seed_used}"
            else:
                output_dir = self.output_dir
                prefix = self.output_name or "chatterbox"
                suffix = "_partial" if self.partial_info else (
                    "" if self.output_name else "_full_stitched")
                file_stem = f"{prefix}_{timestamp}_seed{self.actual_seed_used}{suffix}"
            output_base = os.path.join(output_dir, file_stem)
            output_path = audio_effects.save_audio(
                output_base, final_audio, sr, finishing.output_format)
            print(f"Final stitched audio saved to: {output_path}")
            if finishing.save_subtitles and not self.preview:
                try:
                    cues = subtitles.build_cues(final_chunks[:len(sections)], spans, joined_audio, sr,
                                                timing.get("scale", 1.0), timing.get("offset", 0.0),
                                                len(final_audio) / sr)
                    self.subtitle_path = subtitles.save(output_base, cues, finishing.subtitle_format)
                    print(f"Subtitles saved to: {self.subtitle_path} ({len(cues)} captions)")
                except Exception as exc:  # subtitles never cost the audio
                    print(f"Could not write subtitles: {safe_console_text(exc)}")
            self.generation_complete.emit(output_path, sr)
        except Exception as e:
            if not self._is_stopped:
                tb_str = traceback.format_exc()
                print(
                    "Error in AudioGeneratorThread: "
                    f"{safe_console_text(e)}\n{safe_console_text(tb_str)}"
                )
                self.error_occurred.emit(
                    f"Generation/stitching error: {str(e)}")

# --- Reference audio recording ---


def choose_recording_format(device):
    audio_format = QAudioFormat()
    audio_format.setSampleRate(RECORDING_SAMPLE_RATE)
    audio_format.setChannelCount(1)
    audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    if device.isFormatSupported(audio_format):
        return audio_format
    return device.preferredFormat()


def pcm_to_mono_float(data, audio_format):
    sample_format = audio_format.sampleFormat()
    dtypes = {
        QAudioFormat.SampleFormat.UInt8: (np.uint8, 128.0, 128.0),
        QAudioFormat.SampleFormat.Int16: (np.int16, 0.0, 32768.0),
        QAudioFormat.SampleFormat.Int32: (np.int32, 0.0, 2147483648.0),
        QAudioFormat.SampleFormat.Float: (np.float32, 0.0, 1.0),
    }
    if sample_format not in dtypes:
        raise ValueError(f"Unsupported microphone sample format: {sample_format}")
    dtype, offset, scale = dtypes[sample_format]
    channels = max(1, audio_format.channelCount())
    frame_size = np.dtype(dtype).itemsize * channels
    data = data[:len(data) - len(data) % frame_size]
    samples = np.frombuffer(data, dtype=dtype).astype(np.float32)
    samples = (samples - offset) / scale
    return samples.reshape(-1, channels).mean(axis=1)


def dialog_accepted(result):
    return int(getattr(result, "value", result)) == QDialog.DialogCode.Accepted.value


class LevelHistoryWidget(QWidget):
    """Scrolling bar graph of recent microphone peak levels."""

    def __init__(self, bars=72, parent=None):
        super().__init__(parent)
        self.levels = deque([0.0] * bars, maxlen=bars)
        self.active = False
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def push(self, level):
        self.levels.append(level)
        self.update()

    def set_active(self, active):
        self.active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()).darker(108))
        if self.active:
            normal = self.palette().color(QPalette.ColorRole.Highlight)
        else:
            normal = self.palette().color(QPalette.ColorRole.Mid)
        bar_width = width / len(self.levels)
        middle = height / 2
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        theme = ui_theme.current()
        for index, level in enumerate(self.levels):
            # Square-root scaling so normal speech fills a useful part of the height.
            bar_height = max(2.0, min(1.0, level) ** 0.5 * (height - 6))
            top = middle - bar_height / 2
            if level >= 0.98:
                brush = QColor("#d9534f")
            elif self.active:
                brush = QLinearGradient(0, top, 0, top + bar_height)
                brush.setColorAt(0, QColor(theme["accent_top"]))
                brush.setColorAt(0.5, QColor(theme["accent_bottom"]))
                brush.setColorAt(1, QColor(theme["accent_top"]))
            else:
                brush = normal
            painter.setBrush(brush)
            bar = max(1.0, bar_width - 2)
            painter.drawRoundedRect(index * bar_width + 1, top, bar, bar_height, bar / 2, bar / 2)
        painter.end()


class RecordingDialog(QDialog):
    """Modal recorder: countdown with live mic check, then timed capture."""

    COUNTDOWN_SECONDS = 3
    TICK_MS = 50

    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Reference Audio")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.device = device
        self.audio_format = choose_recording_format(device)
        self.recorded_bytes = bytearray()
        self.audio_source = None
        self.audio_io = None
        self.phase = "countdown"
        self.countdown_started = None
        self.recent_peaks = deque(maxlen=int(1500 / self.TICK_MS))

        layout = QVBoxLayout(self)
        # Let wrapped labels grow the dialog instead of being clipped.
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        mic_label = QLabel(f"Microphone: {device.description()}")
        mic_label.setTextFormat(Qt.TextFormat.PlainText)
        mic_label.setStyleSheet("color: gray;")
        layout.addWidget(mic_label)

        self.phase_label = QLabel("Get ready...")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setObjectName("RecordingPhase")
        layout.addWidget(self.phase_label)

        self.big_label = QLabel(str(self.COUNTDOWN_SECONDS))
        self.big_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.big_label.setObjectName("RecordingClock")
        layout.addWidget(self.big_label)

        self.hint_label = QLabel(
            f"Read the text below at your normal pace (about 15 seconds). Recording "
            f"stops automatically at {MAX_RECORDING_SECONDS} seconds.")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self.script_group = script_group = QGroupBox("Read this aloud")
        script_layout = QVBoxLayout(script_group)
        self.script_index = 0
        self.script_label = QLabel()
        self.script_label.setWordWrap(True)
        self.script_label.setTextFormat(Qt.TextFormat.PlainText)
        self.script_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.script_label.setObjectName("ReadAloud")
        script_layout.addWidget(self.script_label)
        script_footer = QHBoxLayout()
        script_note = QLabel("Read with natural expression. Any language works.")
        script_note.setStyleSheet("color: gray;")
        self.next_script_button = QPushButton("Different text")
        self.next_script_button.clicked.connect(self.show_next_script)
        script_footer.addWidget(script_note, 1)
        script_footer.addWidget(self.next_script_button)
        script_layout.addLayout(script_footer)
        layout.addWidget(script_group)
        self.show_script(0)

        self.level_widget = LevelHistoryWidget(parent=self)
        layout.addWidget(self.level_widget)

        self.level_status_label = QLabel("Checking microphone...")
        self.level_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.level_status_label)

        self.time_bar = QProgressBar()
        self.time_bar.setRange(0, MAX_RECORDING_SECONDS * 1000)
        self.time_bar.setValue(0)
        self.time_bar.setTextVisible(False)
        self.time_bar.setFixedHeight(8)
        layout.addWidget(self.time_bar)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.stop_button = QPushButton("Stop && Use")
        self.stop_button.setEnabled(False)
        self.stop_button.setDefault(True)
        self.stop_button.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        QTimer.singleShot(0, self._open_microphone)

    def show_script(self, index):
        self.script_index = index % len(REFERENCE_READING_SCRIPTS)
        self.script_label.setText(REFERENCE_READING_SCRIPTS[self.script_index])

    def show_next_script(self):
        self.show_script(self.script_index + 1)

    def _open_microphone(self):
        # Open during the countdown: Bluetooth headsets need a moment to switch
        # profiles, and the live meter doubles as a mic check.
        self.audio_source = QAudioSource(self.device, self.audio_format, self)
        self.audio_io = self.audio_source.start()
        error = self.audio_source.error()
        if self.audio_io is None or getattr(error, "name", "NoError") != "NoError":
            self._close_microphone()
            self.phase = "error"
            self.phase_label.setText("Microphone unavailable")
            self.big_label.setText("!")
            self.hint_label.setText(
                f"Could not open '{self.device.description()}' "
                f"({getattr(error, 'name', error)}). Check that it is connected and awake, "
                "and that Windows allows desktop apps to use the microphone "
                "(Settings > Privacy & security > Microphone).")
            self.level_status_label.setText("")
            self.script_group.setVisible(False)
            self.level_widget.setVisible(False)
            self.time_bar.setVisible(False)
            self.cancel_button.setText("Close")
            return
        print(
            f"Recording dialog opened '{self.device.description()}' "
            f"({self.audio_format.sampleRate()} Hz, {self.audio_format.channelCount()} ch).")
        self.countdown_started = time.monotonic()
        self.level_widget.set_active(False)
        self.timer.start(self.TICK_MS)

    def _close_microphone(self):
        self.timer.stop()
        if self.audio_source is not None:
            self.audio_source.stop()
            self.audio_source.deleteLater()
        self.audio_source = None
        self.audio_io = None

    def recorded_seconds(self):
        bytes_per_frame = self.audio_format.bytesPerFrame()
        if not bytes_per_frame:
            return 0.0
        return len(self.recorded_bytes) / bytes_per_frame / self.audio_format.sampleRate()

    def _tick(self):
        chunk = bytes(self.audio_io.readAll().data()) if self.audio_io is not None else b""
        peak = 0.0
        if chunk:
            samples = pcm_to_mono_float(chunk, self.audio_format)
            if samples.size:
                peak = float(np.max(np.abs(samples)))
        self.level_widget.push(peak)
        self.recent_peaks.append(peak)
        self._update_level_status()

        if self.phase == "countdown":
            remaining = self.COUNTDOWN_SECONDS - (time.monotonic() - self.countdown_started)
            if remaining > 0:
                self.big_label.setText(str(int(remaining) + 1))
                return
            # Countdown audio is a mic check only; capture starts now.
            self.phase = "recording"
            self.phase_label.setText("Recording")
            self.phase_label.setStyleSheet("color: #d9534f;")
            self.level_widget.set_active(True)
            self.next_script_button.setEnabled(False)
            return

        self.recorded_bytes.extend(chunk)
        elapsed = self.recorded_seconds()
        self.big_label.setText(
            f"{int(elapsed) // 60}:{int(elapsed) % 60:02d} / "
            f"{MAX_RECORDING_SECONDS // 60}:{MAX_RECORDING_SECONDS % 60:02d}")
        self.time_bar.setValue(int(min(elapsed, MAX_RECORDING_SECONDS) * 1000))
        if elapsed >= MIN_RECORDING_SECONDS:
            self.stop_button.setEnabled(True)
            self.hint_label.setText("Click Stop & Use when you're done.")
        else:
            self.hint_label.setText(
                f"Keep talking - at least {MIN_RECORDING_SECONDS} seconds are needed.")
        if elapsed >= MAX_RECORDING_SECONDS:
            self.accept()

    def _update_level_status(self):
        recent = max(self.recent_peaks) if self.recent_peaks else 0.0
        if recent >= 0.98:
            text, color = "Too loud - clipping. Move back a little.", "#d9534f"
        elif recent < 0.02:
            text, color = "Too quiet - speak up or check the microphone.", "#e0a030"
        else:
            text, color = "Good level", "#3c9a3c"
        self.level_status_label.setText(text)
        self.level_status_label.setStyleSheet(f"color: {color};")

    def done(self, result):
        if self.audio_io is not None and self.phase == "recording":
            self.recorded_bytes.extend(bytes(self.audio_io.readAll().data()))
        self._close_microphone()
        if not dialog_accepted(result):
            self.recorded_bytes = bytearray()
        super().done(result)


# --- Hugging Face model search ---


class FindModelsDialog(QDialog):
    """Search Hugging Face for repos this app can load and pick one to add."""

    ENGINE_FILTERS = (("All engines", "all"), ("Chatterbox", "chatterbox"), ("Qwen3-TTS", "qwen3"),
                      ("Kokoro", "kokoro"), ("VoxCPM", "voxcpm"), ("OmniVoice", "omnivoice"),
                      ("VibeVoice", "vibevoice"))

    def __init__(self, token, existing_repos, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find models on Hugging Face")
        self.setMinimumSize(640, 460)
        self.token = token
        self.existing_repos = existing_repos
        self.selected_repo = None

        layout = QVBoxLayout(self)
        intro = QLabel("Only models this app can load are listed. Pick one to add it; nothing "
                       "downloads until you load it.")
        intro.setObjectName("Muted")
        layout.addWidget(intro)
        search_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Search by name or language, e.g. norwegian, arabic, 0.6B")
        self.query_input.returnPressed.connect(self.run_search)
        search_row.addWidget(self.query_input, 1)
        self.engine_combo = QComboBox()
        for label, key in self.ENGINE_FILTERS:
            self.engine_combo.addItem(label, key)
        self.engine_combo.currentIndexChanged.connect(lambda _i: self.run_search())
        search_row.addWidget(self.engine_combo)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.run_search)
        search_row.addWidget(search_button)
        layout.addLayout(search_row)

        self.results_list = QListWidget()
        self.results_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.results_list.itemDoubleClicked.connect(lambda _item: self.use_selected())
        self.results_list.currentRowChanged.connect(lambda _row: self._update_buttons())
        layout.addWidget(self.results_list, 1)
        self.status_label = QLabel()
        self.status_label.setObjectName("Muted")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.open_page_button = QPushButton("Open on Hugging Face")
        self.open_page_button.clicked.connect(self.open_selected_page)
        buttons.addWidget(self.open_page_button)
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.use_button = QPushButton("Add selected...")
        self.use_button.setProperty("accent", True)
        self.use_button.clicked.connect(self.use_selected)
        buttons.addWidget(self.use_button)
        layout.addLayout(buttons)
        self._update_buttons()
        QTimer.singleShot(0, self.run_search)

    def run_search(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = model_registry.search_models(
                self.query_input.text(), self.engine_combo.currentData(), self.token)
            error = None
        except Exception as exc:
            results, error = [], str(exc)
        finally:
            QApplication.restoreOverrideCursor()
        self.results_list.clear()
        for result in results:
            details = [result.summary]
            if result.languages:
                shown = ", ".join(result.languages[:6]) + (" ..." if len(result.languages) > 6 else "")
                details.append(shown)
            details.append(f"{result.downloads:,} downloads")
            if result.updated:
                details.append(f"updated {result.updated}")
            if result.gated:
                details.append("gated: token and accepted terms needed")
            added = "   (already in your list)" if result.repo_id in self.existing_repos else ""
            item = QListWidgetItem(f"{result.repo_id}{added}\n      " + " · ".join(details))
            item.setData(Qt.ItemDataRole.UserRole, result.repo_id)
            item.setToolTip(f"{result.repo_id}\n{' · '.join(details)}")
            self.results_list.addItem(item)
        if error:
            self.status_label.setText(error)
        elif not results:
            self.status_label.setText("No loadable models found. Try another word or engine.")
        else:
            noun = "model" if len(results) == 1 else "models"
            self.status_label.setText(f"{len(results)} loadable {noun}, most downloaded first.")
        self._update_buttons()

    def _selected(self):
        item = self.results_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _update_buttons(self):
        has = bool(self._selected())
        self.use_button.setEnabled(has)
        self.open_page_button.setEnabled(has)

    def open_selected_page(self):
        repo = self._selected()
        if repo:
            QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{repo}"))

    def use_selected(self):
        self.selected_repo = self._selected()
        if self.selected_repo:
            self.accept()


# --- Model entry editor ---


class ModelEntryDialog(QDialog):
    """Add or edit one models.json entry, with a live Hugging Face check."""

    def __init__(self, entry, other_entries, token, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit model" if entry else "Add model")
        self.setMinimumWidth(560)
        self.other_entries = other_entries
        self.token = token
        self.original = dict(entry or {})
        self.checked = {}
        entry = dict(entry or {"backend": "multilingual", "multilingual_t3_model": "v3",
                               "enabled": True, "language_id": "en"})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        repo_row = QHBoxLayout()
        self.repo_input = QLineEdit(entry.get("repo_id", ""))
        self.repo_input.setPlaceholderText("owner/model-name")
        self.repo_input.textEdited.connect(self._repo_edited)
        repo_row.addWidget(self.repo_input, 1)
        self.check_button = QPushButton("Check")
        self.check_button.setToolTip("Look the repo up on Hugging Face and detect its engine.")
        self.check_button.clicked.connect(self.check_repo)
        repo_row.addWidget(self.check_button)
        find_button = QPushButton("Find...")
        find_button.setToolTip("Search Hugging Face for models this app can load.")
        find_button.clicked.connect(self.find_repo)
        repo_row.addWidget(find_button)
        form.addRow("Hugging Face repo", repo_row)
        self.check_label = QLabel("Enter a repo and click Check to confirm it can be loaded.")
        self.check_label.setObjectName("Muted")
        self.check_label.setWordWrap(True)
        form.addRow("", self.check_label)

        self.name_input = QLineEdit(entry.get("label", ""))
        self.name_input.setPlaceholderText("Shown in the model switcher")
        form.addRow("Name", self.name_input)

        self.engine_combo = QComboBox()
        for engine in model_registry.ENGINES.values():
            self.engine_combo.addItem(engine.label, engine.key)
            self.engine_combo.setItemData(
                self.engine_combo.count() - 1, engine.description, Qt.ItemDataRole.ToolTipRole)
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(entry.get("backend"))))
        self.engine_combo.currentIndexChanged.connect(self._engine_changed)
        form.addRow("Engine", self.engine_combo)

        self.weights_combo = QComboBox()
        for short in model_registry.WEIGHT_VERSIONS:
            self.weights_combo.addItem(short.upper(), short)
        current_weights = str(entry.get("multilingual_t3_model") or "v3")
        for short, filename in model_registry.WEIGHT_VERSIONS.items():
            if current_weights in (short, filename):
                self.weights_combo.setCurrentIndex(self.weights_combo.findData(short))
        self.weights_combo.setToolTip("Which multilingual weights to load. V3 is the newest.")
        self.weights_label = QLabel("Weights")
        form.addRow(self.weights_label, self.weights_combo)

        self.variant_combo = QComboBox()
        for key, label in model_registry.QWEN_VARIANTS.items():
            self.variant_combo.addItem(label, key)
            self.variant_combo.setItemData(self.variant_combo.count() - 1,
                                           qwen_engine.VARIANTS[key], Qt.ItemDataRole.ToolTipRole)
        variant_index = self.variant_combo.findData(entry.get("qwen_variant"))
        self.variant_combo.setCurrentIndex(max(0, variant_index))
        self.variant_label = QLabel("Qwen variant")
        form.addRow(self.variant_label, self.variant_combo)

        self.mode_combo = QComboBox()
        for key, label in model_registry.VOICE_MODES.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(model_registry.entry_mode(entry))))
        self.mode_combo.setToolTip("This model does both; add one entry for each to switch between them.")
        self.mode_label = QLabel("Use for")
        form.addRow(self.mode_label, self.mode_combo)

        self.language_combo = QComboBox()
        self.language_combo.setToolTip("Language selected by default when this model is loaded.")
        form.addRow("Default language", self.language_combo)
        self._preferred_language = entry.get("language_id", "en")

        self.test_text_input = QLineEdit(entry.get("test_text", ""))
        self.test_text_input.setPlaceholderText("Optional sentence used by 'Sample text'")
        form.addRow("Sample sentence", self.test_text_input)
        self.notes_input = QLineEdit(entry.get("notes", ""))
        self.notes_input.setPlaceholderText("Optional")
        form.addRow("Notes", self.notes_input)
        self.enabled_checkbox = QCheckBox("Show in the model switcher")
        self.enabled_checkbox.setChecked(bool(entry.get("enabled", True)))
        form.addRow("", self.enabled_checkbox)
        layout.addLayout(form)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #d9534f;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._engine_changed()
        if entry.get("repo_id") and not entry.get("label"):
            QTimer.singleShot(0, self.check_repo)

    def find_repo(self):
        existing = {e.get("repo_id") for e in self.other_entries}
        finder = FindModelsDialog(self.token, existing, self)
        if dialog_accepted(finder.exec()) and finder.selected_repo:
            self.repo_input.setText(finder.selected_repo)
            self.name_input.clear()
            self.check_repo()

    def _repo_edited(self, text):
        self.check_label.setText("Click Check to confirm this repo can be loaded.")
        self.check_label.setStyleSheet("")

    def _engine_changed(self, *_args):
        engine = model_registry.ENGINES[self.engine_combo.currentData()]
        self.weights_combo.setVisible(engine.uses_weights_version)
        self.weights_label.setVisible(engine.uses_weights_version)
        self.variant_combo.setVisible(engine.key == QWEN_BACKEND)
        self.variant_label.setVisible(engine.key == QWEN_BACKEND)
        self.mode_combo.setVisible(engine.key in DUAL_MODE_BACKENDS)
        self.mode_label.setVisible(engine.key in DUAL_MODE_BACKENDS)
        current = self.language_combo.currentData() or self._preferred_language
        self.language_combo.clear()
        for language_id, name in languages_for_backend(engine.key).items():
            self.language_combo.addItem(f"{name} [{language_id}]", language_id)
        index = self.language_combo.findData(current)
        self.language_combo.setCurrentIndex(index if index >= 0 else max(0, self.language_combo.findData("en")))

    def check_repo(self):
        repo_id = self.repo_input.text().strip()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = model_registry.check_repo(repo_id, self.token)
        finally:
            QApplication.restoreOverrideCursor()
        self.check_label.setText(result.message)
        self.check_label.setStyleSheet("color: #3c9a3c;" if result.ok else "color: #d9534f;")
        self.adjustSize()
        if result.ok:
            self.checked = {"download_bytes": result.download_bytes}
            if result.license:
                self.checked["license"] = result.license
            self.engine_combo.setCurrentIndex(self.engine_combo.findData(result.detected_backend))
            if result.qwen_variant:
                self.variant_combo.setCurrentIndex(self.variant_combo.findData(result.qwen_variant))
            if result.weight_versions and self.weights_combo.currentData() not in result.weight_versions:
                self.weights_combo.setCurrentIndex(self.weights_combo.findData(result.weight_versions[0]))
            if not self.name_input.text().strip():
                self.name_input.setText(repo_id.split("/")[-1].replace("-", " ").replace("_", " "))
        return result

    def result_entry(self):
        engine = self.engine_combo.currentData()
        entry = dict(self.original)
        entry.update({
            "repo_id": self.repo_input.text().strip(),
            "label": self.name_input.text().strip(),
            "backend": engine,
            "language_id": self.language_combo.currentData() or "en",
            "test_text": self.test_text_input.text().strip(),
            "notes": self.notes_input.text().strip(),
            "enabled": self.enabled_checkbox.isChecked(),
            "experimental": entry.get("experimental", False),
            "multilingual_t3_model": self.weights_combo.currentData() if engine == "multilingual" else "",
            "qwen_variant": self.variant_combo.currentData() if engine == QWEN_BACKEND else "",
            "mode": self.mode_combo.currentData() if engine in DUAL_MODE_BACKENDS else "",
            "test_texts": entry.get("test_texts", {}),
        })
        if self.repo_input.text().strip() == self.original.get("repo_id") or self.checked:
            entry.update(self.checked)
        else:
            entry.pop("license", None)  # a different repo: don't keep the old one's license
            entry.pop("download_bytes", None)
        return entry

    def _save(self):
        entry = self.result_entry()
        problem = None
        if not model_registry.is_valid_repo_id(entry["repo_id"]):
            problem = "Enter the Hugging Face repo as owner/name."
        elif not entry["label"]:
            problem = "Give the model a name."
        elif any(other.get("label") == entry["label"] for other in self.other_entries):
            problem = "Another model already uses this name."
        if problem:
            self.error_label.setText(problem)
            self.error_label.setVisible(True)
            self.adjustSize()
            return
        self.accept()


class VoiceDetailsDialog(QDialog):
    """Name, tags and notes for a library voice (and the transcript of a clip)."""

    def __init__(self, title, voice, library, transcript=None, offer_clip=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.voice = voice
        self.library = library
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.name_input = QLineEdit(voice.name)
        form.addRow("Name", self.name_input)
        self.tags_input = QLineEdit(", ".join(voice.tags))
        self.tags_input.setPlaceholderText("Optional, e.g. narrator, warm, project name")
        form.addRow("Tags", self.tags_input)
        self.notes_input = QLineEdit(voice.notes)
        self.notes_input.setPlaceholderText("Optional")
        form.addRow("Notes", self.notes_input)
        self.transcript_input = None
        if transcript is not None:
            self.transcript_input = QLineEdit(transcript)
            self.transcript_input.setPlaceholderText("What is said in the clip (needed by some cloning models)")
            form.addRow("Transcript", self.transcript_input)
        layout.addLayout(form)
        self.clip_checkbox = None
        if offer_clip:
            self.clip_checkbox = QCheckBox("Also make a clip of this voice reading a short passage")
            self.clip_checkbox.setChecked(True)
            self.clip_checkbox.setToolTip(
                "Generates about 15 seconds with this voice and keeps it with its exact transcript, "
                "so every cloning model (Chatterbox, Qwen, VoxCPM, OmniVoice, VibeVoice) can use "
                "the same voice.")
            layout.addWidget(self.clip_checkbox)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #d9534f;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self):
        name = self.name_input.text().strip()
        if not name:
            self.error_label.setText("Give the voice a name.")
        elif any(other.name.lower() == name.lower() for other in self.library.voices if other is not self.voice):
            self.error_label.setText("Another voice already has this name.")
        else:
            self.accept()
            return
        self.error_label.setVisible(True)

    def apply(self):
        self.voice.name = self.name_input.text().strip()
        self.voice.tags = [tag.strip() for tag in self.tags_input.text().split(",") if tag.strip()]
        self.voice.notes = self.notes_input.text().strip()
        return self.transcript_input.text().strip() if self.transcript_input is not None else None

    def make_clip(self):
        return bool(self.clip_checkbox and self.clip_checkbox.isChecked())


class PronunciationDialog(QDialog):
    """Edit the pronunciation dictionary: "write this" -> "say it as"."""

    COLUMNS = ("Write", "Say it as", "Whole word", "Match case", "On")

    def __init__(self, dictionary, speak, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pronunciation dictionary")
        self.setMinimumSize(640, 460)
        self.dictionary = dictionary
        self.speak = speak  # callable(text): plays text with the loaded model
        layout = QVBoxLayout(self)
        intro = QLabel("Respell words the way they should sound, e.g. Nguyen \u2192 Win, SQL \u2192 sequel, "
                       "Siobhan \u2192 Shiv-awn. Works with every model; subtitles keep your spelling.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(lambda _item: self.update_try())
        for rule in dictionary.rules:
            self._add_row(rule)
        layout.addWidget(self.table, 1)
        row_actions = QHBoxLayout()
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_rule)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_rules)
        self.hear_button = QPushButton("Hear it")
        self.hear_button.setToolTip("Speak the selected respelling with the loaded model and voice.")
        self.hear_button.clicked.connect(self.hear_selected)
        for button in (add_button, remove_button, self.hear_button):
            row_actions.addWidget(button)
        row_actions.addStretch(1)
        import_button = self._link_button("Import\u2026", self.import_rules)
        export_button = self._link_button("Export\u2026", self.export_rules)
        row_actions.addWidget(import_button)
        row_actions.addWidget(export_button)
        layout.addLayout(row_actions)
        try_row = QHBoxLayout()
        try_row.addWidget(QLabel("Try"))
        self.try_input = QLineEdit()
        self.try_input.setPlaceholderText("Type a sentence to see (and hear) how it will be read")
        self.try_input.textChanged.connect(lambda _text: self.update_try())
        try_row.addWidget(self.try_input, 1)
        hear_try = QPushButton("Hear")
        hear_try.clicked.connect(lambda: self.speak(self.try_result()) if self.try_input.text().strip() else None)
        try_row.addWidget(hear_try)
        layout.addLayout(try_row)
        self.try_output = QLabel()
        self.try_output.setObjectName("Muted")
        self.try_output.setWordWrap(True)
        layout.addWidget(self.try_output)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _link_button(self, text, slot):
        button = QPushButton(text)
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        return button

    def _add_row(self, rule):
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(rule.word))
        self.table.setItem(row, 1, QTableWidgetItem(rule.say))
        for column, value in ((2, rule.whole_word), (3, rule.match_case), (4, rule.enabled)):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
            self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        return row

    def rules(self):
        rules = []
        for row in range(self.table.rowCount()):
            text = lambda column: (self.table.item(row, column).text() if self.table.item(row, column) else "").strip()
            checked = lambda column: self.table.item(row, column).checkState() == Qt.CheckState.Checked
            if text(0):
                rules.append(pronunciation.Rule(text(0), text(1), checked(2), checked(3), checked(4)))
        return rules

    def add_rule(self):
        row = self._add_row(pronunciation.Rule("", ""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def remove_rules(self):
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
        self.update_try()

    def hear_selected(self):
        row = self.table.currentRow()
        if row >= 0 and self.table.item(row, 1) and self.table.item(row, 1).text().strip():
            self.speak(self.table.item(row, 1).text().strip())

    def try_result(self):
        preview = pronunciation.Dictionary.__new__(pronunciation.Dictionary)
        preview.rules, preview.enabled, preview._pattern, preview._key = self.rules(), True, None, None
        return preview.apply(self.try_input.text())[0]

    def update_try(self):
        text = self.try_input.text().strip()
        self.try_output.setText(f"Read as: {self.try_result()}" if text else "")

    def import_rules(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import pronunciations", "", "Word lists (*.json *.txt *.csv *.tsv);;All files (*.*)")
        if not path:
            return
        try:
            imported = pronunciation.import_rules(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import", f"Couldn't read {os.path.basename(path)}:\n{exc}")
            return
        existing = {rule.word.lower() for rule in self.rules()}
        added = [rule for rule in imported if rule.word.lower() not in existing]
        for rule in added:
            self._add_row(rule)
        self.update_try()
        QMessageBox.information(self, "Import", f"Added {len(added)} of {len(imported)} words "
                                f"({len(imported) - len(added)} were already in the dictionary).")

    def export_rules(self):
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export pronunciations", "pronunciations.txt", "Text list (*.txt);;JSON (*.json)")
        if path:
            pronunciation.export_rules(path, self.rules())


class ApiBridge(QObject):
    """The local API's backend. HTTP requests arrive on server threads; anything that
    touches the app's state runs on the UI thread, and generation itself runs on
    the request's thread through the same generator the Generate button uses."""

    run_requested = Signal(object)
    OPENAI_MODEL_NAMES = {"", "tts-1", "tts-1-hd", "gpt-4o-mini-tts", "default"}
    LOAD_TIMEOUT = 900

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.job_lock = threading.Lock()  # one API generation at a time
        self.run_requested.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, call):
        function, box, done = call
        try:
            box["value"] = function()
        except Exception as exc:
            box["error"] = exc
        done.set()

    def on_ui(self, function, timeout=30):
        box, done = {}, threading.Event()
        self.run_requested.emit((function, box, done))
        if not done.wait(timeout):
            raise local_api.ApiError(503, "The app didn't respond in time.", "server_error")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def health(self):
        return self.on_ui(self.app.api_health)

    def models(self):
        return self.on_ui(self.app.api_models)

    def voices(self):
        return self.on_ui(self.app.api_voices)

    def synthesize(self, request):
        if not self.job_lock.acquire(timeout=self.LOAD_TIMEOUT):
            raise local_api.ApiError(503, "Another API request is still running.", "server_error")
        try:
            deadline = time.monotonic() + self.LOAD_TIMEOUT
            while True:
                job = self.on_ui(lambda: self.app.api_begin(request))
                if not job.get("loading"):
                    break
                if time.monotonic() > deadline:
                    raise local_api.ApiError(504, "The model took too long to load.", "server_error")
                time.sleep(0.5)  # a model is loading for this request
            try:
                return self._generate(job)
            finally:
                self.on_ui(self.app.api_end)
        finally:
            self.job_lock.release()

    def _generate(self, job):
        outcome = {}
        thread = AudioGeneratorThread(**job["generator"])
        thread.pronunciations = job["pronunciations"]
        thread.generation_complete.connect(
            lambda path, sr: outcome.update(path=path, sr=sr), Qt.ConnectionType.DirectConnection)
        thread.error_occurred.connect(
            lambda message: outcome.update(error=message), Qt.ConnectionType.DirectConnection)
        started = time.monotonic()
        thread.run()  # in this request's thread, not a new one
        if "path" not in outcome:
            raise local_api.ApiError(500, outcome.get("error", "Generation failed."), "server_error")
        import soundfile
        info = soundfile.info(outcome["path"])
        return {
            "path": outcome["path"], "subtitles": thread.subtitle_path,
            "seconds": round(info.duration, 2), "sample_rate": outcome["sr"],
            "generation_seconds": round(time.monotonic() - started, 2),
            "model": job["model_label"], "voice": job["voice_label"],
            "mime": job["mime"], "temporary": job["temporary"], "seed": thread.actual_seed_used,
        }


class TaskThread(QThread):
    """Runs one function off the UI thread and reports (result, error message)."""

    done = Signal(object, str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.done.emit(self.function(), "")
        except Exception as exc:
            self.done.emit(None, str(exc))


class GoogleDocsDialog(QDialog):
    """Open a Google Doc: a shared link, or your own docs after signing in."""

    def __init__(self, account, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open from Google Docs")
        self.setMinimumSize(620, 480)
        self.account = account
        self.result_docx = None  # (bytes, title) once a doc is chosen
        self.threads = []
        self.cancel_sign_in = False
        layout = QVBoxLayout(self)

        link_title = QLabel("Shared link")
        link_title.setObjectName("CardTitle")
        layout.addWidget(link_title)
        link_hint = QLabel("For docs shared as \u201cAnyone with the link can view\u201d. No sign-in needed.")
        link_hint.setObjectName("Muted")
        layout.addWidget(link_hint)
        link_row = QHBoxLayout()
        self.link_input = QLineEdit()
        self.link_input.setPlaceholderText("https://docs.google.com/document/d/...")
        self.link_input.returnPressed.connect(self.open_link)
        link_row.addWidget(self.link_input, 1)
        self.link_button = QPushButton("Open link")
        self.link_button.clicked.connect(self.open_link)
        link_row.addWidget(self.link_button)
        layout.addLayout(link_row)

        layout.addSpacing(8)
        account_row = QHBoxLayout()
        mine_title = QLabel("Your Google Docs")
        mine_title.setObjectName("CardTitle")
        account_row.addWidget(mine_title)
        account_row.addStretch(1)
        self.account_label = QLabel()
        self.account_label.setObjectName("Muted")
        account_row.addWidget(self.account_label)
        self.setup_button = self._link("Set up...", self.set_up_client)
        self.setup_button.setToolTip("Load the OAuth client (client_secret_....json) from Google Cloud Console.")
        account_row.addWidget(self.setup_button)
        self.sign_in_button = QPushButton("Sign in with Google")
        self.sign_in_button.clicked.connect(self.sign_in)
        account_row.addWidget(self.sign_in_button)
        self.sign_out_button = self._link("Sign out", self.sign_out)
        account_row.addWidget(self.sign_out_button)
        layout.addLayout(account_row)
        self.setup_hint = QLabel(
            "Signing in needs a free Google Cloud OAuth client (Desktop app) with the Google Drive API "
            "enabled; the README walks through it in about 5 minutes. Access is read-only, and your "
            "sign-in stays on this PC.")
        self.setup_hint.setObjectName("Note")
        self.setup_hint.setWordWrap(True)
        layout.addWidget(self.setup_hint)
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search your docs by name")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.refresh_docs)
        search_row.addWidget(self.search_input, 1)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.refresh_docs)
        search_row.addWidget(self.search_button)
        layout.addLayout(search_row)
        self.docs_list = QListWidget()
        self.docs_list.itemDoubleClicked.connect(lambda _item: self.open_selected())
        layout.addWidget(self.docs_list, 1)
        self.status_label = QLabel()
        self.status_label.setObjectName("Muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.open_button = QPushButton("Open selected")
        self.open_button.setProperty("accent", True)
        self.open_button.clicked.connect(self.open_selected)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.update_account()
        if self.account.signed_in:
            QTimer.singleShot(0, self.refresh_docs)

    def _link(self, text, slot):
        button = QPushButton(text)
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        return button

    def run(self, function, on_done, status):
        self.status_label.setText(status)
        self.set_busy(True)
        thread = TaskThread(function, self)

        def finished(result, error):
            self.set_busy(False)
            on_done(result, error)

        thread.done.connect(finished)
        self.threads.append(thread)
        thread.start()

    def set_busy(self, busy):
        for widget in (self.link_button, self.search_button, self.open_button, self.setup_button):
            widget.setEnabled(not busy)
        self.sign_in_button.setEnabled(not busy and self.account.configured)

    def update_account(self):
        signed_in = self.account.signed_in
        self.account_label.setText(f"Signed in as {self.account.email}" if signed_in and self.account.email
                                   else "Signed in" if signed_in else "")
        self.sign_in_button.setVisible(not signed_in)
        self.sign_in_button.setEnabled(self.account.configured)
        self.sign_in_button.setToolTip("" if self.account.configured else "Set up a Google client first.")
        self.sign_out_button.setVisible(signed_in)
        self.setup_button.setVisible(not signed_in)
        self.setup_hint.setVisible(not self.account.configured)
        for widget in (self.search_input, self.search_button, self.docs_list):
            widget.setEnabled(signed_in)
        if not signed_in:
            self.docs_list.clear()

    def set_up_client(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Google OAuth client", os.path.expanduser("~/Downloads"), "Client file (*.json)")
        if not path:
            return
        try:
            self.account.load_client_file(path)
        except (google_docs.GoogleDocsError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Google", f"Couldn't use that file:\n{exc}")
            return
        self.status_label.setText("Client loaded. Click Sign in with Google.")
        self.update_account()

    def sign_in(self):
        self.cancel_sign_in = False

        def done(_result, error):
            self.update_account()
            if error:
                self.status_label.setText(error)
            else:
                self.refresh_docs()

        self.run(lambda: self.account.sign_in(lambda: self.cancel_sign_in), done,
                 "Finish signing in in your browser\u2026 (this window waits up to 5 minutes)")

    def sign_out(self):
        self.account.sign_out()
        self.update_account()
        self.status_label.setText("Signed out. The app's access to your Google account was revoked.")

    def refresh_docs(self):
        search = self.search_input.text()

        def done(docs, error):
            self.docs_list.clear()
            if error:
                self.status_label.setText(error)
                self.update_account()
                return
            for doc_id, name, modified in docs:
                item = QListWidgetItem(f"{name}    {modified[:10]}")
                item.setData(Qt.ItemDataRole.UserRole, doc_id)
                self.docs_list.addItem(item)
            self.status_label.setText(f"{len(docs)} doc{'s' if len(docs) != 1 else ''}, newest first."
                                      if docs else "No docs found.")

        self.run(lambda: self.account.list_documents(search), done, "Loading your docs\u2026")

    def finish(self, result, error):
        if error:
            self.status_label.setText(error)
            return
        self.result_docx = result
        self.accept()

    def open_link(self):
        doc_id = google_docs.doc_id_from_url(self.link_input.text())
        if not doc_id:
            self.status_label.setText("Paste a Google Docs link (docs.google.com/document/d/...).")
            return

        def fetch():
            try:
                return google_docs.download_shared(doc_id)
            except google_docs.GoogleDocsError:
                if self.account.signed_in:  # not public, but maybe your account can open it
                    return self.account.export_docx(doc_id)
                raise

        self.run(fetch, self.finish, "Downloading\u2026")

    def open_selected(self):
        item = self.docs_list.currentItem()
        if item is None:
            if self.link_input.text().strip():
                self.open_link()
            return
        doc_id = item.data(Qt.ItemDataRole.UserRole)
        self.run(lambda: self.account.export_docx(doc_id), self.finish, "Downloading\u2026")

    def reject(self):
        self.cancel_sign_in = True
        super().reject()


class SpeakThread(QThread):
    """Speak a short text with the loaded model (for trying respellings)."""

    finished_with = Signal(object, int, str)  # waveform, sample rate, error

    def __init__(self, model, text, kwargs, parent=None):
        super().__init__(parent)
        self.model, self.text, self.kwargs = model, text, kwargs

    def run(self):
        try:
            try:
                wav = self.model.generate(self.text, **self.kwargs)
            except TypeError:
                wav = self.model.generate(self.text)  # older loaders take fewer options
            data = wav.squeeze(0).detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            self.finished_with.emit(np.asarray(data, dtype=np.float32).reshape(-1), int(self.model.sr), "")
        except Exception as exc:
            self.finished_with.emit(None, 0, str(exc))


class MakeClipThread(QThread):
    """Generates a reference clip with the current voice: a phonetically rich passage, so
    the clip comes with an exact transcript."""

    finished_with = Signal(object, int, str)  # waveform (numpy) or None, sample rate, error

    def __init__(self, model, text, language_id, parent=None):
        super().__init__(parent)
        self.model = model
        self.text = text
        self.language_id = language_id

    def run(self):
        try:
            wav = self.model.generate(self.text, language_id=self.language_id)
            data = wav.squeeze(0).detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            self.finished_with.emit(np.asarray(data, dtype=np.float32).reshape(-1), int(self.model.sr), "")
        except Exception as exc:
            self.finished_with.emit(None, 0, str(exc))


class CastDialog(QDialog):
    """Pick a voice for each speaker in a conversation script."""

    BROWSE = "__browse__"

    def __init__(self, speakers, cast, sample_paths, recordings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cast")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        intro = QLabel("Choose a voice for each speaker in the script. Sample voices come with "
                       "VibeVoice; clip voices from your library and any clip work too. Only clone "
                       "voices of people who have agreed to it.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.combos = {}
        for row, speaker in enumerate(speakers):
            grid.addWidget(QLabel(speaker), row, 0)
            combo = QComboBox()
            for name, path in sample_paths.items():
                combo.addItem(f"{name}  (sample)", path)
            if recordings:
                combo.insertSeparator(combo.count())
                for label, path in recordings:
                    combo.addItem(label, path)
            combo.insertSeparator(combo.count())
            combo.addItem("Other clip\u2026", self.BROWSE)
            current = cast.get(speaker)
            if current and combo.findData(current) < 0:
                combo.insertItem(combo.count() - 2, os.path.basename(current), current)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            combo.setProperty("previous", combo.currentIndex())
            combo.activated.connect(lambda _index, combo=combo: self._maybe_browse(combo))
            grid.addWidget(combo, row, 1)
            self.combos[speaker] = combo
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _maybe_browse(self, combo):
        if combo.currentData() != self.BROWSE:
            combo.setProperty("previous", combo.currentIndex())
            return
        path, _filter = QFileDialog.getOpenFileName(self, "Choose a voice clip", "",
                                                    "Audio Files (*.wav *.mp3 *.flac)")
        if path:
            combo.insertItem(combo.count() - 2, os.path.basename(path), path)
            combo.setCurrentIndex(combo.count() - 3)
            combo.setProperty("previous", combo.currentIndex())
        else:
            combo.setCurrentIndex(combo.property("previous") or 0)

    def result_cast(self):
        return {speaker: combo.currentData() for speaker, combo in self.combos.items()
                if combo.currentData() and combo.currentData() != self.BROWSE}


class EngineInstallThread(QThread):
    finished_with = Signal(str)

    def __init__(self, engine_module, parent=None):
        super().__init__(parent)
        self.engine_module = engine_module

    def run(self):
        try:
            self.engine_module.install(log=print)
            self.finished_with.emit("")
        except Exception as exc:
            self.finished_with.emit(str(exc))


# --- ChatterboxApp ---


class ChatterboxApp(QMainWindow):
    log_message_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox TTS Interface")
        self.setWindowIcon(QIcon(ui_theme.LOGO))
        self.resize(1000, 820)
        self.model = None
        self.device_used = "cpu"
        self.current_model_repo = DEFAULT_MODEL_REPO
        self.current_model_backend = BACKEND_MULTILINGUAL
        self.current_multilingual_t3_model = resolve_multilingual_t3_model(DEFAULT_MULTILINGUAL_T3_MODEL)
        self.selected_model_repo = DEFAULT_MODEL_REPO
        self.system_has_nvidia_gpu = has_system_nvidia_gpu()
        self.cuda_runtime_issue = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_config_path = os.path.join(self.script_dir, MODEL_CONFIG_FILENAME)
        self.app_settings_path = os.path.join(self.script_dir, APP_SETTINGS_FILENAME)
        self.model_entries = load_models_config(self.model_config_path)
        self.app_settings = self.load_app_settings()
        self.apply_hf_token_setting()
        self.current_audio_file = None
        self.output_directory = os.path.join(os.getcwd(), "chatterbox_outputs")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.last_reference_audio_dir = self.script_dir
        self.recordings_directory = os.path.join(
            self.script_dir, REFERENCE_RECORDINGS_DIRNAME)
        self.voice_library = voice_library.VoiceLibrary(self.script_dir, self.recordings_directory)
        self.pronunciations = pronunciation.Dictionary(os.path.join(self.script_dir, pronunciation.FILENAME))
        self.api_settings = dict({"enabled": False, "port": local_api.DEFAULT_PORT, "token": ""},
                                 **self.app_settings.get("api", {}))
        self.api_server = None
        self.api_bridge = ApiBridge(self)
        self.api_busy = False
        self.api_loading_entry = None
        self.google_account = google_docs.GoogleAccount(self.script_dir)
        self.active_voice_id = None
        self.pending_voice = None
        self.media_devices = QMediaDevices(self)
        self.recording_format = None
        self.recording_buffer = bytearray()

        self.preview_player = QMediaPlayer(self)
        self.preview_audio_output = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio_output)
        self.preview_button_playing = None
        self.preview_player.playbackStateChanged.connect(self._on_preview_state_changed)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.is_seeking_audio = False
        self.paused_position = 0  # Added paused_position here

        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_duration_info)
        self.media_player.playbackStateChanged.connect(
            self.handle_playback_state_changed)
        self.media_player.errorOccurred.connect(self.handle_media_error)

        self.generation_timer = QTimer(self)
        self.generation_timer.timeout.connect(
            self.update_generation_time_display)  # Renamed for clarity
        self.generation_start_time = None
        self.is_generating = False
        self.model_is_warm = False
        self.generation_is_preview = False
        self.generation_started_at = None
        self.generation_char_count = 0
        self.current_document_name = None
        self.last_preview_seed = None
        self.text_stats_timer = QTimer(self)
        self.text_stats_timer.setSingleShot(True)
        self.text_stats_timer.setInterval(350)
        self.text_stats_timer.timeout.connect(self.update_text_stats)
        sampling = self.app_settings.get("sampling", {})
        self.repetition_penalty = float(sampling.get("repetition_penalty", 1.2))
        self.min_p = float(sampling.get("min_p", 0.05))
        self.top_p = float(sampling.get("top_p", 1.0))

        self.log_message_signal.connect(self.append_console_log)
        self._init_ui()
        self.fit_default_geometry()
        self.restore_window_settings()
        self.update_minimum_size()
        self.attach_log_sink()
        self.update_output_log()
        if CHATTERBOX_AVAILABLE:
            if self.system_has_nvidia_gpu and not torch.cuda.is_available():
                print(
                    "WARNING: NVIDIA GPU detected, but the installed PyTorch build "
                    "does not have CUDA enabled. The app will run on CPU."
                )
            self.set_status_message(
                "Status: App ready. Default model load will start shortly. First run may download model files and can take several minutes."
            )
            QTimer.singleShot(0, self.start_default_model_load)
        else:
            self.set_status_message("Status: Chatterbox library not found.")
            self.generate_button.setEnabled(False)
        QTimer.singleShot(0, self.restart_api_server)

    PAGE_GENERATE, PAGE_VOICE, PAGE_MODEL, PAGE_ADVANCED, PAGE_LOG = range(5)

    def _make_card(self, title=None):
        card = ui_theme.CardFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("CardTitle")
            layout.addWidget(title_label)
        return card, layout

    def _make_page(self, title, subtitle):
        page = QWidget()
        layout = QVBoxLayout(page)
        # Cards carry their own shadow margin, so page spacing is tighter.
        layout.setContentsMargins(17, 14, 17, 10)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PageSubtitle")
        # Line up with the cards' visible edge (inside their shadow margin).
        for label in (title_label, subtitle_label):
            label.setContentsMargins(ui_theme.SHADOW, 0, ui_theme.SHADOW, 0)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(6)
        return page, layout

    @staticmethod
    def _accent(button):
        button.setProperty("accent", True)
        return button

    @staticmethod
    def _link(button):
        button.setFlat(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QHBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar_panel = ui_theme.SidebarPanel()
        sidebar_panel.setObjectName("SidebarPanel")
        sidebar_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar_layout = QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(180)
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label in ("Generate", "Voice", "Model", "Advanced", "Log"):
            self.sidebar.addItem(QListWidgetItem(label))
        title_row = QHBoxLayout()
        title_row.setContentsMargins(18, 16, 12, 10)
        title_row.setSpacing(9)
        logo = QLabel()
        logo.setPixmap(QIcon(ui_theme.LOGO).pixmap(26, 26))
        title_row.addWidget(logo)
        app_title = QLabel("Chatterbox")
        app_title.setObjectName("AppTitle")
        title_row.addWidget(app_title)
        title_row.addStretch(1)
        sidebar_layout.addLayout(title_row)
        sidebar_layout.addWidget(self.sidebar)
        self.pages = QStackedWidget()
        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.sidebar.currentRowChanged.connect(self.on_page_changed)
        content_area = ui_theme.TexturedArea()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.pages)
        root_layout.addWidget(sidebar_panel)
        root_layout.addWidget(content_area, 1)

        # ---------- Generate page ----------
        generate_page, generate_layout = self._make_page(
            "Generate", "Write your text, choose the delivery, then generate.")

        voice_row = QHBoxLayout()
        voice_row.setContentsMargins(ui_theme.SHADOW, 0, ui_theme.SHADOW, 2)
        voice_row.addWidget(QLabel("Voice"))
        self.voice_chip = ElidingChip("Default voice")
        self.voice_chip.setObjectName("VoiceChip")
        self.voice_chip.setTextFormat(Qt.TextFormat.PlainText)
        voice_row.addWidget(self.voice_chip)
        change_voice_button = self._link(QPushButton("Change..."))
        change_voice_button.clicked.connect(
            lambda: self.sidebar.setCurrentRow(self.PAGE_VOICE))
        voice_row.addWidget(change_voice_button)
        voice_row.addStretch(1)
        voice_row.addWidget(QLabel("Model"))
        self.model_repo_combo = QComboBox()
        self.model_repo_combo.setMinimumWidth(190)
        self.model_repo_combo.currentIndexChanged.connect(self.on_model_repo_changed)
        voice_row.addWidget(self.model_repo_combo)
        generate_layout.addLayout(voice_row)

        text_card, text_card_layout = self._make_card()
        text_header = QHBoxLayout()
        text_title = QLabel("Text")
        text_title.setObjectName("CardTitle")
        text_header.addWidget(text_title)
        self.document_label = QLabel()
        self.document_label.setObjectName("Muted")
        self.document_label.setTextFormat(Qt.TextFormat.PlainText)
        self.document_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text_header.addWidget(self.document_label, 1)
        text_card_layout.addLayout(text_header)
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Enter text to synthesize, or open a document. Long text is split where a reader "
            "would pause and stitched back together."
        )
        self.text_input.setMinimumHeight(90)
        self.text_input.setAcceptRichText(False)
        self.text_input.textChanged.connect(self.on_text_changed)
        # Fill the leftover height instead of forcing the page to scroll.
        self.text_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        text_card_layout.addWidget(self.text_input, 1)

        text_status_row = QHBoxLayout()
        text_status_row.setSpacing(10)
        self.estimate_button = self._link(QPushButton())
        self.estimate_button.setToolTip(
            "Estimated generation time with the active model. Click to compare every model.")
        self.estimate_button.clicked.connect(self.show_estimate_menu)
        self.estimate_button.setVisible(False)
        text_status_row.addWidget(self.estimate_button)
        self.text_stats_label = QLabel()
        self.text_stats_label.setObjectName("Muted")
        self.text_stats_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_stats_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.text_stats_label.setToolTip(
            "Updates as you edit. Times are learned per model from the sections you generate.")
        text_status_row.addWidget(self.text_stats_label, 1)
        self.activity_label = QLabel()
        self.activity_label.setTextFormat(Qt.TextFormat.PlainText)
        self.activity_label.setToolTip("Section being generated / total sections, and time left.")
        text_status_row.addWidget(self.activity_label)
        self.generation_progress = QProgressBar()
        self.generation_progress.setTextVisible(False)
        self.generation_progress.setFixedHeight(8)
        self.generation_progress.setFixedWidth(100)
        self.generation_progress.setVisible(False)
        text_status_row.addWidget(self.generation_progress)
        self.keep_take_button = self._link(QPushButton("Keep this take"))
        self.keep_take_button.setToolTip(
            "Lock the take number used by the preview so the full render matches it.")
        self.keep_take_button.clicked.connect(self.keep_preview_take)
        self.keep_take_button.setVisible(False)
        text_status_row.addWidget(self.keep_take_button)
        self.keep_voice_button = self._link(QPushButton("Keep this voice"))
        self.keep_voice_button.setToolTip(
            "Save the designed voice you just heard to the voice library and lock it in, so the full "
            "render (and later ones) use exactly this voice instead of designing a new one.")
        self.keep_voice_button.clicked.connect(self.keep_designed_voice)
        self.keep_voice_button.setVisible(False)
        text_status_row.addWidget(self.keep_voice_button)
        status_row_widget = QWidget()
        status_row_widget.setLayout(text_status_row)
        text_status_row.setContentsMargins(0, 0, 0, 0)
        status_row_widget.setFixedHeight(QPushButton("X").sizeHint().height())
        text_card_layout.addWidget(status_row_widget)

        generate_actions_layout = QHBoxLayout()
        self.open_document_button = QPushButton("Open...")
        self.open_document_button.setToolTip("Load a .txt, .md or .docx file, or a Google Doc, to read aloud.")
        open_menu = QMenu(self.open_document_button)
        open_menu.addAction("From this computer...").triggered.connect(lambda _checked=False: self.open_document())
        open_menu.addAction("From Google Docs...").triggered.connect(lambda _checked=False: self.open_google_doc())
        self.open_document_button.setMenu(open_menu)
        self.open_menu = open_menu
        generate_actions_layout.addWidget(self.open_document_button)
        self.use_preset_button = QPushButton("Sample")
        self.use_preset_button.setToolTip("Fill in a short test sentence for the selected language.")
        self.use_preset_button.clicked.connect(self.apply_selected_text_preset)
        generate_actions_layout.addWidget(self.use_preset_button)
        generate_actions_layout.addStretch()
        self.preview_length_combo = QComboBox()
        for label, characters in PREVIEW_LENGTHS:
            self.preview_length_combo.addItem(label, characters)
        saved_length = self.preview_length_combo.findData(self.app_settings.get("preview_chars", DEFAULT_PREVIEW_CHARS))
        self.preview_length_combo.setCurrentIndex(max(0, saved_length))
        self.preview_length_combo.setToolTip(
            "How much of the text Preview reads, from the start (whole sections, so it can run a "
            "little longer). Select text to preview exactly that instead.")
        self.preview_length_combo.currentIndexChanged.connect(
            lambda _index: self.app_settings.update(preview_chars=self.preview_length_combo.currentData()))
        self.preview_length_combo.setFixedWidth(84)
        generate_actions_layout.addWidget(self.preview_length_combo)
        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip(
            "Generate a short sample with the current settings before rendering everything: "
            "the selected text, or the opening of the text (length chosen on the left).")
        self.preview_button.clicked.connect(lambda: self.start_generation(preview=True))
        self.preview_button.setEnabled(False)
        generate_actions_layout.addWidget(self.preview_button)
        self.generate_button = self._accent(QPushButton("Generate Audio"))
        self.generate_button.clicked.connect(self.handle_generate_stop_toggle)
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumWidth(140)
        generate_actions_layout.addWidget(self.generate_button)
        text_card_layout.addLayout(generate_actions_layout)
        generate_layout.addWidget(text_card, 3)

        # Qwen controls share Delivery's first row with the Chatterbox-only sliders,
        # so switching engines never changes the window's minimum height.
        self.qwen_row = QWidget()
        qwen_row_layout = QHBoxLayout(self.qwen_row)
        qwen_row_layout.setContentsMargins(0, 0, 0, 0)
        qwen_row_layout.setSpacing(10)
        qwen_settings = self.app_settings.get("qwen", {})
        self.qwen_speaker_combo = QComboBox()
        # Kokoro's voice names are long ("Heart (US English, female)"); the list opens wide
        # anyway, so the box itself stays compact instead of widening the window.
        self.qwen_speaker_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.qwen_speaker_combo.setMinimumContentsLength(14)
        self.qwen_speaker_combo.view().setMinimumWidth(260)
        self.qwen_speaker_combo.setToolTip("Built-in Qwen speaker.")
        self.qwen_speaker_combo.currentIndexChanged.connect(lambda _i: self.refresh_voice_chip())
        self.qwen_speaker_label = QLabel("Speaker")
        qwen_row_layout.addWidget(self.qwen_speaker_label)
        qwen_row_layout.addWidget(self.qwen_speaker_combo)
        self.qwen_instruct_input = QLineEdit()
        self.qwen_instruct_label = QLabel("Style")
        qwen_row_layout.addWidget(self.qwen_instruct_label)
        qwen_row_layout.addWidget(self.qwen_instruct_input, 1)
        self.design_attributes_button = self._link(QPushButton("Attributes\u2026"))
        self.design_attributes_button.setToolTip("Pick the voice's gender, age, pitch, accent and more.")
        attributes_menu = QMenu(self)
        # Keep Python references: PySide can otherwise free submenus made by addMenu(title).
        self.design_attribute_menus = [attributes_menu]
        for group, items in omnivoice_engine.DESIGN_ATTRIBUTES.items():
            submenu = QMenu(group, attributes_menu)
            attributes_menu.addMenu(submenu)
            self.design_attribute_menus.append(submenu)
            for item in items:
                action = submenu.addAction(item)
                action.triggered.connect(lambda _checked=False, item=item: self.qwen_instruct_input.setText(
                    omnivoice_engine.set_attribute(self.qwen_instruct_input.text(), item)))
        attributes_menu.addSeparator()
        attributes_menu.addAction("Clear").triggered.connect(lambda: self.qwen_instruct_input.clear())
        self.design_attributes_button.setMenu(attributes_menu)
        self.design_attributes_button.setVisible(False)
        qwen_row_layout.addWidget(self.design_attributes_button)
        self.qwen_transcript_label = QLabel("Clip transcript")
        self.qwen_transcript_input = QLineEdit()
        self.qwen_transcript_input.setPlaceholderText(
            "What is said in the reference clip (optional, improves likeness)")
        self.qwen_transcript_input.setToolTip(
            "With a transcript, Qwen and VoxCPM clone more closely. It must match what is said in "
            "the clip: a wrong transcript can garble VoxCPM's output. Recordings made with "
            "Record... fill this in with the passage you read; edit it if you said something different.")
        self.qwen_transcript_input.editingFinished.connect(self.save_reference_transcript)
        qwen_row_layout.addWidget(self.qwen_transcript_label)
        qwen_row_layout.addWidget(self.qwen_transcript_input, 1)
        self.cast_label = QLabel()
        self.cast_label.setObjectName("Muted")
        self.cast_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.cast_button = QPushButton("Cast\u2026")
        self.cast_button.setToolTip("Choose a voice for each speaker in the script.")
        self.cast_button.clicked.connect(self.edit_cast)
        self.cast_title = QLabel("Speakers")
        for widget in (self.cast_title, self.cast_label, self.cast_button):
            widget.setVisible(False)
        qwen_row_layout.addWidget(self.cast_title)
        qwen_row_layout.addWidget(self.cast_label, 1)
        qwen_row_layout.addWidget(self.cast_button)
        self.qwen_watermark_checkbox = QCheckBox("Add AI watermark")
        self.qwen_watermark_checkbox.setChecked(bool(qwen_settings.get("watermark", True)))
        self.qwen_watermark_checkbox.setToolTip(
            "Qwen and Kokoro don't watermark their audio. When ticked, the same inaudible Perth watermark "
            "Chatterbox uses is added, so output from every engine is marked the same way.")
        self.qwen_settings = qwen_settings
        self.kokoro_settings = self.app_settings.get("kokoro", {})
        self.voxcpm_settings = self.app_settings.get("voxcpm", {})
        self.omnivoice_settings = self.app_settings.get("omnivoice", {})
        self.vibevoice_settings = self.app_settings.get("vibevoice", {})
        self.qwen_row.setVisible(False)
        self.qwen_watermark_checkbox.setVisible(False)

        delivery_card, delivery_layout = self._make_card("Delivery")
        delivery_layout.addWidget(self.qwen_row)
        params_layout = QGridLayout()
        params_layout.setHorizontalSpacing(14)
        params_layout.setVerticalSpacing(8)
        params_layout.setColumnStretch(1, 1)
        params_layout.setColumnStretch(3, 1)

        def add_control(row, column, title, widget, tooltip):
            label = QLabel(title)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            params_layout.addWidget(label, row, column)
            params_layout.addWidget(widget, row, column + 1)
            return label

        self.exaggeration_slider = self._create_slider(0.25, 2.0, 0.05, 0.5)
        self.exaggeration_label = add_control(0, 0, "Expressiveness", self.exaggeration_slider,
                    "How animated and emotional the delivery sounds. 0.5 is neutral; "
                    "higher is more dramatic (and often a bit faster). [exaggeration]")
        self.cfg_slider = self._create_slider(0.2, 1.0, 0.05, 0.5)
        self.cfg_label = add_control(0, 2, "Pacing", self.cfg_slider,
                    "Lower gives slower, more deliberate speech; higher is brisker and "
                    "follows the reference voice's style more closely. Try 0.3 for "
                    "expressive or fast-talking voices. [cfg_weight]")
        self.temp_slider = self._create_slider(0.05, 5.0, 0.05, 0.8)
        add_control(1, 0, "Variation", self.temp_slider,
                    "How different each take sounds. Higher is livelier but can become "
                    "unstable; lower is steadier and more predictable. [temperature]")
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1_000_000_000)
        self.seed_input.setValue(0)
        self.seed_input.setSpecialValueText("New take each time")
        self.seed_input.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.language_combo = QComboBox()
        self.language_combo.currentIndexChanged.connect(lambda _i: self.on_language_changed())
        add_control(2, 0, "Language", self.language_combo,
                    "Language of the text. The list depends on the selected model.")
        add_control(1, 2, "Take number", self.seed_input,
                    "Leave on 'New take each time' for a fresh result on every run. Enter a "
                    "number to reproduce the same take exactly; the number used is shown in "
                    "the file name. [seed]")
        params_layout.addWidget(self.qwen_watermark_checkbox, 2, 2, 1, 2)
        delivery_layout.addLayout(params_layout)
        delivery_hint = QLabel(
            "Tip: commas and ellipses add pauses; question marks lift the ending.")
        delivery_hint.setObjectName("Muted")
        delivery_hint.setToolTip("Advanced sampling options are under Model > Sampling.")
        delivery_layout.addWidget(delivery_hint)

        finishing_header = QHBoxLayout()
        self.finishing_toggle = self._link(QPushButton())
        self.finishing_toggle.setToolTip("Adjustments applied to the audio after it is generated.")
        self.finishing_toggle.clicked.connect(
            lambda: self.set_finishing_expanded(self.finishing_panel.isHidden()))
        finishing_header.addWidget(self.finishing_toggle)
        self.finishing_summary_label = QLabel()
        self.finishing_summary_label.setObjectName("Muted")
        finishing_header.addWidget(self.finishing_summary_label)
        finishing_header.addStretch(1)
        delivery_layout.addLayout(finishing_header)

        self.finishing_panel = QWidget()
        finishing_grid = QGridLayout(self.finishing_panel)
        finishing_grid.setContentsMargins(0, 0, 0, 0)
        finishing_grid.setHorizontalSpacing(14)
        finishing_grid.setColumnStretch(1, 1)
        finishing_grid.setColumnStretch(3, 1)

        def add_finishing(row, column, title, widget, tooltip):
            label = QLabel(title)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            finishing_grid.addWidget(label, row, column)
            finishing_grid.addWidget(widget, row, column + 1)

        self.pause_slider = self._create_slider(
            *audio_effects.PAUSE_RANGE, 0.1, 0.6, "{:.1f} s")
        add_finishing(0, 0, "Paragraph pause", self.pause_slider,
                      "Silence between paragraphs (headings get a little more). Pauses between "
                      "sentences and inside long sentences are kept short and even automatically.")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(LOSSLESS_FORMATS)
        add_finishing(0, 2, "Save as", self.output_format_combo,
                      "WAV is uncompressed; FLAC is lossless and about half the size. "
                      "MP3 is on the Advanced page.")
        finishing_checks = QHBoxLayout()
        self.even_volume_checkbox = QCheckBox("Even out volume")
        self.even_volume_checkbox.setToolTip(
            "Bring every result to a consistent, comfortable loudness without clipping.")
        self.trim_silence_checkbox = QCheckBox("Trim silence")
        self.trim_silence_checkbox.setToolTip(
            "Remove dead air before the first word and after the last.")
        finishing_checks.setSpacing(18)
        self.subtitles_checkbox = QCheckBox("Save subtitles")
        self.subtitles_checkbox.setToolTip(
            "Also save captions timed to the audio, next to it (.srt, or .vtt: the format is on the "
            "Advanced page). Timing comes from the generated sections and the pauses in them.")
        finishing_checks.addWidget(self.even_volume_checkbox)
        finishing_checks.addWidget(self.trim_silence_checkbox)
        finishing_checks.addWidget(self.subtitles_checkbox)
        finishing_checks.addStretch(1)
        reset_finishing_button = QPushButton("Reset")
        reset_finishing_button.setToolTip(
            "Restore the default finishing settings (Advanced effects are kept).")
        reset_finishing_button.clicked.connect(self.reset_finishing)
        finishing_checks.addWidget(reset_finishing_button)
        finishing_grid.addLayout(finishing_checks, 1, 0, 1, 4)
        delivery_layout.addWidget(self.finishing_panel)
        self.finishing_toggle.setToolTip(
            "Adjustments applied to the audio after it is generated. Speed, pitch and MP3 "
            "are on the Advanced page.")

        self.pause_slider.slider.valueChanged.connect(self.update_finishing_summary)
        self.output_format_combo.currentTextChanged.connect(self.update_finishing_summary)
        self.even_volume_checkbox.toggled.connect(self.update_finishing_summary)
        self.trim_silence_checkbox.toggled.connect(self.update_finishing_summary)
        self.subtitles_checkbox.toggled.connect(self.update_finishing_summary)
        generate_layout.addWidget(delivery_card)

        player_card, player_layout = self._make_card()
        player_header = QHBoxLayout()
        player_title = QLabel("Player")
        player_title.setObjectName("CardTitle")
        player_header.addWidget(player_title)
        player_header.addSpacing(12)
        self.autoplay_checkbox = QCheckBox("Auto-play results")
        self.autoplay_checkbox.setChecked(True)
        player_header.addWidget(self.autoplay_checkbox)
        player_header.addStretch(1)
        self.current_file_label = QLabel("Currently playing: None")
        self.current_file_label.setObjectName("Muted")
        player_header.addWidget(self.current_file_label)
        player_layout.addLayout(player_header)
        player_controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setMinimumWidth(80)
        player_controls_layout.addWidget(self.play_pause_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumWidth(80)
        player_controls_layout.addWidget(self.stop_button)
        self.current_time_label = QLabel("00:00")
        self.playhead_slider = QSlider(Qt.Orientation.Horizontal)
        self.playhead_slider.sliderPressed.connect(self.slider_pressed)
        self.playhead_slider.sliderMoved.connect(self.seek_audio_on_move)
        self.playhead_slider.sliderReleased.connect(self.slider_released)
        self.playhead_slider.setEnabled(False)
        self.duration_label = QLabel("00:00")
        player_controls_layout.addSpacing(8)
        player_controls_layout.addWidget(self.current_time_label)
        player_controls_layout.addWidget(self.playhead_slider, 1)
        player_controls_layout.addWidget(self.duration_label)
        player_layout.addLayout(player_controls_layout)
        history_label = QLabel("Generated files (double-click to play)")
        history_label.setObjectName("Muted")
        player_layout.addWidget(history_label)
        self.output_log_listwidget = QListWidget()
        self.output_log_listwidget.itemDoubleClicked.connect(
            self.play_selected_from_log)
        self.output_log_listwidget.setMinimumHeight(70)
        self.output_log_listwidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        player_layout.addWidget(self.output_log_listwidget, 1)
        generate_layout.addWidget(player_card, 2)
        self.pages.addWidget(generate_page)

        # ---------- Voice page ----------
        voice_page, voice_layout = self._make_page(
            "Voice", "Your voice library: recordings, clips, presets and designed voices.")

        current_card, current_layout = self._make_card("Current voice")
        current_row = QHBoxLayout()
        self.ref_audio_path_label = QLabel("None selected.")
        self.ref_audio_path_label.setWordWrap(False)
        self.ref_audio_path_label.setTextFormat(Qt.TextFormat.PlainText)
        # Long file names are clipped (full path in the tooltip) instead of widening the window.
        self.ref_audio_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        current_row.addWidget(self.ref_audio_path_label, 1)
        self.preview_reference_button = QPushButton("Preview")
        self.preview_reference_button.clicked.connect(self.toggle_reference_preview)
        current_row.addWidget(self.preview_reference_button)
        self.clear_reference_button = QPushButton("Use default voice")
        self.clear_reference_button.clicked.connect(self.clear_reference_audio)
        current_row.addWidget(self.clear_reference_button)
        save_current_button = QPushButton("Save to library...")
        save_current_button.setToolTip("Save the voice you're using: a clip, a preset or a designed voice.")
        save_current_button.clicked.connect(self.save_current_voice)
        current_row.addWidget(save_current_button)
        current_layout.addLayout(current_row)
        voice_layout.addWidget(current_card)

        library_card, library_layout = self._make_card()
        library_header = QHBoxLayout()
        self.voice_filter_tabs = QTabBar()
        self.voice_filter_tabs.setObjectName("CapabilityTabs")
        self.voice_filter_tabs.setDrawBase(False)
        self.voice_filter_tabs.setExpanding(False)
        self.voice_filter_tabs.setUsesScrollButtons(False)
        self.voice_filter_tabs.setCursor(Qt.CursorShape.PointingHandCursor)
        for key, title in (("", "All"), ("clip", "Clips"), ("preset", "Presets"), ("design", "Designed")):
            self.voice_filter_tabs.setTabData(self.voice_filter_tabs.addTab(title), key)
        self.voice_filter_tabs.currentChanged.connect(lambda _index: self.render_voice_tiles())
        library_header.addWidget(self.voice_filter_tabs)
        library_header.addStretch(1)
        self.voice_search = QLineEdit()
        self.voice_search.setPlaceholderText("Search voices")
        self.voice_search.setToolTip("Matches names, tags and notes.")
        self.voice_search.setClearButtonEnabled(True)
        self.voice_search.setFixedWidth(170)
        self.voice_search.textChanged.connect(lambda _text: self.render_voice_tiles())
        library_header.addWidget(self.voice_search)
        library_layout.addLayout(library_header)
        library_hint = QLabel("Click a voice to use it. Clip voices work with every cloning model; "
                              "presets and designed voices load their model.")
        library_hint.setObjectName("Muted")
        library_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        library_layout.addWidget(library_hint)
        self.voice_tiles = model_tiles.TileArea(min_rows=1)
        library_layout.addWidget(self.voice_tiles, 1)
        library_actions = QHBoxLayout()
        self.mic_combo = QComboBox()
        self.mic_combo.setToolTip("Microphone used for Record...")
        self.mic_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mic_combo.setMinimumContentsLength(10)
        library_actions.addWidget(self.mic_combo, 1)
        self.record_button = self._accent(QPushButton("Record..."))
        self.record_button.setToolTip(
            "Record a new clip voice: read about 15 seconds in a quiet room; the first few seconds "
            f"matter most ({MIN_RECORDING_SECONDS}-{MAX_RECORDING_SECONDS} s).")
        self.record_button.clicked.connect(self.open_recording_dialog)
        library_actions.addWidget(self.record_button)
        self.media_devices.audioInputsChanged.connect(self.populate_microphones)
        self.populate_microphones()
        browse_ref_button = QPushButton("Add a file...")
        browse_ref_button.setToolTip("Add a .wav, .mp3 or .flac clip to the library and use it.")
        browse_ref_button.clicked.connect(self.browse_reference_audio)
        library_actions.addWidget(browse_ref_button)
        open_recordings_button = self._link(QPushButton("Open folder"))
        open_recordings_button.clicked.connect(self.open_recordings_folder)
        library_actions.addWidget(open_recordings_button)
        library_layout.addLayout(library_actions)
        voice_layout.addWidget(library_card, 1)
        self.pages.addWidget(voice_page)

        # ---------- Model page ----------
        model_page, model_layout = self._make_page(
            "Model", "Models download once, then load from the local cache.")
        models_card, models_layout = self._make_card()
        tabs_row = QHBoxLayout()
        tabs_row.setSpacing(6)
        self.capability_tabs = QTabBar()
        self.capability_tabs.setObjectName("CapabilityTabs")
        self.capability_tabs.setDrawBase(False)
        self.capability_tabs.setExpanding(False)
        self.capability_tabs.setUsesScrollButtons(False)
        self.capability_tabs.setCursor(Qt.CursorShape.PointingHandCursor)
        for key, (title, description) in model_registry.CAPABILITIES.items():
            index = self.capability_tabs.addTab(model_registry.CAPABILITY_TABS[key])
            self.capability_tabs.setTabData(index, key)
            self.capability_tabs.setTabToolTip(index, f"{title}: {description}")
        self.capability_tabs.currentChanged.connect(lambda _index: self.render_model_tiles())
        tabs_row.addWidget(self.capability_tabs)
        tabs_row.addStretch(1)
        add_model_button = self._link(QPushButton("+ Add repo..."))
        add_model_button.setToolTip("Add a Hugging Face repo you already know, e.g. owner/model-name.")
        add_model_button.clicked.connect(self.add_model)
        tabs_row.addWidget(add_model_button)
        models_layout.addLayout(tabs_row)
        self.capability_note = QLabel()
        self.capability_note.setObjectName("Muted")
        self.capability_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        models_layout.addWidget(self.capability_note)

        your_label = QLabel("YOUR MODELS")
        your_label.setObjectName("SectionLabel")
        your_label.setToolTip("Click a model to load it. Right-click or \u22ef for edit, hide and remove.")
        models_layout.addWidget(your_label)
        self.your_tiles = model_tiles.TileArea(min_rows=1)
        models_layout.addWidget(self.your_tiles, 3)

        discover_row = QHBoxLayout()
        discover_label = QLabel("DISCOVER ON HUGGING FACE")
        discover_label.setObjectName("SectionLabel")
        discover_row.addWidget(discover_label)
        discover_row.addStretch(1)
        self.discover_input = QLineEdit()
        self.discover_input.setPlaceholderText("Search, e.g. norwegian, arabic, 0.6B")
        self.discover_input.setClearButtonEnabled(True)
        self.discover_input.setFixedWidth(210)
        self.discover_input.returnPressed.connect(self.start_discover)
        discover_row.addWidget(self.discover_input)
        discover_button = QPushButton("Search")
        discover_button.clicked.connect(self.start_discover)
        discover_row.addWidget(discover_button)
        models_layout.addLayout(discover_row)
        self.discover_tiles = model_tiles.TileArea(min_rows=1)
        models_layout.addWidget(self.discover_tiles, 2)
        self.discover_status = QLabel()
        self.discover_status.setObjectName("Muted")
        self.discover_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        models_layout.addWidget(self.discover_status)
        model_layout.addWidget(models_card, 1)

        hf_card, hf_layout = self._make_card("Hugging Face access")
        hf_row = QHBoxLayout()
        self.hf_token_input = QLineEdit(str(self.app_settings.get("hf_token", "")))
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setPlaceholderText("hf_... (optional)")
        self.hf_token_input.returnPressed.connect(self.save_hf_token)
        hf_row.addWidget(self.hf_token_input, 1)
        save_token_button = QPushButton("Save")
        save_token_button.clicked.connect(self.save_hf_token)
        hf_row.addWidget(save_token_button)
        test_token_button = QPushButton("Test")
        test_token_button.clicked.connect(self.test_hf_token)
        hf_row.addWidget(test_token_button)
        hf_layout.addLayout(hf_row)
        self.hf_token_status = QLabel()
        self.hf_token_status.setObjectName("Muted")
        self.hf_token_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hf_layout.addWidget(self.hf_token_status)
        model_layout.addWidget(hf_card)

        tuning_card, tuning_layout = self._make_card()
        tuning_header = QHBoxLayout()
        self.tuning_toggle = self._link(QPushButton())
        self.tuning_toggle.clicked.connect(lambda: self.set_tuning_expanded(self.tuning_panel.isHidden()))
        tuning_header.addWidget(self.tuning_toggle)
        self.tuning_summary_label = QLabel()
        self.tuning_summary_label.setObjectName("Muted")
        tuning_header.addWidget(self.tuning_summary_label)
        tuning_header.addStretch(1)
        tuning_layout.addLayout(tuning_header)
        self.tuning_panel = QWidget()
        tuning_grid = QGridLayout(self.tuning_panel)
        tuning_grid.setContentsMargins(0, 0, 0, 0)
        tuning_grid.setHorizontalSpacing(14)

        def tuning_spin(minimum, maximum, step, value):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.valueChanged.connect(self.on_tuning_changed)
            return spin

        self.repetition_spin = tuning_spin(0.5, 3.0, 0.05, self.repetition_penalty)
        self.min_p_spin = tuning_spin(0.0, 1.0, 0.01, self.min_p)
        self.top_p_spin = tuning_spin(0.0, 1.0, 0.01, self.top_p)
        for column, (title, spin, tip) in enumerate((
                ("Repetition control", self.repetition_spin,
                 "Discourages repeated words and stutters. Default 1.20; raise slightly if "
                 "phrases repeat. [repetition_penalty]"),
                ("Unlikely-sound filter", self.min_p_spin,
                 "Skips very unlikely sounds. Default 0.05; higher is steadier but flatter. [min_p]"),
                ("Top-p", self.top_p_spin,
                 "Limits choices to the most likely sounds. 1.00 means off. [top_p]"))):
            caption = QLabel(title)
            caption.setToolTip(tip)
            spin.setToolTip(tip)
            row, col = divmod(column, 2)
            tuning_grid.addWidget(caption, row, col * 2)
            tuning_grid.addWidget(spin, row, col * 2 + 1)
        tuning_grid.setColumnStretch(1, 1)
        tuning_grid.setColumnStretch(3, 1)
        tuning_reset = QPushButton("Reset")
        tuning_reset.clicked.connect(self.reset_tuning)
        tuning_grid.addWidget(tuning_reset, 1, 3, Qt.AlignmentFlag.AlignRight)
        tuning_layout.addWidget(self.tuning_panel)
        model_layout.addWidget(tuning_card)
        self.pages.addWidget(model_page)

        # ---------- Advanced page ----------
        advanced_page, advanced_layout = self._make_page(
            "Advanced", "Extra processing applied to results after they are generated.")
        effects_card, effects_layout = self._make_card("Voice effects")
        watermark_note = QLabel(
            "Pitch, speed and MP3 compression change the audio after it is generated and can "
            "weaken the inaudible watermark that marks it as AI-generated. Leave them at their "
            "defaults if that matters to you.")
        watermark_note.setObjectName("Note")
        watermark_note.setWordWrap(True)
        effects_layout.addWidget(watermark_note)
        effects_grid = QGridLayout()
        effects_grid.setHorizontalSpacing(14)
        effects_grid.setColumnStretch(1, 1)
        effects_grid.setColumnStretch(3, 1)
        self.speed_slider = self._create_slider(
            *audio_effects.SPEED_RANGE, 0.05, 1.0, "{:.2f}x")
        self.pitch_slider = self._create_slider(
            *audio_effects.PITCH_RANGE, 0.5, 0.0, "{:+.1f} st")
        for column, (title, slider, tip) in enumerate((
                ("Speed", self.speed_slider,
                 "Speaking speed without changing the pitch. 1.00x is unchanged."),
                ("Pitch", self.pitch_slider,
                 "Raise or lower the voice in semitones while keeping its natural character. "
                 "Small changes (1-2 st) sound most natural."))):
            caption = QLabel(title)
            caption.setToolTip(tip)
            slider.setToolTip(tip)
            effects_grid.addWidget(caption, 0, column * 2)
            effects_grid.addWidget(slider, 0, column * 2 + 1)
        effects_layout.addLayout(effects_grid)
        advanced_layout.addWidget(effects_card)

        pronunciation_card, pronunciation_layout = self._make_card("Pronunciation")
        pronunciation_row = QHBoxLayout()
        self.pronunciation_checkbox = QCheckBox("Use the pronunciation dictionary")
        self.pronunciation_checkbox.setChecked(self.pronunciations.enabled)
        self.pronunciation_checkbox.setToolTip(
            "Respell words before they're spoken (names, acronyms, jargon). Applies to every model; "
            "subtitles keep the original spelling.")
        self.pronunciation_checkbox.toggled.connect(self.on_pronunciation_toggled)
        pronunciation_row.addWidget(self.pronunciation_checkbox)
        self.pronunciation_summary = QLabel()
        self.pronunciation_summary.setObjectName("Muted")
        pronunciation_row.addWidget(self.pronunciation_summary)
        pronunciation_row.addStretch(1)
        edit_pronunciations = QPushButton("Edit dictionary...")
        edit_pronunciations.clicked.connect(self.edit_pronunciations)
        pronunciation_row.addWidget(edit_pronunciations)
        pronunciation_layout.addLayout(pronunciation_row)
        advanced_layout.addWidget(pronunciation_card)
        self.update_pronunciation_summary()

        api_card, api_layout = self._make_card("Local API")
        api_top = QHBoxLayout()
        self.api_checkbox = QCheckBox("Let other programs on this PC use the app")
        self.api_checkbox.setToolTip(
            "Starts a small web server on 127.0.0.1 (this computer only). Scripts and tools that speak "
            "the OpenAI speech API, such as Open WebUI or SillyTavern, can then use your models and voices.")
        self.api_checkbox.setChecked(bool(self.api_settings.get("enabled")))
        self.api_checkbox.toggled.connect(self.on_api_toggled)
        api_top.addWidget(self.api_checkbox)
        api_top.addStretch(1)
        self.api_status_label = QLabel()
        self.api_status_label.setObjectName("Muted")
        self.api_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        api_top.addWidget(self.api_status_label)
        api_layout.addLayout(api_top)
        api_row = QHBoxLayout()
        api_row.addWidget(QLabel("Port"))
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(int(self.api_settings.get("port") or local_api.DEFAULT_PORT))
        self.api_port_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.api_port_spin.setFixedWidth(70)
        self.api_port_spin.editingFinished.connect(self.on_api_settings_changed)
        api_row.addWidget(self.api_port_spin)
        api_row.addWidget(QLabel("Token"))
        self.api_token_input = QLineEdit(str(self.api_settings.get("token") or ""))
        self.api_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_token_input.setPlaceholderText("Optional: required as a Bearer token")
        self.api_token_input.editingFinished.connect(self.on_api_settings_changed)
        api_row.addWidget(self.api_token_input, 1)
        copy_example = self._link(QPushButton("Copy example"))
        copy_example.setToolTip("Copy a curl command that saves speech to speech.mp3.")
        copy_example.clicked.connect(self.copy_api_example)
        api_row.addWidget(copy_example)
        api_layout.addLayout(api_row)
        advanced_layout.addWidget(api_card)

        export_card, export_layout = self._make_card("Export")
        self.mp3_checkbox = QCheckBox("Save results as MP3")
        self.mp3_checkbox.setToolTip(
            "Smallest files and plays everywhere, but lossy. Replaces the WAV/FLAC choice in "
            "Finishing touches while ticked.")
        export_layout.addWidget(self.mp3_checkbox)
        mp3_hint = QLabel("Lossy compression; replaces the WAV/FLAC choice on the Generate page.")
        mp3_hint.setObjectName("Muted")
        export_layout.addWidget(mp3_hint)
        subtitle_row = QHBoxLayout()
        subtitle_row.addWidget(QLabel("Subtitle format"))
        self.subtitle_format_combo = QComboBox()
        self.subtitle_format_combo.addItems(list(subtitles.FORMATS))
        self.subtitle_format_combo.setToolTip(
            "SRT works almost everywhere (video editors, YouTube, VLC). WebVTT is for web players; "
            "in conversations it tags each caption with the speaker.")
        self.subtitle_format_combo.currentTextChanged.connect(self.update_finishing_summary)
        subtitle_row.addWidget(self.subtitle_format_combo)
        subtitle_row.addStretch(1)
        export_layout.addLayout(subtitle_row)
        subtitle_hint = QLabel("Used when Save subtitles is ticked in Finishing touches.")
        subtitle_hint.setObjectName("Muted")
        export_layout.addWidget(subtitle_hint)
        advanced_layout.addWidget(export_card)

        advanced_actions = QHBoxLayout()
        advanced_actions.setContentsMargins(ui_theme.SHADOW, 0, ui_theme.SHADOW, 0)
        advanced_actions.addStretch(1)
        reset_advanced_button = QPushButton("Reset effects")
        reset_advanced_button.setToolTip("Speed 1.00x, pitch 0, lossless output.")
        reset_advanced_button.clicked.connect(self.reset_advanced)
        advanced_actions.addWidget(reset_advanced_button)
        advanced_layout.addLayout(advanced_actions)
        advanced_layout.addStretch(1)
        self.pages.addWidget(advanced_page)

        for slider in (self.speed_slider, self.pitch_slider):
            slider.slider.valueChanged.connect(self.update_finishing_summary)
        self.mp3_checkbox.toggled.connect(self.on_mp3_toggled)
        self.apply_finishing_settings(
            audio_effects.FinishingSettings.from_dict(self.app_settings.get("finishing")))
        self.set_finishing_expanded(bool(self.app_settings.get("finishing_expanded", False)))

        # ---------- Log page ----------
        log_page, log_layout = self._make_page(
            "Log", "Technical output from model loading and generation.")
        self.console_log_view = QPlainTextEdit()
        self.console_log_view.setReadOnly(True)
        self.console_log_view.setMaximumBlockCount(1000)
        self.console_log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.console_log_view.setObjectName("LogView")
        log_layout.addWidget(self.console_log_view, 1)
        self.pages.addWidget(log_page)

        self.sidebar.setCurrentRow(self.PAGE_GENERATE)

        qt_status_bar = self.statusBar()
        qt_status_bar.setSizeGripEnabled(False)
        self.status_bar = QLabel("Status: Initializing...")
        self.status_bar.setWordWrap(False)
        self.status_bar.setTextFormat(Qt.TextFormat.PlainText)
        # Status text is elided to fit; it must never set the window's minimum width.
        self.status_bar.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.status_bar.setMinimumWidth(120)
        self.status_bar.setFixedHeight(self.status_bar.sizeHint().height() + 4)
        qt_status_bar.addWidget(self.status_bar, 1)
        self.model_load_progress = QProgressBar()
        self.model_load_progress.setRange(0, 1)
        self.model_load_progress.setValue(0)
        self.model_load_progress.setTextVisible(False)
        self.model_load_progress.setFixedHeight(8)
        self.model_load_progress.setFixedWidth(120)
        self.model_load_progress.setEnabled(False)
        qt_status_bar.addPermanentWidget(self.model_load_progress)
        self.refresh_model_repo_options()
        self.refresh_language_options()
        self.refresh_models_page()
        self.update_hf_token_status()
        self.set_tuning_expanded(bool(self.app_settings.get("tuning_expanded", False)))
        self.refresh_recordings_list()
        self.set_reference_audio(None)

    def attach_log_sink(self):
        global APP_LOG_SINK
        APP_LOG_SINK = self.log_message_signal.emit

    def append_console_log(self, text):
        if not hasattr(self, "console_log_view"):
            return
        cursor = self.console_log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(str(text))
        self.console_log_view.setTextCursor(cursor)
        self.console_log_view.ensureCursorVisible()

    def load_app_settings(self):
        payload, load_error = read_json_payload(self.app_settings_path)
        if load_error:
            print(load_error)
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def update_minimum_size(self):
        """Keep the window at least as large as its content needs, so nothing
        ever clips or scrolls; grow the window if the content just got bigger."""
        if self.centralWidget() is None:
            return
        # Layout changes propagate upward one event-loop pass per level, so
        # invalidate the whole chain to measure the current content right now.
        for layout in self.findChildren(QLayout):
            layout.invalidate()
        self.layout().activate()
        hint = self.minimumSizeHint()
        self.setMinimumSize(hint)
        if self.isMaximized() or self.isFullScreen():
            return
        if self.width() < hint.width() or self.height() < hint.height():
            self.resize(max(self.width(), hint.width()), max(self.height(), hint.height()))

    def fit_default_geometry(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        hint = self.minimumSizeHint()
        width = max(hint.width(), min(self.width(), available.width() - 40))
        height = max(hint.height(), min(self.height(), available.height() - 60))
        self.resize(width, height)
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def restore_window_settings(self):
        geometry_b64 = self.app_settings.get("window_geometry")
        if isinstance(geometry_b64, str) and geometry_b64:
            try:
                geometry_bytes = base64.b64decode(geometry_b64.encode("ascii"))
                self.restoreGeometry(geometry_bytes)
            except Exception as exc:
                print(f"Failed to restore saved window geometry: {exc}")

        if self.app_settings.get("window_maximized"):
            self.showMaximized()
        elif self.app_settings.get("window_fullscreen"):
            self.showFullScreen()

    def save_window_settings(self):
        try:
            self.app_settings["window_geometry"] = base64.b64encode(
                bytes(self.saveGeometry())
            ).decode("ascii")
            self.app_settings["window_maximized"] = self.isMaximized()
            self.app_settings["window_fullscreen"] = self.isFullScreen()
        except Exception as exc:
            print(f"Failed to save window settings: {exc}")

    def save_app_settings(self):
        if hasattr(self, "speed_slider"):
            self.app_settings["finishing"] = self.current_finishing_settings().to_dict()
            self.app_settings["finishing_expanded"] = not self.finishing_panel.isHidden()
        try:
            write_json_payload(self.app_settings_path, self.app_settings)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Settings Error",
                f"Failed to save {APP_SETTINGS_FILENAME}: {exc}",
            )

    def apply_hf_token_setting(self):
        hf_token = str(self.app_settings.get("hf_token", "")).strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        else:
            os.environ.pop("HF_TOKEN", None)

    def set_status_message(self, message):
        compact_message = " ".join(str(message).split())
        self.status_bar.setToolTip(compact_message)
        metrics = self.status_bar.fontMetrics()
        elided_message = metrics.elidedText(
            compact_message,
            Qt.TextElideMode.ElideRight,
            max(120, self.status_bar.width() - 8),
        )
        self.status_bar.setText(elided_message)

    def start_default_model_load(self):
        if not CHATTERBOX_AVAILABLE:
            return
        self.set_status_message(
            "Status: Starting default model load. First run may download model files and can take several minutes. Watch Activity Log for progress."
        )
        self.load_model()

    def handle_generate_stop_toggle(self):
        if not self.is_generating:
            self.start_generation(preview=False)
            return
        if hasattr(self, 'audio_generator_thread') and self.audio_generator_thread.isRunning():
            print("UI: Requesting stop for audio_generator_thread")
            self.audio_generator_thread.stop()
            self.generate_button.setText("Stopping...")
            self.generate_button.setEnabled(False)
            self.set_status_message(
                "Status: Stopping after the current section. Finished sections will be kept.")
        else:
            print("UI: Stop requested, but no active generation thread found. Resetting UI.")
            self.on_generation_thread_finished()

    def preview_text(self):
        """(text, character budget): a selection is previewed whole; otherwise the text's
        opening sections, up to the length picked next to Preview."""
        selected = self.text_input.textCursor().selectedText().replace("\u2029", "\n").strip()
        if selected:
            return selected, None
        return self.text_input.toPlainText().strip(), self.preview_length_combo.currentData()

    def start_generation(self, preview=False):
        if self.is_generating:
            return
        if self.api_busy:
            self.set_status_message("Status: Busy with a request from the local API; try again in a moment.")
            return
        if self.model is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load the model first.")
            return
        preview_budget = None
        if preview:
            text, preview_budget = self.preview_text()
        else:
            text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to synthesize.")
            return
        qwen_problem = self.prepare_qwen_generation()
        if qwen_problem:
            QMessageBox.information(self, "Voice", qwen_problem)
            return

        self.is_generating = True
        self.generation_is_preview = preview
        self.generation_char_count = len(text)
        plan_entry = self.loaded_entry() or self.get_selected_model_entry()
        lengths = self.section_lengths(text, plan_entry)
        if preview:
            lengths = lengths[:preview_cut(lengths, preview_budget)]
        self.generation_plan = self.batch_plan(plan_entry, lengths)
        self.generation_estimate = sum(cost for _f, _l, cost in self.generation_plan)
        self.progress_range = None
        self.progress_done_cost = 0.0
        self.progress_done_time = 0.0
        self.generation_started_at = time.monotonic()
        self.keep_take_button.setVisible(False)
        self.keep_voice_button.setVisible(False)
        self.generate_button.setText("Stop")
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(False)
        self.open_document_button.setEnabled(False)
        self.model_repo_combo.setEnabled(False)
        self.generation_progress.setValue(0)
        self.generation_progress.setVisible(True)
        self.activity_label.setText("Previewing..." if preview else "Starting...")

        self.generation_start_time = QTime.currentTime()
        self.generation_timer.start(1000)
        self.update_generation_time_display()

        self.audio_generator_thread = AudioGeneratorThread(
            self.model, text,
            self.ref_audio_path_label.toolTip(),
            self.exaggeration_slider.get_value(),
            self.temp_slider.get_value(),
            self.cfg_slider.get_value(),
            self.seed_input.value(),
            self.output_directory,
            language_id=self.language_combo.currentData() or "en",
            repetition_penalty=self.repetition_penalty,
            min_p=self.min_p,
            top_p=self.top_p,
            finishing=self.current_finishing_settings(),
            output_name=self.current_document_name,
            preview=preview,
        )
        self.audio_generator_thread.pronunciations = self.pronunciations
        self.audio_generator_thread.preview_chars = preview_budget
        self.audio_generator_thread.generation_complete.connect(self.on_generation_complete)
        self.audio_generator_thread.error_occurred.connect(self.on_generation_error)
        self.audio_generator_thread.chunk_generated.connect(self.on_chunk_generated_progress)
        self.audio_generator_thread.section_timed.connect(self.on_section_timed)
        self.audio_generator_thread.finished.connect(self.on_generation_thread_finished)
        self.audio_generator_thread.start()

    def keep_designed_voice(self):
        """Save the voice a design model just made (the preview's first section) to the
        library, and lock it in for every render."""
        model = self.active_qwen_model()
        anchor = getattr(model, "_anchor", None)
        if model is None or model.mode != "voice_design" or not anchor or not os.path.exists(anchor[0]):
            self.keep_voice_button.setVisible(False)
            return
        entry = self.loaded_entry()
        description = self.qwen_instruct_input.text().strip()
        voice = voice_library.Voice(
            name="Designed voice", kind="design", backend=entry["backend"], repo_id=entry["repo_id"],
            mode=model_registry.entry_mode(entry) if entry["backend"] in DUAL_MODE_BACKENDS else "",
            description=description, language=self.language_combo.currentData() or "")
        dialog = VoiceDetailsDialog("Keep this voice", voice, self.voice_library, parent=self)
        if not dialog_accepted(dialog.exec()):
            return
        dialog.apply()
        path = self.voice_library.new_clip_path(voice.name)
        import soundfile
        wav, sr = soundfile.read(anchor[0], dtype="float32")
        soundfile.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        voice_library.write_transcript(path, anchor[1])
        voice.clip = self.voice_library.to_stored(path)
        self.voice_library.add(voice)
        model.locked_anchor = (path, anchor[1])
        self.locked_description = description
        self.locked_voice_name = voice.name
        self.active_voice_id = voice.id
        self.keep_voice_button.setVisible(False)
        self.refresh_voice_chip()
        self.render_voice_tiles()
        self.set_status_message(f"Status: Kept {voice.name}. Every section now uses this voice; "
                                "it's in the voice library too.")

    def keep_preview_take(self):
        if self.last_preview_seed:
            self.seed_input.setValue(self.last_preview_seed)
            self.keep_take_button.setVisible(False)
            self.activity_label.setText(f"Take {self.last_preview_seed} locked")

    # --- Documents ---

    def open_document(self):
        start_dir = self.app_settings.get("last_document_dir") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", start_dir, documents.DOCUMENT_FILTER)
        if not path:
            return
        try:
            text = documents.load_document(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could Not Open Document",
                                f"{os.path.basename(path)} could not be read:\n{exc}")
            return
        if not text:
            QMessageBox.warning(self, "Empty Document",
                                f"No readable text was found in {os.path.basename(path)}.")
            return
        self.app_settings["last_document_dir"] = os.path.dirname(path)
        self.show_document(text, os.path.basename(path), documents.safe_file_stem(path))

    def show_document(self, text, label, file_stem):
        self.text_input.setPlainText(text)
        self.current_document_name = file_stem
        self.document_label.setText(label)
        self.update_text_stats()
        self.set_status_message(f"Status: Loaded {label}. Try Preview before generating.")

    def open_google_doc(self):
        dialog = GoogleDocsDialog(self.google_account, self)
        if not dialog_accepted(dialog.exec()) or not dialog.result_docx:
            return
        data, title = dialog.result_docx
        path = os.path.join(tempfile.gettempdir(), "google_doc_import.docx")
        try:
            with open(path, "wb") as handle:
                handle.write(data)
            text = documents.load_document(path)
        except Exception as exc:
            QMessageBox.warning(self, "Google Docs", f"\u201c{title}\u201d couldn't be read:\n{exc}")
            return
        finally:
            if os.path.exists(path):
                os.remove(path)
        if not text:
            QMessageBox.warning(self, "Google Docs", f"No readable text was found in \u201c{title}\u201d.")
            return
        self.show_document(text, f"{title} (Google Docs)", documents.safe_file_stem(title + ".docx"))

    def on_text_changed(self):
        if not self.text_input.toPlainText().strip():
            self.current_document_name = None
            self.document_label.clear()
        self.text_stats_timer.start()

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

    def _create_slider(self, min_val, max_val, step_val, default_val, value_format="{:.2f}"):
        return SliderWithValue(min_val, max_val, step_val, default_val, value_format)

    def browse_reference_audio(self):
        default_dir = self.last_reference_audio_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", default_dir, "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            voice = self.voice_library.add_clip(file_path)
            self.active_voice_id = voice.id
            self.set_reference_audio(file_path)
            self.last_reference_audio_dir = os.path.dirname(file_path)
            self.render_voice_tiles()
            self.set_status_message(f"Status: Added {voice.name} to the voice library and selected it.")

    # --- Engine-specific controls ---

    def active_qwen_model(self):
        """The loaded worker-engine model (Qwen or Kokoro), which uses the speaker/style row."""
        return self.model if isinstance(self.model, WORKER_MODEL_TYPES) else None

    def engine_settings(self, model):
        if isinstance(model, kokoro_engine.KokoroModel):
            return self.kokoro_settings
        if isinstance(model, voxcpm_engine.VoxCPMModel):
            return self.voxcpm_settings
        if isinstance(model, omnivoice_engine.OmniVoiceModel):
            return self.omnivoice_settings
        if isinstance(model, vibevoice_engine.VibeVoiceModel):
            return self.vibevoice_settings
        return self.qwen_settings

    def fill_speaker_combo(self, model):
        """Built-in voices; Kokoro's are filtered to the selected language."""
        kokoro = isinstance(model, kokoro_engine.KokoroModel)
        language = self.language_combo.currentData() or "en"
        speakers = model.voices_for(language) if kokoro else model.speakers
        if kokoro:
            saved = self.kokoro_settings.get("voice_by_language", {}).get(language)
        else:
            saved = self.qwen_settings.get("speaker")
        self.qwen_speaker_combo.blockSignals(True)
        self.qwen_speaker_combo.clear()
        for speaker in speakers:
            label = model.speaker_label(speaker) if kokoro else speaker.replace("_", " ").title()
            self.qwen_speaker_combo.addItem(label, speaker)
        index = self.qwen_speaker_combo.findData(saved)
        if index < 0 and kokoro:
            index = self.qwen_speaker_combo.findData("af_heart")
        self.qwen_speaker_combo.setCurrentIndex(max(0, index))
        self.qwen_speaker_combo.setToolTip(
            "Built-in Kokoro voice for the selected language." if kokoro else "Built-in Qwen speaker.")
        self.qwen_speaker_combo.blockSignals(False)
        self.refresh_voice_chip()

    def on_language_changed(self):
        if isinstance(self.model, kokoro_engine.KokoroModel) and hasattr(self, "qwen_speaker_combo"):
            self.fill_speaker_combo(self.model)

    def update_engine_controls(self):
        qwen = self.active_qwen_model()
        self.qwen_row.setVisible(qwen is not None)
        self.qwen_watermark_checkbox.setVisible(qwen is not None)
        for widget in (self.exaggeration_label, self.exaggeration_slider, self.cfg_label, self.cfg_slider):
            widget.setVisible(qwen is None)
        if qwen is not None:
            mode = qwen.mode
            self.fill_speaker_combo(qwen)
            self.qwen_watermark_checkbox.setChecked(bool(self.engine_settings(qwen).get("watermark", True)))
            settings = self.engine_settings(qwen)
            voxcpm = isinstance(qwen, voxcpm_engine.VoxCPMModel)
            omnivoice = isinstance(qwen, omnivoice_engine.OmniVoiceModel)
            if mode == "voice_design" and omnivoice:
                self.qwen_instruct_label.setText("Voice attributes")
                self.qwen_instruct_input.setPlaceholderText("e.g. female, young adult, low pitch, british accent")
                self.qwen_instruct_input.setText(settings.get("description", ""))
            elif mode == "voice_design":
                self.qwen_instruct_label.setText("Voice description")
                self.qwen_instruct_input.setPlaceholderText(
                    "e.g. a calm, low male voice with a slight rasp, unhurried and warm")
                self.qwen_instruct_input.setText(settings.get("description", ""))
            else:
                self.qwen_instruct_label.setText("Style")
                self.qwen_instruct_input.setPlaceholderText(
                    "Optional, e.g. slightly faster and cheerful" if voxcpm else
                    "Optional, e.g. excited and upbeat, or whisper softly")
                self.qwen_instruct_input.setText(settings.get("style", ""))
            self.qwen_speaker_label.setText("Voice" if mode == "preset" else "Speaker")
            for widget in (self.qwen_speaker_label, self.qwen_speaker_combo):
                widget.setVisible(mode in ("custom_voice", "preset"))
            for widget in (self.qwen_instruct_label, self.qwen_instruct_input):
                widget.setVisible(mode in ("custom_voice", "voice_design") or (voxcpm and mode == "base"))
            for widget in (self.qwen_transcript_label, self.qwen_transcript_input):
                widget.setVisible(mode == "base")
            self.design_attributes_button.setVisible(omnivoice and mode == "voice_design")
            conversation = mode == "conversation"
            for widget in (self.cast_title, self.cast_label, self.cast_button):
                widget.setVisible(conversation)
            if conversation:
                self.refresh_cast_label()
            self.qwen_transcript_input.setPlaceholderText(
                "What is said in the reference clip (required by OmniVoice)" if omnivoice else
                "What is said in the reference clip (optional, improves likeness)")
        else:
            self.design_attributes_button.setVisible(False)
            for widget in (self.cast_title, self.cast_label, self.cast_button):
                widget.setVisible(False)
        self.text_input.setPlaceholderText(
            vibevoice_engine.SCRIPT_HINT if qwen is not None and qwen.mode == "conversation" else
            "Enter text to synthesize, or open a document. Long text is split where a reader "
            "would pause and stitched back together.")
        self.refresh_voice_chip()
        if self.isVisible():
            self.update_minimum_size()

    def refresh_voice_chip(self):
        qwen = self.active_qwen_model()
        reference = self.ref_audio_path_label.toolTip()
        if qwen is not None and qwen.mode == "conversation":
            speakers = self.script_speakers()
            text = f"Cast: {len(speakers)} voice{'s' if len(speakers) != 1 else ''}"
            tip = "Each speaker in the script has their own voice. Change them with Cast\u2026 in Delivery."
        elif qwen is not None and qwen.mode in ("custom_voice", "preset"):
            name = self.qwen_speaker_combo.currentText().split(" (")[0]
            text = f"Preset: {name or 'speaker'}"
            tip = "A built-in voice. Reference clips aren't used by this model."
        elif qwen is not None and qwen.mode == "voice_design" and getattr(qwen, "locked_anchor", None):
            name = getattr(self, "locked_voice_name", None) or "kept voice"
            text, tip = f"Designed: {name}", ("Locked to a voice you kept: every section uses it. Change "
                                              "the description to design a new voice.")
        elif qwen is not None and qwen.mode == "voice_design":
            text, tip = "Designed voice", "Described in the Delivery card below. Preview, then Keep this voice to lock it."
        elif reference:
            saved = self.voice_library.find_clip(reference)
            text, tip = (saved.name if saved else os.path.basename(reference)), reference
        else:
            text = "Default voice"
            tip = "The model's built-in voice. Pick a reference clip on the Voice page to clone a voice."
            if qwen is not None:
                text, tip = "No clip selected", "This cloning model needs a reference clip from the Voice page."
        self.voice_chip.setText(text)
        self.voice_chip.setToolTip(tip)

    def prepare_qwen_generation(self):
        """Copy the Qwen card into the model; returns an error message or None."""
        qwen = self.active_qwen_model()
        if qwen is None:
            return None
        instruct = self.qwen_instruct_input.text().strip()
        if qwen.mode == "voice_design" and not instruct:
            return "Describe the voice you want (Voice description, in the Delivery card) first."
        if qwen.mode == "base" and not self.ref_audio_path_label.toolTip():
            return "Choose a reference clip on the Voice page; this cloning model needs one."
        if isinstance(qwen, vibevoice_engine.VibeVoiceModel):
            speakers = self.script_speakers()
            if len(speakers) > vibevoice_engine.MAX_SPEAKERS:
                return (f"VibeVoice handles up to {vibevoice_engine.MAX_SPEAKERS} speakers; this script has "
                        f"{len(speakers)}: {', '.join(speakers)}.")
            qwen.cast = self.current_cast(speakers)
            qwen.watermark = self.qwen_watermark_checkbox.isChecked()
            qwen.begin_run()
            self.vibevoice_settings["watermark"] = qwen.watermark
            self.app_settings["vibevoice"] = self.vibevoice_settings
            return None
        if isinstance(qwen, omnivoice_engine.OmniVoiceModel):
            if qwen.mode == "base" and not self.qwen_transcript_input.text().strip():
                return ("OmniVoice needs the Clip transcript: type exactly what is said in the "
                        "reference clip. Recordings made with Record\u2026 fill it in for you.")
            if qwen.mode == "voice_design":
                problem = omnivoice_engine.check_description(instruct)
                if problem:
                    return problem
        qwen.speaker = self.qwen_speaker_combo.currentData() or qwen.speaker
        qwen.watermark = self.qwen_watermark_checkbox.isChecked()
        if isinstance(qwen, kokoro_engine.KokoroModel):
            language = self.language_combo.currentData() or "en"
            self.kokoro_settings.setdefault("voice_by_language", {})[language] = qwen.speaker
            self.kokoro_settings["watermark"] = qwen.watermark
            self.app_settings["kokoro"] = self.kokoro_settings
            return None
        qwen.instruct = instruct
        qwen.ref_text = self.qwen_transcript_input.text().strip() if qwen.mode == "base" else ""
        if getattr(qwen, "locked_anchor", None) and instruct != getattr(self, "locked_description", instruct):
            qwen.locked_anchor = None  # the description changed: design a new voice
            self.locked_voice_name = None
            self.set_status_message("Status: Description changed, so a new voice will be designed.")
        if hasattr(qwen, "begin_run"):
            qwen.begin_run()
        key = "description" if qwen.mode == "voice_design" else "style"
        settings = self.engine_settings(qwen)
        settings.update({key: instruct, "watermark": qwen.watermark})
        if isinstance(qwen, DUAL_MODE_TYPES):
            self.app_settings[qwen.backend] = settings
        else:
            settings["speaker"] = qwen.speaker
            self.app_settings["qwen"] = settings
        return None

    # --- Local API ---

    def save_api_settings(self):
        self.api_settings.update(enabled=self.api_checkbox.isChecked(), port=self.api_port_spin.value(),
                                 token=self.api_token_input.text().strip())
        self.app_settings["api"] = dict(self.api_settings)
        self.save_app_settings()

    def on_api_toggled(self, checked):
        self.save_api_settings()
        self.restart_api_server()

    def on_api_settings_changed(self):
        changed = (self.api_port_spin.value() != self.api_settings.get("port")
                   or self.api_token_input.text().strip() != self.api_settings.get("token"))
        self.save_api_settings()
        if changed and self.api_checkbox.isChecked():
            self.restart_api_server()

    def restart_api_server(self):
        if self.api_server is not None:
            self.api_server.stop()
            self.api_server = None
        if self.api_checkbox.isChecked():
            server = local_api.LocalApiServer(self.api_bridge, self.api_port_spin.value(),
                                              self.api_token_input.text(), log=print)
            try:
                server.start()
            except OSError as exc:
                self.api_status_label.setText(f"Port {self.api_port_spin.value()} is in use")
                self.api_status_label.setToolTip(str(exc))
                print(f"Local API could not start: {exc}")
                return
            self.api_server = server
            self.api_status_label.setText(f"On: {server.url}")
            self.api_status_label.setToolTip("Only programs on this computer can connect.")
        else:
            self.api_status_label.setText("Off")
            self.api_status_label.setToolTip("")

    def copy_api_example(self):
        url = f"http://127.0.0.1:{self.api_port_spin.value()}"
        token = self.api_token_input.text().strip()
        auth = f' -H "Authorization: Bearer {token}"' if token else ""
        command = (f'curl {url}/v1/audio/speech -H "Content-Type: application/json"{auth} '
                   '-d "{\\"input\\": \\"Hello from my own computer.\\", \\"response_format\\": \\"mp3\\"}" '
                   "-o speech.mp3")
        QApplication.clipboard().setText(command)
        self.set_status_message("Status: Copied an example curl command.")

    def api_health(self):
        entry = self.loaded_entry() if self.model is not None else None
        return {
            "status": "busy" if self.model_busy() else "ready" if self.model is not None else "no model loaded",
            "model": entry["label"] if entry else None,
            "device": self.device_used if self.model is not None else None,
            "voice": self.voice_chip.text(),
            "endpoints": ["GET /v1/health", "GET /v1/models", "GET /v1/voices",
                          "POST /v1/audio/speech (OpenAI-compatible)", "POST /v1/speech"],
        }

    def api_models(self):
        rows = []
        for entry in self.model_entries:
            rows.append({
                "id": entry["label"], "object": "model", "repo_id": entry["repo_id"],
                "engine": model_registry.engine_label(entry),
                "capability": model_registry.capability_for(entry),
                "loaded": self.is_active_entry(entry), "downloaded": model_registry.is_downloaded(entry),
                "installed": self.engine_installed(entry),
            })
        return rows

    def api_voices(self):
        rows = [{"id": voice.id, "name": voice.name, "kind": voice.kind, "tags": voice.tags,
                 "engine": voice.backend or "any cloning model", "has_clip": voice.has_clip}
                for voice in self.voice_library.voices]
        model = self.active_qwen_model()
        for speaker in getattr(model, "speakers", []) or []:
            rows.append({"id": speaker, "name": speaker, "kind": "built-in",
                         "engine": getattr(model, "backend", "")})
        for name in (getattr(model, "sample_paths", None) or {}):
            rows.append({"id": name.split(" (")[0], "name": name, "kind": "sample", "engine": "vibevoice"})
        return rows

    def api_find_entry(self, name):
        name = str(name or "").strip()
        if name.lower() in ApiBridge.OPENAI_MODEL_NAMES:
            return None
        for entry in self.model_entries:
            if name.lower() in (entry["label"].lower(), entry["repo_id"].lower()):
                return entry
        raise local_api.ApiError(404, f"No model called {name!r}. See GET /v1/models.", "not_found")

    def api_begin(self, request):
        """Resolve a request's model and voice, and claim the GPU. Runs on the UI thread.
        Returns {"loading": True} while a model it asked for is still loading."""
        if getattr(self, "model_is_loading", False):
            if self.api_loading_entry is not None:
                return {"loading": True}
            raise local_api.ApiError(503, "A model is loading in the app; try again shortly.", "server_error")
        self.api_loading_entry = None
        if self.is_generating or self.api_busy:
            raise local_api.ApiError(503, "The app is generating right now; try again shortly.", "server_error")
        entry = self.api_find_entry(request.get("model"))
        voice = self.api_find_voice(request.get("voice"))
        if voice is not None and voice.kind != "clip" and entry is None:
            entry = self.entry_for_voice(voice)  # a preset or designed voice brings its model
        if entry is not None and not self.is_active_entry(entry):
            if not self.engine_installed(entry):
                raise local_api.ApiError(409, f"{entry['label']} needs its engine installed; load it once in "
                                              "the app first.")
            self.api_loading_entry = entry
            self.set_status_message(f"Status: Loading {entry['label']} for an API request...")
            self.load_entry(entry)
            return {"loading": True}
        if self.model is None:
            raise local_api.ApiError(409, "No model is loaded in the app.")
        model = self.model
        text = request["text"]
        reference = self.ref_audio_path_label.toolTip() or None
        voice_label = self.voice_chip.text()
        worker = self.active_qwen_model()
        if worker is not None:
            problem = self.prepare_qwen_generation()  # the app's current voice settings first...
            if problem and voice is None and not request.get("style"):
                raise local_api.ApiError(400, problem)
        if voice is not None:  # ...then the requested voice on top
            reference, voice_label = self.api_apply_voice(voice, reference)
        elif request.get("voice") and worker is not None:
            voice_label = self.api_apply_speaker(str(request["voice"]))
        elif request.get("voice") and str(request["voice"]).lower() not in self.OPENAI_VOICES | {"default"}:
            raise local_api.ApiError(404, f"No voice called {request['voice']!r}. See GET /v1/voices.", "not_found")
        style = str(request.get("style") or "").strip()
        if style and worker is not None and worker.mode in ("custom_voice", "voice_design") or (
                style and isinstance(worker, voxcpm_engine.VoxCPMModel)):
            worker.instruct = style
        if isinstance(worker, vibevoice_engine.VibeVoiceModel):
            speakers = documents.script_speakers(documents.parse_script(text))
            if len(speakers) > vibevoice_engine.MAX_SPEAKERS:
                raise local_api.ApiError(400, f"VibeVoice handles up to {vibevoice_engine.MAX_SPEAKERS} speakers.")
            worker.cast = self.current_cast(speakers)
        if worker is not None and hasattr(worker, "begin_run"):
            worker.begin_run()
        if worker is not None and worker.mode == "voice_design" and not worker.instruct.strip():
            raise local_api.ApiError(400, "This is a voice design model: give a designed voice or a \"style\" "
                                          "(the voice description).")
        if worker is not None and worker.mode == "base" and not reference:
            raise local_api.ApiError(400, "This cloning model needs a voice: name a clip voice, or pick one in the app.")
        finishing = self.current_finishing_settings()
        output_format = str(request.get("format") or "WAV").upper()
        if output_format not in audio_effects.OUTPUT_FORMATS:
            raise local_api.ApiError(400, f"format must be one of {', '.join(audio_effects.OUTPUT_FORMATS)}.")
        finishing.output_format = output_format
        if request.get("speed") not in (None, ""):
            try:
                finishing.speed = float(np.clip(float(request["speed"]), *audio_effects.SPEED_RANGE))
            except (TypeError, ValueError):
                raise local_api.ApiError(400, "speed must be a number.")
        subtitle_format = str(request.get("subtitles") or "").strip().lower()
        finishing.save_subtitles = subtitle_format in ("srt", "vtt", "webvtt")
        finishing.subtitle_format = "WebVTT" if subtitle_format in ("vtt", "webvtt") else "SRT"
        language = request.get("language") or self.language_combo.currentData() or "en"
        if request.get("save"):
            output_dir = os.path.join(self.output_directory, "api")
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = tempfile.mkdtemp(prefix="tts_api_")
        name = documents.safe_file_stem(str(request.get("name") or "api")) if request.get("save") else "api"
        self.api_busy = True
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.model_repo_combo.setEnabled(False)
        self.set_status_message("Status: Generating for a local API request...")
        loaded = self.loaded_entry()
        return {
            "generator": dict(
                model=model, text=text, audio_prompt_path=reference,
                exaggeration=self.exaggeration_slider.get_value(), temperature=self.temp_slider.get_value(),
                cfg_weight=self.cfg_slider.get_value(), seed=0, output_dir=output_dir, language_id=language,
                repetition_penalty=self.repetition_penalty, min_p=self.min_p, top_p=self.top_p,
                finishing=finishing, output_name=name, preview=False),
            "pronunciations": self.pronunciations,
            "model_label": loaded["label"] if loaded else None,
            "voice_label": voice_label,
            "mime": {"WAV": "audio/wav", "FLAC": "audio/flac", "MP3": "audio/mpeg"}.get(output_format, "audio/wav"),
            "temporary": not request.get("save"),
        }

    def api_find_voice(self, name):
        name = str(name or "").strip()
        if not name:
            return None
        lowered = name.lower()
        for voice in self.voice_library.voices:
            if lowered in (voice.id.lower(), voice.name.lower()):
                return voice
        return None  # maybe a built-in speaker of the loaded model (handled later)

    def api_apply_voice(self, voice, reference):
        """Point the loaded model at a library voice; returns (reference clip, label)."""
        model = self.active_qwen_model()
        path = self.voice_library.clip_path(voice)
        if voice.kind == "clip" or (model is None or model.mode == "base"):
            if not voice.has_clip or not os.path.exists(path):
                raise local_api.ApiError(400, f"{voice.name} has no clip; this model clones from a clip.")
            if model is not None:
                model.ref_text = voice_library.read_transcript(path)
            return path, voice.name
        if voice.kind == "preset" and model.mode in ("custom_voice", "preset"):
            model.speaker = voice.speaker
            if model.mode == "custom_voice":
                model.instruct = voice.style
        elif voice.kind == "design" and model.mode == "voice_design":
            model.instruct = voice.description
        else:
            raise local_api.ApiError(400, f"{voice.name} doesn't fit the loaded model.")
        return reference, voice.name

    OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"}

    def api_apply_speaker(self, name):
        """A built-in speaker of the loaded model (Kokoro/Qwen), by id or plain name."""
        model = self.active_qwen_model()
        speakers = getattr(model, "speakers", []) or []
        lowered = name.lower()
        match = next((speaker for speaker in speakers if speaker.lower() == lowered), None)
        if match is None:  # "alloy" -> Kokoro's af_alloy, "heart" -> af_heart
            match = next((speaker for speaker in speakers if speaker.lower().split("_", 1)[-1] == lowered), None)
        if match is not None:
            model.speaker = match
            return match
        if lowered in self.OPENAI_VOICES:
            return self.voice_chip.text()  # no such voice here: use the app's current voice
        raise local_api.ApiError(404, f"No voice called {name!r}. See GET /v1/voices.", "not_found")

    def api_end(self):
        self.api_busy = False
        has_model = self.model is not None
        self.generate_button.setEnabled(has_model)
        self.preview_button.setEnabled(has_model)
        self.model_repo_combo.setEnabled(not getattr(self, "model_is_loading", False))
        self.set_status_message("Status: Finished a local API request. Ready.")

    # --- Pronunciation dictionary ---

    def update_pronunciation_summary(self):
        rules = self.pronunciations.rules
        active = sum(1 for rule in rules if rule.enabled)
        self.pronunciation_summary.setText(
            f"{active} word{'s' if active != 1 else ''}" + (f" ({len(rules) - active} off)" if active != len(rules) else "")
            if rules else "Empty")

    def on_pronunciation_toggled(self, checked):
        self.pronunciations.enabled = checked
        self.pronunciations.save()
        self.update_text_stats()

    def edit_pronunciations(self):
        dialog = PronunciationDialog(self.pronunciations, self.speak_text, self)
        if dialog_accepted(dialog.exec()):
            self.pronunciations.set_rules(dialog.rules())
            self.pronunciations.save()
            self.update_pronunciation_summary()
            self.update_text_stats()
            self.set_status_message(f"Status: Saved {len(self.pronunciations.rules)} pronunciations.")

    def speak_text(self, text):
        """Say a short text with the loaded model and voice (used by Hear it)."""
        if self.model is None or self.model_busy() or not text.strip():
            self.set_status_message("Status: Load a model (and wait for any generation) to hear it.")
            return
        if getattr(self, "speak_thread", None) is not None and self.speak_thread.isRunning():
            return
        kwargs = {"language_id": self.language_combo.currentData() or "en"}
        reference = self.ref_audio_path_label.toolTip()
        if self.active_qwen_model() is not None:
            problem = self.prepare_qwen_generation()
            if problem:
                QMessageBox.information(self, "Hear It", problem)
                return
        if reference:
            kwargs["audio_prompt_path"] = reference
        self.set_status_message(f"Status: Speaking \u201c{text[:40]}\u201d...")
        self.speak_thread = SpeakThread(self.model, text, kwargs, self)
        self.speak_thread.finished_with.connect(self.on_spoken)
        self.speak_thread.start()

    def on_spoken(self, wav, sr, error):
        if error or wav is None:
            self.set_status_message(f"Status: Couldn't speak it: {error}")
            return
        import soundfile
        path = os.path.join(tempfile.gettempdir(), "tts_pronunciation_check.wav")
        soundfile.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        self.stop_reference_preview()
        self.preview_player.setSource(QUrl())
        self.preview_player.setSource(QUrl.fromLocalFile(path))
        self.preview_player.play()
        self.set_status_message("Status: Playing the pronunciation check.")

    # --- Conversations (VibeVoice) ---

    def script_speakers(self):
        return documents.script_speakers(documents.parse_script(self.text_input.toPlainText()))

    def current_cast(self, speakers):
        model = self.active_qwen_model()
        samples = getattr(model, "sample_paths", {}) or {}
        return vibevoice_engine.resolve_cast(speakers, self.vibevoice_settings.get("cast", {}), samples)

    def refresh_cast_label(self):
        model = self.active_qwen_model()
        if not isinstance(model, vibevoice_engine.VibeVoiceModel):
            return
        speakers = self.script_speakers()
        cast = self.current_cast(speakers)
        def cast_name(path):
            voice = self.voice_library.find_clip(path)
            return voice.name if voice else vibevoice_engine.voice_name(path, model.sample_paths)

        parts = [f"{speaker} \u2192 {cast_name(cast[speaker])}" for speaker in speakers if speaker in cast]
        text = " \u00b7 ".join(parts) if parts else "Write lines like \u201cLinda: Hello.\u201d"
        if len(speakers) > vibevoice_engine.MAX_SPEAKERS:
            text = f"{len(speakers)} speakers: VibeVoice handles up to {vibevoice_engine.MAX_SPEAKERS}"
        self.cast_label.setText(text)
        self.cast_label.setToolTip("\n".join(f"{speaker}: {cast.get(speaker, '')}" for speaker in speakers))
        self.refresh_voice_chip()

    def edit_cast(self):
        model = self.active_qwen_model()
        if not isinstance(model, vibevoice_engine.VibeVoiceModel):
            return
        speakers = self.script_speakers()[:vibevoice_engine.MAX_SPEAKERS]
        if not speakers:
            QMessageBox.information(self, "Cast", "Write the script first, one speaker per line, e.g.\n\n"
                                    + vibevoice_engine.SCRIPT_HINT.split("\n", 1)[1])
            return
        self.voice_library.import_recordings()
        recordings = [(voice.name, self.voice_library.clip_path(voice)) for voice in self.voice_library.clip_voices()]
        reference = self.ref_audio_path_label.toolTip()
        if reference and reference not in {path for _name, path in recordings}:
            recordings.insert(0, (os.path.basename(reference), reference))
        dialog = CastDialog(speakers, self.current_cast(speakers), model.sample_paths, recordings, self)
        if dialog_accepted(dialog.exec()):
            chosen = dict(self.vibevoice_settings.get("cast", {}))
            chosen.update(dialog.result_cast())
            self.vibevoice_settings["cast"] = chosen
            self.app_settings["vibevoice"] = self.vibevoice_settings
            self.refresh_cast_label()

    @staticmethod
    def transcript_path(audio_path):
        return os.path.splitext(audio_path)[0] + ".txt"

    def load_reference_transcript(self, audio_path):
        text = ""
        if audio_path and os.path.exists(self.transcript_path(audio_path)):
            with open(self.transcript_path(audio_path), encoding="utf-8") as handle:
                text = handle.read().strip()
        self.qwen_transcript_input.setText(text)

    def save_reference_transcript(self):
        audio_path = self.ref_audio_path_label.toolTip()
        if not audio_path:
            return
        text = self.qwen_transcript_input.text().strip()
        path = self.transcript_path(audio_path)
        try:
            if text:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(text + "\n")
            elif os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f"Could not save transcript for {os.path.basename(audio_path)}: {exc}")

    # --- Finishing touches ---

    def current_finishing_settings(self):
        return audio_effects.FinishingSettings(
            speed=round(self.speed_slider.get_value(), 2),
            pitch_semitones=round(self.pitch_slider.get_value(), 1),
            paragraph_pause=round(self.pause_slider.get_value(), 1),
            even_volume=self.even_volume_checkbox.isChecked(),
            trim_silence=self.trim_silence_checkbox.isChecked(),
            output_format="MP3" if self.mp3_checkbox.isChecked() else self.output_format_combo.currentText(),
            save_subtitles=self.subtitles_checkbox.isChecked(),
            subtitle_format=self.subtitle_format_combo.currentText(),
        )

    def apply_finishing_settings(self, settings):
        self.speed_slider.set_value(settings.speed)
        self.pitch_slider.set_value(settings.pitch_semitones)
        self.pause_slider.set_value(settings.paragraph_pause)
        self.even_volume_checkbox.setChecked(settings.even_volume)
        self.trim_silence_checkbox.setChecked(settings.trim_silence)
        self.mp3_checkbox.setChecked(settings.output_format == "MP3")
        if settings.output_format in LOSSLESS_FORMATS:
            self.output_format_combo.setCurrentText(settings.output_format)
        self.subtitles_checkbox.setChecked(settings.save_subtitles)
        self.subtitle_format_combo.setCurrentText(settings.subtitle_format)
        self.update_finishing_summary()

    def reset_finishing(self):
        """Finishing touches only; Advanced effects are left as they are."""
        defaults = audio_effects.FinishingSettings()
        self.pause_slider.set_value(defaults.paragraph_pause)
        self.even_volume_checkbox.setChecked(defaults.even_volume)
        self.trim_silence_checkbox.setChecked(defaults.trim_silence)
        self.output_format_combo.setCurrentText(defaults.output_format)
        self.subtitles_checkbox.setChecked(defaults.save_subtitles)
        self.update_finishing_summary()

    def reset_advanced(self):
        self.speed_slider.set_value(1.0)
        self.pitch_slider.set_value(0.0)
        self.mp3_checkbox.setChecked(False)
        self.update_finishing_summary()

    def on_mp3_toggled(self, checked):
        self.output_format_combo.setEnabled(not checked)
        self.output_format_combo.setToolTip(
            "MP3 is selected on the Advanced page." if checked else
            "WAV is uncompressed; FLAC is lossless and about half the size. "
            "MP3 is on the Advanced page.")
        self.update_finishing_summary()

    def update_finishing_summary(self, *_args):
        if not hasattr(self, "subtitle_format_combo"):
            return  # Advanced page not built yet
        settings = self.current_finishing_settings()
        summary = settings.summary()
        advanced = (abs(settings.speed - 1.0) > 1e-6 or abs(settings.pitch_semitones) > 1e-6
                    or settings.output_format == "MP3")
        self.finishing_summary_label.setText(summary + ("  (effects on Advanced page)" if advanced else ""))

    def set_finishing_expanded(self, expanded):
        self.finishing_panel.setVisible(expanded)
        arrow = "\u25be" if expanded else "\u25b8"
        self.finishing_toggle.setText(f"{arrow} Finishing touches")
        self.finishing_summary_label.setVisible(not expanded)
        if self.isVisible():
            self.update_minimum_size()

    # --- Voice selection ---

    def set_reference_audio(self, path):
        if path:
            saved = self.voice_library.find_clip(path) if hasattr(self, "voice_library") else None
            name = saved.name if saved else os.path.basename(path)
            self.ref_audio_path_label.setText(name)
            self.ref_audio_path_label.setToolTip(path)
            self.voice_chip.setText(name)
            self.voice_chip.setToolTip(path)
        else:
            self.ref_audio_path_label.setText("Default voice (no reference clip)")
            self.ref_audio_path_label.setToolTip("")
            self.voice_chip.setText("Default voice")
            self.voice_chip.setToolTip("The model's built-in voice. Pick a reference clip on the Voice page to clone a voice.")
        self.preview_reference_button.setEnabled(bool(path))
        self.clear_reference_button.setEnabled(bool(path))
        if hasattr(self, "qwen_transcript_input"):
            self.load_reference_transcript(path)
            self.refresh_voice_chip()
        self.render_voice_tiles()

    def clear_reference_audio(self):
        self.stop_reference_preview()
        self.set_reference_audio(None)
        self.set_status_message("Status: Using the default voice.")

    def refresh_recordings_list(self):
        """Bring new recordings into the library and redraw it."""
        self.voice_library.import_recordings()
        self.render_voice_tiles()

    def _start_reference_preview(self, path, button):
        self.stop_reference_preview()
        self.preview_button_playing = button
        self.preview_player.setSource(QUrl.fromLocalFile(path))
        self.preview_player.play()
        button.setText("Stop preview")

    def stop_reference_preview(self):
        self.preview_player.stop()

    def _on_preview_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState and self.preview_button_playing:
            self.preview_reference_button.setText("Preview")
            self.preview_button_playing = None

    def toggle_reference_preview(self):
        if self.preview_button_playing is self.preview_reference_button:
            self.stop_reference_preview()
            return
        path = self.ref_audio_path_label.toolTip()
        if path:
            self._start_reference_preview(path, self.preview_reference_button)


    def open_recordings_folder(self):
        os.makedirs(self.recordings_directory, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.recordings_directory))

    # --- Voice library ---

    def voice_is_active(self, voice):
        if voice.kind == "clip":
            path = self.voice_library.clip_path(voice)
            return bool(path) and os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                os.path.abspath(self.ref_audio_path_label.toolTip() or "~none~"))
        model = self.active_qwen_model()
        return (voice.id == self.active_voice_id and model is not None
                and getattr(model, "backend", "") == voice.backend)

    def render_voice_tiles(self):
        if not hasattr(self, "voice_tiles"):
            return
        kind = self.voice_filter_tabs.tabData(self.voice_filter_tabs.currentIndex())
        query = self.voice_search.text().strip().lower()
        voices = [voice for voice in self.voice_library.voices
                  if (not kind or voice.kind == kind or (kind == "clip" and voice.has_clip))
                  and (not query or query in " ".join([voice.name, voice.notes, *voice.tags]).lower())]
        voices.sort(key=lambda voice: voice.created, reverse=True)
        counts = {key: sum(1 for voice in self.voice_library.voices
                           if not key or voice.kind == key or (key == "clip" and voice.has_clip))
                  for key in ("", "clip", "preset", "design")}
        for index in range(self.voice_filter_tabs.count()):
            key = self.voice_filter_tabs.tabData(index)
            title = {"": "All", "clip": "Clips", "preset": "Presets", "design": "Designed"}[key]
            self.voice_filter_tabs.setTabText(index, f"{title}  {counts[key]}" if counts[key] else title)
        empty = ("No voices match." if query or kind else
                 "No voices yet. Record one, add a file, or save the voice you're using.")
        self.voice_tiles.set_tiles([self.voice_tile(voice) for voice in voices], empty)

    def voice_tile(self, voice):
        library = self.voice_library
        active = self.voice_is_active(voice)
        engine = model_registry.ENGINES.get(voice.backend)
        path = library.clip_path(voice)
        if voice.kind == "clip":
            exists = os.path.exists(path)
            seconds = voice_library.clip_seconds(path) if exists else 0
            subtitle = f"Clip \u00b7 {seconds:.0f} s" if exists else "Clip \u00b7 file missing"
            transcript = voice_library.read_transcript(path) if exists else ""
            detail = f"\u201c{transcript}\u201d" if transcript else "No transcript"
        elif voice.kind == "preset":
            label = kokoro_engine.voice_label(voice.speaker) if voice.backend == KOKORO_BACKEND \
                else voice.speaker.replace("_", " ").title()
            subtitle = f"Preset \u00b7 {engine.label if engine else voice.backend} \u00b7 {label}"
            detail = voice.style or voice.notes or "Built-in voice"
        else:
            subtitle = f"Designed \u00b7 {engine.label if engine else voice.backend}"
            detail = voice.description
        badges = [("active", "In use", "This is the voice you're using.")] if active else []
        if voice.kind != "clip" and voice.has_clip:
            badges.append(("status", "Has clip", "Also usable as a clip voice by cloning models."))
        badges += [("status", tag, "Tag") for tag in voice.tags[:2]]
        tooltip = "\n".join(line for line in (voice.name, subtitle, detail if voice.kind != "clip" else "",
                                               voice.notes, path, "Click to use. Right-click for more.") if line)
        tile = model_tiles.ModelTile(voice.name, subtitle, badges, detail, tooltip, active=active, with_menu=True)
        tile.clicked.connect(lambda v=voice: self.use_voice(v))
        tile.menu_requested.connect(lambda pos, v=voice: self.show_voice_menu(v, pos))
        return tile

    def show_voice_menu(self, voice, pos):
        menu = QMenu(self)
        path = self.voice_library.clip_path(voice)
        playing = self.preview_button_playing is self.voice_tiles and getattr(self, "previewing_voice", None) is voice
        entries = [
            ("Use", lambda: self.use_voice(voice), True),
            ("Stop preview" if playing else "Preview clip", lambda: self.preview_voice(voice),
             bool(path) and os.path.exists(path)),
        ]
        if voice.kind != "clip":
            entries.append(("Use as a clip voice", lambda: self.use_voice(voice, as_clip=True),
                            bool(path) and os.path.exists(path)))
            entries.append(("Make clip..." if not voice.has_clip else "Remake clip...",
                            lambda: self.make_voice_clip(voice), not self.model_busy()))
        entries += [None, ("Edit...", lambda: self.edit_voice(voice), True),
                    ("Show file in folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path))),
                     bool(path) and os.path.exists(path)),
                    None, ("Remove from library...", lambda: self.remove_voice(voice), True)]
        for item in entries:
            if item is None:
                menu.addSeparator()
                continue
            text, callback, enabled = item
            action = menu.addAction(text)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False, callback=callback: callback())
        menu.exec(pos)

    def preview_voice(self, voice):
        if self.preview_button_playing is self.voice_tiles and getattr(self, "previewing_voice", None) is voice:
            self.stop_reference_preview()
            return
        self.stop_reference_preview()
        self.previewing_voice = voice
        self.preview_button_playing = self.voice_tiles
        self.preview_player.setSource(QUrl.fromLocalFile(self.voice_library.clip_path(voice)))
        self.preview_player.play()

    def entry_for_voice(self, voice):
        for entry in self.model_entries:
            if entry.get("backend") != voice.backend or entry.get("repo_id") != voice.repo_id:
                continue
            if voice.backend in DUAL_MODE_BACKENDS and model_registry.entry_mode(entry) != (voice.mode or "design"):
                continue
            return entry
        return None

    def use_voice(self, voice, as_clip=False):
        path = self.voice_library.clip_path(voice)
        if voice.kind == "clip" or as_clip:
            if not os.path.exists(path):
                QMessageBox.warning(self, "Voice", f"The clip for {voice.name} is missing:\n{path}")
                return
            self.set_reference_audio(path)
            self.active_voice_id = voice.id
            model = self.active_qwen_model()
            note = ""
            if model is not None and model.mode not in ("base",):
                note = " It's used by cloning models; the loaded model doesn't clone."
            self.set_status_message(f"Status: Voice set to {voice.name}.{note}")
            self.render_voice_tiles()
            return
        entry = self.entry_for_voice(voice)
        if entry is None:
            engine = model_registry.ENGINES.get(voice.backend)
            QMessageBox.information(
                self, "Voice", f"{voice.name} needs {engine.label if engine else voice.backend} "
                f"({voice.repo_id}), which isn't in your model list. Add it on the Model page first.")
            return
        if self.model_busy():
            self.set_status_message("Status: Wait for the current load or generation to finish.")
            return
        self.pending_voice = voice
        if self.is_active_entry(entry):
            self.apply_pending_voice()
        else:
            self.set_status_message(f"Status: Loading {entry['label']} for {voice.name}...")
            self.load_entry(entry)

    def apply_pending_voice(self):
        voice, self.pending_voice = self.pending_voice, None
        model = self.active_qwen_model()
        if voice is None or model is None or getattr(model, "backend", "") != voice.backend:
            return
        if voice.language:
            index = self.language_combo.findData(voice.language)
            if index >= 0:
                self.language_combo.setCurrentIndex(index)
        if voice.kind == "preset":
            if isinstance(model, kokoro_engine.KokoroModel):
                language = voice.language or kokoro_engine.voice_language(voice.speaker) or "en"
                self.kokoro_settings.setdefault("voice_by_language", {})[language] = voice.speaker
            else:
                self.qwen_settings.update(speaker=voice.speaker, style=voice.style)
        else:
            self.engine_settings(model)["description"] = voice.description
            path = self.voice_library.clip_path(voice)
            if hasattr(model, "locked_anchor"):
                model.locked_anchor = None
                self.locked_voice_name = None
            if voice.has_clip and os.path.exists(path) and hasattr(model, "locked_anchor"):
                model.locked_anchor = (path, voice_library.read_transcript(path))
                self.locked_description = voice.description
                self.locked_voice_name = voice.name
        self.update_engine_controls()
        if voice.kind == "preset":
            index = self.qwen_speaker_combo.findData(voice.speaker)
            if index >= 0:
                self.qwen_speaker_combo.setCurrentIndex(index)
        self.active_voice_id = voice.id
        self.set_status_message(f"Status: Voice set to {voice.name}.")
        self.refresh_voice_chip()
        self.render_voice_tiles()

    def current_voice_spec(self):
        """(Voice, error): the voice in use, as an unsaved library voice."""
        model = self.active_qwen_model()
        entry = self.loaded_entry()
        language = self.language_combo.currentData() or ""
        if model is not None and model.mode == "conversation":
            return None, ("A conversation uses a cast of voices. Save each speaker's voice as a clip "
                          "voice instead (record it, or add the file), then pick it in Cast\u2026.")
        if model is not None and entry is not None and model.mode in ("custom_voice", "preset"):
            speaker = self.qwen_speaker_combo.currentData() or ""
            style = self.qwen_instruct_input.text().strip() if model.mode == "custom_voice" else ""
            name = self.qwen_speaker_combo.currentText().split(" (")[0]
            return voice_library.Voice(name=name, kind="preset", backend=entry["backend"], repo_id=entry["repo_id"],
                                       speaker=speaker, style=style, language=language), None
        if model is not None and entry is not None and model.mode == "voice_design":
            description = self.qwen_instruct_input.text().strip()
            if not description:
                return None, "Describe the voice first (Voice description, in the Delivery card)."
            return voice_library.Voice(name="Designed voice", kind="design", backend=entry["backend"],
                                       repo_id=entry["repo_id"], mode=model_registry.entry_mode(entry)
                                       if entry["backend"] in DUAL_MODE_BACKENDS else "",
                                       description=description, language=language), None
        path = self.ref_audio_path_label.toolTip()
        if not path:
            return None, "You're using the model's default voice. Record a clip or add a file to save a voice."
        existing = self.voice_library.find_clip(path)
        if existing:
            return existing, None
        return voice_library.Voice(name=os.path.splitext(os.path.basename(path))[0], kind="clip",
                                   clip=self.voice_library.to_stored(path)), None

    def save_current_voice(self):
        voice, problem = self.current_voice_spec()
        if problem:
            QMessageBox.information(self, "Save Voice", problem)
            return
        if voice in self.voice_library.voices:
            self.edit_voice(voice)
            return
        transcript = None
        if voice.kind == "clip":
            transcript = voice_library.read_transcript(self.voice_library.clip_path(voice))
        dialog = VoiceDetailsDialog("Save voice", voice, self.voice_library, transcript,
                                    offer_clip=voice.kind != "clip", parent=self)
        if not dialog_accepted(dialog.exec()):
            return
        transcript = dialog.apply()
        self.voice_library.add(voice)
        if transcript is not None:
            voice_library.write_transcript(self.voice_library.clip_path(voice), transcript)
            self.load_reference_transcript(self.voice_library.clip_path(voice))
        self.active_voice_id = voice.id
        self.set_status_message(f"Status: Saved {voice.name} to the voice library.")
        self.render_voice_tiles()
        if dialog.make_clip():
            self.make_voice_clip(voice)

    def edit_voice(self, voice):
        path = self.voice_library.clip_path(voice)
        transcript = voice_library.read_transcript(path) if path and os.path.exists(path) else None
        dialog = VoiceDetailsDialog("Edit voice", voice, self.voice_library, transcript, parent=self)
        if not dialog_accepted(dialog.exec()):
            return
        transcript = dialog.apply()
        if transcript is not None:
            voice_library.write_transcript(path, transcript)
            if self.voice_is_active(voice):
                self.load_reference_transcript(path)
        self.voice_library.save()
        self.render_voice_tiles()

    def remove_voice(self, voice):
        path = self.voice_library.clip_path(voice)
        owned = bool(path) and os.path.abspath(path).startswith(os.path.abspath(self.voice_library.clips_dir))
        detail = ("Its clip, made by the library, is deleted too." if owned else
                  "The audio file stays where it is." if path else "")
        answer = QMessageBox.question(self, "Remove Voice", f"Remove {voice.name} from the library?\n\n{detail}")
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.preview_button_playing is self.voice_tiles:
            self.stop_reference_preview()
        if owned and self.voice_is_active(voice):
            self.set_reference_audio(None)
        self.voice_library.remove(voice)
        self.render_voice_tiles()

    def make_voice_clip(self, voice):
        """Generate a clip of a preset or designed voice reading a passage, so cloning
        models can use it. The voice has to be the one in use (it is loaded first)."""
        if not self.voice_is_active(voice):
            self.use_voice(voice)
            if not self.voice_is_active(voice):
                self.pending_clip_voice = voice  # made once the model has loaded
                return
        problem = self.prepare_qwen_generation()
        if problem:
            QMessageBox.information(self, "Make Clip", problem)
            return
        language = self.language_combo.currentData() or "en"
        passage = REFERENCE_READING_SCRIPTS[0]
        self.set_status_message(f"Status: Making a clip of {voice.name}...")
        self.generate_button.setEnabled(False)
        self.clip_thread = MakeClipThread(self.model, passage, language, self)
        self.clip_thread.finished_with.connect(
            lambda wav, sr, error, v=voice, text=passage: self.on_voice_clip_made(v, text, wav, sr, error))
        self.clip_thread.start()

    def on_voice_clip_made(self, voice, text, wav, sr, error):
        self.generate_button.setEnabled(self.model is not None)
        if error or wav is None:
            self.set_status_message("Status: Could not make the clip. See the Log page.")
            QMessageBox.warning(self, "Make Clip", f"Could not make the clip:\n{error}")
            return
        old = self.voice_library.clip_path(voice)
        path = self.voice_library.new_clip_path(voice.name)
        import soundfile
        soundfile.write(path, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        voice_library.write_transcript(path, text)
        voice.clip = self.voice_library.to_stored(path)
        self.voice_library.save()
        if old and old != path and old.startswith(os.path.abspath(self.voice_library.clips_dir)):
            for target in (old, voice_library.transcript_path(old)):
                if os.path.exists(target):
                    os.remove(target)
        self.set_status_message(f"Status: Made a {len(wav) / sr:.0f} s clip of {voice.name}. "
                                "Cloning models can use it now.")
        self.render_voice_tiles()

    # --- Reference audio recording ---

    def populate_microphones(self):
        previous = self.mic_combo.currentData()
        previous_id = previous.id() if previous is not None else None
        self.mic_combo.blockSignals(True)
        self.mic_combo.clear()
        default_id = QMediaDevices.defaultAudioInput().id()
        for device in QMediaDevices.audioInputs():
            label = device.description()
            if device.id() == default_id:
                label += " (default)"
            self.mic_combo.addItem(label, device)
        selected_index = 0
        for index in range(self.mic_combo.count()):
            device_id = self.mic_combo.itemData(index).id()
            if device_id == previous_id or (previous_id is None and device_id == default_id):
                selected_index = index
                break
        self.mic_combo.setCurrentIndex(selected_index)
        self.mic_combo.blockSignals(False)
        has_inputs = self.mic_combo.count() > 0
        if not has_inputs:
            self.mic_combo.addItem("No microphone found")
        self.mic_combo.setEnabled(has_inputs)
        self.record_button.setEnabled(has_inputs)

    def open_recording_dialog(self):
        device = self.mic_combo.currentData()
        if device is None or device.isNull():
            QMessageBox.warning(self, "No Microphone",
                                "No audio input device is available.")
            return
        dialog = RecordingDialog(device, self)
        if not dialog_accepted(dialog.exec()):
            self.set_status_message("Status: Recording cancelled.")
            return
        self.recording_format = dialog.audio_format
        self.recording_buffer = dialog.recorded_bytes
        self.recording_script = dialog.script_label.text()
        self._save_recording()
        self.recording_buffer = bytearray()

    def _save_recording(self):
        audio_format = self.recording_format
        mono = pcm_to_mono_float(bytes(self.recording_buffer), audio_format)
        duration = mono.size / audio_format.sampleRate()
        if duration < MIN_RECORDING_SECONDS:
            self.set_status_message("Status: Recording discarded (too short).")
            QMessageBox.warning(
                self, "Recording Too Short",
                f"The recording was {duration:.1f}s. Please record at least "
                f"{MIN_RECORDING_SECONDS}s; about 10-15s of clear speech works best.")
            return
        os.makedirs(self.recordings_directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.recordings_directory, f"reference_{timestamp}.wav")
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio_format.sampleRate())
            wav_file.writeframes(pcm16.tobytes())
        script = getattr(self, "recording_script", "")
        if script:
            with open(self.transcript_path(output_path), "w", encoding="utf-8") as handle:
                handle.write(script + "\n")
        self.voice_library.import_recordings()
        self.active_voice_id = getattr(self.voice_library.find_clip(output_path), "id", None)
        self.set_reference_audio(output_path)
        self.refresh_recordings_list()
        self.last_reference_audio_dir = self.recordings_directory
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        print(f"Saved reference recording ({duration:.1f}s, peak {peak:.3f}): {output_path}")
        self.set_status_message(
            f"Status: Recorded {duration:.1f}s reference clip and selected it.")
        if peak < SILENT_RECORDING_PEAK:
            QMessageBox.warning(
                self, "Recording Is Nearly Silent",
                "The clip was saved and selected, but almost no sound was captured.\n\n"
                "Check that the right microphone is selected, that it isn't muted, and that "
                "Windows allows desktop apps to use it (Settings > Privacy & security > Microphone).")

    def on_model_repo_changed(self, _index):
        entry = self.get_selected_model_entry()
        self.selected_model_repo = entry["repo_id"]
        self.refresh_model_repo_tooltip()
        self.refresh_language_options()
        if self.entry_key(entry) != self.loaded_entry_key():
            self.load_model(entry)

    # --- Model page ---

    @staticmethod
    def entry_key(entry):
        if entry.get("backend") in DUAL_MODE_BACKENDS:
            return (entry.get("repo_id"), entry.get("backend"), model_registry.entry_mode(entry))
        return (entry.get("repo_id"), entry.get("backend"), entry.get("multilingual_t3_model") or "")

    def loaded_entry_key(self):
        backend = self.current_model_backend
        weights = self.current_multilingual_t3_model if backend == BACKEND_MULTILINGUAL else ""
        if backend in DUAL_MODE_BACKENDS:
            weights = getattr(self, "current_mode", "clone")
        return (self.current_model_repo, backend, weights or "")

    def refresh_models_page(self, select_entry=None):
        if not hasattr(self, "capability_tabs"):
            return
        self.model_cache_sizes = model_registry.cached_repo_sizes()
        if select_entry is not None:
            self.show_capability(model_registry.capability_for(select_entry))
        self.render_model_tiles()

    def show_capability(self, capability):
        for index in range(self.capability_tabs.count()):
            if self.capability_tabs.tabData(index) == capability:
                self.capability_tabs.setCurrentIndex(index)  # renders via currentChanged

    def current_capability(self):
        return self.capability_tabs.tabData(self.capability_tabs.currentIndex())

    def on_page_changed(self, page):
        if page == self.PAGE_MODEL and getattr(self, "discover_results", None) is None \
                and getattr(self, "discover_thread", None) is None:
            self.start_discover()

    def is_active_entry(self, entry):
        return self.model is not None and self.entry_key(entry) == self.loaded_entry_key()

    def model_busy(self):
        return getattr(self, "model_is_loading", False) or self.is_generating or getattr(self, "api_busy", False)

    def render_model_tiles(self):
        if not hasattr(self, "capability_tabs"):
            return
        if getattr(self, "model_cache_sizes", None) is None:
            self.model_cache_sizes = model_registry.cached_repo_sizes()
        groups = {capability: members for capability, _title, members
                  in model_registry.group_by_capability(self.model_entries)}
        for index in range(self.capability_tabs.count()):
            capability = self.capability_tabs.tabData(index)
            title = model_registry.CAPABILITY_TABS[capability]
            count = len(groups.get(capability, []))
            self.capability_tabs.setTabText(index, f"{title}  {count}" if count else title)
        capability = self.current_capability()
        self.capability_note.setText(model_registry.CAPABILITIES[capability][1])
        self.your_tiles.set_tiles(
            [self.model_tile(entry) for entry in groups.get(capability, [])],
            "None yet. Add one from Discover below.")
        self.render_discover_tiles()

    def engine_installed(self, entry):
        module = ENGINE_MODULES.get(entry.get("backend"))
        return module is None or module.is_installed()

    def gpu_memory(self):
        """(name, GB) of the first CUDA GPU, or None."""
        if not hasattr(self, "_gpu_memory"):
            self._gpu_memory = None
            try:
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    self._gpu_memory = (props.name, props.total_memory / 1024 ** 3)
            except Exception:
                pass
        return self._gpu_memory

    def hardware_line(self, backend, repo_id):
        """(kind, text, tooltip) comparing a model's GPU memory needs with this PC."""
        needs = model_registry.hardware_needs(backend, repo_id)
        wanted = f"{needs.min_gb:g} GB" if needs.good_gb == needs.min_gb else \
            f"{needs.min_gb:g}\u2013{needs.good_gb:g} GB"
        gpu = self.gpu_memory()
        detail = (f"Needs about {needs.min_gb:g} GB of GPU memory"
                  + ("" if needs.good_gb == needs.min_gb else f", {needs.good_gb:g} GB for full speed")
                  + f". {needs.note}")
        if gpu is None:
            if needs.cpu_ok:
                return "tight", "No GPU found: runs on the CPU", detail + "\nNo NVIDIA GPU was found."
            return "short", f"Needs an NVIDIA GPU ({needs.min_gb:g} GB+)", detail
        name, memory = gpu
        detail += f"\nYour GPU: {name}, {memory:.0f} GB."
        if memory + 0.5 >= needs.good_gb:
            return "good", f"\u2713 GPU {wanted} \u00b7 yours {memory:.0f} GB", detail
        if memory + 0.5 >= needs.min_gb:
            return "tight", f"! GPU {wanted} \u00b7 yours {memory:.0f} GB", \
                detail + "\nIt runs, but slower than on a bigger GPU."
        fallback = " (CPU, slow)" if needs.cpu_ok else ""
        return "short", f"\u2717 Needs {needs.min_gb:g} GB GPU{fallback} \u00b7 yours {memory:.0f} GB", detail

    def typical_speed_text(self, entry):
        """Estimated time for 1,000 characters, split the way generation would split them."""
        count = max(1, math.ceil(1000 / (self.max_section_chars_for(entry) * 0.85)))
        seconds, measured = self.estimate_seconds(entry, [1000 // count] * count)
        amount = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"
        return f"\u2248 {amount} per 1,000 characters", measured

    def model_tile(self, entry):
        engine = model_registry.engine_for(entry)
        active = self.is_active_entry(entry)
        repo = entry["repo_id"]
        subtitle = model_registry.engine_label(entry)
        if engine.uses_weights_version:
            subtitle += f" {model_registry.weights_file(entry).split('_')[-1].split('.')[0].upper()}"
        subtitle += f" \u00b7 {engine.languages_summary}"
        badges = []
        if active:
            badges.append(("active", "Loaded", "This model is loaded and ready to generate."))
        elif not self.engine_installed(entry):
            badges.append(("status", "Needs engine", "Click to install this engine (it runs in its own environment)."))
        elif model_registry.is_downloaded(entry):
            size = model_registry.format_size(self.model_cache_sizes.get(repo, 0))
            badges.append(("status", f"Ready \u00b7 {size}", "Downloaded; loads from the local cache."))
        elif entry.get("download_bytes"):
            size = model_registry.format_size(entry["download_bytes"])
            badges.append(("status", f"Download {size}", "Downloads the first time you load it."))
        else:
            badges.append(("status", "Not downloaded", "Downloads the first time you load it."))
        badges.append(model_tiles.license_badge(model_registry.license_of(entry)))
        speed, measured = self.typical_speed_text(entry)
        if not entry.get("enabled", True):
            speed = f"Hidden · {speed}"
        tooltip = "\n".join(line for line in (
            f"{entry['label']}  ({repo})",
            engine.description,
            entry.get("notes", ""),
            "" if entry.get("enabled", True) else "Hidden from the model switcher on the Generate page.",
            f"Speed {'measured from your runs' if measured else 'estimated until you generate with it'}.",
            "" if active else "Click to load. Right-click for more.") if line)
        needs = self.hardware_line(entry.get("backend"), repo)
        tooltip += "\n" + needs[2]
        tile = model_tiles.ModelTile(entry["label"], subtitle, badges, speed, tooltip,
                                     active=active, with_menu=True, needs=needs)
        tile.clicked.connect(lambda e=entry: self.load_entry(e))
        tile.menu_requested.connect(lambda pos, e=entry: self.show_model_menu(e, pos))
        return tile

    def show_model_menu(self, entry, pos):
        menu = QMenu(self)
        active = self.is_active_entry(entry)
        hidden = not entry.get("enabled", True)
        is_default = entry["repo_id"] == DEFAULT_MODEL_REPO and entry.get("backend") == BACKEND_MULTILINGUAL \
            and sum(1 for e in self.model_entries if self.entry_key(e) == self.entry_key(entry)) == 1
        visible = sum(1 for e in self.model_entries if e.get("enabled", True))
        actions = (
            ("Loaded" if active else "Load", lambda: self.load_entry(entry),
             not active and not self.model_busy()),
            None,
            ("Edit...", lambda: self.edit_model(entry), True),
            ("Duplicate...", lambda: self.duplicate_model(entry), True),
            ("Show in model switcher" if hidden else "Hide from model switcher",
             lambda: self.toggle_model_hidden(entry), hidden or visible > 1),
            ("Open on Hugging Face",
             lambda: QDesktopServices.openUrl(QUrl(f"https://huggingface.co/{entry['repo_id']}")), True),
            None,
            ("Remove (official fallback)" if is_default else "Remove (unload it first)" if active
             else "Remove...", lambda: self.remove_model(entry), not is_default and not active),
        )
        for item in actions:
            if item is None:
                menu.addSeparator()
                continue
            text, callback, enabled = item
            action = menu.addAction(text)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False, callback=callback: callback())
        menu.exec(pos)

    def update_model_details(self):
        self.render_model_tiles()

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

    def persist_model_entries(self, entries, select_entry=None):
        try:
            model_registry.save_models_config(self.model_config_path, entries)
        except OSError as exc:
            QMessageBox.warning(self, "Could Not Save", f"{MODEL_CONFIG_FILENAME} could not be written:\n{exc}")
            return False
        self.model_entries = load_models_config(self.model_config_path)
        self.refresh_model_repo_options()
        self.refresh_model_repo_tooltip()
        self.refresh_models_page(select_entry)
        self.set_status_message(f"Status: Saved model list to {MODEL_CONFIG_FILENAME}.")
        return True

    def _edit_entry_dialog(self, entry, replacing=None):
        others = [e for e in self.model_entries if e is not replacing]
        dialog = ModelEntryDialog(entry, others, self.app_settings.get("hf_token"), self)
        if not dialog_accepted(dialog.exec()):
            return None
        return dialog.result_entry()

    def add_model(self):
        new_entry = self._edit_entry_dialog(None)
        if new_entry:
            self.persist_model_entries(self.model_entries + [new_entry], new_entry)

    def edit_model(self, entry):
        updated = self._edit_entry_dialog(entry, replacing=entry)
        if updated:
            entries = [updated if e is entry else e for e in self.model_entries]
            self.persist_model_entries(entries, updated)

    def duplicate_model(self, entry):
        copy = dict(entry)
        base, n = f"{entry['label']} copy", 2
        copy["label"] = base
        while any(e.get("label") == copy["label"] for e in self.model_entries):
            copy["label"] = f"{base} {n}"
            n += 1
        updated = self._edit_entry_dialog(copy)
        if updated:
            self.persist_model_entries(self.model_entries + [updated], updated)

    def toggle_model_hidden(self, entry):
        updated = dict(entry, enabled=not entry.get("enabled", True))
        self.persist_model_entries([updated if e is entry else e for e in self.model_entries], updated)

    def remove_model(self, entry):
        answer = QMessageBox.question(
            self, "Remove Model",
            f"Remove '{entry['label']}' from the model list?\n\nDownloaded files stay in the "
            "Hugging Face cache, so adding it back later won't download again.")
        if answer == QMessageBox.StandardButton.Yes:
            self.persist_model_entries([e for e in self.model_entries if e is not entry])

    def load_entry(self, entry):
        if self.is_active_entry(entry):
            return
        if self.model_busy():
            self.set_status_message("Status: Wait for the current load or generation to finish.")
            return
        if not self.engine_installed(entry):
            self.install_engine(entry)
            return
        index = self.model_repo_combo.findText(entry["label"])
        if index >= 0 and index != self.model_repo_combo.currentIndex():
            self.model_repo_combo.setCurrentIndex(index)  # loads via the switcher
        elif self.entry_key(entry) != self.loaded_entry_key() or self.model is None:
            self.load_model(entry)

    def install_engine(self, entry):
        backend = entry.get("backend")
        name, note = ENGINE_INSTALL_NOTES[backend]
        if getattr(self, "engine_install_thread", None) is not None and self.engine_install_thread.isRunning():
            self.set_status_message("Status: An engine is still installing. Progress is on the Log page.")
            return
        answer = QMessageBox.question(self, f"Install {name} Engine", f"{note}\n\nInstall now?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.set_status_message(f"Status: Installing the {name} engine. Progress is on the Log page.")
        self.pending_install_entry = entry
        self.engine_install_thread = EngineInstallThread(ENGINE_MODULES[backend])
        self.engine_install_thread.finished_with.connect(
            lambda error, name=name: self.on_engine_install_finished(name, error))
        self.engine_install_thread.start()

    def on_engine_install_finished(self, name, error):
        if error:
            self.set_status_message(f"Status: {name} engine install failed. See the Log page.")
            QMessageBox.warning(self, f"{name} Engine", f"The install failed:\n{error}")
        else:
            self.set_status_message(f"Status: {name} engine installed. Loading the model...")
            self.refresh_models_page()
            if self.pending_install_entry is not None:
                self.load_entry(self.pending_install_entry)
        self.render_model_tiles()

    def save_hf_token(self):
        token = self.hf_token_input.text().strip()
        if token:
            self.app_settings["hf_token"] = token
        else:
            self.app_settings.pop("hf_token", None)
        self.apply_hf_token_setting()
        self.save_app_settings()
        self.update_hf_token_status("Token saved." if token else "Token removed.")

    def test_hf_token(self):
        token = self.hf_token_input.text().strip()
        if not token:
            self.update_hf_token_status("Enter a token to test.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, message = model_registry.whoami(token)
        finally:
            QApplication.restoreOverrideCursor()
        self.update_hf_token_status(message, ok)

    def update_hf_token_status(self, message=None, ok=None):
        saved = bool(self.app_settings.get("hf_token"))
        base = ("A token is saved and used for downloads." if saved else
                "Optional: only needed for gated or private repos.")
        self.hf_token_status.setText(f"{message}  {base}" if message else base)
        color = "#3c9a3c" if ok else "#d9534f" if ok is False else ""
        self.hf_token_status.setStyleSheet(f"color: {color};" if color else "")
        self.hf_token_status.setToolTip(
            "Public models need no token. A token unlocks gated or private repos and higher "
            "download limits. It is stored only in app_settings.json on this computer (ignored "
            "by git), never in models.json.")

    def on_tuning_changed(self, *_args):
        self.repetition_penalty = self.repetition_spin.value()
        self.min_p = self.min_p_spin.value()
        self.top_p = self.top_p_spin.value()
        self.app_settings["sampling"] = {
            "repetition_penalty": self.repetition_penalty, "min_p": self.min_p, "top_p": self.top_p}
        defaults = (abs(self.repetition_penalty - 1.2) < 1e-9 and abs(self.min_p - 0.05) < 1e-9
                    and abs(self.top_p - 1.0) < 1e-9)
        self.tuning_summary_label.setText("defaults" if defaults else
                                          f"repetition {self.repetition_penalty:.2f}, "
                                          f"min-p {self.min_p:.2f}, top-p {self.top_p:.2f}")

    def reset_tuning(self):
        self.repetition_spin.setValue(1.2)
        self.min_p_spin.setValue(0.05)
        self.top_p_spin.setValue(1.0)

    def set_tuning_expanded(self, expanded):
        self.tuning_panel.setVisible(expanded)
        arrow = "\u25be" if expanded else "\u25b8"
        self.tuning_toggle.setText(f"{arrow} Fine-tuning")
        self.app_settings["tuning_expanded"] = expanded
        self.on_tuning_changed()
        if self.isVisible():
            self.update_minimum_size()

    def get_selected_model_repo(self):
        return self.get_selected_model_entry()["repo_id"]

    def get_visible_model_entries(self):
        enabled_entries = [
            entry for entry in self.model_entries
            if entry.get("enabled", True)
        ]
        return enabled_entries or self.model_entries[:1]

    def refresh_model_repo_options(self):
        selected_label = None
        if hasattr(self, "model_repo_combo") and self.model_repo_combo.count() > 0:
            selected_label = self.model_repo_combo.currentText()

        visible_entries = self.get_visible_model_entries()
        self.model_repo_combo.blockSignals(True)
        self.model_repo_combo.clear()
        header_font = QFont(self.model_repo_combo.font())
        header_font.setBold(True)
        for _capability, title, members in model_registry.group_by_capability(visible_entries):
            self.model_repo_combo.addItem(title.upper(), None)
            header = self.model_repo_combo.model().item(self.model_repo_combo.count() - 1)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setFont(header_font)
            header.setForeground(QApplication.palette().color(QPalette.ColorRole.Link))
            for entry in members:
                self.model_repo_combo.addItem(entry["label"], visible_entries.index(entry))
                self.model_repo_combo.setItemData(
                    self.model_repo_combo.count() - 1,
                    f"{entry['repo_id']} \u00b7 {model_registry.engine_label(entry)}",
                    Qt.ItemDataRole.ToolTipRole)

        selected_index = -1
        if selected_label:
            selected_index = self.model_repo_combo.findText(selected_label)
            if selected_index >= 0 and self.model_repo_combo.itemData(selected_index) is None:
                selected_index = -1
        if selected_index < 0:
            for index, entry in enumerate(visible_entries):
                if self.entry_key(entry) == self.loaded_entry_key():
                    selected_index = self.model_repo_combo.findData(index)
                    break

        if selected_index < 0:
            for index, entry in enumerate(visible_entries):
                if entry["repo_id"] == DEFAULT_MODEL_REPO and entry.get("backend") == BACKEND_MULTILINGUAL:
                    selected_index = self.model_repo_combo.findData(index)
                    break

        if selected_index < 0 and visible_entries:
            selected_index = self.model_repo_combo.findData(0)

        if selected_index >= 0:
            self.model_repo_combo.setCurrentIndex(selected_index)
        self.model_repo_combo.blockSignals(False)

    def get_selected_model_entry(self):
        visible_entries = self.get_visible_model_entries()
        selected_index = self.model_repo_combo.currentData()
        if isinstance(selected_index, int) and 0 <= selected_index < len(visible_entries):
            return visible_entries[selected_index]
        return visible_entries[0]

    def refresh_model_repo_tooltip(self):
        selected_entry = self.get_selected_model_entry()
        tooltip = (f"{selected_entry['repo_id']} \u00b7 "
                   f"{model_registry.engine_for(selected_entry).label}\n"
                   "Switching loads the model. Manage models on the Model page.")
        if selected_entry.get("notes"):
            tooltip += f"\n\n{selected_entry['notes']}"
        self.model_repo_combo.setToolTip(tooltip)

    def refresh_language_options(self):
        selected_entry = self.get_selected_model_entry()
        supported_languages = languages_for_backend(
            selected_entry.get("backend", BACKEND_MULTILINGUAL)
        )
        preferred_language = selected_entry.get("language_id", "en")

        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        for language_id, language_name in supported_languages.items():
            self.language_combo.addItem(f"{language_name} [{language_id}]", language_id)

        preferred_index = self.language_combo.findData(preferred_language)
        if preferred_index < 0:
            preferred_index = self.language_combo.findData("en")
        if preferred_index < 0 and self.language_combo.count() > 0:
            preferred_index = 0
        if preferred_index >= 0:
            self.language_combo.setCurrentIndex(preferred_index)

        is_multilingual = selected_entry.get("backend") in (BACKEND_MULTILINGUAL, QWEN_BACKEND, KOKORO_BACKEND,
                                                            VIBEVOICE_BACKEND, *DUAL_MODE_BACKENDS)
        self.language_combo.setEnabled(is_multilingual)
        self.language_combo.setToolTip(
            "Language used by the multilingual Chatterbox backend."
            if is_multilingual else
            "Legacy English backend only."
        )
        self.language_combo.blockSignals(False)

    def apply_selected_text_preset(self):
        selected_entry = self.get_selected_model_entry()
        selected_language = self.language_combo.currentData() or selected_entry.get("language_id", "en")
        preset_text = selected_entry.get("test_texts", {}).get(selected_language)
        if not preset_text and selected_entry.get("backend") == BACKEND_MULTILINGUAL:
            preset_text = DEFAULT_LANGUAGE_TEST_TEXTS.get(
                selected_language,
                DEFAULT_LANGUAGE_TEST_TEXTS["en"],
            )
        if not preset_text:
            preset_text = selected_entry.get("test_text")
        if preset_text:
            self.current_document_name = None
            self.document_label.clear()
            self.text_input.setPlainText(preset_text)

    def set_model_loading_state(self, is_loading):
        if is_loading:
            self.model_load_progress.setEnabled(True)
            self.model_load_progress.setRange(0, 0)
        else:
            self.model_load_progress.setRange(0, 1)
            self.model_load_progress.setValue(0)
            self.model_load_progress.setEnabled(False)
        self.model_is_loading = is_loading
        self.model_repo_combo.setEnabled(not is_loading)
        self.use_preset_button.setEnabled(not is_loading)
        self.update_model_details()
        if is_loading:
            self.language_combo.setEnabled(False)
        else:
            self.refresh_language_options()

    def release_model(self):
        """Free the current model (and stop a Qwen worker) before loading another."""
        old_model, self.model = self.model, None
        if isinstance(old_model, WORKER_MODEL_TYPES):
            old_model.close()
        del old_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, "qwen_row"):
            self.update_engine_controls()

    def load_model(self, selected_entry=None):
        if not CHATTERBOX_AVAILABLE:
            QMessageBox.critical(
                self, "Error", "ChatterboxTTS library not installed.")
            return
        if getattr(self, "model_is_loading", False):
            return
        selected_entry = selected_entry or self.get_selected_model_entry()
        selected_repo = selected_entry["repo_id"]
        selected_backend = selected_entry.get("backend", BACKEND_MULTILINGUAL)
        if (selected_backend in DUAL_MODE_BACKENDS and isinstance(self.model, DUAL_MODE_TYPES)
                and self.model.backend == selected_backend and self.model.repo_id == selected_repo):
            self.switch_voice_mode(selected_entry)
            return
        selected_multilingual_t3_model = selected_entry.get(
            "multilingual_t3_model",
            DEFAULT_MULTILINGUAL_T3_MODEL,
        )
        self.current_model_repo = selected_repo
        self.current_model_backend = selected_backend
        self.current_multilingual_t3_model = selected_multilingual_t3_model
        self.set_status_message(
            f"Status: Loading {selected_entry['label']}. A model that isn't downloaded yet "
            "is fetched first; progress appears on the Log page."
        )
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.release_model()
        self.set_model_loading_state(True)
        self.current_mode = model_registry.entry_mode(selected_entry)
        self.model_loader_thread = ModelLoaderThread(
            selected_repo,
            selected_backend,
            selected_multilingual_t3_model,
        )
        self.model_loader_thread.mode = self.current_mode
        self.cuda_runtime_issue = self.model_loader_thread.cuda_probe_error
        self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loader_thread.error_occurred.connect(
            self.on_model_load_error)
        self.model_loader_thread.start()

    def switch_voice_mode(self, entry):
        """Switch a loaded dual-mode model between cloning and design without reloading."""
        self.current_mode = model_registry.entry_mode(entry)
        self.model.set_mode(self.current_mode)
        self.set_status_message(f"Status: Switched to {entry['label']} (same model, no reload). Ready.")
        self.update_engine_controls()
        self.refresh_language_options()
        self.update_text_stats()
        self.render_model_tiles()
        self.after_voice_model_ready()

    def after_voice_model_ready(self):
        if self.pending_voice is not None:
            self.apply_pending_voice()
        clip_voice = getattr(self, "pending_clip_voice", None)
        if clip_voice is not None:
            self.pending_clip_voice = None
            if self.voice_is_active(clip_voice):
                self.make_voice_clip(clip_voice)
        self.render_voice_tiles()

    def on_model_loaded(self, model_instance, device_used):
        self.model_is_warm = False
        self.model = model_instance
        self.device_used = device_used
        status_message = (
            f"Status: Model loaded from {self.current_model_repo} "
            f"using {self.current_model_backend} on {self.device_used}. Ready."
        )
        if self.system_has_nvidia_gpu and self.device_used == "cpu":
            status_message += " NVIDIA GPU detected, but PyTorch CUDA is unavailable."
        self.set_status_message(status_message)
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.set_model_loading_state(False)
        self.update_text_stats()
        self.refresh_models_page()
        self.update_engine_controls()
        self.after_voice_model_ready()
        if self.system_has_nvidia_gpu and self.device_used == "cpu":
            details = self.cuda_runtime_issue or (
                "This Python environment is using a CPU-only PyTorch build."
            )
            QMessageBox.warning(
                self,
                "CUDA Not Active",
                "An NVIDIA GPU was detected, but this Python environment is using "
                "a PyTorch configuration that cannot run on the detected GPU.\n\n"
                f"Details: {details}\n\n"
                "Re-run the installer to repair the PyTorch installation for CUDA.",
            )

    def on_model_load_error(self, error_msg):
        self.model = None
        self.set_status_message(f"Status: Model load failed. {error_msg}")
        self.generate_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.set_model_loading_state(False)
        self.refresh_models_page()
        QMessageBox.critical(self, "Model Load Error", error_msg)

    def on_generation_thread_finished(self):
        print("UI: audio_generator_thread.finished signal received.")

        # Stop the timer regardless of how the thread finished
        if self.generation_timer.isActive():
            print("UI: Stopping generation timer.")
            self.generation_timer.stop()

        # Reset UI elements
        self.is_generating = False
        self.generate_button.setText("Generate Audio")
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.open_document_button.setEnabled(True)
        self.model_repo_combo.setEnabled(not getattr(self, "model_is_loading", False))
        self.update_model_details()
        self.generation_progress.setVisible(False)
        if not self.keep_take_button.isVisible() and not self.keep_voice_button.isVisible():
            self.activity_label.clear()
        self.update_text_stats()

        # Final status update based on how the thread might have ended,
        # if not already set by on_generation_complete or on_generation_error.
        # This ensures "Stopping..." doesn't linger.
        current_status = self.status_bar.text()
        if "stopping generation..." in current_status.lower() or \
           "stop requested." in current_status.lower():
            self.set_status_message("Status: Generation stopped by user.")
        elif not any(marker in current_status.lower() for marker in (
                "full audio generated", "failed", "stopped by user",
                "preview ready", "stopped. saved")):
            # If no specific completion or error message was set, default to Ready
            self.set_status_message("Status: Ready.")

    def update_generation_time_display(self):
        self.refresh_progress_activity()
        if self.generation_start_time and self.is_generating:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            # Only update if not showing chunk progress, to avoid flicker
            # and if the button still says "Stop Generation" (i.e. not "Stopping...")
            if "chunk" not in self.status_bar.text().lower() and \
               self.generate_button.text() == "Stop Generation":
                self.set_status_message(
                    f"Status: Generating... (Elapsed: {self.format_time(elapsed_ms)})")
        elif not self.is_generating and self.generation_timer.isActive():
            # This is a failsafe, should be stopped by on_generation_thread_finished
            print(
                "UI: Generation timer stopped by failsafe in update_generation_time_display.")
            self.generation_timer.stop()

    def update_generation_time(self):
        if self.generation_start_time and self.is_generating:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            seconds = int((elapsed_ms / 1000) % 60)
            minutes = int((elapsed_ms / (1000 * 60)) % 60)
            self.set_status_message(
                f"Status: Generating audio... {minutes:02}:{seconds:02}")

    def on_chunk_generated_progress(self, first, last, total):
        if not self.is_generating:
            return
        done = first - 1
        elapsed = time.monotonic() - self.generation_started_at
        # Everything before this batch has just finished: calibrate against the plan.
        self.progress_done_cost = sum(cost for _f, end, cost in getattr(self, "generation_plan", [])
                                      if end <= done)
        self.progress_done_time = elapsed
        self.progress_range = (first, last, total)
        self.generation_progress.setMaximum(total)
        self.generation_progress.setValue(done)
        span = f"{first}" if first == last else f"{first}\u2013{last}"
        self.set_status_message(f"Status: Generating section {span} of {total}...")
        self.refresh_progress_activity()

    def refresh_progress_activity(self):
        """Activity text with a countdown; called on progress and every second."""
        if not self.is_generating or not getattr(self, "progress_range", None):
            return
        first, last, total = self.progress_range
        elapsed = time.monotonic() - self.generation_started_at
        estimate = getattr(self, "generation_estimate", 0.0)
        if self.progress_done_cost > 0:
            projected = self.progress_done_time / self.progress_done_cost * estimate
        else:
            projected = estimate
        remaining = max(projected - elapsed, 0.0)
        span = f"{first}" if first == last else f"{first}\u2013{last}"
        activity = f"{span}/{total}"
        if self.generation_is_preview:
            activity = f"Preview {activity}"
        if estimate > 0:
            activity += f" \u00b7 {self.format_clock(remaining)} left" if remaining >= 1 else " \u00b7 finishing"
        self.activity_label.setText(activity)

    def on_generation_complete(self, output_path, sample_rate):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        total_generation_time_str = ""
        if self.generation_start_time:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            total_generation_time_str = f" (Total time: {self.format_time(elapsed_ms)})"

        thread = self.audio_generator_thread
        self.generation_progress.setValue(self.generation_progress.maximum())
        if thread.preview:
            self.last_preview_seed = thread.actual_seed_used
            self.activity_label.setText(f"Preview take {self.last_preview_seed}")
            self.keep_take_button.setVisible(self.seed_input.value() == 0)
            designed = self.active_qwen_model()
            self.keep_voice_button.setVisible(
                designed is not None and designed.mode == "voice_design"
                and getattr(designed, "_anchor", None) is not None and not getattr(designed, "locked_anchor", None))
            self.set_status_message(f"Status: Preview ready{total_generation_time_str}.")
        elif thread.partial_info:
            done, total = thread.partial_info
            self.set_status_message(
                f"Status: Stopped. Saved {done} of {total} sections: {os.path.basename(output_path)}")
        else:
            captions = f" + {os.path.basename(thread.subtitle_path)}" if thread.subtitle_path else ""
            self.set_status_message(
                f"Status: Full audio generated: {os.path.basename(output_path)}{captions}{total_generation_time_str}")

        self.current_audio_file = output_path
        # ... (rest of the method same as your working version)
        self.current_file_label.setText(
            f"Last generated: {os.path.basename(output_path)}")
        self.media_player.setSource(QUrl.fromLocalFile(output_path))
        self.play_pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.playhead_slider.setEnabled(True)
        self.update_output_log()
        if self.autoplay_checkbox.isChecked() or thread.preview:
            self.media_player.play()

    def on_generation_error(self, error_msg):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        is_user_stop = "stopped by user" in error_msg.lower()

        final_status_msg = f"Status: {'Generation stopped by user.' if is_user_stop else 'Generation failed.'}"
        if not is_user_stop and error_msg:
            first_line_error = error_msg.splitlines()[0]
            if len(first_line_error) > 70:
                # Adjusted length
                first_line_error = first_line_error[:67] + "..."
            final_status_msg += f" ({first_line_error})"

        self.set_status_message(final_status_msg)

        if not is_user_stop:
            QMessageBox.critical(self, "Generation Error", error_msg)
        else:
            print(f"User stop confirmed by error signal: {error_msg}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current_tooltip = self.status_bar.toolTip()
        if current_tooltip:
            self.set_status_message(current_tooltip)

    def update_output_log(self):
        self.output_log_listwidget.clear()
        try:
            files = sorted([os.path.join(self.output_directory, f)for f in os.listdir(
                self.output_directory)if f.lower().endswith(audio_effects.AUDIO_EXTENSIONS)], key=os.path.getmtime, reverse=True)
            for f_path in files:
                item = QListWidgetItem(os.path.basename(f_path))
                item.setData(Qt.ItemDataRole.UserRole, f_path)
                self.output_log_listwidget.addItem(item)
        except Exception as e:
            print(f"Error updating output log: {e}")

    def play_selected_from_log(self, item: QListWidgetItem):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and os.path.exists(file_path):
            self.current_audio_file = file_path
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.current_file_label.setText(
                f"Playing from log: {os.path.basename(file_path)}")
            self.play_pause_button.setText("Pause")
            self.media_player.play()
            self.play_pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.playhead_slider.setEnabled(True)
        else:
            QMessageBox.warning(self, "File Error",
                                "Could not find or play selected audio file.")
            self.update_output_log()

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            # Store current position before pausing
            self.paused_position = self.media_player.position()
            self.media_player.pause()
            # self.play_pause_button.setText("Play") # Done by handle_playback_state_changed
        else:  # Was Paused or Stopped
            if not self.current_audio_file or not os.path.exists(self.current_audio_file):
                QMessageBox.warning(
                    self, "No Audio", "No audio file loaded to play.")
                return

            # Ensure media source is set
            if self.media_player.source().isEmpty() or \
               self.media_player.source().toLocalFile() != self.current_audio_file:
                self.media_player.setSource(
                    QUrl.fromLocalFile(self.current_audio_file))
                # If source changed or was empty, assume play from start or last paused pos
                if hasattr(self, 'paused_position') and self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                    self.media_player.setPosition(self.paused_position)
                else:  # Play from start if no paused_position or at end of media
                    self.media_player.setPosition(0)
                    self.paused_position = 0  # Reset paused position

            # If resuming from a paused state (and not end of media)
            elif hasattr(self, 'paused_position') and self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                # Only set position if it's significantly different (avoids tiny jumps if already paused)
                # or if it was explicitly stopped and then play is hit again.
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                    self.media_player.setPosition(self.paused_position)

            # If it was fully stopped (and not at end of media), or if no paused_position, play from current slider or 0
            elif self.media_player.playbackState() == QMediaPlayer.PlaybackState.StoppedState and \
                    self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                # If stopped, it might have a valid position already (e.g. user seeked then stopped)
                # If no paused_position exists, it implies fresh play or play after stop
                if not hasattr(self, 'paused_position'):
                    self.media_player.setPosition(0)  # Default to start
                # else: play from current position (which might be 0 if stopped at start)

            self.media_player.play()
            # self.play_pause_button.setText("Pause") # Done by handle_playback_state_changed

    def stop_audio(self):
        self.media_player.stop()
        self.play_pause_button.setText("Play")

    # --- Slider Seeking Logic ---
    def slider_pressed(self): self.is_seeking_audio = True

    def seek_audio_on_move(self, position):
        if self.is_seeking_audio:
            self.media_player.setPosition(position)
            self.current_time_label.setText(self.format_time(position))

    def slider_released(self):
        if self.is_seeking_audio:
            self.is_seeking_audio = False
            self.media_player.setPosition(self.playhead_slider.value())

    def update_slider_position(self, position):
        if not self.is_seeking_audio:
            self.playhead_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def update_duration_info(self, duration):
        if duration > 0:  # Only set range if duration is valid
            self.playhead_slider.setRange(0, duration)
            self.duration_label.setText(self.format_time(duration))
        else:  # Reset if duration is 0 or invalid (e.g. after stop or error)
            self.playhead_slider.setRange(0, 0)
            self.duration_label.setText("00:00")
            self.current_time_label.setText("00:00")

    def format_time(self, ms: int) -> str:  # Your improved version
        if ms < 0:
            ms = 0
        total_seconds_val = ms // 1000
        minutes_val = total_seconds_val // 60
        seconds_remainder_val = total_seconds_val % 60
        return f"{minutes_val:02}:{seconds_remainder_val:02}"

    def handle_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setText("Pause")
        else:
            self.play_pause_button.setText("Play")
        # This logic might conflict with explicit stop, if positionChanged handle slider reset mainly
        # if state==QMediaPlayer.PlaybackState.StoppedState and self.media_player.position()==0:   # Reset slider and time if stopped at start
        #     if self.media_player.mediaStatus()==QMediaPlayer.MediaStatus.EndOfMedia: # End of media reached
        #         self.playhead_slider.setValue(0);self.current_time_label.setText("00:00") # Reset to start

    def handle_media_error(self):
        error_string = self.media_player.errorString()
        if error_string:
            QMessageBox.warning(self, "Media Player Error",
                                f"Error playing audio: {error_string}")
        self.set_status_message("Status: Media player error.")
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.playhead_slider.setEnabled(False)
        self.update_duration_info(0)  # Reset duration display on error

    def closeEvent(self, event):
        global APP_LOG_SINK
        if self.api_server is not None:
            self.api_server.stop()
        if APP_LOG_SINK == self.log_message_signal.emit:
            APP_LOG_SINK = None
        self.save_window_settings()
        self.save_app_settings()
        if hasattr(self, 'model_loader_thread') and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait()
        if hasattr(self, 'audio_generator_thread') and self.audio_generator_thread.isRunning():
            self.audio_generator_thread.stop()  # Request stop
            self.audio_generator_thread.wait()  # Wait for it to finish
        if isinstance(self.model, WORKER_MODEL_TYPES):
            self.model.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui_theme.apply_theme(app)
    window = ChatterboxApp()
    window.show()
    sys.exit(app.exec())

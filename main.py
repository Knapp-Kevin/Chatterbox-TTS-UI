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
    QMenu
)
# QStandardPaths was in your full file, good.
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QTime
from PySide6.QtMultimedia import (
    QMediaPlayer, QAudioOutput, QAudioSource, QAudioFormat, QMediaDevices
)
from PySide6.QtGui import QDesktopServices, QPainter, QColor, QFont, QPalette
import ui_theme
import audio_effects
import documents
import model_registry
import qwen_engine
import gc
import time
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
QWEN_BACKEND = "qwen3"


def languages_for_backend(backend):
    if backend == QWEN_BACKEND:
        return dict(qwen_engine.LANGUAGE_LABELS)
    return get_supported_languages_for_backend(backend)
MODEL_CONFIG_FILENAME = "models.json"
APP_SETTINGS_FILENAME = "app_settings.json"
REFERENCE_RECORDINGS_DIRNAME = "reference_recordings"
RECORDING_SAMPLE_RATE = 48000
MIN_RECORDING_SECONDS = 3
MAX_RECORDING_SECONDS = 30
PREVIEW_MAX_SECTIONS = 2
# Initial generation-speed guesses (seconds per character), refined by measurement.
# Initial speed guesses (seconds of generation per character of text) per engine
# and device; replaced by measurements as each model is used.
DEFAULT_SECONDS_PER_CHAR = {
    ("chatterbox", "cuda"): 0.045, ("chatterbox", "cpu"): 0.35,
    ("qwen3", "cuda"): 0.2, ("qwen3", "cpu"): 2.0,
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
        {
            "repo_id": DEFAULT_MODEL_REPO,
            "label": "Legacy English compatibility",
            "enabled": True,
            "experimental": True,
            "test_text": "Hello. This is a quick test of the legacy English Chatterbox loader.",
            "notes": "Uses the older English-focused loader path for compatibility testing.",
            "backend": BACKEND_LEGACY,
            "language_id": "en"
        }
    ]
}


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
            multilingual_t3_model,
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
        normalized_models.append(
            {
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
    chunk_generated = Signal(int, int)
    section_timed = Signal(int, float)  # characters, seconds

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

            print("Using NLTK for sentence tokenization/combining...")
            final_chunks = documents.split_into_sections(self.original_text)
            if self.preview:
                final_chunks = final_chunks[:PREVIEW_MAX_SECTIONS]

            if not final_chunks:
                self.error_occurred.emit(
                    "Input text empty or resulted in no chunks.")
                return

            total_chunks = len(final_chunks)
            print(f"Processed into {total_chunks} chunks.")
            # (Optional debug print for chunks can go here)

            all_audio_tensors = []
            sr = self.model.sr
            for i, chunk_text in enumerate(final_chunks):
                if self._is_stopped:
                    if all_audio_tensors and not self.preview:
                        # Keep the finished sections of a long render.
                        self.partial_info = (i, total_chunks)
                        print(f"Stopped at section {i + 1}/{total_chunks}; saving {i} finished sections.")
                        break
                    self.error_occurred.emit(
                        f"Generation stopped by user at chunk {i+1}/{total_chunks}.")
                    return
                current_chunk_num = i + 1
                self.chunk_generated.emit(current_chunk_num, total_chunks)
                print(
                    f"\nGenerating chunk {current_chunk_num}/{total_chunks} (seed: {self.actual_seed_used}).")
                section_started = time.monotonic()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    wav_tensor_chunk = self.model.generate(
                        chunk_text,
                        audio_prompt_path=self.audio_prompt_path if self.audio_prompt_path else None,
                        exaggeration=self.exaggeration,
                        temperature=self.temperature,
                        cfg_weight=self.cfg_weight,
                        language_id=self.language_id,
                        repetition_penalty=self.repetition_penalty,
                        min_p=self.min_p,
                        top_p=self.top_p,
                    )
                if wav_tensor_chunk.ndim == 1:
                    wav_tensor_chunk = wav_tensor_chunk.unsqueeze(0)
                all_audio_tensors.append(wav_tensor_chunk.cpu())
                self.section_timed.emit(len(chunk_text), time.monotonic() - section_started)

            if self._is_stopped and self.partial_info is None and len(all_audio_tensors) < total_chunks:
                self.error_occurred.emit("Stopped before final concat.")
                return
            if not all_audio_tensors:
                self.error_occurred.emit("No audio data generated.")
                return

            print("\nConcatenating audio chunks...")
            finishing = self.finishing
            sections = [chunk.reshape(-1).float().numpy() for chunk in all_audio_tensors]
            final_audio = audio_effects.join_sections(sections, sr, finishing.section_pause)
            print(f"Applying finishing touches: {finishing.summary()}")
            final_audio = audio_effects.apply_finishing(final_audio, sr, finishing)
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
        for index, level in enumerate(self.levels):
            # Square-root scaling so normal speech fills a useful part of the height.
            bar_height = max(2.0, min(1.0, level) ** 0.5 * (height - 6))
            color = QColor("#d9534f") if level >= 0.98 else normal
            painter.fillRect(
                int(index * bar_width + 1), int(middle - bar_height / 2),
                max(1, int(bar_width) - 2), int(bar_height), color)
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
        phase_font = QFont(self.phase_label.font())
        phase_font.setPointSize(phase_font.pointSize() + 2)
        phase_font.setBold(True)
        self.phase_label.setFont(phase_font)
        layout.addWidget(self.phase_label)

        self.big_label = QLabel(str(self.COUNTDOWN_SECONDS))
        self.big_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big_font = QFont(self.big_label.font())
        big_font.setPointSize(big_font.pointSize() + 18)
        big_font.setBold(True)
        self.big_label.setFont(big_font)
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
        script_font = QFont(self.script_label.font())
        script_font.setPointSize(script_font.pointSize() + 3)
        self.script_label.setFont(script_font)
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

    ENGINE_FILTERS = (("All engines", "all"), ("Chatterbox", "chatterbox"), ("Qwen3-TTS", "qwen3"))

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
            "test_texts": entry.get("test_texts", {}),
        })
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


class QwenInstallThread(QThread):
    finished_with = Signal(str)

    def run(self):
        try:
            qwen_engine.install(log=print)
            self.finished_with.emit("")
        except Exception as exc:
            self.finished_with.emit(str(exc))


# --- ChatterboxApp ---


class ChatterboxApp(QMainWindow):
    log_message_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox TTS Interface")
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
            self.load_model_button.setEnabled(False)

    PAGE_GENERATE, PAGE_VOICE, PAGE_MODEL, PAGE_LOG = range(4)

    def _make_card(self, title=None):
        card = QFrame()
        card.setObjectName("Card")
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
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PageSubtitle")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
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

        sidebar_panel = QWidget()
        sidebar_panel.setObjectName("SidebarPanel")
        sidebar_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar_layout = QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(180)
        self.sidebar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label in ("Generate", "Voice", "Model", "Log"):
            self.sidebar.addItem(QListWidgetItem(label))
        app_title = QLabel("Chatterbox")
        app_title.setObjectName("AppTitle")
        app_title.setAutoFillBackground(False)
        sidebar_layout.addSpacing(14)
        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(self.sidebar)
        self.pages = QStackedWidget()
        self.sidebar.currentRowChanged.connect(self.pages.setCurrentIndex)
        root_layout.addWidget(sidebar_panel)
        root_layout.addWidget(self.pages, 1)

        # ---------- Generate page ----------
        generate_page, generate_layout = self._make_page(
            "Generate", "Write your text, choose the delivery, then generate.")

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice"))
        self.voice_chip = QLabel("Default voice")
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
            "Enter text to synthesize. Long text is split into segments of "
            f"~{MAX_TEXT_INPUT_LENGTH} characters and stitched together."
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
        status_row_widget = QWidget()
        status_row_widget.setLayout(text_status_row)
        text_status_row.setContentsMargins(0, 0, 0, 0)
        status_row_widget.setFixedHeight(QPushButton("X").sizeHint().height())
        text_card_layout.addWidget(status_row_widget)

        generate_actions_layout = QHBoxLayout()
        self.open_document_button = QPushButton("Open document...")
        self.open_document_button.setToolTip("Load a .txt, .md or .docx file to read aloud.")
        self.open_document_button.clicked.connect(self.open_document)
        generate_actions_layout.addWidget(self.open_document_button)
        self.use_preset_button = QPushButton("Sample text")
        self.use_preset_button.setToolTip("Fill in a short test sentence for the selected language.")
        self.use_preset_button.clicked.connect(self.apply_selected_text_preset)
        generate_actions_layout.addWidget(self.use_preset_button)
        generate_actions_layout.addStretch()
        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip(
            "Generate a short sample with the current settings before rendering everything: "
            "the selected text, or the opening section if nothing is selected.")
        self.preview_button.clicked.connect(lambda: self.start_generation(preview=True))
        self.preview_button.setEnabled(False)
        generate_actions_layout.addWidget(self.preview_button)
        self.generate_button = self._accent(QPushButton("Generate Audio"))
        self.generate_button.clicked.connect(self.handle_generate_stop_toggle)
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumWidth(160)
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
        self.qwen_speaker_combo.setToolTip("Built-in Qwen speaker.")
        self.qwen_speaker_combo.currentIndexChanged.connect(lambda _i: self.refresh_voice_chip())
        self.qwen_speaker_label = QLabel("Speaker")
        qwen_row_layout.addWidget(self.qwen_speaker_label)
        qwen_row_layout.addWidget(self.qwen_speaker_combo)
        self.qwen_instruct_input = QLineEdit()
        self.qwen_instruct_label = QLabel("Style")
        qwen_row_layout.addWidget(self.qwen_instruct_label)
        qwen_row_layout.addWidget(self.qwen_instruct_input, 1)
        self.qwen_transcript_label = QLabel("Clip transcript")
        self.qwen_transcript_input = QLineEdit()
        self.qwen_transcript_input.setPlaceholderText(
            "What is said in the reference clip (optional, improves likeness)")
        self.qwen_transcript_input.setToolTip(
            "With a transcript Qwen clones more closely. Recordings made with Record... fill this "
            "in with the passage you read; edit it if you said something different.")
        self.qwen_transcript_input.editingFinished.connect(self.save_reference_transcript)
        qwen_row_layout.addWidget(self.qwen_transcript_label)
        qwen_row_layout.addWidget(self.qwen_transcript_input, 1)
        self.qwen_watermark_checkbox = QCheckBox("Add AI watermark")
        self.qwen_watermark_checkbox.setChecked(bool(qwen_settings.get("watermark", True)))
        self.qwen_watermark_checkbox.setToolTip(
            "Qwen doesn't watermark its audio. When ticked, the same inaudible Perth watermark "
            "Chatterbox uses is added, so output from every engine is marked the same way.")
        self.qwen_settings = qwen_settings
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

        self.speed_slider = self._create_slider(
            *audio_effects.SPEED_RANGE, 0.05, 1.0, "{:.2f}x")
        add_finishing(0, 0, "Speed", self.speed_slider,
                      "Speaking speed without changing the pitch. 1.00x is unchanged.")
        self.pitch_slider = self._create_slider(
            *audio_effects.PITCH_RANGE, 0.5, 0.0, "{:+.1f} st")
        add_finishing(0, 2, "Pitch", self.pitch_slider,
                      "Raise or lower the voice in semitones while keeping its natural "
                      "character. Small changes (1-2 st) sound most natural.")
        self.pause_slider = self._create_slider(
            *audio_effects.PAUSE_RANGE, 0.1, 0.0, "{:.1f} s")
        add_finishing(1, 0, "Section pause", self.pause_slider,
                      f"Long text is generated in sections of about {MAX_TEXT_INPUT_LENGTH} "
                      "characters. This adds a pause where the sections are joined.")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(list(audio_effects.OUTPUT_FORMATS))
        add_finishing(1, 2, "Save as", self.output_format_combo,
                      "WAV is uncompressed, FLAC is lossless and smaller, MP3 is "
                      "smallest and plays everywhere.")
        finishing_checks = QHBoxLayout()
        self.even_volume_checkbox = QCheckBox("Even out volume")
        self.even_volume_checkbox.setToolTip(
            "Bring every result to a consistent, comfortable loudness without clipping.")
        self.trim_silence_checkbox = QCheckBox("Trim silence at start and end")
        self.trim_silence_checkbox.setToolTip(
            "Remove dead air before the first word and after the last.")
        finishing_checks.setSpacing(24)
        finishing_checks.addWidget(self.even_volume_checkbox)
        finishing_checks.addWidget(self.trim_silence_checkbox)
        finishing_checks.addStretch(1)
        reset_finishing_button = QPushButton("Reset")
        reset_finishing_button.setToolTip("Restore the default finishing settings.")
        reset_finishing_button.clicked.connect(
            lambda: self.apply_finishing_settings(audio_effects.FinishingSettings()))
        finishing_checks.addWidget(reset_finishing_button)
        finishing_grid.addLayout(finishing_checks, 2, 0, 1, 4)
        delivery_layout.addWidget(self.finishing_panel)

        for slider in (self.speed_slider, self.pitch_slider, self.pause_slider):
            slider.slider.valueChanged.connect(self.update_finishing_summary)
        self.output_format_combo.currentTextChanged.connect(self.update_finishing_summary)
        self.even_volume_checkbox.toggled.connect(self.update_finishing_summary)
        self.trim_silence_checkbox.toggled.connect(self.update_finishing_summary)
        self.apply_finishing_settings(
            audio_effects.FinishingSettings.from_dict(self.app_settings.get("finishing")))
        self.set_finishing_expanded(bool(self.app_settings.get("finishing_expanded", False)))
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
            "Voice", "Use the default voice, record your own, or pick an audio file.")

        current_card, current_layout = self._make_card("Current voice")
        current_row = QHBoxLayout()
        self.ref_audio_path_label = QLabel("None selected.")
        self.ref_audio_path_label.setWordWrap(False)
        self.ref_audio_path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.ref_audio_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        current_row.addWidget(self.ref_audio_path_label, 1)
        self.preview_reference_button = QPushButton("Preview")
        self.preview_reference_button.clicked.connect(self.toggle_reference_preview)
        current_row.addWidget(self.preview_reference_button)
        self.clear_reference_button = QPushButton("Use default voice")
        self.clear_reference_button.clicked.connect(self.clear_reference_audio)
        current_row.addWidget(self.clear_reference_button)
        current_layout.addLayout(current_row)
        voice_layout.addWidget(current_card)

        record_card, record_layout = self._make_card("Record a new reference")
        record_hint = QLabel(
            "Read about 15 seconds in a quiet room; the first few seconds matter most.")
        record_hint.setObjectName("Muted")
        record_layout.addWidget(record_hint)
        record_row = QHBoxLayout()
        record_row.addWidget(QLabel("Microphone"))
        self.mic_combo = QComboBox()
        self.mic_combo.setToolTip("Microphone used for recording a reference clip.")
        self.mic_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        record_row.addWidget(self.mic_combo, 1)
        self.record_button = self._accent(QPushButton("Record..."))
        self.record_button.setToolTip(
            "Record a reference clip from the selected microphone "
            f"({MIN_RECORDING_SECONDS}-{MAX_RECORDING_SECONDS} s).")
        self.record_button.clicked.connect(self.open_recording_dialog)
        record_row.addWidget(self.record_button)
        record_layout.addLayout(record_row)
        self.media_devices.audioInputsChanged.connect(self.populate_microphones)
        self.populate_microphones()
        voice_layout.addWidget(record_card)

        saved_card, saved_layout = self._make_card("Saved recordings and files")
        self.recordings_listwidget = QListWidget()
        self.recordings_listwidget.setToolTip("Double-click a recording to use it.")
        self.recordings_listwidget.itemDoubleClicked.connect(
            lambda _item: self.use_selected_recording())
        self.recordings_listwidget.currentRowChanged.connect(
            lambda _row: self.update_recording_buttons())
        self.recordings_listwidget.setMinimumHeight(90)
        self.recordings_listwidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        saved_layout.addWidget(self.recordings_listwidget, 1)
        saved_actions = QHBoxLayout()
        self.use_recording_button = QPushButton("Use selected")
        self.use_recording_button.clicked.connect(self.use_selected_recording)
        saved_actions.addWidget(self.use_recording_button)
        self.preview_recording_button = QPushButton("Preview selected")
        self.preview_recording_button.clicked.connect(self.preview_selected_recording)
        saved_actions.addWidget(self.preview_recording_button)
        saved_actions.addStretch(1)
        open_recordings_button = QPushButton("Open folder")
        open_recordings_button.clicked.connect(self.open_recordings_folder)
        saved_actions.addWidget(open_recordings_button)
        browse_ref_button = QPushButton("Browse for a file...")
        browse_ref_button.clicked.connect(self.browse_reference_audio)
        saved_actions.addWidget(browse_ref_button)
        saved_layout.addLayout(saved_actions)
        voice_layout.addWidget(saved_card, 1)
        self.pages.addWidget(voice_page)

        # ---------- Model page ----------
        model_page, model_layout = self._make_page(
            "Model", "Models download once, then load from the local cache.")
        models_card, models_layout = self._make_card("Models")
        self.models_list = QListWidget()
        self.models_list.setMinimumHeight(96)
        self.models_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.models_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.models_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.models_list.currentRowChanged.connect(lambda _row: self.update_model_details())
        self.models_list.itemDoubleClicked.connect(lambda _item: self.load_selected_list_model())
        models_layout.addWidget(self.models_list, 1)

        details_header = QHBoxLayout()
        self.model_name_label = QLabel()
        self.model_name_label.setObjectName("CardTitle")
        self.model_name_label.setTextFormat(Qt.TextFormat.PlainText)
        details_header.addWidget(self.model_name_label)
        self.model_active_chip = QLabel("Active")
        self.model_active_chip.setObjectName("VoiceChip")
        details_header.addWidget(self.model_active_chip)
        details_header.addStretch(1)
        find_models_button = self._link(QPushButton("Find models..."))
        find_models_button.setToolTip("Search Hugging Face for models this app can load.")
        find_models_button.clicked.connect(self.find_models)
        details_header.addWidget(find_models_button)
        add_model_button = self._link(QPushButton("+ Add model..."))
        add_model_button.clicked.connect(self.add_model)
        details_header.addWidget(add_model_button)
        models_layout.addLayout(details_header)
        details_grid = QGridLayout()
        details_grid.setHorizontalSpacing(14)
        details_grid.setColumnStretch(1, 1)
        self.model_engine_label = QLabel()
        self.model_repo_label = QLabel()
        self.model_repo_label.setOpenExternalLinks(True)
        self.model_status_label = QLabel()
        # Long notes are clipped (full text in the tooltip) instead of widening the window.
        for label in (self.model_engine_label, self.model_repo_label, self.model_status_label):
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for row, (title, widget) in enumerate((
                ("Engine", self.model_engine_label),
                ("Repo", self.model_repo_label),
                ("Status", self.model_status_label))):
            caption = QLabel(title)
            caption.setObjectName("Muted")
            details_grid.addWidget(caption, row, 0)
            details_grid.addWidget(widget, row, 1)
        models_layout.addLayout(details_grid)
        model_actions = QHBoxLayout()
        self.load_model_button = self._accent(QPushButton("Load this model"))
        self.load_model_button.clicked.connect(self.load_selected_list_model)
        model_actions.addWidget(self.load_model_button)
        model_actions.addStretch(1)
        self.edit_model_button = QPushButton("Edit...")
        self.edit_model_button.clicked.connect(self.edit_model)
        self.duplicate_model_button = QPushButton("Duplicate")
        self.duplicate_model_button.setToolTip("Copy this entry, e.g. to try other weights or a default language.")
        self.duplicate_model_button.clicked.connect(self.duplicate_model)
        self.remove_model_button = QPushButton("Remove")
        self.remove_model_button.setToolTip("Remove from the list. Downloaded files stay in the cache.")
        self.remove_model_button.clicked.connect(self.remove_model)
        for button in (self.edit_model_button, self.duplicate_model_button, self.remove_model_button):
            model_actions.addWidget(button)
        models_layout.addLayout(model_actions)
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

        # ---------- Log page ----------
        log_page, log_layout = self._make_page(
            "Log", "Technical output from model loading and generation.")
        self.console_log_view = QPlainTextEdit()
        self.console_log_view.setReadOnly(True)
        self.console_log_view.setMaximumBlockCount(1000)
        self.console_log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_font = QFont("Consolas")
        log_font.setStyleHint(QFont.StyleHint.Monospace)
        self.console_log_view.setFont(log_font)
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
        selected = self.text_input.textCursor().selectedText().replace("\u2029", "\n").strip()
        if selected:
            return selected
        sections = documents.split_into_sections(self.text_input.toPlainText())
        return sections[0] if sections else ""

    def start_generation(self, preview=False):
        if self.is_generating:
            return
        if self.model is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load the model first.")
            return
        text = self.preview_text() if preview else self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Error", "Please enter some text to synthesize.")
            return
        qwen_problem = self.prepare_qwen_generation()
        if qwen_problem:
            QMessageBox.information(self, "Qwen Voice", qwen_problem)
            return

        self.is_generating = True
        self.generation_is_preview = preview
        self.generation_char_count = len(text)
        self.generation_started_at = time.monotonic()
        self.keep_take_button.setVisible(False)
        self.generate_button.setText("Stop")
        self.generate_button.setEnabled(True)
        self.preview_button.setEnabled(False)
        self.open_document_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
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
        self.audio_generator_thread.generation_complete.connect(self.on_generation_complete)
        self.audio_generator_thread.error_occurred.connect(self.on_generation_error)
        self.audio_generator_thread.chunk_generated.connect(self.on_chunk_generated_progress)
        self.audio_generator_thread.section_timed.connect(self.on_section_timed)
        self.audio_generator_thread.finished.connect(self.on_generation_thread_finished)
        self.audio_generator_thread.start()

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
        self.text_input.setPlainText(text)
        self.current_document_name = documents.safe_file_stem(path)
        self.document_label.setText(os.path.basename(path))
        self.update_text_stats()
        self.set_status_message(f"Status: Loaded {os.path.basename(path)}. Try Preview before generating.")

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
        return f"{self.speed_device()}|{entry.get('repo_id')}|{entry.get('backend')}|{variant}"

    def seconds_per_char_for(self, entry):
        """(seconds per character, measured?) for an entry on the current device."""
        device = self.speed_device()
        measured = self.app_settings.get("speed_by_model", {}).get(self.speed_key(entry))
        if measured:
            return measured, True
        engine = "qwen3" if entry.get("backend") == QWEN_BACKEND else "chatterbox"
        if engine == "chatterbox":
            legacy = self.app_settings.get("seconds_per_char", {}).get(device)  # older single rate
            if legacy:
                return legacy, False
        return DEFAULT_SECONDS_PER_CHAR.get((engine, device), 0.35), False

    def loaded_entry(self):
        return next((e for e in self.model_entries if self.entry_key(e) == self.loaded_entry_key()), None)

    def model_estimates(self, characters):
        """[(entry, seconds, measured, active)] for every model in the switcher, fastest first."""
        rows = []
        active_key = self.loaded_entry_key() if self.model is not None else None
        for entry in self.get_visible_model_entries():
            rate, measured = self.seconds_per_char_for(entry)
            rows.append((entry, characters * rate, measured, self.entry_key(entry) == active_key))
        return sorted(rows, key=lambda row: row[1])

    def update_text_stats(self):
        # Always current, including while a preview or render runs; a running
        # render keeps using the text it started with.
        text = self.text_input.toPlainText().strip()
        if not text:
            self.estimate_button.setVisible(False)
            self.text_stats_label.setText("Type or paste text, or open a document.")
            return
        sections = len(documents.split_into_sections(text))
        entry = self.loaded_entry() or self.get_selected_model_entry()
        rate, measured = self.seconds_per_char_for(entry)
        self.estimate_button.setText(f"About {self.format_duration(len(text) * rate)} \u25be")
        self.estimate_button.setVisible(True)
        self.text_stats_label.setText(
            f"{sections} section{'s' if sections != 1 else ''} \u00b7 {len(text):,} characters")
        for index in range(self.model_repo_combo.count()):
            visible = self.get_visible_model_entries()
            if index < len(visible):
                item_rate, item_measured = self.seconds_per_char_for(visible[index])
                self.model_repo_combo.setItemData(
                    index,
                    f"About {self.format_duration(len(text) * item_rate)} for the current text"
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
        for entry, seconds, measured, active in self.model_estimates(len(text)):
            marker = "\u25cf " if active else "    "
            label = f"{marker}{entry['label']}  \u2014  about {self.format_duration(seconds)}"
            label += "" if measured else "  (estimate)"
            if not active:
                label += "  + load"
            action = menu.addAction(label)
            action.setEnabled(not active and not self.is_generating and not getattr(self, "model_is_loading", False))
            action.triggered.connect(lambda _checked=False, e=entry: self.switch_to_entry(e))
        menu.addSeparator()
        note = menu.addAction("Estimates become measurements once a model has generated a few sections.")
        note.setEnabled(False)
        menu.exec(self.estimate_button.mapToGlobal(self.estimate_button.rect().bottomLeft()))

    def switch_to_entry(self, entry):
        index = self.model_repo_combo.findText(entry["label"])
        if index >= 0:
            self.model_repo_combo.setCurrentIndex(index)

    def on_section_timed(self, characters, seconds):
        # The first section after a load includes warm-up, so it isn't a fair sample.
        if not self.model_is_warm:
            self.model_is_warm = True
            return
        entry = self.loaded_entry()
        if entry is None or characters < 20:
            return
        measured = seconds / characters
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
            self.set_reference_audio(file_path)
            self.last_reference_audio_dir = os.path.dirname(file_path)

    # --- Engine-specific controls ---

    def active_qwen_model(self):
        return self.model if isinstance(self.model, qwen_engine.QwenModel) else None

    def update_engine_controls(self):
        qwen = self.active_qwen_model()
        self.qwen_row.setVisible(qwen is not None)
        self.qwen_watermark_checkbox.setVisible(qwen is not None)
        for widget in (self.exaggeration_label, self.exaggeration_slider, self.cfg_label, self.cfg_slider):
            widget.setVisible(qwen is None)
        if qwen is not None:
            mode = qwen.mode
            self.qwen_speaker_combo.blockSignals(True)
            self.qwen_speaker_combo.clear()
            for speaker in qwen.speakers:
                self.qwen_speaker_combo.addItem(speaker.replace("_", " ").title(), speaker)
            saved = self.qwen_speaker_combo.findData(self.qwen_settings.get("speaker"))
            self.qwen_speaker_combo.setCurrentIndex(max(0, saved))
            self.qwen_speaker_combo.blockSignals(False)
            if mode == "voice_design":
                self.qwen_instruct_label.setText("Voice description")
                self.qwen_instruct_input.setPlaceholderText(
                    "e.g. a calm, low male voice with a slight rasp, unhurried and warm")
                self.qwen_instruct_input.setText(self.qwen_settings.get("description", ""))
            else:
                self.qwen_instruct_label.setText("Style")
                self.qwen_instruct_input.setPlaceholderText(
                    "Optional, e.g. excited and upbeat, or whisper softly")
                self.qwen_instruct_input.setText(self.qwen_settings.get("style", ""))
            for widget in (self.qwen_speaker_label, self.qwen_speaker_combo):
                widget.setVisible(mode == "custom_voice")
            for widget in (self.qwen_instruct_label, self.qwen_instruct_input):
                widget.setVisible(mode in ("custom_voice", "voice_design"))
            for widget in (self.qwen_transcript_label, self.qwen_transcript_input):
                widget.setVisible(mode == "base")
        self.refresh_voice_chip()
        if self.isVisible():
            self.update_minimum_size()

    def refresh_voice_chip(self):
        qwen = self.active_qwen_model()
        reference = self.ref_audio_path_label.toolTip()
        if qwen is not None and qwen.mode == "custom_voice":
            text = f"Preset: {self.qwen_speaker_combo.currentText() or 'speaker'}"
            tip = "A built-in Qwen speaker. Reference clips aren't used by this model."
        elif qwen is not None and qwen.mode == "voice_design":
            text, tip = "Designed voice", "Described in the Delivery card below."
        elif reference:
            text, tip = os.path.basename(reference), reference
        else:
            text = "Default voice"
            tip = "The model's built-in voice. Pick a reference clip on the Voice page to clone a voice."
            if qwen is not None:
                text, tip = "No clip selected", "Qwen voice cloning needs a reference clip from the Voice page."
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
            return "Choose a reference clip on the Voice page; Qwen cloning needs one."
        qwen.speaker = self.qwen_speaker_combo.currentData() or qwen.speaker
        qwen.instruct = instruct
        qwen.ref_text = self.qwen_transcript_input.text().strip() if qwen.mode == "base" else ""
        qwen.watermark = self.qwen_watermark_checkbox.isChecked()
        key = "description" if qwen.mode == "voice_design" else "style"
        self.qwen_settings.update({key: instruct, "speaker": qwen.speaker, "watermark": qwen.watermark})
        self.app_settings["qwen"] = self.qwen_settings
        return None

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
            section_pause=round(self.pause_slider.get_value(), 1),
            even_volume=self.even_volume_checkbox.isChecked(),
            trim_silence=self.trim_silence_checkbox.isChecked(),
            output_format=self.output_format_combo.currentText(),
        )

    def apply_finishing_settings(self, settings):
        self.speed_slider.set_value(settings.speed)
        self.pitch_slider.set_value(settings.pitch_semitones)
        self.pause_slider.set_value(settings.section_pause)
        self.even_volume_checkbox.setChecked(settings.even_volume)
        self.trim_silence_checkbox.setChecked(settings.trim_silence)
        self.output_format_combo.setCurrentText(settings.output_format)
        self.update_finishing_summary()

    def update_finishing_summary(self, *_args):
        self.finishing_summary_label.setText(self.current_finishing_settings().summary())

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
            name = os.path.basename(path)
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

    def clear_reference_audio(self):
        self.stop_reference_preview()
        self.set_reference_audio(None)
        self.set_status_message("Status: Using the default voice.")

    def refresh_recordings_list(self):
        self.recordings_listwidget.clear()
        paths = []
        if os.path.isdir(self.recordings_directory):
            paths = [os.path.join(self.recordings_directory, name)
                     for name in os.listdir(self.recordings_directory)
                     if name.lower().endswith(".wav")]
        for path in sorted(paths, key=os.path.getmtime, reverse=True):
            try:
                with wave.open(path, "rb") as wav_file:
                    seconds = wav_file.getnframes() / float(wav_file.getframerate())
                length = f"{seconds:.0f} s"
            except Exception:
                length = "unreadable"
            when = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%b %d, %I:%M %p")
            item = QListWidgetItem(f"{os.path.basename(path)}    {length}  ·  {when}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.recordings_listwidget.addItem(item)
        if not paths:
            placeholder = QListWidgetItem("No recordings yet. Use Record... above to make one.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recordings_listwidget.addItem(placeholder)
        self.update_recording_buttons()

    def selected_recording_path(self):
        item = self.recordings_listwidget.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def update_recording_buttons(self):
        has_selection = bool(self.selected_recording_path())
        self.use_recording_button.setEnabled(has_selection)
        self.preview_recording_button.setEnabled(has_selection)

    def use_selected_recording(self):
        path = self.selected_recording_path()
        if path:
            self.set_reference_audio(path)
            self.set_status_message(f"Status: Voice set to {os.path.basename(path)}.")

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
            self.preview_recording_button.setText("Preview selected")
            self.preview_button_playing = None

    def toggle_reference_preview(self):
        if self.preview_button_playing is self.preview_reference_button:
            self.stop_reference_preview()
            return
        path = self.ref_audio_path_label.toolTip()
        if path:
            self._start_reference_preview(path, self.preview_reference_button)

    def preview_selected_recording(self):
        if self.preview_button_playing is self.preview_recording_button:
            self.stop_reference_preview()
            return
        path = self.selected_recording_path()
        if path:
            self._start_reference_preview(path, self.preview_recording_button)

    def open_recordings_folder(self):
        os.makedirs(self.recordings_directory, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.recordings_directory))

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
        return (entry.get("repo_id"), entry.get("backend"), entry.get("multilingual_t3_model") or "")

    def loaded_entry_key(self):
        backend = self.current_model_backend
        weights = self.current_multilingual_t3_model if backend == BACKEND_MULTILINGUAL else ""
        return (self.current_model_repo, backend, weights or "")

    def refresh_models_page(self, select_entry=None):
        if not hasattr(self, "models_list"):
            return
        previous = select_entry or self.selected_list_entry()
        sizes = model_registry.cached_repo_sizes()
        self.models_list.blockSignals(True)
        self.models_list.clear()
        select_row = 0
        for row, entry in enumerate(self.model_entries):
            engine = model_registry.engine_for(entry)
            languages = engine.languages_summary
            engine_text = model_registry.engine_label(entry)
            if entry.get("backend") == QWEN_BACKEND and not qwen_engine.is_installed():
                status = "engine not installed"
            elif model_registry.is_downloaded(entry):
                status = f"downloaded ({model_registry.format_size(sizes.get(entry['repo_id'], 0))} repo cache)"
            else:
                status = "not downloaded"
            active = self.entry_key(entry) == self.loaded_entry_key() and self.model is not None
            hidden = "" if entry.get("enabled", True) else " \u00b7 hidden"
            marker = "\u25cf " if active else "    "
            item = QListWidgetItem(f"{marker}{entry['label']}\n      {engine_text} \u00b7 {languages} \u00b7 {status}{hidden}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setToolTip(f"{entry['repo_id']}\n{engine_text} \u00b7 {languages} \u00b7 {status}"
                            + ("\nHidden from the model switcher" if hidden else ""))
            entry["_status"] = status
            if active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.models_list.addItem(item)
            if previous is not None and self.entry_key(entry) == self.entry_key(previous) \
                    and entry.get("label") == previous.get("label"):
                select_row = row
        self.models_list.setCurrentRow(select_row)
        self.models_list.blockSignals(False)
        self.update_model_details()

    def selected_list_entry(self):
        if not hasattr(self, "models_list"):
            return None
        item = self.models_list.currentItem()
        if item is None:
            return None
        row = item.data(Qt.ItemDataRole.UserRole)
        return self.model_entries[row] if isinstance(row, int) and row < len(self.model_entries) else None

    def update_model_details(self):
        entry = self.selected_list_entry()
        has_entry = entry is not None
        for widget in (self.edit_model_button, self.duplicate_model_button, self.remove_model_button):
            widget.setEnabled(has_entry)
        if not has_entry:
            self.model_name_label.setText("No models")
            self.model_active_chip.setVisible(False)
            return
        engine = model_registry.engine_for(entry)
        self.model_name_label.setText(entry["label"])
        active = self.entry_key(entry) == self.loaded_entry_key() and self.model is not None
        self.model_active_chip.setVisible(active)
        engine_text = model_registry.engine_label(entry)
        if engine.uses_weights_version:
            engine_text += f" \u00b7 weights {model_registry.weights_file(entry).split('_')[-1].split('.')[0].upper()}"
        self.model_engine_label.setText(engine_text)
        self.model_engine_label.setToolTip(engine.description)
        repo = entry["repo_id"]
        self.model_repo_label.setText(f'<a href="https://huggingface.co/{repo}">{repo}</a>')
        status = entry.get("_status", "")
        if entry.get("notes"):
            status += f" \u00b7 {entry['notes']}"
        self.model_status_label.setText(status[:1].upper() + status[1:])
        self.model_status_label.setToolTip(status)
        busy = getattr(self, "model_is_loading", False) or self.is_generating
        self.load_model_button.setEnabled(not busy and not active)
        needs_install = entry.get("backend") == QWEN_BACKEND and not qwen_engine.is_installed()
        self.load_model_button.setText(
            "Loaded" if active else "Install Qwen engine..." if needs_install else "Load this model")
        is_default = repo == DEFAULT_MODEL_REPO and entry.get("backend") == BACKEND_MULTILINGUAL \
            and sum(1 for e in self.model_entries if self.entry_key(e) == self.entry_key(entry)) == 1
        self.remove_model_button.setEnabled(not is_default and not active)
        self.remove_model_button.setToolTip(
            "The official model can't be removed; it is the fallback." if is_default else
            "Unload it first by switching to another model." if active else
            "Remove from the list. Downloaded files stay in the cache.")

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

    def find_models(self):
        existing = {e.get("repo_id") for e in self.model_entries}
        finder = FindModelsDialog(self.app_settings.get("hf_token"), existing, self)
        if not dialog_accepted(finder.exec()) or not finder.selected_repo:
            return
        new_entry = self._edit_entry_dialog({"repo_id": finder.selected_repo})
        if new_entry:
            self.persist_model_entries(self.model_entries + [new_entry], new_entry)

    def add_model(self):
        new_entry = self._edit_entry_dialog(None)
        if new_entry:
            self.persist_model_entries(self.model_entries + [new_entry], new_entry)

    def edit_model(self):
        entry = self.selected_list_entry()
        if entry is None:
            return
        updated = self._edit_entry_dialog(entry, replacing=entry)
        if updated:
            entries = [updated if e is entry else e for e in self.model_entries]
            self.persist_model_entries(entries, updated)

    def duplicate_model(self):
        entry = self.selected_list_entry()
        if entry is None:
            return
        copy = dict(entry)
        base, n = f"{entry['label']} copy", 2
        copy["label"] = base
        while any(e.get("label") == copy["label"] for e in self.model_entries):
            copy["label"] = f"{base} {n}"
            n += 1
        updated = self._edit_entry_dialog(copy)
        if updated:
            self.persist_model_entries(self.model_entries + [updated], updated)

    def remove_model(self):
        entry = self.selected_list_entry()
        if entry is None:
            return
        answer = QMessageBox.question(
            self, "Remove Model",
            f"Remove '{entry['label']}' from the model list?\n\nDownloaded files stay in the "
            "Hugging Face cache, so adding it back later won't download again.")
        if answer == QMessageBox.StandardButton.Yes:
            self.persist_model_entries([e for e in self.model_entries if e is not entry])

    def load_selected_list_model(self):
        entry = self.selected_list_entry()
        if entry is None:
            return
        if entry.get("backend") == QWEN_BACKEND and not qwen_engine.is_installed():
            self.install_qwen_engine()
            return
        index = self.model_repo_combo.findText(entry["label"])
        if index >= 0 and index != self.model_repo_combo.currentIndex():
            self.model_repo_combo.setCurrentIndex(index)  # loads via the switcher
        elif self.entry_key(entry) != self.loaded_entry_key() or self.model is None:
            self.load_model(entry)

    def install_qwen_engine(self):
        answer = QMessageBox.question(
            self, "Install Qwen Engine",
            "Qwen3-TTS runs in its own Python environment (engines/qwen) because it needs "
            "different library versions than Chatterbox.\n\nInstalling downloads about 3 GB of "
            "PyTorch and Qwen packages (less if PyTorch is already cached). Model weights "
            "download the first time each Qwen model is loaded.\n\nInstall now?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.set_status_message("Status: Installing the Qwen engine. Progress is on the Log page.")
        self.load_model_button.setEnabled(False)
        self.qwen_install_thread = QwenInstallThread()
        self.qwen_install_thread.finished_with.connect(self.on_qwen_install_finished)
        self.qwen_install_thread.start()

    def on_qwen_install_finished(self, error):
        if error:
            self.set_status_message("Status: Qwen engine install failed. See the Log page.")
            QMessageBox.warning(self, "Qwen Engine", f"The install failed:\n{error}")
        else:
            self.set_status_message("Status: Qwen engine installed. Loading the model...")
            self.refresh_models_page()
            self.load_selected_list_model()
        self.update_model_details()

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
        for index, entry in enumerate(visible_entries):
            self.model_repo_combo.addItem(entry["label"], index)
            self.model_repo_combo.setItemData(
                index, f"{entry['repo_id']} \u00b7 {model_registry.engine_for(entry).label}",
                Qt.ItemDataRole.ToolTipRole)

        selected_index = -1
        if selected_label:
            selected_index = self.model_repo_combo.findText(selected_label)
        if selected_index < 0:
            for index, entry in enumerate(visible_entries):
                if self.entry_key(entry) == self.loaded_entry_key():
                    selected_index = index
                    break

        if selected_index < 0:
            for index, entry in enumerate(visible_entries):
                if entry["repo_id"] == DEFAULT_MODEL_REPO and entry.get("backend") == BACKEND_MULTILINGUAL:
                    selected_index = index
                    break

        if selected_index < 0 and visible_entries:
            selected_index = 0

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

        is_multilingual = selected_entry.get("backend") in (BACKEND_MULTILINGUAL, QWEN_BACKEND)
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
        if isinstance(old_model, qwen_engine.QwenModel):
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
        self.model_loader_thread = ModelLoaderThread(
            selected_repo,
            selected_backend,
            selected_multilingual_t3_model,
        )
        self.cuda_runtime_issue = self.model_loader_thread.cuda_probe_error
        self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loader_thread.error_occurred.connect(
            self.on_model_load_error)
        self.model_loader_thread.start()

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
        if not self.keep_take_button.isVisible():
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

    def on_chunk_generated_progress(self, current_chunk, total_chunks):
        if not self.is_generating:
            return
        done = current_chunk - 1
        elapsed = time.monotonic() - self.generation_started_at
        self.generation_progress.setMaximum(total_chunks)
        self.generation_progress.setValue(done)
        activity = f"{current_chunk}/{total_chunks}"
        if self.generation_is_preview:
            activity = f"Preview {activity}"
        if done:
            remaining = elapsed / done * (total_chunks - done)
            activity += f" \u00b7 {self.format_clock(remaining)} left"
        else:
            activity += f" \u00b7 {self.format_clock(elapsed)}"
        self.activity_label.setText(activity)
        self.set_status_message(f"Status: Generating section {current_chunk}/{total_chunks}...")

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
            self.set_status_message(f"Status: Preview ready{total_generation_time_str}.")
        elif thread.partial_info:
            done, total = thread.partial_info
            self.set_status_message(
                f"Status: Stopped. Saved {done} of {total} sections: {os.path.basename(output_path)}")
        else:
            self.set_status_message(
                f"Status: Full audio generated: {os.path.basename(output_path)}{total_generation_time_str}")

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
        if isinstance(self.model, qwen_engine.QwenModel):
            self.model.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui_theme.apply_theme(app)
    window = ChatterboxApp()
    window.show()
    sys.exit(app.exec())

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
    QCheckBox, QComboBox, QProgressBar, QSizePolicy, QFrame, QStackedWidget,
    QScrollArea
)
# QStandardPaths was in your full file, good.
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QTime
from PySide6.QtMultimedia import (
    QMediaPlayer, QAudioOutput, QAudioSource, QAudioFormat, QMediaDevices
)
from PySide6.QtGui import QDesktopServices, QPainter, QColor, QFont, QPalette
import ui_theme
import audio_effects
from collections import deque
import time
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

MAX_TEXT_INPUT_LENGTH = 280
EFFECTIVE_MAX_CHUNK_LENGTH = MAX_TEXT_INPUT_LENGTH - 20
MODEL_CONFIG_FILENAME = "models.json"
APP_SETTINGS_FILENAME = "app_settings.json"
REFERENCE_RECORDINGS_DIRNAME = "reference_recordings"
RECORDING_SAMPLE_RATE = 48000
MIN_RECORDING_SECONDS = 3
MAX_RECORDING_SECONDS = 30
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
    ):
        super().__init__()
        self.finishing = finishing or audio_effects.FinishingSettings()
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

    def _chunk_long_sentence(self, sentence, max_len):
        sub_chunks = []
        current_pos = 0
        sentence_len = len(sentence)
        while current_pos < sentence_len:
            end_pos = min(current_pos + max_len, sentence_len)
            if end_pos == sentence_len:
                sub_chunks.append(sentence[current_pos:end_pos].strip())
                current_pos = end_pos
            else:
                last_space_idx = sentence.rfind(' ', current_pos, end_pos)
                if last_space_idx != -1 and last_space_idx > current_pos:
                    sub_chunks.append(
                        sentence[current_pos:last_space_idx].strip())
                    current_pos = last_space_idx + 1
                else:
                    sub_chunks.append(sentence[current_pos:end_pos].strip())
                    current_pos = end_pos
        return [sc for sc in sub_chunks if sc]

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

            final_chunks = []
            text_to_process = self.original_text.strip()
            print("Using NLTK for sentence tokenization/combining...")
            sentences = nltk.sent_tokenize(text_to_process)
            current_chunk_sents = []
            current_chunk_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > MAX_TEXT_INPUT_LENGTH:
                    if current_chunk_sents:
                        final_chunks.append(" ".join(current_chunk_sents))
                    current_chunk_sents = []
                    current_chunk_len = 0
                    print(f"Sentence too long ({len(sentence)}), sub-chunking.")
                    final_chunks.extend(self._chunk_long_sentence(
                        sentence, EFFECTIVE_MAX_CHUNK_LENGTH))
                    continue
                potential_len = current_chunk_len + \
                    (1 if current_chunk_sents else 0) + len(sentence)
                if potential_len <= MAX_TEXT_INPUT_LENGTH:
                    current_chunk_sents.append(sentence)
                    current_chunk_len = potential_len
                else:
                    if current_chunk_sents:
                        final_chunks.append(" ".join(current_chunk_sents))
                    current_chunk_sents = [sentence]
                    current_chunk_len = len(sentence)
            if current_chunk_sents:
                final_chunks.append(" ".join(current_chunk_sents))
            final_chunks = [c.strip() for c in final_chunks if c.strip()]

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
                    self.error_occurred.emit(
                        f"Generation stopped by user at chunk {i+1}/{total_chunks}.")
                    return
                current_chunk_num = i + 1
                self.chunk_generated.emit(current_chunk_num, total_chunks)
                print(
                    f"\nGenerating chunk {current_chunk_num}/{total_chunks} (seed: {self.actual_seed_used}).")
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

            if self._is_stopped:
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
            output_base = os.path.join(
                self.output_dir,
                f"chatterbox_{timestamp}_seed{self.actual_seed_used}_full_stitched")
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


# --- ChatterboxApp ---


class ChatterboxApp(QMainWindow):
    log_message_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox TTS Interface")
        self.setGeometry(100, 100, 1000, 820)
        self.model = None
        self.device_used = "cpu"
        self.current_model_repo = DEFAULT_MODEL_REPO
        self.current_model_backend = BACKEND_MULTILINGUAL
        self.current_multilingual_t3_model = DEFAULT_MULTILINGUAL_T3_MODEL
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
        self.repetition_penalty = 1.2
        self.min_p = 0.05
        self.top_p = 1.0

        self.log_message_signal.connect(self.append_console_log)
        self._init_ui()
        self.restore_window_settings()
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
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return page, layout

    @staticmethod
    def _scrollable(page):
        # Pages scroll rather than clip when the window is short.
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

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
        self.setMinimumSize(940, 660)
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
            "Generate", "Write the text, pick a language and delivery, then generate.")

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
        voice_row.addWidget(QLabel("Language"))
        self.language_combo = QComboBox()
        self.language_combo.setMinimumWidth(200)
        voice_row.addWidget(self.language_combo)
        generate_layout.addLayout(voice_row)

        text_card, text_card_layout = self._make_card("Text")
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Enter text to synthesize. Long text is split into segments of "
            f"~{MAX_TEXT_INPUT_LENGTH} characters and stitched together."
        )
        self.text_input.setMinimumHeight(90)
        # Fill the leftover height instead of forcing the page to scroll.
        self.text_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        text_card_layout.addWidget(self.text_input, 1)

        generate_actions_layout = QHBoxLayout()
        self.autoplay_checkbox = QCheckBox("Auto-play generated audio")
        self.autoplay_checkbox.setChecked(True)
        generate_actions_layout.addWidget(self.autoplay_checkbox)
        generate_actions_layout.addStretch()
        self.use_preset_button = QPushButton("Use Test Preset")
        self.use_preset_button.clicked.connect(self.apply_selected_text_preset)
        generate_actions_layout.addWidget(self.use_preset_button)
        self.generate_button = self._accent(QPushButton("Generate Audio"))
        self.generate_button.clicked.connect(self.handle_generate_stop_toggle)
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumWidth(180)
        generate_actions_layout.addWidget(self.generate_button)
        text_card_layout.addLayout(generate_actions_layout)
        generate_layout.addWidget(text_card, 3)

        delivery_card, delivery_layout = self._make_card("Delivery")
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

        self.exaggeration_slider = self._create_slider(0.25, 2.0, 0.05, 0.5)
        add_control(0, 0, "Expressiveness", self.exaggeration_slider,
                    "How animated and emotional the delivery sounds. 0.5 is neutral; "
                    "higher is more dramatic (and often a bit faster). [exaggeration]")
        self.cfg_slider = self._create_slider(0.2, 1.0, 0.05, 0.5)
        add_control(0, 2, "Pacing", self.cfg_slider,
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
        add_control(1, 2, "Take number", self.seed_input,
                    "Leave on 'New take each time' for a fresh result on every run. Enter a "
                    "number to reproduce the same take exactly; the number used is shown in "
                    "the file name. [seed]")
        delivery_layout.addLayout(params_layout)
        delivery_hint = QLabel(
            "Tip: punctuation steers delivery too. Commas and ellipses add pauses; "
            "question marks lift the ending. Advanced options are under Model > Sampling.")
        delivery_hint.setObjectName("Muted")
        delivery_hint.setWordWrap(True)
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
        self.pages.addWidget(self._scrollable(generate_page))

        # ---------- Voice page ----------
        voice_page, voice_layout = self._make_page(
            "Voice", "Choose the voice to clone. Leave it on the default voice, "
            "record yourself, or use an existing audio file.")

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
            "Read a short passage (about 15 seconds) in a quiet room. "
            "The first 6-10 seconds matter most, so start speaking right away.")
        record_hint.setObjectName("Muted")
        record_hint.setWordWrap(True)
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
        self.pages.addWidget(self._scrollable(voice_page))

        # ---------- Model page ----------
        model_page, model_layout = self._make_page(
            "Model", "Choose which Chatterbox model to run. Models download once, "
            "then load from the local cache.")
        model_card, model_card_layout = self._make_card("Active model")
        repo_layout = QHBoxLayout()
        self.model_repo_combo = QComboBox()
        self.model_repo_combo.currentTextChanged.connect(self.on_model_repo_changed)
        repo_layout.addWidget(self.model_repo_combo, 1)
        self.load_model_button = self._accent(QPushButton("Load Selected Model"))
        self.load_model_button.clicked.connect(self.load_model)
        repo_layout.addWidget(self.load_model_button)
        model_card_layout.addLayout(repo_layout)
        self.experimental_models_checkbox = QCheckBox("Show experimental user models")
        self.experimental_models_checkbox.toggled.connect(
            self.on_experimental_models_toggled)
        model_card_layout.addWidget(self.experimental_models_checkbox)
        model_layout.addWidget(model_card)

        config_card, config_layout = self._make_card("Configuration")
        self.model_config_help_label = QLabel(
            "Edit repo_id, enabled, backend, and language_id in models.json. "
            "Set enabled=true to make an entry appear in the picker, then use Reload Model List."
        )
        self.model_config_help_label.setWordWrap(True)
        self.model_config_help_label.setTextFormat(Qt.TextFormat.PlainText)
        self.model_config_help_label.setObjectName("Muted")
        config_layout.addWidget(self.model_config_help_label)
        self.model_details_label = QLabel("")
        self.model_details_label.setWordWrap(True)
        self.model_details_label.setTextFormat(Qt.TextFormat.PlainText)
        config_grid = QGridLayout()
        self.open_models_config_button = QPushButton("Open models.json")
        self.open_models_config_button.clicked.connect(self.open_models_config)
        self.reload_models_button = QPushButton("Reload Model List")
        self.reload_models_button.clicked.connect(self.reload_models_config)
        self.model_details_button = QPushButton("Model Details")
        self.model_details_button.clicked.connect(self.show_model_details_dialog)
        self.model_help_button = QPushButton("Custom Models Help")
        self.model_help_button.clicked.connect(self.show_model_help_dialog)
        self.hf_token_button = QPushButton("HF Token...")
        self.hf_token_button.clicked.connect(self.open_hf_token_dialog)
        self.sampling_settings_button = QPushButton("Sampling...")
        self.sampling_settings_button.clicked.connect(self.open_sampling_settings_dialog)
        for index, button in enumerate((
                self.open_models_config_button, self.reload_models_button,
                self.model_details_button, self.model_help_button,
                self.hf_token_button, self.sampling_settings_button)):
            config_grid.addWidget(button, index // 3, index % 3)
        config_layout.addLayout(config_grid)
        model_layout.addWidget(config_card)
        model_layout.addStretch(1)
        self.pages.addWidget(self._scrollable(model_page))

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
        self.pages.addWidget(self._scrollable(log_page))

        self.sidebar.setCurrentRow(self.PAGE_GENERATE)

        qt_status_bar = self.statusBar()
        qt_status_bar.setSizeGripEnabled(False)
        self.status_bar = QLabel("Status: Initializing...")
        self.status_bar.setWordWrap(False)
        self.status_bar.setTextFormat(Qt.TextFormat.PlainText)
        self.status_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
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
        self.on_experimental_models_toggled(False)
        self.refresh_language_options()
        self.refresh_hf_token_button_tooltip()
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
            self.hf_token_button_tooltip = "Saved token is active for Hugging Face downloads."
        else:
            os.environ.pop("HF_TOKEN", None)
            self.hf_token_button_tooltip = (
                "Optional. Set an HF token for higher rate limits and authenticated downloads."
            )

    def refresh_hf_token_button_tooltip(self):
        if hasattr(self, "hf_token_button"):
            self.hf_token_button.setToolTip(self.hf_token_button_tooltip)

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
            # --- Start Generation Part ---
            # (Same as your last full working version, ensures button is enabled for stop)
            if self.model is None:
                QMessageBox.warning(self, "Model Not Loaded",
                                    "Please load the model first.")
                return
            text = self.text_input.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Input Error",
                                    "Please enter some text to synthesize.")
                return

            self.is_generating = True
            self.generate_button.setText("Stop Generation")
            # Keep enabled to click "Stop"
            self.generate_button.setEnabled(True)
            self.load_model_button.setEnabled(False)

            self.generation_start_time = QTime.currentTime()
            self.generation_timer.start(1000)
            self.update_generation_time_display()  # Initial status update

            # ... (rest of parameter fetching and thread creation/start same as your file)
            ref_audio_full_path = self.ref_audio_path_label.toolTip()
            exaggeration = self.exaggeration_slider.get_value()
            cfg = self.cfg_slider.get_value()
            temperature = self.temp_slider.get_value()
            seed = self.seed_input.value()

            self.audio_generator_thread = AudioGeneratorThread(
                self.model, text, 
                ref_audio_full_path,
                exaggeration, temperature, cfg, seed, self.output_directory,
                language_id=self.language_combo.currentData() or "en",
                repetition_penalty=self.repetition_penalty,
                min_p=self.min_p,
                top_p=self.top_p,
                finishing=self.current_finishing_settings(),
            )
            self.audio_generator_thread.generation_complete.connect(
                self.on_generation_complete)
            self.audio_generator_thread.error_occurred.connect(
                self.on_generation_error)
            self.audio_generator_thread.chunk_generated.connect(
                self.on_chunk_generated_progress)
            self.audio_generator_thread.finished.connect(
                self.on_generation_thread_finished)
            self.audio_generator_thread.start()

        else:  # self.is_generating is True, so this is a Stop request
            if hasattr(self, 'audio_generator_thread') and self.audio_generator_thread.isRunning():
                print("UI: Requesting stop for audio_generator_thread")
                self.audio_generator_thread.stop()  # Signal the thread
                self.generate_button.setText("Stopping...")
                # Disable button while waiting for thread to acknowledge stop
                self.generate_button.setEnabled(False)
                self.set_status_message(
                    "Status: Stop requested. Waiting for current chunk to finish...")
            else:  # Should not happen if is_generating is True
                print(
                    "UI: Stop requested, but no active generation thread found. Resetting UI.")
                self.on_generation_thread_finished()  # Manually trigger UI reset

    def _create_slider(self, min_val, max_val, step_val, default_val, value_format="{:.2f}"):
        return SliderWithValue(min_val, max_val, step_val, default_val, value_format)

    def browse_reference_audio(self):
        default_dir = self.last_reference_audio_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", default_dir, "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            self.set_reference_audio(file_path)
            self.last_reference_audio_dir = os.path.dirname(file_path)

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

    def on_experimental_models_toggled(self, checked):
        self.refresh_model_repo_options()
        self.model_repo_combo.setEnabled(checked)
        self.use_preset_button.setEnabled(True)
        self.refresh_model_repo_tooltip()
        self.refresh_model_details()
        self.refresh_language_options()

    def on_model_repo_changed(self, _text):
        self.selected_model_repo = self.get_selected_model_entry()["repo_id"]
        self.refresh_model_repo_tooltip()
        self.refresh_model_details()
        self.refresh_language_options()

    def get_selected_model_repo(self):
        return self.get_selected_model_entry()["repo_id"]

    def get_visible_model_entries(self):
        enabled_entries = [
            entry for entry in self.model_entries
            if entry.get("enabled", True)
        ]
        if self.experimental_models_checkbox.isChecked():
            return enabled_entries

        default_entries = [
            entry for entry in enabled_entries
            if entry["repo_id"] == DEFAULT_MODEL_REPO and
            entry.get("backend") == BACKEND_MULTILINGUAL
        ]
        return default_entries or enabled_entries[:1]

    def refresh_model_repo_options(self):
        selected_label = None
        if hasattr(self, "model_repo_combo") and self.model_repo_combo.count() > 0:
            selected_label = self.model_repo_combo.currentText()

        visible_entries = self.get_visible_model_entries()
        self.model_repo_combo.blockSignals(True)
        self.model_repo_combo.clear()
        for index, entry in enumerate(visible_entries):
            self.model_repo_combo.addItem(
                f"{entry['label']} [{entry['repo_id']}]",
                index,
            )

        selected_index = -1
        if selected_label:
            for index, entry in enumerate(visible_entries):
                display_text = f"{entry['label']} [{entry['repo_id']}]"
                if display_text == selected_label:
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
        notes = selected_entry.get("notes", "")
        if self.experimental_models_checkbox.isChecked():
            tooltip = (
                f"Edit {MODEL_CONFIG_FILENAME} to add or change model repos. "
                "ResembleAI/chatterbox remains the default official multilingual repo."
            )
            if notes:
                tooltip += f"\n\nNotes: {notes}"
        else:
            tooltip = (
                f"Enable 'Experimental user models' to select a repo from {MODEL_CONFIG_FILENAME}."
            )
        self.model_repo_combo.setToolTip(tooltip)

    def refresh_model_details(self):
        if not self.model_entries:
            self.model_details_label.setText(
                f"No active models found in {MODEL_CONFIG_FILENAME}."
            )
            return

        selected_entry = self.get_selected_model_entry()
        detail_lines = [
            f"Backend: {selected_entry.get('backend', BACKEND_MULTILINGUAL)}",
            f"Repo: {selected_entry['repo_id']}",
        ]
        if selected_entry.get("backend") == BACKEND_MULTILINGUAL:
            detail_lines.append(
                f"Multilingual T3 model: {selected_entry.get('multilingual_t3_model', DEFAULT_MULTILINGUAL_T3_MODEL)}"
            )
        if selected_entry.get("experimental"):
            detail_lines.append("Experimental entry.")
        notes = selected_entry.get("notes", "")
        if notes:
            detail_lines.append(f"Notes: {notes}")
        self.model_details_label.setText("\n".join(detail_lines))

    def show_model_details_dialog(self):
        self.refresh_model_details()
        QMessageBox.information(
            self,
            "Model Details",
            self.model_details_label.text() or "No model details available.",
        )

    def show_model_help_dialog(self):
        QMessageBox.information(
            self,
            "Custom Models Help",
            (
                self.model_config_help_label.text()
                + "\n\nOptional field: multilingual_t3_model can be set to v3 or v2 for multilingual entries."
                + "\n\nOptional: set an HF token from the UI if you want authenticated Hugging Face downloads."
            ),
        )

    def open_hf_token_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Hugging Face Token")
        layout = QFormLayout(dialog)

        token_input = QLineEdit(dialog)
        token_input.setEchoMode(QLineEdit.EchoMode.Password)
        token_input.setPlaceholderText("hf_...")
        token_input.setText(str(self.app_settings.get("hf_token", "")))
        layout.addRow("HF_TOKEN", token_input)

        note_label = QLabel(
            "Optional. Used for higher rate limits and authenticated Hugging Face downloads. "
            "Leave blank to remove the saved token."
        )
        note_label.setWordWrap(True)
        note_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addRow(note_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            token_value = token_input.text().strip()
            if token_value:
                self.app_settings["hf_token"] = token_value
            else:
                self.app_settings.pop("hf_token", None)
            self.apply_hf_token_setting()
            self.save_app_settings()
            self.refresh_hf_token_button_tooltip()
            self.set_status_message(
                "Status: Hugging Face token settings updated."
            )

    def open_sampling_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Sampling Controls")
        layout = QFormLayout(dialog)

        repetition_penalty_input = QDoubleSpinBox(dialog)
        repetition_penalty_input.setRange(0.5, 3.0)
        repetition_penalty_input.setSingleStep(0.05)
        repetition_penalty_input.setDecimals(2)
        repetition_penalty_input.setValue(self.repetition_penalty)
        layout.addRow("Repetition Penalty", repetition_penalty_input)

        min_p_input = QDoubleSpinBox(dialog)
        min_p_input.setRange(0.0, 1.0)
        min_p_input.setSingleStep(0.01)
        min_p_input.setDecimals(2)
        min_p_input.setValue(self.min_p)
        layout.addRow("Min P", min_p_input)

        top_p_input = QDoubleSpinBox(dialog)
        top_p_input.setRange(0.0, 1.0)
        top_p_input.setSingleStep(0.01)
        top_p_input.setDecimals(2)
        top_p_input.setValue(self.top_p)
        layout.addRow("Top P", top_p_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            parent=dialog,
        )
        restore_defaults_button = buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        restore_defaults_button.clicked.connect(
            lambda: (
                repetition_penalty_input.setValue(1.2),
                min_p_input.setValue(0.05),
                top_p_input.setValue(1.0),
            )
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.repetition_penalty = repetition_penalty_input.value()
            self.min_p = min_p_input.value()
            self.top_p = top_p_input.value()
            self.sampling_settings_button.setToolTip(
                f"repetition_penalty={self.repetition_penalty:.2f}, "
                f"min_p={self.min_p:.2f}, top_p={self.top_p:.2f}"
            )

    def refresh_language_options(self):
        selected_entry = self.get_selected_model_entry()
        supported_languages = get_supported_languages_for_backend(
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

        is_multilingual = selected_entry.get("backend") == BACKEND_MULTILINGUAL
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
            self.text_input.setPlainText(preset_text)

    def open_models_config(self):
        opened = QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.model_config_path)
        )
        if not opened:
            QMessageBox.warning(
                self,
                "Open Config Failed",
                f"Could not open {self.model_config_path}.",
            )

    def reload_models_config(self):
        _payload, load_error = read_models_config_payload(self.model_config_path)
        if load_error:
            QMessageBox.warning(
                self,
                "Model Config Error",
                f"{load_error}\n\nThe current in-memory model list was left unchanged.",
            )
            return

        reloaded_entries = load_models_config(self.model_config_path)
        active_entries = [
            entry for entry in reloaded_entries
            if entry.get("enabled", True)
        ]
        if not active_entries:
            QMessageBox.warning(
                self,
                "No Active Models",
                f"{MODEL_CONFIG_FILENAME} does not contain any enabled model entries.",
            )
            return

        previous_selection = self.model_repo_combo.currentText()
        self.model_entries = reloaded_entries
        self.refresh_model_repo_options()
        if previous_selection:
            index = self.model_repo_combo.findText(previous_selection)
            if index >= 0:
                self.model_repo_combo.setCurrentIndex(index)
        self.refresh_model_repo_tooltip()
        self.refresh_model_details()
        self.refresh_language_options()
        self.set_status_message(
            f"Status: Reloaded model list from {MODEL_CONFIG_FILENAME}."
        )

    def set_model_loading_state(self, is_loading):
        if is_loading:
            self.model_load_progress.setEnabled(True)
            self.model_load_progress.setRange(0, 0)
        else:
            self.model_load_progress.setRange(0, 1)
            self.model_load_progress.setValue(0)
            self.model_load_progress.setEnabled(False)
        self.load_model_button.setEnabled(not is_loading)
        self.experimental_models_checkbox.setEnabled(not is_loading)
        self.model_repo_combo.setEnabled(
            not is_loading and self.experimental_models_checkbox.isChecked()
        )
        self.use_preset_button.setEnabled(not is_loading)
        self.reload_models_button.setEnabled(not is_loading)
        self.open_models_config_button.setEnabled(not is_loading)
        if is_loading:
            self.language_combo.setEnabled(False)
        else:
            self.refresh_language_options()

    def load_model(self):
        if not CHATTERBOX_AVAILABLE:
            QMessageBox.critical(
                self, "Error", "ChatterboxTTS library not installed.")
            return
        selected_entry = self.get_selected_model_entry()
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
            f"Status: Loading model from {selected_repo} using {selected_backend} backend"
            f"{f' ({selected_multilingual_t3_model})' if selected_backend == BACKEND_MULTILINGUAL else ''}. "
            "If this is the first run or a new repo, model files may still be downloading in the console."
        )
        self.generate_button.setEnabled(False)
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
        self.set_model_loading_state(False)
        self.on_experimental_models_toggled(
            self.experimental_models_checkbox.isChecked())
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
        self.set_model_loading_state(False)
        self.on_experimental_models_toggled(
            self.experimental_models_checkbox.isChecked())
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
        self.load_model_button.setEnabled(True)

        # Final status update based on how the thread might have ended,
        # if not already set by on_generation_complete or on_generation_error.
        # This ensures "Stopping..." doesn't linger.
        current_status = self.status_bar.text()
        if "stopping generation..." in current_status.lower() or \
           "stop requested." in current_status.lower():
            self.set_status_message("Status: Generation stopped by user.")
        elif not ("full audio generated" in current_status.lower() or
                  "failed" in current_status.lower() or
                  "stopped by user" in current_status.lower()):
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
        # This will be the primary status updater during active generation
        if not self.is_generating:
            return  # Don't update if we're trying to stop

        elapsed_str = ""
        if self.generation_start_time:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            elapsed_str = f" (Elapsed: {self.format_time(elapsed_ms)})"
        self.set_status_message(
            f"Status: Generating chunk {current_chunk}/{total_chunks}{elapsed_str}...")

    def on_generation_complete(self, output_path, sample_rate):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        total_generation_time_str = ""
        if self.generation_start_time:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            total_generation_time_str = f" (Total time: {self.format_time(elapsed_ms)})"

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
        if self.autoplay_checkbox.isChecked():
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
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui_theme.apply_theme(app)
    window = ChatterboxApp()
    window.show()
    sys.exit(app.exec())

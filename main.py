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
    QCheckBox, QComboBox, QProgressBar, QSizePolicy
)
# QStandardPaths was in your full file, good.
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QTime
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QDesktopServices

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
    def __init__(self, min_val, max_val, step_val, default_val):
        super().__init__()
        self._step_val = step_val
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val / step_val))
        self.slider.setMaximum(int(max_val / step_val))
        self.slider.setValue(int(default_val / step_val))
        self.slider.setSingleStep(1)
        self.value_label = QLabel(f"{default_val:.2f}")
        self.slider.valueChanged.connect(
            lambda val, lbl=self.value_label, s=step_val: lbl.setText(f"{val * s:.2f}")
        )
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)

    def get_value(self):
        return self.slider.value() * self._step_val


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
    ):
        super().__init__()
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
            final_audio_tensor = torch.cat(all_audio_tensors, dim=1)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chatterbox_{timestamp}_seed{self.actual_seed_used}_full_stitched.wav"
            output_path = os.path.join(self.output_dir, filename)
            torchaudio.save(output_path, final_audio_tensor, sr)
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

# --- ChatterboxApp ---


class ChatterboxApp(QMainWindow):
    log_message_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox TTS Interface")
        self.setGeometry(100, 100, 800, 720)
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

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        inputs_group = QGroupBox("Inputs")
        inputs_layout = QVBoxLayout()

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Enter text to synthesize. "
            "Optionally load a reference clip using the browse button. "
            f"Segments are ~{MAX_TEXT_INPUT_LENGTH} chars."
        )
        self.text_input.setMaximumHeight(200)
        self.text_input.setFixedHeight(88)
        text_label = QLabel("Text")
        inputs_layout.addWidget(text_label)
        inputs_layout.addWidget(self.text_input)

        ref_audio_layout = QHBoxLayout()
        self.ref_audio_path_label = QLabel("None selected.")
        self.ref_audio_path_label.setWordWrap(False)
        self.ref_audio_path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.ref_audio_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        browse_ref_button = QPushButton("Browse Reference Audio...")
        browse_ref_button.clicked.connect(self.browse_reference_audio)
        ref_audio_layout.addWidget(self.ref_audio_path_label, 1)
        ref_audio_layout.addWidget(browse_ref_button)
        top_controls_layout = QGridLayout()
        top_controls_layout.setColumnStretch(1, 1)
        top_controls_layout.setColumnStretch(3, 1)
        top_controls_layout.addWidget(QLabel("Reference Audio"), 0, 0)
        top_controls_layout.addLayout(ref_audio_layout, 0, 1, 1, 3)

        repo_layout = QHBoxLayout()
        self.model_repo_combo = QComboBox()
        self.model_repo_combo.currentTextChanged.connect(
            self.on_model_repo_changed)
        repo_layout.addWidget(self.model_repo_combo, 1)
        top_controls_layout.addWidget(QLabel("Model Repo"), 1, 0)
        top_controls_layout.addLayout(repo_layout, 1, 1)

        self.load_model_button = QPushButton("Load Selected Model")
        self.load_model_button.clicked.connect(self.load_model)
        top_controls_layout.addWidget(self.load_model_button, 1, 2)

        self.language_combo = QComboBox()
        top_controls_layout.addWidget(QLabel("Language"), 2, 0)
        top_controls_layout.addWidget(self.language_combo, 2, 1)

        self.experimental_models_checkbox = QCheckBox(
            "Experimental user models")
        self.experimental_models_checkbox.toggled.connect(
            self.on_experimental_models_toggled)
        top_controls_layout.addWidget(self.experimental_models_checkbox, 2, 2, 1, 2)

        model_config_actions_layout = QHBoxLayout()
        self.open_models_config_button = QPushButton("Open models.json")
        self.open_models_config_button.clicked.connect(self.open_models_config)
        model_config_actions_layout.addWidget(self.open_models_config_button)
        self.reload_models_button = QPushButton("Reload Model List")
        self.reload_models_button.clicked.connect(self.reload_models_config)
        model_config_actions_layout.addWidget(self.reload_models_button)
        self.model_details_button = QPushButton("Model Details")
        self.model_details_button.clicked.connect(self.show_model_details_dialog)
        model_config_actions_layout.addWidget(self.model_details_button)
        self.model_help_button = QPushButton("Custom Models Help")
        self.model_help_button.clicked.connect(self.show_model_help_dialog)
        model_config_actions_layout.addWidget(self.model_help_button)
        self.hf_token_button = QPushButton("HF Token...")
        self.hf_token_button.clicked.connect(self.open_hf_token_dialog)
        model_config_actions_layout.addWidget(self.hf_token_button)
        self.sampling_settings_button = QPushButton("Sampling...")
        self.sampling_settings_button.clicked.connect(self.open_sampling_settings_dialog)
        model_config_actions_layout.addWidget(self.sampling_settings_button)
        model_config_actions_layout.addStretch()
        top_controls_layout.addWidget(QLabel("Config"), 3, 0)
        top_controls_layout.addLayout(model_config_actions_layout, 3, 1, 1, 3)
        inputs_layout.addLayout(top_controls_layout)

        self.model_config_help_label = QLabel(
            "Edit repo_id, enabled, backend, and language_id in models.json. "
            "Set enabled=true to make an entry appear in the picker, then use Reload Model List."
        )
        self.model_config_help_label.setWordWrap(True)
        self.model_config_help_label.setTextFormat(Qt.TextFormat.PlainText)

        self.model_details_label = QLabel("")
        self.model_details_label.setWordWrap(True)
        self.model_details_label.setTextFormat(Qt.TextFormat.PlainText)

        params_layout = QGridLayout()
        params_layout.setColumnStretch(1, 1)
        params_layout.setColumnStretch(3, 1)
        self.exaggeration_slider = self._create_slider(0.25, 2.0, 0.05, 0.5)
        params_layout.addWidget(QLabel("Exaggeration"), 0, 0)
        params_layout.addWidget(self.exaggeration_slider, 0, 1)

        self.cfg_slider = self._create_slider(0.2, 1.0, 0.05, 0.5)
        params_layout.addWidget(QLabel("CFG/Pace"), 0, 2)
        params_layout.addWidget(self.cfg_slider, 0, 3)

        self.temp_slider = self._create_slider(0.05, 5.0, 0.05, 0.8)
        params_layout.addWidget(QLabel("Temperature"), 1, 0)
        params_layout.addWidget(self.temp_slider, 1, 1)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1_000_000_000)
        self.seed_input.setValue(0)
        params_layout.addWidget(QLabel("Seed (0 = random)"), 1, 2)
        params_layout.addWidget(self.seed_input, 1, 3)

        inputs_layout.addLayout(params_layout)

        generate_actions_layout = QHBoxLayout()
        self.autoplay_checkbox = QCheckBox("Auto-play generated audio")
        self.autoplay_checkbox.setChecked(True)
        generate_actions_layout.addWidget(self.autoplay_checkbox)
        generate_actions_layout.addStretch()

        self.use_preset_button = QPushButton("Use Test Preset")
        self.use_preset_button.clicked.connect(self.apply_selected_text_preset)
        generate_actions_layout.addWidget(self.use_preset_button)

        self.generate_button = QPushButton("Generate Audio")
        self.generate_button.clicked.connect(self.handle_generate_stop_toggle)
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumWidth(180)
        generate_actions_layout.addWidget(self.generate_button)
        inputs_layout.addLayout(generate_actions_layout)

        inputs_group.setLayout(inputs_layout)
        main_layout.addWidget(inputs_group)

        playback_group = QGroupBox("Playback & Output")
        playback_v_layout = QVBoxLayout()
        self.current_file_label = QLabel("Currently playing: None")
        playback_v_layout.addWidget(self.current_file_label)

        player_controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(
            self.toggle_play_pause)  # Connection is correct
        self.play_pause_button.setEnabled(False)
        player_controls_layout.addWidget(self.play_pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        player_controls_layout.addWidget(self.stop_button)
        playback_v_layout.addLayout(player_controls_layout)

        playhead_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.playhead_slider = QSlider(Qt.Orientation.Horizontal)

        self.playhead_slider.sliderPressed.connect(self.slider_pressed)
        self.playhead_slider.sliderMoved.connect(self.seek_audio_on_move)
        self.playhead_slider.sliderReleased.connect(self.slider_released)

        self.playhead_slider.setEnabled(False)
        self.duration_label = QLabel("00:00")
        playhead_layout.addWidget(self.current_time_label)
        playhead_layout.addWidget(self.playhead_slider)
        playhead_layout.addWidget(self.duration_label)
        playback_v_layout.addLayout(playhead_layout)

        playback_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.output_log_listwidget = QListWidget()
        self.output_log_listwidget.itemDoubleClicked.connect(
            self.play_selected_from_log)
        history_row_height = self.output_log_listwidget.sizeHintForRow(0)
        if history_row_height <= 0:
            history_row_height = self.output_log_listwidget.fontMetrics().height() + 8
        panel_list_height = (history_row_height * 6) + 8
        self.output_log_listwidget.setMaximumHeight(panel_list_height)
        history_panel = QWidget()
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.addWidget(
            QLabel("Generated Files History (double-click to play):")
        )
        history_layout.addWidget(self.output_log_listwidget)
        playback_splitter.addWidget(history_panel)

        self.console_log_view = QPlainTextEdit()
        self.console_log_view.setReadOnly(True)
        self.console_log_view.setMaximumBlockCount(1000)
        self.console_log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.console_log_view.setMaximumHeight(panel_list_height)
        console_panel = QWidget()
        console_layout = QVBoxLayout(console_panel)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.addWidget(QLabel("Activity Log:"))
        console_layout.addWidget(self.console_log_view)
        playback_splitter.addWidget(console_panel)
        playback_splitter.setSizes([320, 520])
        playback_v_layout.addWidget(playback_splitter)

        playback_group.setLayout(playback_v_layout)
        main_layout.addWidget(playback_group)

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
        self.model_load_progress.setFixedHeight(10)
        self.model_load_progress.setFixedWidth(120)
        self.model_load_progress.setEnabled(False)
        qt_status_bar.addPermanentWidget(self.model_load_progress)
        self.refresh_model_repo_options()
        self.on_experimental_models_toggled(False)
        self.refresh_language_options()
        self.refresh_hf_token_button_tooltip()

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

    def _create_slider(self, min_val, max_val, step_val, default_val):
        return SliderWithValue(min_val, max_val, step_val, default_val)

    def browse_reference_audio(self):
        default_dir = self.last_reference_audio_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", default_dir, "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            self.ref_audio_path_label.setText(os.path.basename(file_path))
            self.ref_audio_path_label.setToolTip(file_path)
            self.last_reference_audio_dir = os.path.dirname(file_path)
        else:
            self.ref_audio_path_label.setText("None selected.")
            self.ref_audio_path_label.setToolTip("")

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
                self.output_directory)if f.endswith(".wav")], key=os.path.getmtime, reverse=True)
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
    window = ChatterboxApp()
    window.show()
    sys.exit(app.exec())

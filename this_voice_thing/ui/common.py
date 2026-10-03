# ruff: noqa: E402
"""Start-up setup and shared constants for the window modules: console logging,
NLTK, the Chatterbox backend, model config and engine tables."""

import sys
import os
import json
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
import torchaudio  # noqa: F401  (imported at start-up, as before the split)


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

from this_voice_thing.core import documents
from this_voice_thing.core import model_registry
from this_voice_thing.engines import qwen as qwen_engine
from this_voice_thing.engines import kokoro as kokoro_engine
from this_voice_thing.engines import voxcpm as voxcpm_engine
from this_voice_thing.engines import omnivoice as omnivoice_engine
from this_voice_thing.engines import vibevoice as vibevoice_engine

try:
    from this_voice_thing.engines import chatterbox_backend as chatterbox_backends
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
# 15 characters a second; a few seconds is enough to judge a voice.
PREVIEW_LENGTHS = (("3 s", 45), ("5 s", 75), ("10 s", 150))
DEFAULT_PREVIEW_CHARS = 75


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

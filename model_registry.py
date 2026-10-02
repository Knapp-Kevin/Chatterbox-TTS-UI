"""Engine definitions, models.json persistence and Hugging Face lookups.

Engine-neutral by design: each models.json entry names an engine through its
"backend" key, and each engine declares which fields it uses. New engines
(e.g. Qwen3-TTS) are added here without changing the entry format.
"""

from dataclasses import dataclass
import json
import os
import re

from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir, try_to_load_from_cache
from huggingface_hub.errors import (
    GatedRepoError, HfHubHTTPError, RepositoryNotFoundError)

ENTRY_KEYS = (
    "repo_id", "label", "enabled", "experimental", "test_text", "test_texts",
    "notes", "backend", "language_id", "multilingual_t3_model", "qwen_variant",
)


@dataclass(frozen=True)
class Engine:
    key: str                 # value stored in models.json "backend"
    label: str
    description: str
    languages_summary: str
    uses_weights_version: bool = False


ENGINES = {
    "multilingual": Engine(
        key="multilingual",
        label="Chatterbox multilingual",
        description="23 languages, voice cloning from a reference clip.",
        languages_summary="23 languages",
        uses_weights_version=True,
    ),
    "legacy": Engine(
        key="legacy",
        label="Chatterbox original",
        description="The original single-language Chatterbox layout: English for the official "
                    "weights, or the language a community fine-tune was trained on.",
        languages_summary="single language",
    ),
    "qwen3": Engine(
        key="qwen3",
        label="Qwen3-TTS",
        description="10 languages; preset voices with style instructions, voice design from a "
                    "description, or cloning. Runs in its own environment (Apache-2.0).",
        languages_summary="10 languages",
    ),
}

QWEN_VARIANTS = {
    "custom_voice": "Preset voices",
    "voice_design": "Voice design",
    "base": "Voice cloning",
}

WEIGHT_VERSIONS = {
    "v3": "t3_mtl23ls_v3.safetensors",
    "v2": "t3_mtl23ls_v2.safetensors",
}

# Files each engine downloads (mirrors model_backends / chatterbox loaders).
ENGINE_FILES = {
    "multilingual": ["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
                     "conds.pt", "Cangjie5_TC.json"],
    "legacy": ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors",
               "tokenizer.json", "conds.pt"],
}
REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def engine_for(entry):
    return ENGINES.get(entry.get("backend", "multilingual"), ENGINES["multilingual"])


def weights_file(entry):
    version = str(entry.get("multilingual_t3_model") or "v3")
    if version.endswith(".safetensors"):
        return version
    return WEIGHT_VERSIONS.get(version.lower(), WEIGHT_VERSIONS["v3"])


def key_weight_file(entry):
    """The large file whose presence means the entry's model is downloaded."""
    if entry.get("backend") == "legacy":
        return "t3_cfg.safetensors"
    if entry.get("backend") == "qwen3":
        return "model.safetensors"
    return weights_file(entry)


# What each model is for. The order is the order groups appear in the UI.
CAPABILITIES = {
    "clone": ("Voice cloning", "Speak in the voice of a reference clip from the Voice page."),
    "preset": ("Preset voices", "Pick a built-in speaker and steer it with a style."),
    "design": ("Voice design", "Describe a voice in words and the model creates it."),
}
QWEN_CAPABILITY = {"base": "clone", "custom_voice": "preset", "voice_design": "design"}


def capability_for(entry):
    if entry.get("backend") == "qwen3":
        return QWEN_CAPABILITY.get(entry.get("qwen_variant"), "clone")
    return "clone"  # Chatterbox clones, or uses its built-in voice with no clip


def group_by_capability(entries):
    """[(capability, title, [entries])] in CAPABILITIES order, skipping empty groups."""
    groups = []
    for capability, (title, _description) in CAPABILITIES.items():
        members = [entry for entry in entries if capability_for(entry) == capability]
        if members:
            groups.append((capability, title, members))
    return groups


def engine_label(entry):
    engine = engine_for(entry)
    if engine.key == "qwen3":
        return f"{engine.label} · {QWEN_VARIANTS.get(entry.get('qwen_variant'), 'unknown variant')}"
    return engine.label


def entry_to_json(entry):
    payload = {key: entry[key] for key in ENTRY_KEYS if key in entry}
    if payload.get("backend") != "multilingual":
        payload.pop("multilingual_t3_model", None)
    if payload.get("backend") != "qwen3":
        payload.pop("qwen_variant", None)
    elif payload.get("multilingual_t3_model", "").endswith(".safetensors"):
        for short, filename in WEIGHT_VERSIONS.items():
            if payload["multilingual_t3_model"] == filename:
                payload["multilingual_t3_model"] = short
    if not payload.get("test_texts"):
        payload.pop("test_texts", None)
    return payload


def save_models_config(path, entries):
    payload = {"models": [entry_to_json(entry) for entry in entries]}
    temp_path = path + ".tmp"
    # LF endings to match the tracked file (no CRLF churn on Windows).
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, path)


def is_valid_repo_id(repo_id):
    return bool(REPO_ID_PATTERN.match(repo_id or ""))


# ---------- local cache ----------

def cached_repo_sizes():
    """{repo_id: bytes on disk} for every model repo in the local HF cache."""
    try:
        info = scan_cache_dir()
    except Exception:
        return {}
    return {repo.repo_id: repo.size_on_disk for repo in info.repos if repo.repo_type == "model"}


def is_downloaded(entry):
    try:
        cached = try_to_load_from_cache(entry["repo_id"], key_weight_file(entry))
    except Exception:
        return False
    return isinstance(cached, str) and os.path.exists(cached)


def format_size(num_bytes):
    if not num_bytes:
        return "0 MB"
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.1f} GB"
    return f"{num_bytes / 1024 ** 2:.0f} MB"


# ---------- Hugging Face lookups ----------

@dataclass
class RepoCheck:
    ok: bool
    message: str
    detected_backend: str = ""
    weight_versions: tuple = ()
    download_bytes: int = 0
    gated: bool = False
    private: bool = False
    qwen_variant: str = ""


def check_repo(repo_id, token=None):
    """Inspect a Hugging Face repo and work out which engine can load it."""
    if not is_valid_repo_id(repo_id):
        return RepoCheck(False, "Enter a repo in the form owner/name.")
    api = HfApi()
    try:
        info = api.model_info(repo_id, files_metadata=True, token=token or None)
    except GatedRepoError:
        return RepoCheck(False, "This repo is gated: accept its terms on huggingface.co, "
                                "then add a token under Hugging Face access.", gated=True)
    except RepositoryNotFoundError:
        return RepoCheck(False, "Repo not found. Check the name, or add a token if it is private.")
    except HfHubHTTPError as exc:
        return RepoCheck(False, f"Hugging Face returned an error: {exc}")
    except Exception as exc:
        return RepoCheck(False, f"Could not reach Hugging Face: {exc}")

    sizes = {sibling.rfilename: (sibling.size or 0) for sibling in (info.siblings or [])}
    versions = tuple(short for short, filename in WEIGHT_VERSIONS.items() if filename in sizes)
    gated = bool(getattr(info, "gated", False))
    private = bool(getattr(info, "private", False))
    if versions or "t3_23lang.safetensors" in sizes:
        backend = "multilingual"
        best = versions[0] if versions else None
        files = ENGINE_FILES["multilingual"] + ([WEIGHT_VERSIONS[best]] if best else ["t3_23lang.safetensors"])
    elif "t3_cfg.safetensors" in sizes:
        backend, files = "legacy", ENGINE_FILES["legacy"]
    elif "config.json" in sizes and "model.safetensors" in sizes:
        return _check_qwen_repo(repo_id, token, sizes, gated, private)
    else:
        weights = sorted(name for name in sizes
                         if name.endswith((".safetensors", ".pt", ".bin", ".gguf", ".onnx")))
        found = ", ".join(weights[:4]) + (" ..." if len(weights) > 4 else "") if weights else "no model files"
        return RepoCheck(False, f"Not a layout this app can load (found: {found}). Chatterbox "
                                "multilingual repos have t3_mtl23ls_v3.safetensors; English ones "
                                "have t3_cfg.safetensors.", gated=gated, private=private)
    download = sum(sizes.get(name, 0) for name in files)
    engine = ENGINES[backend]
    access = "gated (token needed)" if gated else "private (token needed)" if private else "public"
    detail = f"Found: {engine.label}"
    if versions:
        detail += f", weights {', '.join(v.upper() for v in versions)}"
    detail += f" · about {format_size(download)} to download · {access}"
    return RepoCheck(True, detail, backend, versions, download, gated, private)


def _check_qwen_repo(repo_id, token, sizes, gated, private):
    try:
        with open(hf_hub_download(repo_id, "config.json", token=token or None), encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        return RepoCheck(False, f"Could not read config.json: {exc}", gated=gated, private=private)
    variant = config.get("tts_model_type", "")
    if config.get("model_type") != "qwen3_tts" or variant not in QWEN_VARIANTS:
        return RepoCheck(False, "This repo has a config.json but is not a Qwen3-TTS speech model "
                                "this app can load.", gated=gated, private=private)
    download = sum(sizes.values())
    access = "gated (token needed)" if gated else "private (token needed)" if private else "public"
    return RepoCheck(True, f"Found: Qwen3-TTS, {QWEN_VARIANTS[variant].lower()} "
                           f"· about {format_size(download)} to download · {access}",
                     "qwen3", (), download, gated, private, variant)


# ---------- discovery ----------

CHATTERBOX_KEY_FILES = {"t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors",
                        "t3_23lang.safetensors", "t3_cfg.safetensors"}
# Same names, different formats (Apple MLX, GGUF, ONNX ...) or test fixtures.
EXCLUDED_TAGS = {"mlx", "mlx-audio", "gguf", "onnx", "openvino", "coreml", "coremltools", "ggml"}
QWEN_NAME_HINTS = {"customvoice": "custom_voice", "voicedesign": "voice_design", "base": "base"}
KNOWN_LANGUAGE_TAGS = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
    "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh", "id", "bn", "fa", "uk", "vi", "th", "cs",
    "ro", "hu", "ur", "ta", "te", "mos",
}


@dataclass
class SearchResult:
    repo_id: str
    backend: str
    summary: str
    downloads: int
    likes: int
    gated: bool
    languages: tuple
    updated: str
    qwen_variant: str = ""


def _classify(model):
    tags = {tag.lower() for tag in (model.tags or [])}
    if tags & EXCLUDED_TAGS or "tiny-random" in model.id.lower():
        return None
    files = {sibling.rfilename for sibling in (model.siblings or [])}
    if files & CHATTERBOX_KEY_FILES:
        if files & (CHATTERBOX_KEY_FILES - {"t3_cfg.safetensors"}):
            versions = [v.upper() for v, f in WEIGHT_VERSIONS.items() if f in files]
            return "multilingual", "", "Chatterbox multilingual" + (f" {'/'.join(versions)}" if versions else "")
        return "legacy", "", "Chatterbox original (single language)"
    if "qwen3_tts" in tags and "model.safetensors" in files and "config.json" in files:
        name = model.id.split("/")[-1].lower().replace("-", "").replace("_", "")
        variant = next((v for hint, v in QWEN_NAME_HINTS.items() if hint in name), "")
        size = "0.6B " if "0.6b" in model.id.lower() else "1.7B " if "1.7b" in model.id.lower() else ""
        label = QWEN_VARIANTS.get(variant, "variant confirmed by Check").lower()
        return "qwen3", variant, f"Qwen3-TTS {size}· {label}"
    return None


def search_models(query="", engine="all", token=None, limit=40):
    """Find Hugging Face repos this app can load, most downloaded first."""
    api = HfApi()
    expand = ["siblings", "downloads", "likes", "gated", "tags", "lastModified"]
    query = (query or "").strip()
    listings = []
    if engine in ("all", "chatterbox"):
        listings.append(dict(filter="chatterbox", search=query or None))
        listings.append(dict(search=query or "chatterbox"))
    if engine in ("all", "qwen3"):
        listings.append(dict(filter="qwen3_tts", search=query or None))
        if query:
            listings.append(dict(search=query))
    seen, results = set(), []
    for kwargs in listings:
        try:
            models = api.list_models(sort="downloads", limit=200, expand=expand,
                                     token=token or None, **kwargs)
            for model in models:
                if model.id in seen:
                    continue
                seen.add(model.id)
                classified = _classify(model)
                if not classified:
                    continue
                backend, variant, summary = classified
                if engine == "chatterbox" and backend == "qwen3" or engine == "qwen3" and backend != "qwen3":
                    continue
                tags = {tag.lower() for tag in (model.tags or [])}
                languages = tuple(sorted(tags & KNOWN_LANGUAGE_TAGS))
                updated = model.last_modified.strftime("%b %Y") if getattr(model, "last_modified", None) else ""
                results.append(SearchResult(model.id, backend, summary, model.downloads or 0,
                                            model.likes or 0, bool(model.gated), languages, updated, variant))
        except Exception as exc:
            raise RuntimeError(f"Hugging Face search failed: {exc}") from exc
    results.sort(key=lambda result: result.downloads, reverse=True)
    return results[:limit]


def whoami(token):
    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:
        return False, f"Token not accepted: {exc}".split("\n")[0][:160]
    return True, f"Signed in as {info.get('name', 'unknown')}"

"""Engine definitions, models.json persistence and Hugging Face lookups.

Engine-neutral by design: each models.json entry names an engine through its
"backend" key, and each engine declares which fields it uses. New engines
(e.g. Qwen3-TTS) are added here without changing the entry format.
"""

from dataclasses import dataclass
import json
import os
import re

from huggingface_hub import HfApi, scan_cache_dir, try_to_load_from_cache
from huggingface_hub.errors import (
    GatedRepoError, HfHubHTTPError, RepositoryNotFoundError)

ENTRY_KEYS = (
    "repo_id", "label", "enabled", "experimental", "test_text", "test_texts",
    "notes", "backend", "language_id", "multilingual_t3_model",
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
        label="Chatterbox English",
        description="The original English-only Chatterbox layout.",
        languages_summary="English",
    ),
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
    return weights_file(entry)


def entry_to_json(entry):
    payload = {key: entry[key] for key in ENTRY_KEYS if key in entry}
    if payload.get("backend") != "multilingual":
        payload.pop("multilingual_t3_model", None)
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


def whoami(token):
    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:
        return False, f"Token not accepted: {exc}".split("\n")[0][:160]
    return True, f"Signed in as {info.get('name', 'unknown')}"

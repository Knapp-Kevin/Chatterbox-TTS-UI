from dataclasses import dataclass
from pathlib import Path
import inspect
import os
import subprocess

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS


DEFAULT_MODEL_REPO = "ResembleAI/chatterbox"
BACKEND_LEGACY = "legacy"
BACKEND_MULTILINGUAL = "multilingual"
DEFAULT_MULTILINGUAL_T3_MODEL = "v3"

LEGACY_MODEL_FILES = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
]

MULTILINGUAL_MODEL_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v3.safetensors",
    "t3_23lang.safetensors",
    "s3gen.pt",
    "s3gen_v3.pt",
    "s3gen_v3.safetensors",
    "grapheme_mtl_merged_expanded_v1.json",
    "mtl_tokenizer.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

MULTILINGUAL_T3_MODEL_ALIASES = {
    "v2": "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v2": "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v2.safetensors": "t3_mtl23ls_v2.safetensors",
    "v3": "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v3": "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v3.safetensors": "t3_mtl23ls_v3.safetensors",
}


class UnsupportedChatterboxRepoError(RuntimeError):
    pass


@dataclass
class LoadedChatterboxModel:
    instance: object
    backend: str
    repo_id: str
    supported_languages: dict

    @property
    def sr(self):
        return self.instance.sr

    @property
    def device(self):
        return getattr(self.instance, "device", "N/A")

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        temperature=0.8,
        cfg_weight=0.5,
        language_id=None,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    ):
        if self.backend == BACKEND_MULTILINGUAL:
            effective_language = language_id or "en"
            return self.instance.generate(
                text,
                language_id=effective_language,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        return self.instance.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )


def has_system_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def probe_usable_torch_cuda():
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned False."

    try:
        arch_list = set(torch.cuda.get_arch_list())
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
        if current_arch not in arch_list:
            return (
                False,
                f"Installed PyTorch CUDA kernels do not include {current_arch}. "
                f"Supported architectures: {sorted(arch_list)}",
            )

        x = torch.randn((8, 8), device="cuda")
        y = x @ x
        _ = y.sum().item()
        gru = torch.nn.GRU(4, 4).to("cuda")
        gru(torch.randn((2, 1, 4), device="cuda"))
        return True, None
    except Exception as exc:
        return False, str(exc)


def get_supported_languages_for_backend(backend):
    if backend == BACKEND_MULTILINGUAL:
        return ChatterboxMultilingualTTS.get_supported_languages()
    return {"en": "English"}


def resolve_multilingual_t3_model(multilingual_t3_model):
    if not multilingual_t3_model:
        multilingual_t3_model = DEFAULT_MULTILINGUAL_T3_MODEL
    normalized = str(multilingual_t3_model).strip().lower()
    return MULTILINGUAL_T3_MODEL_ALIASES.get(normalized, str(multilingual_t3_model).strip())


def _download_repo_snapshot(repo_id, allow_patterns):
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            allow_patterns=allow_patterns,
            token=os.getenv("HF_TOKEN"),
        )
    )


def load_chatterbox_model(repo_id, backend, device, multilingual_t3_model=None):
    try:
        if backend == BACKEND_MULTILINGUAL:
            resolved_t3_model = resolve_multilingual_t3_model(multilingual_t3_model)
            instance = None
            # Prefer the higher-level official path for the default repo if the
            # installed package exposes explicit V3 selection there.
            if repo_id == DEFAULT_MODEL_REPO:
                from_pretrained = getattr(ChatterboxMultilingualTTS, "from_pretrained", None)
                if callable(from_pretrained):
                    from_pretrained_signature = inspect.signature(from_pretrained)
                    if "t3_model" in from_pretrained_signature.parameters:
                        print(
                            "Using ChatterboxMultilingualTTS.from_pretrained for the official multilingual repo "
                            f"with t3_model='{resolved_t3_model}'."
                        )
                        instance = from_pretrained(device=device, t3_model=resolved_t3_model)

            if instance is None:
                ckpt_dir = _download_repo_snapshot(repo_id, MULTILINGUAL_MODEL_FILES)
                from_local_signature = inspect.signature(ChatterboxMultilingualTTS.from_local)
                if "t3_model" in from_local_signature.parameters:
                    print(
                        "Using ChatterboxMultilingualTTS.from_local with explicit multilingual "
                        f"T3 selection '{resolved_t3_model}'."
                    )
                    instance = ChatterboxMultilingualTTS.from_local(
                        ckpt_dir,
                        device,
                        t3_model=resolved_t3_model,
                    )
                else:
                    if resolved_t3_model != "t3_mtl23ls_v2.safetensors":
                        print(
                            "Installed chatterbox-tts package does not support explicit multilingual "
                            f"T3 selection on from_local. Falling back to the package default instead of '{resolved_t3_model}'."
                        )
                    instance = ChatterboxMultilingualTTS.from_local(ckpt_dir, device)
            return LoadedChatterboxModel(
                instance=instance,
                backend=backend,
                repo_id=repo_id,
                supported_languages=ChatterboxMultilingualTTS.get_supported_languages(),
            )

        ckpt_dir = _download_repo_snapshot(repo_id, LEGACY_MODEL_FILES)
        instance = ChatterboxTTS.from_local(ckpt_dir, device)
        return LoadedChatterboxModel(
            instance=instance,
            backend=backend,
            repo_id=repo_id,
            supported_languages={"en": "English"},
        )
    except EntryNotFoundError as exc:
        if backend == BACKEND_MULTILINGUAL:
            raise UnsupportedChatterboxRepoError(
                "This repo does not publish the multilingual checkpoint layout "
                "expected by the current Chatterbox multilingual loader. "
                "If this is an older repo, try setting multilingual_t3_model to v2 in models.json."
            ) from exc
        raise UnsupportedChatterboxRepoError(
            "This repo does not publish the legacy Chatterbox checkpoint layout "
            "expected by the current English loader."
        ) from exc

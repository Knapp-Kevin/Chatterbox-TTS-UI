"""Qwen3-TTS worker: runs inside engines/qwen/.venv and serves the main app.

Protocol: one JSON request per line on stdin, one JSON reply per line on the
original stdout. Everything libraries print goes to stderr, which the app
shows on its Log page, so replies can never be corrupted by log output.

Requests:
  {"cmd": "load", "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
  {"cmd": "generate", "mode": "custom_voice" | "voice_design" | "base",
   "text": ..., "language": "English" | null, "speaker": ..., "instruct": ...,
   "ref_audio": path, "ref_text": str | null, "seed": int, "out_path": path,
   "temperature": f, "top_p": f, "repetition_penalty": f}
  {"cmd": "ping"} / {"cmd": "shutdown"}
"""

import json
import os
import sys
import traceback
import warnings

REPLY = sys.stdout
sys.stdout = sys.stderr
warnings.filterwarnings("ignore")


def reply(**payload):
    REPLY.write(json.dumps(payload) + "\n")
    REPLY.flush()


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    state = {"model": None, "model_id": None, "mode": None, "clone_prompts": {}}

    def load(model_id):
        if state["model_id"] == model_id:
            return
        state["model"] = None
        state["clone_prompts"].clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cuda = torch.cuda.is_available()
        # Load from the local cache when the model is already downloaded, so loading
        # (and therefore generation) works offline; qwen-tts otherwise asks the Hub for
        # the latest revision and fails without a connection.
        try:
            from huggingface_hub import snapshot_download
            source = snapshot_download(model_id, local_files_only=True)
        except Exception:
            source = model_id  # not cached yet: download it
        model = Qwen3TTSModel.from_pretrained(
            source,
            device_map="cuda:0" if cuda else "cpu",
            dtype=torch.bfloat16 if cuda else torch.float32,
            attn_implementation="sdpa",
        )
        state.update(model=model, model_id=model_id,
                     mode=getattr(model.model, "tts_model_type", None)
                     or getattr(model.model.config, "tts_model_type", None))

    def clone_prompt(ref_audio, ref_text):
        key = (ref_audio, os.path.getmtime(ref_audio), ref_text or "")
        if key not in state["clone_prompts"]:
            state["clone_prompts"].clear()
            state["clone_prompts"][key] = state["model"].create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text or None,
                x_vector_only_mode=not ref_text)
        return state["clone_prompts"][key]

    def generate(req):
        model = state["model"]
        if model is None:
            raise RuntimeError("No Qwen model is loaded.")
        if req.get("seed"):
            torch.manual_seed(int(req["seed"]))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(req["seed"]))
        sampling = {key: req[key] for key in ("temperature", "top_p", "repetition_penalty")
                    if req.get(key) is not None}
        # A batch of texts runs as one call: per-step overhead (not GPU compute) dominates
        # single-sequence decoding, so batching sections multiplies throughput.
        texts = req.get("texts") or [req["text"]]
        mode = req["mode"]
        wavs, sr = run_batch(model, mode, texts, req, sampling)
        out_paths = req.get("out_paths") or [req["out_path"]]
        seconds = []
        for wav, path in zip(wavs, out_paths):
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            sf.write(path, wav, sr, subtype="FLOAT")
            seconds.append(round(len(wav) / sr, 2))
        return {"paths": out_paths[:len(wavs)], "path": out_paths[0], "sr": int(sr),
                "seconds": seconds}

    def run_batch(model, mode, texts, req, sampling):
        """Generate a batch; if the GPU runs out of memory, split it and retry."""
        try:
            return generate_texts(model, mode, texts, req, sampling)
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(texts) // 2
            print(f"GPU memory full for a batch of {len(texts)}; retrying as {half} + "
                  f"{len(texts) - half}.", file=sys.stderr)
            first, sr = run_batch(model, mode, texts[:half], req, sampling)
            second, _ = run_batch(model, mode, texts[half:], req, sampling)
            return list(first) + list(second), sr

    def generate_texts(model, mode, texts, req, sampling):
        count = len(texts)
        language = [req.get("language") or "auto"] * count
        if mode == "custom_voice":
            instruct = req.get("instruct") or None
            wavs, sr = model.generate_custom_voice(
                text=texts, speaker=[req["speaker"]] * count, language=language,
                instruct=[instruct] * count if instruct else None, **sampling)
        elif mode == "voice_design":
            if not req.get("instruct"):
                raise ValueError("Describe the voice you want before generating.")
            wavs, sr = model.generate_voice_design(
                text=texts, instruct=[req["instruct"]] * count, language=language, **sampling)
        elif mode == "base":
            if not req.get("ref_audio"):
                raise ValueError("Pick a reference clip on the Voice page to clone.")
            prompt = clone_prompt(req["ref_audio"], req.get("ref_text"))
            wavs, sr = model.generate_voice_clone(
                text=texts, language=language, voice_clone_prompt=list(prompt) * count, **sampling)
        else:
            raise ValueError(f"Unknown Qwen mode: {mode}")
        return wavs, sr

    reply(ok=True, event="ready", cuda=torch.cuda.is_available(),
          device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "load":
                load(req["model_id"])
                model = state["model"]
                reply(ok=True, mode=state["mode"],
                      speakers=list(model.get_supported_speakers() or []),
                      languages=list(model.get_supported_languages() or []))
            elif cmd == "generate":
                reply(ok=True, **generate(req))
            elif cmd == "ping":
                reply(ok=True)
            elif cmd == "shutdown":
                reply(ok=True)
                break
            else:
                reply(ok=False, error=f"Unknown command: {cmd}")
        except Exception as exc:
            traceback.print_exc()
            reply(ok=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        reply(ok=False, event="fatal", error=f"{type(exc).__name__}: {exc}")

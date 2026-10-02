"""OmniVoice worker: runs inside engines/omnivoice/.venv and serves the main app.

Protocol: one JSON request per line on stdin, one JSON reply per line on the
original stdout. Everything libraries print goes to stderr (the app's Log page).

Requests:
  {"cmd": "load", "model_id": "k2-fsa/OmniVoice"}
  {"cmd": "generate", "texts": [...], "out_paths": [...], "language": "en" | null,
   "instruct": "female, low pitch" | absent, "ref_audio": path | absent,
   "ref_text": str | absent, "seed": int}
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
    from huggingface_hub import snapshot_download
    from omnivoice import OmniVoice

    cuda = torch.cuda.is_available()
    state = {"model": None, "model_id": None, "languages": set(), "prompts": {}}

    def load(model_id):
        if state["model_id"] == model_id:
            return
        state.update(model=None, model_id=None, prompts={})
        if cuda:
            torch.cuda.empty_cache()
        # Use the local copy when it's there so loading works offline.
        try:
            source = snapshot_download(model_id, local_files_only=True)
        except Exception:
            source = snapshot_download(model_id)
        model = OmniVoice.from_pretrained(source, device_map="cuda:0" if cuda else "cpu",
                                          dtype=torch.float16 if cuda else torch.float32)
        state.update(model=model, model_id=model_id,
                     languages={code.lower() for code in model.supported_language_ids()})
        # The first generation pays a one-time warm-up (~15 s); do it while loading.
        model.generate(text="Ready.", language="en", instruct="female")

    def clone_prompt(ref_audio, ref_text):
        key = (ref_audio, os.path.getmtime(ref_audio), ref_text)
        if key not in state["prompts"]:
            state["prompts"].clear()
            state["prompts"][key] = state["model"].create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text)
        return state["prompts"][key]

    def run_batch(texts, kwargs):
        """Generate a batch; if the GPU runs out of memory, split it and retry."""
        try:
            prompt = kwargs.get("voice_clone_prompt")
            call = dict(kwargs)
            if prompt is not None:
                call["voice_clone_prompt"] = [prompt] * len(texts)
            return list(state["model"].generate(text=texts, **call))
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(texts) // 2
            print(f"GPU memory full for a batch of {len(texts)}; retrying as {half} + "
                  f"{len(texts) - half}.", file=sys.stderr)
            return run_batch(texts[:half], kwargs) + run_batch(texts[half:], kwargs)

    def generate(req):
        if state["model"] is None:
            raise RuntimeError("No OmniVoice model is loaded.")
        if req.get("seed"):
            torch.manual_seed(int(req["seed"]))
            if cuda:
                torch.cuda.manual_seed_all(int(req["seed"]))
        kwargs = {}
        language = (req.get("language") or "").lower()
        if language and language in state["languages"]:
            kwargs["language"] = language
        if req.get("ref_audio"):
            if not req.get("ref_text"):
                raise ValueError("OmniVoice cloning needs the Clip transcript.")
            kwargs["voice_clone_prompt"] = clone_prompt(req["ref_audio"], req["ref_text"])
        elif req.get("instruct"):
            kwargs["instruct"] = req["instruct"]
        wavs = run_batch(list(req["texts"]), kwargs)
        seconds = []
        for wav, path in zip(wavs, req["out_paths"]):
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            sf.write(path, wav, 24000, subtype="FLOAT")
            seconds.append(round(len(wav) / 24000, 2))
        return {"paths": req["out_paths"][:len(wavs)], "sr": 24000, "seconds": seconds}

    reply(ok=True, event="ready", cuda=cuda, device=torch.cuda.get_device_name(0) if cuda else "cpu")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "load":
                load(req["model_id"])
                reply(ok=True)
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

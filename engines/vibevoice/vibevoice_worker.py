"""VibeVoice worker: runs inside engines/vibevoice/.venv and serves the main app.

Protocol: one JSON request per line on stdin, one JSON reply per line on the
original stdout. Everything libraries print goes to stderr (the app's Log page).

Requests:
  {"cmd": "load", "model_id": "vibevoice/VibeVoice-1.5B-hf",
   "sample_repo": "bezzam/vibevoice_samples", "samples": {name: path in repo}}
  {"cmd": "generate", "turns": [{"role": "0", "text": ..., "audio": path | null}],
   "out_path": path, "seed": int}
  {"cmd": "ping"} / {"cmd": "shutdown"}
"""

import json
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
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoModelForTextToWaveform, AutoProcessor, set_seed

    cuda = torch.cuda.is_available()
    state = {"model": None, "processor": None, "model_id": None}

    def fetch(repo_id, filename, repo_type=None):
        # Use the local copy when it's there so loading works offline.
        try:
            return hf_hub_download(repo_id, filename, repo_type=repo_type, local_files_only=True)
        except Exception:
            return hf_hub_download(repo_id, filename, repo_type=repo_type)

    def load(req):
        model_id = req["model_id"]
        if state["model_id"] != model_id:
            state.update(model=None, processor=None, model_id=None)
            if cuda:
                torch.cuda.empty_cache()
            try:
                source = snapshot_download(model_id, local_files_only=True)
            except Exception:
                source = snapshot_download(model_id, ignore_patterns=["figures/*"])
            processor = AutoProcessor.from_pretrained(source)
            model = AutoModelForTextToWaveform.from_pretrained(
                source, dtype=torch.bfloat16 if cuda else torch.float32,
                device_map="cuda" if cuda else "cpu").eval()
            state.update(model=model, processor=processor, model_id=model_id)
            # The first generation pays a one-time warm-up; do it while loading.
            warmup = processor.apply_chat_template(
                [{"role": "0", "content": [{"type": "text", "text": "Ready."}]}],
                return_dict=True, tokenize=True, add_generation_prompt=True).to(model.device, model.dtype)
            with torch.inference_mode():
                model.generate(**warmup, max_new_tokens=20)
        samples = {}
        for name, path in (req.get("samples") or {}).items():
            try:
                samples[name] = fetch(req["sample_repo"], path, repo_type="dataset")
            except Exception as exc:
                print(f"Sample voice {name} unavailable: {exc}", file=sys.stderr)
        return samples

    def generate(req):
        model, processor = state["model"], state["processor"]
        if model is None:
            raise RuntimeError("No VibeVoice model is loaded.")
        set_seed(int(req.get("seed") or 0))
        conversation = []
        for turn in req["turns"]:
            content = [{"type": "text", "text": turn["text"]}]
            if turn.get("audio"):
                content.append({"type": "audio", "path": turn["audio"]})
            conversation.append({"role": str(turn["role"]), "content": content})
        inputs = processor.apply_chat_template(
            conversation, return_dict=True, tokenize=True, add_generation_prompt=True,
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            audio = model.generate(**inputs)
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        wav = wav.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        sr = int(processor.feature_extractor.sampling_rate)
        sf.write(req["out_path"], wav, sr, subtype="FLOAT")
        return {"path": req["out_path"], "sr": sr, "seconds": round(len(wav) / sr, 2)}

    reply(ok=True, event="ready", cuda=cuda, device=torch.cuda.get_device_name(0) if cuda else "cpu")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "load":
                reply(ok=True, samples=load(req))
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

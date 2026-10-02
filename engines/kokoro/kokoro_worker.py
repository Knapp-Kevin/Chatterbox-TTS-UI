"""Kokoro worker: runs inside engines/kokoro/.venv and serves the main app.

Protocol: one JSON request per line on stdin, one JSON reply per line on the
original stdout. Everything libraries print goes to stderr (the app's Log page).

Requests:
  {"cmd": "load", "model_id": "hexgrad/Kokoro-82M"}
  {"cmd": "generate", "texts": [...], "voice": "af_heart", "out_paths": [...]}
  {"cmd": "ping"} / {"cmd": "shutdown"}
"""

import glob
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
    from kokoro import KModel, KPipeline

    cuda = torch.cuda.is_available()
    state = {"model": None, "model_id": None, "root": None, "pipelines": {}}

    def load(model_id):
        if state["model_id"] == model_id:
            return
        state.update(model=None, pipelines={})
        # Only the weights, config and voices; the repos also hold samples and eval files.
        # Use the local copy when it's complete so loading works offline.
        patterns = ["*.pth", "config.json", "voices/*.pt"]

        def complete(root):
            return glob.glob(os.path.join(root, "*.pth")) and glob.glob(os.path.join(root, "voices", "*.pt"))

        try:
            root = snapshot_download(model_id, allow_patterns=patterns, local_files_only=True)
        except Exception:
            root = None
        if not root or not complete(root):
            root = snapshot_download(model_id, allow_patterns=patterns)
        weights = sorted(glob.glob(os.path.join(root, "*.pth")))
        if not weights:
            raise RuntimeError(f"{model_id} has no Kokoro .pth weights.")
        model = KModel(repo_id=model_id, config=os.path.join(root, "config.json"), model=weights[-1])
        model = model.to("cuda" if cuda else "cpu").eval()
        state.update(model=model, model_id=model_id, root=root)
        # Warm up the English text front end (spaCy loads on first use, several seconds)
        # during loading rather than on the first generation.
        voice = os.path.join(root, "voices", "af_heart.pt")
        if os.path.exists(voice):
            for _result in pipeline("a")("Ready.", voice=voice):
                pass

    def voices():
        names = [os.path.splitext(os.path.basename(path))[0]
                 for path in glob.glob(os.path.join(state["root"], "voices", "*.pt"))]
        return sorted(names)

    def pipeline(lang_code):
        if lang_code not in state["pipelines"]:
            state["pipelines"][lang_code] = KPipeline(
                lang_code=lang_code, repo_id=state["model_id"], model=state["model"])
        return state["pipelines"][lang_code]

    def generate(req):
        if state["model"] is None:
            raise RuntimeError("No Kokoro model is loaded.")
        voice = req["voice"]
        voice_path = os.path.join(state["root"], "voices", f"{voice}.pt")
        if not os.path.exists(voice_path):
            raise ValueError(f"Voice {voice} isn't in {state['model_id']}.")
        speaker = pipeline(voice[0])
        seconds = []
        for text, path in zip(req["texts"], req["out_paths"]):
            chunks = [result.audio.detach().cpu().numpy() for result in
                      speaker(text, voice=voice_path, speed=1.0) if result.audio is not None]
            wav = np.concatenate(chunks) if chunks else np.zeros(2400, dtype=np.float32)
            sf.write(path, wav.astype(np.float32), 24000, subtype="FLOAT")
            seconds.append(round(len(wav) / 24000, 2))
        return {"paths": req["out_paths"], "sr": 24000, "seconds": seconds}

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
                reply(ok=True, voices=voices())
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

"""Speech-to-text with OpenAI's Whisper large-v3-turbo (MIT license).

Runs in the main environment through transformers, on the GPU when there is one
(about 1.7 GB of GPU memory; 20 s of speech in about 1 s on an RTX 5070 Ti once
warm). The model downloads (about 1.6 GB) the first time it's used.
"""

import threading
from dataclasses import dataclass, field

MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_LABEL = "Whisper large-v3 turbo"
DOWNLOAD_BYTES = 1_620_000_000
LICENSE = "mit"
SAMPLE_RATE = 16000
AUDIO_FILTER = "Audio files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.opus *.wma);;All files (*.*)"

# Whisper language names for the app's language codes ("auto" lets Whisper detect it).
LANGUAGES = {
    "auto": "Detect automatically",
    "ar": "Arabic", "bg": "Bulgarian", "ca": "Catalan", "zh": "Chinese", "hr": "Croatian",
    "cs": "Czech", "da": "Danish", "nl": "Dutch", "en": "English", "et": "Estonian",
    "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian", "ms": "Malay", "no": "Norwegian",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sw": "Swahili",
    "sv": "Swedish", "tl": "Tagalog", "ta": "Tamil", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "cy": "Welsh",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: list = field(default_factory=list)
    language: str = "auto"
    task: str = "transcribe"
    seconds: float = 0.0


def is_downloaded():
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(MODEL_ID, "model.safetensors")
        return isinstance(cached, str)
    except Exception:
        return False


def load_audio(path):
    """Mono float32 at 16 kHz. librosa reads WAV/FLAC/OGG/MP3 directly and other
    formats (M4A, AAC...) through FFmpeg when it's installed."""
    import librosa
    audio, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


class Transcriber:
    """Loads Whisper on first use and keeps it; one transcription at a time."""

    def __init__(self):
        self._pipe = None
        self._lock = threading.Lock()
        self.device = None

    @property
    def loaded(self):
        return self._pipe is not None

    def _load(self):
        import torch
        from transformers import pipeline
        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        self._pipe = pipeline("automatic-speech-recognition", model=MODEL_ID,
                              dtype=torch.float16 if cuda else torch.float32,
                              device="cuda:0" if cuda else "cpu")

    def transcribe(self, source, language="auto", task="transcribe"):
        """Transcribe a file path (or 16 kHz mono samples). The turbo model wasn't
        trained to translate, so the app always uses task "transcribe"."""
        with self._lock:
            if self._pipe is None:
                self._load()
            audio = load_audio(source) if isinstance(source, str) else source
            seconds = len(audio) / SAMPLE_RATE
            options = {"task": "translate" if task == "translate" else "transcribe"}
            if language and language != "auto":
                options["language"] = LANGUAGES.get(language, language).lower()
            result = self._pipe({"raw": audio, "sampling_rate": SAMPLE_RATE},
                                return_timestamps=True, generate_kwargs=options)
        segments = []
        for chunk in result.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            text = (chunk.get("text") or "").strip()
            if text and start is not None:
                segments.append(Segment(float(start), float(end if end is not None else seconds), text))
        return Transcript(text=(result.get("text") or "").strip(), segments=segments,
                          language=language or "auto", task=options["task"], seconds=seconds)

    def unload(self):
        with self._lock:
            self._pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def to_cues(transcript):
    """Subtitle cues from a transcript's timed segments."""
    from this_voice_thing.core import subtitles
    return [subtitles.Cue(segment.start, max(segment.end, segment.start + 0.3), segment.text)
            for segment in transcript.segments]


def timestamped_text(transcript):
    """"[0:05] Welcome back..." lines, for reading or pasting."""
    lines = []
    for segment in transcript.segments:
        minutes, seconds = divmod(int(segment.start), 60)
        lines.append(f"[{minutes}:{seconds:02d}] {segment.text}")
    return "\n".join(lines)

"""Post-generation audio finishing: speed, pitch, silence trim, loudness, export.

Speed and pitch use FFmpeg's Rubber Band filter when available (with formant
preservation, so pitch changes keep the voice's character), falling back to
librosa's phase vocoder otherwise.
"""

from dataclasses import asdict, dataclass
import shutil
import subprocess

import numpy as np
import soundfile as sf

OUTPUT_FORMATS = {
    # label: (extension, soundfile format, soundfile subtype)
    "WAV": ("wav", "WAV", "PCM_16"),
    "FLAC": ("flac", "FLAC", "PCM_16"),
    "MP3": ("mp3", "MP3", "MPEG_LAYER_III"),
}
AUDIO_EXTENSIONS = tuple("." + ext for ext, _fmt, _sub in OUTPUT_FORMATS.values())

SPEED_RANGE = (0.75, 1.25)
PITCH_RANGE = (-4.0, 4.0)  # semitones
PAUSE_RANGE = (0.0, 1.5)   # seconds between generated sections
TARGET_RMS_DBFS = -20.0
PEAK_CEILING_DBFS = -1.0
TRIM_TOP_DB = 40.0
TRIM_PADDING_SECONDS = 0.08


@dataclass
class FinishingSettings:
    speed: float = 1.0
    pitch_semitones: float = 0.0
    section_pause: float = 0.0
    even_volume: bool = True
    trim_silence: bool = True
    output_format: str = "WAV"

    @classmethod
    def from_dict(cls, payload):
        settings = cls()
        if isinstance(payload, dict):
            for key, value in payload.items():
                if hasattr(settings, key):
                    setattr(settings, key, type(getattr(settings, key))(value))
        settings.speed = float(np.clip(settings.speed, *SPEED_RANGE))
        settings.pitch_semitones = float(np.clip(settings.pitch_semitones, *PITCH_RANGE))
        settings.section_pause = float(np.clip(settings.section_pause, *PAUSE_RANGE))
        if settings.output_format not in OUTPUT_FORMATS:
            settings.output_format = "WAV"
        return settings

    def to_dict(self):
        return asdict(self)

    def summary(self):
        parts = []
        if abs(self.speed - 1.0) > 1e-6:
            parts.append(f"speed {self.speed:.2f}x")
        if abs(self.pitch_semitones) > 1e-6:
            parts.append(f"pitch {self.pitch_semitones:+g} st")
        if self.section_pause > 0:
            parts.append(f"{self.section_pause:.1f}s pauses")
        if self.even_volume:
            parts.append("even volume")
        if self.trim_silence:
            parts.append("trimmed")
        parts.append(self.output_format)
        return ", ".join(parts)


def _ffmpeg_has_rubberband():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return ffmpeg if " rubberband " in result.stdout else None


_RUBBERBAND_FFMPEG = None
_RUBBERBAND_CHECKED = False


def _rubberband_ffmpeg():
    global _RUBBERBAND_FFMPEG, _RUBBERBAND_CHECKED
    if not _RUBBERBAND_CHECKED:
        _RUBBERBAND_FFMPEG = _ffmpeg_has_rubberband()
        _RUBBERBAND_CHECKED = True
    return _RUBBERBAND_FFMPEG


def _stretch_and_shift(wav, sr, speed, semitones, log):
    ffmpeg = _rubberband_ffmpeg()
    if ffmpeg:
        pitch_scale = 2.0 ** (semitones / 12.0)
        audio_filter = (
            f"rubberband=tempo={speed:.4f}:pitch={pitch_scale:.6f}"
            ":formant=preserved:pitchq=quality:window=standard")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-af", audio_filter,
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1",
        ]
        try:
            result = subprocess.run(
                command, input=wav.astype("<f4").tobytes(), capture_output=True,
                timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode == 0 and result.stdout:
                return np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
            log(f"Rubber Band (FFmpeg) failed, using librosa instead: "
                f"{result.stderr.decode(errors='replace').strip()[:300]}")
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"Rubber Band (FFmpeg) failed, using librosa instead: {exc}")
    else:
        log("FFmpeg with Rubber Band not found; using librosa for speed/pitch "
            "(lower quality).")

    import librosa
    if abs(semitones) > 1e-6:
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=semitones)
    if abs(speed - 1.0) > 1e-6:
        wav = librosa.effects.time_stretch(wav, rate=speed)
    return wav.astype(np.float32)


def _trim_silence(wav, sr):
    import librosa
    _trimmed, (start, end) = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    pad = int(TRIM_PADDING_SECONDS * sr)
    return wav[max(0, start - pad):min(len(wav), end + pad)]


def _even_volume(wav):
    # Measure loudness over the voiced parts only, so pauses don't skew it.
    frame = 1024
    usable = len(wav) - len(wav) % frame
    if usable == 0:
        return wav
    frames = wav[:usable].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    voiced = frame_rms[frame_rms > frame_rms.max() * 0.1]
    if voiced.size == 0 or voiced.mean() <= 0:
        return wav
    gain = 10 ** (TARGET_RMS_DBFS / 20) / float(np.sqrt(np.mean(voiced ** 2)))
    peak = float(np.max(np.abs(wav))) * gain
    ceiling = 10 ** (PEAK_CEILING_DBFS / 20)
    if peak > ceiling:
        gain *= ceiling / peak
    return (wav * gain).astype(np.float32)


def join_sections(sections, sr, pause_seconds):
    """Concatenate mono float sections with optional silence between them."""
    gap = np.zeros(int(round(pause_seconds * sr)), dtype=np.float32)
    joined = []
    for index, section in enumerate(sections):
        if index and gap.size:
            joined.append(gap)
        joined.append(np.asarray(section, dtype=np.float32).reshape(-1))
    return np.concatenate(joined) if joined else np.zeros(0, dtype=np.float32)


def apply_finishing(wav, sr, settings, log=print):
    """Apply speed/pitch, silence trim and loudness to a mono float waveform."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if abs(settings.speed - 1.0) > 1e-6 or abs(settings.pitch_semitones) > 1e-6:
        wav = _stretch_and_shift(wav, sr, settings.speed, settings.pitch_semitones, log)
    if settings.trim_silence:
        wav = _trim_silence(wav, sr)
    if settings.even_volume:
        wav = _even_volume(wav)
    return np.clip(wav, -1.0, 1.0)


def save_audio(path_without_extension, wav, sr, output_format):
    extension, file_format, subtype = OUTPUT_FORMATS.get(output_format, OUTPUT_FORMATS["WAV"])
    path = f"{path_without_extension}.{extension}"
    sf.write(path, np.asarray(wav, dtype=np.float32), sr, format=file_format, subtype=subtype)
    return path

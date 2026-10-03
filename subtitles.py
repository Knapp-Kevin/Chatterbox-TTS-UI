"""Subtitles (SRT / WebVTT) from the app's own generation timeline.

No speech recognition is needed: the app knows which text went into each
section and where that section sits in the final audio. Within a section,
captions are split at sentence and clause breaks, timed by length, then
snapped to the pauses actually found in that section's audio, so a caption
changes where the speaker breathes.
"""

import re
from dataclasses import dataclass

import nltk
import numpy as np

import documents

FORMATS = {"SRT": "srt", "WebVTT": "vtt"}
LINE_CHARS = 42            # per line; two lines per caption
CAPTION_CHARS = 2 * LINE_CHARS
MIN_CAPTION_CHARS = 18     # shorter pieces merge with a neighbour
PAUSE_FRAME = 0.02         # seconds
MIN_PAUSE = 0.12           # seconds of near-silence that count as a pause
PAUSE_LEVEL = 0.06         # fraction of the section's loudest frame
SNAP_RANGE = 0.8           # seconds a caption boundary may move to reach a pause
# A caption shouldn't end on these (they belong with the next word)...
TRAILING_WORDS = {"a", "an", "the", "at", "of", "to", "in", "on", "for", "with", "from", "by", "into",
                  "and", "or", "but", "my", "your", "his", "her", "its", "our", "their", "this", "that",
                  "is", "was", "be", "very", "so", "as"}
# ...and these make a natural start for the next caption.
LEADING_WORDS = {"and", "but", "or", "so", "because", "which", "who", "whether", "when", "while",
                 "where", "that", "though", "although", "until", "unless", "if", "then"}


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str = ""


def _split_long(text, limit):
    """Split text over limit at clause breaks, then at word boundaries."""
    if len(text) <= limit:
        return [text]
    for pattern in documents.CLAUSE_BREAKS:
        cuts = [match.end() for match in pattern.finditer(text) if 0 < match.end() < len(text)]
        if cuts:
            middle = min(cuts, key=lambda cut: abs(cut - len(text) / 2))
            left, right = text[:middle].strip(), text[middle:].strip()
            if left and right and len(left) >= MIN_CAPTION_CHARS and len(right) >= MIN_CAPTION_CHARS:
                return _split_long(left, limit) + _split_long(right, limit)
    # No punctuation to use: split between words near the middle, never leaving a
    # caption ending on "at", "the", "of"..., and preferring to start the next one
    # with a joining word like "and" or "which".
    words = text.split()
    best, best_score = 1, None
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        score = abs(len(left) - len(text) / 2)
        if words[index - 1].lower().strip(",") in TRAILING_WORDS:
            score += 25
        if words[index].lower() in LEADING_WORDS:
            score -= 8
        if best_score is None or score < best_score:
            best, best_score = index, score
    return _split_long(" ".join(words[:best]), limit) + _split_long(" ".join(words[best:]), limit)


def caption_chunks(text, limit=CAPTION_CHARS):
    """Caption-sized pieces of text: whole sentences where they fit."""
    pieces = []
    for sentence in (s.strip() for s in nltk.sent_tokenize(text.replace("\n", " "))):
        if sentence:
            pieces.extend(_split_long(sentence, limit))
    merged = []
    for piece in pieces:
        if merged and (len(piece) < MIN_CAPTION_CHARS or len(merged[-1]) < MIN_CAPTION_CHARS) \
                and len(merged[-1]) + 1 + len(piece) <= limit:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def wrap(text, width=LINE_CHARS):
    """One line, or two balanced lines."""
    if len(text) <= width:
        return text
    words = text.split()
    best, best_score = text, None
    for index in range(1, len(words)):
        first, second = " ".join(words[:index]), " ".join(words[index:])
        score = max(len(first), len(second))
        if words[index - 1].lower().strip(",;:") in TRAILING_WORDS:
            score += 12  # don't end a line on "a", "the", "of"...
        if best_score is None or score < best_score:
            best, best_score = f"{first}\n{second}", score
    return best


def find_pauses(wav, sr):
    """Midpoints (seconds from the start of wav) of pauses in speech."""
    frame = max(1, int(PAUSE_FRAME * sr))
    usable = len(wav) - len(wav) % frame
    if usable <= frame:
        return []
    rms = np.sqrt(np.mean(np.asarray(wav[:usable], dtype=np.float32).reshape(-1, frame) ** 2, axis=1))
    quiet = rms < rms.max() * PAUSE_LEVEL
    pauses, run_start = [], None
    for index, is_quiet in enumerate(np.append(quiet, False)):
        if is_quiet and run_start is None:
            run_start = index
        elif not is_quiet and run_start is not None:
            if (index - run_start) * PAUSE_FRAME >= MIN_PAUSE and run_start > 0 and index < len(quiet):
                pauses.append((run_start + index) / 2 * PAUSE_FRAME)
            run_start = None
    return pauses


def _section_pieces(text):
    """[(speaker, caption text)] for a section; conversation sections keep speakers."""
    lines = text.splitlines()
    if lines and all(documents.SPEAKER_LINE.match(line) for line in lines if line.strip()):
        return [(speaker, chunk) for speaker, words in documents.parse_script(text)
                for chunk in caption_chunks(words)]
    return [("", chunk) for chunk in caption_chunks(text)]


def section_cues(text, start, end, section_wav, sr):
    """Cues for one section spanning [start, end] seconds in the joined audio."""
    pieces = _section_pieces(text)
    if not pieces or end <= start:
        return []
    weights = [len(piece) + 4 for _speaker, piece in pieces]  # +4: every caption has a beat
    total = float(sum(weights))
    duration = end - start
    boundaries, elapsed = [], 0.0
    for weight in weights[:-1]:
        elapsed += weight
        boundaries.append(duration * elapsed / total)
    pauses = find_pauses(section_wav, sr)
    snapped, previous = [], 0.0
    for index, boundary in enumerate(boundaries):
        nearby = [pause for pause in pauses if abs(pause - boundary) <= SNAP_RANGE and pause > previous + 0.3]
        if nearby:
            boundary = min(nearby, key=lambda pause: abs(pause - boundary))
        remaining = len(boundaries) - index
        boundary = min(max(boundary, previous + 0.3), duration - 0.3 * remaining)
        snapped.append(boundary)
        previous = boundary
    edges = [0.0] + snapped + [duration]
    return [Cue(start + edges[index], start + edges[index + 1], piece, speaker)
            for index, (speaker, piece) in enumerate(pieces)]


def build_cues(section_texts, spans, joined_wav, sr, scale=1.0, offset=0.0, length=None):
    """Cues for the whole render. spans[i] = (start, end) samples of section i in the
    joined audio. Finishing can stretch the timeline (scale) and trim its start
    (offset seconds); length clamps to the final audio's duration."""
    cues = []
    for text, (begin, finish) in zip(section_texts, spans):
        for cue in section_cues(text, begin / sr, finish / sr, joined_wav[begin:finish], sr):
            cue.start = max(0.0, cue.start * scale - offset)
            cue.end = max(cue.start, cue.end * scale - offset)
            if length is not None:
                cue.end = min(cue.end, length)
            if cue.end - cue.start >= 0.05:
                cues.append(cue)
    return cues


def _stamp(seconds, separator):
    milliseconds = int(round(seconds * 1000))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_srt(cues):
    blocks = []
    for number, cue in enumerate(cues, 1):
        text = wrap(f"{cue.speaker}: {cue.text}" if cue.speaker else cue.text)
        blocks.append(f"{number}\n{_stamp(cue.start, ',')} --> {_stamp(cue.end, ',')}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(cues):
    blocks = ["WEBVTT\n"]
    for cue in cues:
        text = wrap(cue.text)
        if cue.speaker:
            speaker = re.sub(r"[<>&]", "", cue.speaker)
            text = f"<v {speaker}>{text}"
        blocks.append(f"{_stamp(cue.start, '.')} --> {_stamp(cue.end, '.')}\n{text}\n")
    return "\n".join(blocks)


def save(path_without_extension, cues, subtitle_format):
    extension = FORMATS.get(subtitle_format, "srt")
    path = f"{path_without_extension}.{extension}"
    content = to_vtt(cues) if extension == "vtt" else to_srt(cues)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return path

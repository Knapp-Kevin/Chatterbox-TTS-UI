"""Document loading and section splitting for long-form generation."""

import os
from dataclasses import dataclass
import re
import zipfile
from xml.etree import ElementTree

import nltk

MAX_SECTION_LENGTH = 280
DOCUMENT_FILTER = "Documents (*.txt *.md *.markdown *.docx);;All files (*.*)"
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# What follows a section; decides the pause inserted at that seam.
CLAUSE, SENTENCE, PARAGRAPH, HEADING, END = "clause", "sentence", "paragraph", "heading", "end"
HEADING_MAX_CHARS = 80
# Where an overlong sentence may be split, strongest break first.
CLAUSE_BREAKS = (re.compile(r"[;:]\s+"), re.compile(r"\s[—–]\s|—|\s-\s"), re.compile(r",\s+"))


@dataclass
class Section:
    text: str
    boundary: str  # the kind of seam after this section


def split_paragraphs(text):
    """Paragraphs separated by blank lines. Text without any blank lines (Word
    documents, pasted paragraphs) treats each line as a paragraph; within a
    blank-line block, hard-wrapped lines are joined."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if re.search(r"\n\s*\n", text):
        paragraphs = []
        for block in re.split(r"\n\s*\n", text):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            # A heading line directly above its paragraph stays separate.
            if len(lines) > 1 and is_heading(lines[0], True) and not lines[1][:1].islower():
                paragraphs.append(lines.pop(0))
            if lines:
                paragraphs.append(" ".join(lines))
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_heading(paragraph, has_following):
    stripped = paragraph.strip()
    return (has_following and len(stripped) <= HEADING_MAX_CHARS
            and not re.search(r"[.!?…:;,\"'”)\]]$", stripped)
            and len(nltk.sent_tokenize(stripped)) == 1)


def _split_long_sentence(sentence, max_len):
    """Split at the clause break nearest the middle (recursively); fall back to
    the word boundary nearest the middle only when there is no punctuation."""
    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]
    middle = len(sentence) / 2
    for pattern in CLAUSE_BREAKS:
        cuts = [match.end() for match in pattern.finditer(sentence)
                if 0 < match.end() < len(sentence)]
        cuts = [cut for cut in cuts if min(cut, len(sentence) - cut) >= len(sentence) * 0.2]
        if cuts:
            cut = min(cuts, key=lambda position: abs(position - middle))
            return (_split_long_sentence(sentence[:cut], max_len)
                    + _split_long_sentence(sentence[cut:], max_len))
    spaces = [index for index, char in enumerate(sentence) if char == " "]
    if not spaces:
        return [sentence[i:i + max_len] for i in range(0, len(sentence), max_len)]
    cut = min(spaces, key=lambda position: abs(position - middle))
    return (_split_long_sentence(sentence[:cut], max_len)
            + _split_long_sentence(sentence[cut + 1:], max_len))


def plan_sections(text, max_len=MAX_SECTION_LENGTH):
    """Split text into sections no longer than max_len, with seams placed where a
    reader would pause anyway: whole sentences grouped within a paragraph, never
    across paragraphs or headings, and overlong sentences split at clause breaks."""
    paragraphs = split_paragraphs(text)
    sections = []
    for index, paragraph in enumerate(paragraphs):
        has_following = index < len(paragraphs) - 1
        if is_heading(paragraph, has_following):
            sections.append(Section(paragraph, HEADING))
            continue
        pieces = []  # (text, seam after it)
        group, group_len = [], 0
        for sentence in (s.strip() for s in nltk.sent_tokenize(paragraph)):
            if not sentence:
                continue
            if len(sentence) > max_len:
                if group:
                    pieces.append((" ".join(group), SENTENCE))
                    group, group_len = [], 0
                parts = _split_long_sentence(sentence, max_len)
                pieces.extend((part, CLAUSE) for part in parts[:-1])
                pieces.append((parts[-1], SENTENCE))
                continue
            if group and group_len + 1 + len(sentence) > max_len:
                pieces.append((" ".join(group), SENTENCE))
                group, group_len = [], 0
            group.append(sentence)
            group_len += len(sentence) + (1 if group_len else 0)
        if group:
            pieces.append((" ".join(group), SENTENCE))
        if pieces:
            pieces[-1] = (pieces[-1][0], PARAGRAPH)
        sections.extend(Section(piece, seam) for piece, seam in pieces if piece.strip())
    if sections:
        sections[-1].boundary = END
    return sections


def split_into_sections(text, max_len=MAX_SECTION_LENGTH):
    return [section.text for section in plan_sections(text, max_len)]


# ---------- conversation scripts ----------

# "Linda: Hello there." or "[Linda]: Hello there." A name is up to 30 characters,
# starts with a letter, and holds no sentence punctuation. The colon must be
# followed by a space, so times like "10:30" aren't mistaken for speakers.
SPEAKER_LINE = re.compile(r"^\s*\[?(?P<name>[^\W\d_][\w .'\-]{0,29}?)\]?\s*:\s+(?P<text>\S.*)$")
DEFAULT_SPEAKER = "Speaker 1"


def parse_script(text):
    """[(speaker, text)] turns. A "Name:" line starts a turn; other lines continue the
    current turn; text before any name belongs to "Speaker 1"."""
    turns = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        match = SPEAKER_LINE.match(line)
        if match:
            turns.append([match.group("name").strip(), match.group("text").strip()])
        elif turns:
            turns[-1][1] += " " + line
        else:
            turns.append([DEFAULT_SPEAKER, line])
    return [(speaker, words) for speaker, words in turns if words]


def script_speakers(turns):
    """Speaker names in order of first appearance."""
    names = []
    for speaker, _words in turns:
        if speaker not in names:
            names.append(speaker)
    return names


def plan_script_sections(text, max_len):
    """Sections of whole turns ("Name: text" lines) up to max_len characters, so a
    conversation is only ever split between turns. A turn longer than max_len is
    split at sentence breaks, each part keeping its speaker."""
    lines = []
    for speaker, words in parse_script(text):
        budget = max(40, max_len - len(speaker) - 2)
        parts, group = [], ""
        for sentence in (s.strip() for s in nltk.sent_tokenize(words)):
            for piece in ([sentence] if len(sentence) <= budget else _split_long_sentence(sentence, budget)):
                if group and len(group) + 1 + len(piece) > budget:
                    parts.append(group)
                    group = ""
                group = f"{group} {piece}".strip()
        if group:
            parts.append(group)
        lines.extend(f"{speaker}: {part}" for part in parts)
    sections, current = [], []
    for line in lines:
        if current and sum(len(item) + 1 for item in current) + len(line) > max_len:
            sections.append(Section("\n".join(current), PARAGRAPH))
            current = []
        current.append(line)
    if current:
        sections.append(Section("\n".join(current), END))
    return sections


DANGLING_WORDS = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
                  "from", "by", "that", "which", "who", "whether", "is", "was", "as", "so", "if"}


def excerpt(text, max_chars):
    """The opening of text, about max_chars long: whole sentences when they fit, else
    the first sentence cut at a clause break or between words. A conversation script
    keeps its "Name:" prefix."""
    text = text.strip()
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    match = SPEAKER_LINE.match(first_line)
    if match:
        turns = parse_script(text)
        if turns:
            speaker, words = turns[0]
            return f"{speaker}: {excerpt(words, max(20, max_chars - len(speaker) - 2))}"
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return ""
    sentences = []
    for index, paragraph in enumerate(paragraphs):
        if is_heading(paragraph, index < len(paragraphs) - 1) and index < len(paragraphs) - 1:
            continue  # a heading says little about the voice: preview the prose after it
        else:
            sentences.extend(s.strip() for s in nltk.sent_tokenize(paragraph) if s.strip())
        if sum(len(item) + 1 for item in sentences) > max_chars * 2:
            break
    taken = ""
    for sentence in sentences:
        candidate = f"{taken} {sentence}".strip()
        if taken and len(candidate) > max_chars:
            break
        taken = candidate
        if len(taken) >= max_chars:
            break
    if len(taken) <= max_chars * 1.3:
        return taken
    # One long sentence: stop at the last clause break, else word boundary, before the limit.
    limit = int(max_chars * 1.15)
    for pattern in CLAUSE_BREAKS:
        cuts = [m.start() for m in pattern.finditer(taken[:limit]) if m.start() >= max_chars * 0.5]
        if cuts:
            return taken[:cuts[-1]].rstrip(" ,;:—–-") + "."
    words = taken[:limit].split()[:-1] or taken[:limit].split()
    while len(words) > 1 and words[-1].lower().strip(",;:") in DANGLING_WORDS:
        words.pop()  # don't end on "and", "the", "of"...
    return " ".join(words).rstrip(" ,;:") + "."


def plan_batches(lengths, max_count, char_budget=None):
    """Group consecutive sections into batches of at most max_count sections, and,
    when char_budget is set, at most char_budget characters counted as
    sections x longest section (GPU memory grows with both). Returns
    [(start, end)] index ranges, end exclusive."""
    batches, start = [], 0
    while start < len(lengths):
        end = start + 1
        while end < len(lengths) and end - start < max_count:
            longest = max(lengths[start:end + 1])
            if char_budget and (end + 1 - start) * longest > char_budget:
                break
            end += 1
        batches.append((start, end))
        start = end
    return batches


def _read_text_file(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _markdown_to_text(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)        # code blocks
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)               # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # links -> label
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)   # headings
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)       # quotes
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)       # bullets
    text = re.sub(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|\*|_|`)(.+?)\1", r"\2", text)         # emphasis, code
    return text


def _docx_to_text(path):
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(f"{WORD_NAMESPACE}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{WORD_NAMESPACE}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{WORD_NAMESPACE}tab":
                parts.append(" ")
            elif node.tag in (f"{WORD_NAMESPACE}br", f"{WORD_NAMESPACE}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def tidy_text(text):
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_document(path):
    """Return the readable text of a .txt, .md or .docx file."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".docx":
        text = _docx_to_text(path)
    else:
        text = _read_text_file(path)
        if extension in (".md", ".markdown"):
            text = _markdown_to_text(text)
    return tidy_text(text)


def safe_file_stem(name, max_len=40):
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem[:max_len] or "document"

"""Document loading and section splitting for long-form generation."""

import os
import re
import zipfile
from xml.etree import ElementTree

import nltk

MAX_SECTION_LENGTH = 280
LONG_SENTENCE_CHUNK_LENGTH = MAX_SECTION_LENGTH - 20
DOCUMENT_FILTER = "Documents (*.txt *.md *.markdown *.docx);;All files (*.*)"
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _chunk_long_sentence(sentence, max_len):
    sub_chunks = []
    current_pos = 0
    sentence_len = len(sentence)
    while current_pos < sentence_len:
        end_pos = min(current_pos + max_len, sentence_len)
        if end_pos == sentence_len:
            sub_chunks.append(sentence[current_pos:end_pos].strip())
            current_pos = end_pos
        else:
            last_space_idx = sentence.rfind(' ', current_pos, end_pos)
            if last_space_idx != -1 and last_space_idx > current_pos:
                sub_chunks.append(sentence[current_pos:last_space_idx].strip())
                current_pos = last_space_idx + 1
            else:
                sub_chunks.append(sentence[current_pos:end_pos].strip())
                current_pos = end_pos
    return [sc for sc in sub_chunks if sc]


def split_into_sections(text, max_len=MAX_SECTION_LENGTH):
    """Group whole sentences into sections of at most max_len characters;
    sentences longer than that are split at word boundaries."""
    sections = []
    current_sents = []
    current_len = 0
    for sentence in nltk.sent_tokenize(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_len:
            if current_sents:
                sections.append(" ".join(current_sents))
            current_sents = []
            current_len = 0
            sections.extend(_chunk_long_sentence(sentence, LONG_SENTENCE_CHUNK_LENGTH))
            continue
        potential_len = current_len + (1 if current_sents else 0) + len(sentence)
        if potential_len <= max_len:
            current_sents.append(sentence)
            current_len = potential_len
        else:
            if current_sents:
                sections.append(" ".join(current_sents))
            current_sents = [sentence]
            current_len = len(sentence)
    if current_sents:
        sections.append(" ".join(current_sents))
    return [s.strip() for s in sections if s.strip()]


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

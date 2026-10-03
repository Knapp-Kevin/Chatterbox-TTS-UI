# Changelog

All notable changes to this fork of [AcTePuKc/Chatterbox-TTS-UI](https://github.com/AcTePuKc/Chatterbox-TTS-UI) are listed here.

## [Unreleased] - 2026-10-02

Changes since upstream commit `22460fd` ("Add best-effort macOS/Linux shell launcher flow").

### Added
- **Qwen3-TTS engine (optional).** Three new models alongside Chatterbox:
  - **Preset voices**, with nine speakers and plain-language style instructions.
  - **Voice design**, which creates a voice from a written description.
  - **Voice cloning**, which can use a transcript of the clip for closer likeness.

  Qwen runs in its own environment, which the app offers to install. Its output can carry the same AI watermark as Chatterbox.
- **VibeVoice engine and Conversations (optional, research use).** A new **Conversations** tab. Write a script with up to 4 speakers (`Name: line`), pick a voice for each with **Cast…** (7 sample voices, your recordings or any clip), and VibeVoice performs it as one natural conversation. Long scripts split only between turns, and each speaker keeps their voice throughout. MIT-licensed but limited to research use by its model card, so it's badged amber **Research use**.
- **OmniVoice engine (optional, non-commercial).** Fast voice cloning and voice design in 600+ languages from a 3 GB model (CC BY-NC 4.0 weights, badged red). Cloning needs the clip transcript; voice design uses an **Attributes…** picker (gender, age, pitch, whisper, accent), with unsupported words caught before generating. Sections are batched, about 3 s for four paragraphs on an RTX 5070 Ti, and a designed voice stays the same through a document.
- **License badges look past wrong tags.** Fine-tunes of a non-commercial model inherit its license even when their own tag claims a permissive one.
- **VoxCPM2 engine (optional).** One 5 GB model for voice cloning (with an optional style, and a clip transcript for closer likeness) and voice design, in 30 languages at 48 kHz. The two uses share one download and switch instantly. A designed voice stays the same through a whole document. VoxCPM runs in its own environment, which the app offers to install, and can add the same AI watermark.
- **Kokoro engine (optional).** 49 built-in voices in 7 languages from a 340 MB model, about 20× faster than Chatterbox. Pick a language, then a voice; the choice is remembered per language. Kokoro runs in its own environment, which the app offers to install, and can add the same AI watermark.
- **GPU requirements on every model tile.** Each tile shows the GPU memory the model needs next to your GPU's, in green when it fits, amber when it runs but slower, and red when the GPU is too small.
- **Faster Qwen documents.** Long text is generated in batches (up to 16 sections or about 5,000 characters per call), roughly 5× faster. Oversized batches are split automatically if the GPU runs out of memory.
- **In-app voice recording.** A guided window with countdown, live level meter, too-quiet and clipping warnings, a 30-second limit, and phonetically rich read-aloud passages. Recordings save the passage as a transcript for Qwen cloning.
- **Document narration:**
  - Open `.txt`, `.md` or `.docx` files.
  - **Preview** a sample, and **Keep this take** so the full render matches.
  - A live estimate of the time and number of sections.
  - Progress with time remaining, and **Stop** keeps finished sections as a `_partial` file.
- **Per-model time estimates.** Every model's estimated time for the current text, learned from your own runs. The estimate is a menu: pick a model from it to switch.
- **Model management in the app:**
  - Add, edit, duplicate, hide and remove models.
  - **Check** a Hugging Face repo before downloading.
  - **Find models** searches Hugging Face for repos this app can load.
  - **Model page tabs and tiles.** Tabs for voice cloning, preset voices and voice design. Your models show as tiles with download status, typical speed and a **license badge** (permissive, non-commercial or unclear). Click a tile to load it.
  - **Discover on Hugging Face** lists more loadable models for each tab, searched in the background. Click one to add it.
- **Hugging Face access** section with token **Save** and **Test**. The token is stored locally, never in `models.json`.
- **Finishing touches:** paragraph pauses, even out volume, trim silence, and WAV/FLAC output.
- **Pronunciation dictionary.** Respell names, acronyms and jargon (`Nguyen → Win`, `SQL → sequel`) for every model, with whole-word and match-case options. **Hear it** and **Try** let you test respellings, and word lists can be imported or exported. Subtitles keep the original spelling; the Generate page shows how many words were respelled.
- **Subtitles.** Tick **Save subtitles** to get an `.srt` or `.vtt` file next to the audio. Captions are timed from the generated sections and snapped to the pauses in the speech, with no speech recognition needed. Speed changes and trimmed silence are accounted for, and conversation captions name their speaker.
- **Advanced page:** speed and pitch changes (formant-preserving) and MP3 export, with a note that they can weaken the AI watermark.
- **Voice library** (the Voice page). Save clip voices, preset voices (Kokoro, Qwen speakers) and designed voices (Qwen, VoxCPM, OmniVoice) by name, with tags and notes, and use any of them with a click. A preset or designed voice can **make a clip**: 15 s reading a phonetic passage with its exact transcript, so cloning models can reuse it. Existing recordings join the library without being moved, and VibeVoice's **Cast…** lists library voices by name.

### Changed
- **The shipped model list only holds models you can actually download.** The placeholder "Example custom…" entries and the "Legacy English compatibility" entry are gone; add an original-layout model with **+ Add repo…** if you need one. Each shipped model's download size is shown on its tile before you load it.
- **New interface:**
  - Sidebar pages (Generate, Voice, Model, Advanced, Log).
  - A light/dark theme that follows Windows, with depth and texture.
  - Plain-language controls: Expressiveness, Pacing, Variation, Take number.
- **Models are grouped by what they do:** cloning, presets, design and conversations, as tabs on the Model page.
- **Text is split where a reader would pause:**
  - Never across paragraphs or headings.
  - Overlong sentences split at clause breaks, not mid-phrase.
  - Qwen sections can hold whole paragraphs.
  - Pauses depend on the type of join.
- **The window sizes itself to its content,** so nothing needs scrolling.
- **Generated audio is saved as standard 16-bit WAV** (previously 32-bit float), with optional volume levelling and silence trimming.
- **Model settings** (repetition, min-p, top-p) are now remembered between sessions.

### Fixed
- **Kokoro's voice picker no longer widens the window** (its long voice names set the minimum width).
- **Windows installer:**
  - Batch files are now checked out with Windows line endings. Before, steps could run out of order and report success after a failure.
  - Newer NVIDIA drivers no longer cause a CPU-only PyTorch install.
  - A dependency URL conflict that newer `uv` versions reject is resolved.
- **Qwen works offline** once its models are downloaded.
- **A slider rounding error:** Expressiveness could go below its minimum (0.20 instead of 0.25).
- **Windows-1252 text files** now open correctly.
- **Cancelling the file browser** no longer clears the selected voice.

"""Transcription helpers and the local API's speech-to-text endpoint (no model needed)."""

import json
import os
import sys
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from this_voice_thing.core import transcription  # noqa: E402
from this_voice_thing.integrations import local_api  # noqa: E402

SAMPLE = transcription.Transcript(
    text="Hello there. General Kenobi.",
    segments=[transcription.Segment(0.0, 1.5, "Hello there."),
              transcription.Segment(65.2, 65.2, "General Kenobi.")],
    language="en", task="transcribe", seconds=66.0)


class HelperTests(unittest.TestCase):
    def test_timestamped_text(self):
        self.assertEqual(transcription.timestamped_text(SAMPLE), "[0:00] Hello there.\n[1:05] General Kenobi.")

    def test_cues_have_a_minimum_length(self):
        cues = transcription.to_cues(SAMPLE)
        self.assertEqual([cue.text for cue in cues], ["Hello there.", "General Kenobi."])
        self.assertGreater(cues[1].end, cues[1].start)

    def test_languages_include_auto(self):
        self.assertIn("auto", transcription.LANGUAGES)
        self.assertEqual(transcription.LANGUAGES["es"], "Spanish")


class FormatTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(local_api.format_transcript(SAMPLE, "json")[1], {"text": SAMPLE.text})
        self.assertEqual(local_api.format_transcript(SAMPLE, "text")[1], SAMPLE.text + "\n")
        srt = local_api.format_transcript(SAMPLE, "srt")[1]
        self.assertIn("00:01:05,200 --> ", srt)
        vtt = local_api.format_transcript(SAMPLE, "vtt")
        self.assertTrue(vtt[1].startswith("WEBVTT"))
        self.assertTrue(vtt[2].startswith("text/vtt"))
        verbose = local_api.format_transcript(SAMPLE, "verbose_json")[1]
        self.assertEqual(verbose["duration"], 66.0)
        self.assertEqual(len(verbose["segments"]), 2)

    def test_parse_multipart(self):
        boundary = "XyZ"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nes\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                f"Content-Type: audio/wav\r\n\r\n").encode() + b"RIFF\x00\x01\r\n\xff" + f"\r\n--{boundary}--\r\n".encode()
        fields, files = local_api.parse_multipart(f"multipart/form-data; boundary={boundary}", body)
        self.assertEqual(fields, {"language": "es"})
        self.assertEqual(files["file"], ("a.wav", b"RIFF\x00\x01\r\n\xff"))


class FakeBackend:
    def __init__(self):
        self.calls = []

    def health(self):
        return {"ok": True}

    def transcribe(self, audio, filename, language):
        self.calls.append((audio, filename, language))
        return SAMPLE


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = FakeBackend()
        cls.server = local_api.LocalApiServer(cls.backend, port=0, log=lambda *_: None)
        cls.server.start()
        cls.server.port = cls.server.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def post(self, path, fields, audio=b"RIFFdata"):
        boundary = "b0undary"
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                 for k, v in fields.items()]
        if audio is not None:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
                         f"Content-Type: audio/wav\r\n\r\n".encode() + audio + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(self.server.url + path, data=body, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return urllib.request.urlopen(request, timeout=10)

    def test_transcriptions_json(self):
        with self.post("/v1/audio/transcriptions", {"language": "EN"}) as response:
            self.assertEqual(json.loads(response.read()), {"text": SAMPLE.text})
        self.assertEqual(self.backend.calls[-1], (b"RIFFdata", "clip.wav", "en"))

    def test_transcriptions_srt(self):
        with self.post("/v1/audio/transcriptions", {"response_format": "srt"}) as response:
            self.assertIn("General Kenobi.", response.read().decode())
        self.assertEqual(self.backend.calls[-1][2], "auto")

    def test_missing_file_and_bad_format(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/v1/audio/transcriptions", {}, audio=None)
        self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/v1/audio/transcriptions", {"response_format": "docx"})
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()

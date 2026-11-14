"""Self-check of the engine chain against a fake qwen3-tts. Needs ffmpeg on PATH.
python test_tts.py"""
import io
import json
import os
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient

health = {"default_voice_ready": True}
spoken = []


def silence() -> bytes:
    """A quarter second of 24 kHz mono silence, the format qwen3-tts answers with."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0\0" * 6000)
    return buf.getvalue()


class Qwen(BaseHTTPRequestHandler):
    """Fake qwen3-tts: /health and /v1/audio/speak."""

    def do_GET(self):
        """Health answer as configured by the test."""
        self.answer(json.dumps(health).encode(), "application/json")

    def do_POST(self):
        """Record the request and answer with silence, after a delay if one is set."""
        spoken.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        time.sleep(health.get("delay", 0))
        self.answer(silence(), "audio/wav", {"X-Voice": spoken[-1]["voice"]})

    def answer(self, body, ctype, extra=()):
        """Write one response."""
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in dict(extra).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the output quiet."""


server = HTTPServer(("127.0.0.1", 0), Qwen)
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["QWEN_URL"] = f"http://127.0.0.1:{server.server_port}/"
os.environ["TTS_CHAIN"] = "qwen,azure,edge"

from app import main  # noqa: E402  (reads the environment on import)

azure_got, edge_got = [], []


async def fake_azure(req):
    """Stand-in for Azure that fails until told otherwise."""
    azure_got.append(req.voice)
    if fake_azure.down:
        raise RuntimeError("HTTP 401")
    return "mp3", req.voice or main.AZURE_VOICE, main.once(b"ID3azure")


async def fake_edge(req):
    """Stand-in for Edge."""
    edge_got.append(req.input)
    return "wav", main.EDGE_VOICE, main.once(silence())


async def late_failure(req):
    """An engine whose error only shows with the first chunk, like an Azure 429."""
    async def chunks():
        """Fails before the first byte."""
        raise RuntimeError("HTTP 429")
        yield b""
    return "mp3", "de-DE-SeraphinaMultilingualNeural", chunks()


async def mp3_chunks(req):
    """An engine that sends its mp3 in three pieces."""
    async def chunks():
        """Three pieces of one mp3."""
        for c in (b"ID3", b"-a", b"-b"):
            yield c
    return "mp3", main.EDGE_VOICE, chunks()


fake_azure.down = False
main.ENGINES.update(azure=fake_azure, edge=fake_edge)
api = TestClient(main.app)


def speak(**body):
    """POST /v1/audio/speech."""
    return api.post("/v1/audio/speech", json=body)


# --- qwen first: voice and device hint go through, WAV is re-encoded to Ogg/Opus
r = speak(input="Guten Morgen", voice="portal2", response_format="opus", device="gpu_if_free")
assert r.status_code == 200, r.text
assert r.headers["x-engine"] == "qwen" and r.headers["x-voice"] == "portal2", r.headers
assert r.content[:4] == b"OggS" and r.headers["content-type"] == "audio/ogg", r.content[:8]
assert spoken[-1] == {"text": "Guten Morgen", "language": "German", "voice": "portal2",
                      "device": "gpu_if_free"}, spoken

# --- qwen not ready: next engine, mp3 from Azure passes through untouched
health["default_voice_ready"] = False
r = speak(input="x")
assert r.headers["x-engine"] == "azure" and r.content == b"ID3azure", r.headers
health["default_voice_ready"] = True

# --- the caller's qwen_timeout replaces the length-based budget
health["delay"] = 1.5
r = speak(input="x", qwen_timeout=0.5)
assert r.headers["x-engine"] == "azure", r.headers
health["delay"] = 0
assert speak(input="x", qwen_timeout=0).status_code == 422

# --- a Microsoft voice never goes to qwen, Azure gets it as asked
before = len(spoken)
r = speak(input="x", voice="de-DE-ConradNeural")
assert r.headers["x-engine"] == "azure" and azure_got[-1] == "de-DE-ConradNeural"
assert len(spoken) == before

# --- explicit engine list keeps its order and skips qwen
fake_azure.down = True
r = speak(input="Ja", model="azure,edge")
assert r.headers["x-engine"] == "edge" and len(spoken) == before, r.headers

# --- spaces in the list are fine; a body ffmpeg cannot read moves on to the next engine
fake_azure.down = False
r = speak(input="x", model="azure, edge", response_format="opus")
assert r.status_code == 200 and r.headers["x-engine"] == "edge", r.text
fake_azure.down = True

# --- Edge: an empty first stream gets the second try
streams = [[], [{"type": "audio", "data": b"ID3"}]]


class FakeCommunicate:
    """edge_tts.Communicate replaying queued streams."""

    def __init__(self, *args, **kwargs):
        self.chunks = streams.pop(0)

    async def stream(self):
        """Yield the queued chunks."""
        for c in self.chunks:
            yield c


main.ENGINES["edge"] = main.edge
real_communicate, main.edge_tts.Communicate = main.edge_tts.Communicate, FakeCommunicate
r = speak(input="x", model="edge")
assert r.status_code == 200 and r.content == b"ID3" and not streams, r.text
main.edge_tts.Communicate = real_communicate
main.ENGINES["edge"] = fake_edge

# --- stream: an error with the first chunk moves on, the chosen engine's chunks pass through
main.ENGINES.update(azure=late_failure, edge=mp3_chunks)
with api.stream("POST", "/v1/audio/speech",
                json={"input": "x", "model": "azure,edge", "stream": True}) as r:
    assert r.status_code == 200 and r.headers["x-engine"] == "edge", r.headers
    assert b"".join(r.iter_bytes()) == b"ID3-a-b"
# without stream the same pieces come joined
r = speak(input="x", model="edge")
assert r.content == b"ID3-a-b" and "transfer-encoding" not in r.headers, r.headers
# stream with a format the engine does not produce: converted, in one piece
r = speak(input="x", model="qwen", stream=True)
assert r.headers["x-engine"] == "qwen" and r.content[:3] in (b"ID3", b"\xff\xf3\x64"), r.content[:4]
main.ENGINES.update(azure=fake_azure, edge=fake_edge)

# --- everything down: 503 naming each engine
main.ENGINES["edge"] = fake_azure
r = speak(input="x", model="azure,edge")
assert r.status_code == 503 and r.json()["detail"].count("HTTP 401") == 2, r.text
main.ENGINES["edge"] = fake_edge

# --- input checks
assert speak(input="x", model="whisper").status_code == 400
assert speak(input="").status_code == 422
assert speak(input="x", response_format="flac").status_code == 422

# --- speed and SSML
assert [main.rate(s) for s in (1.1, 0.9, 1.0)] == ["+10%", "-10%", "+0%"]
ssml = main.azure_ssml("AT&T <Q3>", "de-DE-SeraphinaMultilingualNeural", 1.1)
assert "AT&amp;T &lt;Q3&gt;" in ssml and "<lang xml:lang='de-DE'>" in ssml, ssml
assert "<lang" not in main.azure_ssml("Kalimera", "el-GR-AthinaNeural", 1.0)
assert "xml:lang='fil-PH'" in main.azure_ssml("x", "fil-PH-AngeloNeural", 1.0)
assert main.qwen_budget("a" * 10) == 240 and main.qwen_budget("a" * 9000) == 2400

server.shutdown()
print("ok")

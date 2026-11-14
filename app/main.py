"""Speech gateway: one OpenAI-compatible endpoint in front of Qwen3, Azure Speech and Edge."""
import asyncio
import logging
import os
import subprocess
from typing import Literal
from urllib.parse import quote
from xml.sax.saxutils import escape

import edge_tts
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

QWEN_URL = os.environ.get("QWEN_URL", "").rstrip("/")
QWEN_VOICE = os.environ.get("QWEN_VOICE", "random")
AZURE_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.environ.get("AZURE_SPEECH_REGION", "")
AZURE_VOICE = os.environ.get("AZURE_VOICE", "de-DE-SeraphinaMultilingualNeural")
# Edge dropped most German voices in November 2026 and answers unknown names with a
# default voice instead of an error, so only pick names from `edge-tts --list-voices`.
EDGE_VOICE = os.environ.get("EDGE_VOICE", "de-DE-TanjaNeural")
CHAIN = os.environ.get("TTS_CHAIN", "qwen,azure,edge").split(",")

# One Azure request returns at most 10 minutes of audio, roughly 9000 characters.
AZURE_MAX_CHARS = 9000
EDGE_SECS = 240
MEDIA = {"mp3": "audio/mpeg", "opus": "audio/ogg", "wav": "audio/wav"}
ENCODE = {"mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
          "opus": ["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"],
          "wav": ["-f", "wav"]}

log = logging.getLogger("uvicorn.error")
app = FastAPI()


class Speech(BaseModel):
    input: str = Field(min_length=1)
    model: str = "auto"
    voice: str | None = None
    response_format: Literal["mp3", "opus", "wav"] = "mp3"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    device: str | None = None  # qwen3-tts routing hint, e.g. "gpu_if_free"
    qwen_timeout: float | None = Field(None, gt=0, le=7200)  # overrides qwen_budget()


def microsoft_voice(voice: str | None) -> str | None:
    """The requested voice if it names an Azure/Edge voice, otherwise None."""
    return voice if voice and voice.endswith("Neural") else None


def rate(speed: float) -> str:
    """OpenAI speed factor as SSML/Edge rate, 1.1 -> "+10%"."""
    return f"{round((speed - 1) * 100):+d}%"


def qwen_budget(text: str) -> float:
    """Seconds to wait for qwen3-tts. It may fall back to CPU, where a long text takes 20 minutes."""
    return max(240, min(2400, 150 + 0.6 * len(text)))


async def qwen(req: Speech) -> tuple[bytes, str, str]:
    """Voice clone on the GPU box; the health check tells a stopped service apart quickly."""
    if not QWEN_URL:
        raise RuntimeError("QWEN_URL not set")
    if microsoft_voice(req.voice):
        raise RuntimeError(f"{req.voice} is not a qwen voice")
    body = {"text": req.input, "language": "German", "voice": req.voice or QWEN_VOICE}
    if req.device:
        body["device"] = req.device
    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{QWEN_URL}/health", timeout=6)).json()
        if not health.get("default_voice_ready"):
            raise RuntimeError(f"not ready: {health}")
        r = await client.post(f"{QWEN_URL}/v1/audio/speak", json=body,
                              timeout=req.qwen_timeout or qwen_budget(req.input))
        r.raise_for_status()
    return r.content, "wav", r.headers.get("x-voice", body["voice"])


def azure_ssml(text: str, voice: str, speed: float) -> str:
    """SSML for Azure. Multilingual voices guess the language per sentence and read
    short German sentences in English unless <lang> pins the voice's own locale."""
    locale = "-".join(voice.split("-")[:2])
    inner = f"<prosody rate='{rate(speed)}'>{escape(text)}</prosody>"
    if "Multilingual" in voice:
        inner = f"<lang xml:lang='{locale}'>{inner}</lang>"
    return f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice}'>{inner}</voice></speak>"


async def azure(req: Speech) -> tuple[bytes, str, str]:
    """Azure Speech REST API (free tier F0, 500k characters a month)."""
    if not (AZURE_KEY and AZURE_REGION):
        raise RuntimeError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION not set")
    if len(req.input) > AZURE_MAX_CHARS:
        raise RuntimeError(f"text too long for one request ({len(req.input)} characters)")
    voice = microsoft_voice(req.voice) or AZURE_VOICE
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            content=azure_ssml(req.input, voice, req.speed).encode(),
            headers={"Ocp-Apim-Subscription-Key": AZURE_KEY,
                     "Content-Type": "application/ssml+xml",
                     "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                     "User-Agent": "tts"},
            timeout=min(180, 30 + len(req.input) // 50))
        r.raise_for_status()
    return r.content, "mp3", voice


async def edge(req: Speech) -> tuple[bytes, str, str]:
    """Edge read-aloud service. Roughly every fifth request fails, so it gets a second try."""
    voice = microsoft_voice(req.voice) or EDGE_VOICE

    async def collect() -> bytes:
        """Concatenate the audio chunks of one synthesis."""
        out = bytearray()
        async for chunk in edge_tts.Communicate(req.input, voice, rate=rate(req.speed)).stream():
            if chunk["type"] == "audio":
                out += chunk["data"]
        return bytes(out)

    try:
        audio = await asyncio.wait_for(collect(), EDGE_SECS)
        if audio:
            return audio, "mp3", voice
        log.warning("edge attempt 1 returned no audio")
    except Exception as e:
        log.warning("edge attempt 1 failed: %s", e)
    return await asyncio.wait_for(collect(), EDGE_SECS), "mp3", voice


ENGINES = {"qwen": qwen, "azure": azure, "edge": edge}


def convert(audio: bytes, src: str, dst: str) -> bytes:
    """Re-encode through ffmpeg when the engine's format is not the one asked for."""
    if src == dst:
        return audio
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *ENCODE[dst], "pipe:1"],
                          input=audio, capture_output=True, check=True, timeout=120).stdout


@app.post("/v1/audio/speech")
async def speech(req: Speech):
    """Try the engines in order; model is "auto" (TTS_CHAIN), one engine or a comma list."""
    names = CHAIN if req.model in ("auto", "tts-1", "tts-1-hd") else [
        n.strip() for n in req.model.split(",")]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        raise HTTPException(400, f"unknown engine {unknown}, use auto or {sorted(ENGINES)}")
    errors = []
    for name in names:
        try:
            audio, fmt, voice = await ENGINES[name](req)
            if not audio:
                raise RuntimeError("empty audio")
            # A broken body from one engine shows up here; the next engine still gets its turn.
            audio = await asyncio.to_thread(convert, audio, fmt, req.response_format)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log.warning("tts %s", errors[-1])
            continue
        log.info("tts %s voice=%s chars=%d", name, voice, len(req.input))
        return Response(audio, media_type=MEDIA[req.response_format],
                        headers={"X-Engine": name, "X-Voice": quote(voice)})
    raise HTTPException(503, "; ".join(errors))


@app.get("/health")
def health():
    """Liveness plus the configured fallback order."""
    return {"ok": True, "chain": CHAIN}


@app.get("/")
def index():
    """Read-aloud page: type something, it is spoken out loud."""
    return HTMLResponse(HTML)


HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vorleser</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem;
         background: #111; color: #eee; }
  textarea { width: 100%; height: 40vh; font-size: 1.1rem; padding: .75rem; box-sizing: border-box;
             background: #1c1c1c; color: #eee; border: 1px solid #444; border-radius: 8px; }
  button { width: 100%; margin-top: 1rem; padding: 1rem; font-size: 1.2rem; border: 0;
           border-radius: 8px; background: #2e7d32; color: #fff; }
  button:disabled { background: #555; }
  #quick, #main { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
  #quick button { flex: 1 1 30%; width: auto; margin: 0; padding: .75rem; background: #37474f; }
  #main button { width: auto; margin: 0; }
  #play { flex: 2; }
  #cancel { flex: 1; background: #b71c1c; }
</style>
</head>
<body>
<textarea id="text" placeholder="Text hier eingeben …"></textarea>
<div id="main">
  <button id="play">▶ Play</button>
  <button id="cancel">✕ Cancel</button>
</div>
<div id="quick"></div>
<script>
const btn = document.getElementById("play");
const cache = {};  // one text field per visit, so it stays small

async function getAudio(text) {
  if (!cache[text]) {
    // Skips qwen on purpose: a spoken reply has to come within a second.
    const r = await fetch("/v1/audio/speech", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model: "azure,edge", input: text}),
    });
    if (!r.ok) throw new Error(await r.text());
    const a = new Audio(URL.createObjectURL(await r.blob()));
    a.load();
    cache[text] = a;
  }
  return cache[text];
}

async function speak(text, button) {
  if (!text) return;
  const label = button.textContent;
  if (!cache[text]) {
    document.querySelectorAll("button").forEach(b => b.disabled = true);
    button.textContent = "…";
  }
  try {
    const audio = await getAudio(text);
    audio.currentTime = 0;
    await new Promise(done => { audio.onended = audio.onerror = done; audio.play(); });
  } catch (e) {
    alert("Fehler: " + e.message);
  }
  button.textContent = label;
  document.querySelectorAll("button").forEach(b => b.disabled = false);
}

btn.onclick = () => speak(document.getElementById("text").value.trim(), btn);

document.getElementById("cancel").onclick = () => {
  const t = document.getElementById("text");
  t.value = "";
  t.focus();
};

for (const phrase of ["Halt!", "Stop!", "Ja", "Nein", "Warte", "Danke",
                      "Keine Ahnung, musst du Mama fragen."]) {
  const b = document.createElement("button");
  b.textContent = phrase;
  b.onclick = () => speak(phrase, b);
  document.getElementById("quick").append(b);
  getAudio(phrase).catch(() => {});  // preload so the click plays at once
}
</script>
</body>
</html>"""

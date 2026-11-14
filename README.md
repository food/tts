# tts

One speech endpoint for all services. It speaks the OpenAI audio API and tries
three engines in order until one delivers:

1. **qwen** - a [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) voice clone server
   (`/health`, `/v1/audio/speak`). Often switched off; a stopped server is
   detected by the health check within seconds.
2. **azure** - Azure Speech REST API (the free F0 tier covers 500k characters a month).
3. **edge** - the Edge read-aloud service via `edge-tts`, no key needed.

The root page is a small read-aloud tool: type a sentence, it is spoken.

## Request

```
POST /v1/audio/speech
{"input": "Guten Morgen", "model": "auto", "voice": "portal2",
 "response_format": "opus", "speed": 1.1}
```

| Field | Meaning |
|---|---|
| `input` | Text to speak. |
| `model` | `auto` (or `tts-1`) runs `TTS_CHAIN`; `azure` forces one engine; `azure,edge` a list. |
| `voice` | A qwen clone name, or an Azure/Edge voice such as `de-DE-ConradNeural`. Microsoft voices skip qwen. Unset means each engine's default. |
| `response_format` | `mp3` (default), `opus` (Ogg/Opus, what Telegram wants for voice notes) or `wav`. |
| `speed` | 0.5 to 2.0, becomes the SSML rate for Azure and Edge. |
| `device` | Optional routing hint passed to qwen3-tts, e.g. `gpu_if_free`. |

The response is the audio. `X-Engine` and `X-Voice` (URL-encoded) say who spoke. If every
engine fails the answer is 503 with each engine's error in `detail`.

Any OpenAI client works:

```python
from openai import OpenAI
OpenAI(base_url="http://tts-host:8095/v1", api_key="-").audio.speech.create(
    model="auto", voice="portal2", input="Hallo").write_to_file("hallo.mp3")
```

## Run

```
cp .env.example .env   # fill in QWEN_URL and the Azure key
docker compose up -d --build
```

Edge answers unknown voice names with a default voice instead of an error;
take names from `edge-tts --list-voices`.

## Test

```
python test_tts.py
```

Needs ffmpeg on `PATH`; qwen is faked, Azure and Edge are stubbed.

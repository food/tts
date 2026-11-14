import io

import edge_tts
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

# ponytail: bewusst KEINE Multilingual-Stimme — die wechselt die Aussprache
# je nach Sprache des Textes. Katja liest auch Fremdwoerter deutsch.
VOICE = "de-DE-KatjaNeural"

app = FastAPI()

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
const cache = {};  // ponytail: waechst unbegrenzt, egal bei einem Textfeld pro Sitzung

async function getAudio(text) {
  if (!cache[text]) {
    const r = await fetch("/tts", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
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
  getAudio(phrase).catch(() => {});  // vorladen, damit der Klick ohne Verzoegerung spielt
}
</script>
</body>
</html>"""


class Req(BaseModel):
    text: str


@app.get("/")
def index():
    return HTMLResponse(HTML)


_cache: dict[str, bytes] = {}  # ponytail: unbegrenzt; LRU erst wenn der Speicher stoert


@app.post("/tts")
async def tts(req: Req):
    if req.text not in _cache:
        buf = io.BytesIO()
        async for chunk in edge_tts.Communicate(req.text, VOICE).stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        _cache[req.text] = buf.getvalue()
    return Response(_cache[req.text], media_type="audio/mpeg")

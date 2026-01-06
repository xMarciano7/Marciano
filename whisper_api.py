from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile, uvicorn, os, asyncio, subprocess

# ===== CONFIG =====
MODEL_SIZE = os.getenv("MODEL", "medium")
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Cargar modelo una sola vez
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

app = FastAPI()

# ---------- FUNCIÓN BLOQUEANTE ----------
def _transcribe_sync(audio_bytes: bytes):
    # Guardar audio original
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
        t.write(audio_bytes)
        raw_path = t.name

    # Convertir a WAV PCM mono 16k (OBLIGATORIO)
    pcm_path = raw_path + "_16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", pcm_path],
        check=True
    )

    # Transcribir
    segments, info = model.transcribe(pcm_path, word_timestamps=True)

    words = []
    for s in segments:
        if s.words:
            for w in s.words:
                words.append({
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                })

    return {
        "language": info.language,
        "words": words,
    }

# ---------- ENDPOINT ----------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _transcribe_sync, audio_bytes)
    except Exception as e:
        print("POD ERROR:", repr(e))
        raise


@app.get("/")
def root():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

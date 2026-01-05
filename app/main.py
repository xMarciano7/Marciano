import os
import uuid
import json
import shutil
import subprocess
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STORAGE_INPUT = os.path.join(BASE_DIR, "storage", "input")
STORAGE_OUTPUT = os.path.join(BASE_DIR, "storage", "output")
STORAGE_TMP = os.path.join(BASE_DIR, "storage", "tmp")

os.makedirs(STORAGE_INPUT, exist_ok=True)
os.makedirs(STORAGE_OUTPUT, exist_ok=True)
os.makedirs(STORAGE_TMP, exist_ok=True)

# RunPod POD (no serverless)
RUNPOD_POD_URL = os.getenv("RUNPOD_POD_URL")  # ej: http://IP:8000/transcribe

if not RUNPOD_POD_URL:
    raise RuntimeError("RUNPOD_POD_URL no definida")

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="ClipFile Backend (RunPod POD Whisper)")

# ============================================================
# UTILS
# ============================================================

def write_progress(job_id: str, percent: int):
    path = os.path.join(STORAGE_TMP, f"{job_id}.progress.json")
    with open(path, "w") as f:
        json.dump({"percent": percent}, f)


def pod_transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        files = {"file": f}
        r = requests.post(
            RUNPOD_POD_URL,
            files=files,
            timeout=600
        )
    r.raise_for_status()
    data = r.json()
    if "text" not in data:
        raise RuntimeError(f"Respuesta inválida del POD: {data}")
    return data["text"]

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = os.path.join(STORAGE_INPUT, f"{job_id}.mp4")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    write_progress(job_id, 5)
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
def progress(job_id: str):
    path = os.path.join(STORAGE_TMP, f"{job_id}.progress.json")
    if not os.path.exists(path):
        return {"percent": 0}
    with open(path) as f:
        return json.load(f)


@app.post("/process/{job_id}")
def process(job_id: str):
    input_video = os.path.join(STORAGE_INPUT, f"{job_id}.mp4")
    if not os.path.exists(input_video):
        return JSONResponse({"error": "video no encontrado"}, status_code=404)

    write_progress(job_id, 10)

    audio_path = os.path.join(STORAGE_TMP, f"{job_id}.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            audio_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    write_progress(job_id, 30)

    text = pod_transcribe(audio_path)

    write_progress(job_id, 70)

    txt_path = os.path.join(STORAGE_OUTPUT, f"{job_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    write_progress(job_id, 100)
    return {"status": "ok", "job_id": job_id}


@app.get("/download/{job_id}")
def download(job_id: str):
    path = os.path.join(STORAGE_OUTPUT, f"{job_id}.txt")
    if not os.path.exists(path):
        return JSONResponse({"error": "archivo no listo"}, status_code=404)
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}.txt")


@app.get("/")
def root():
    return {"status": "ok", "backend": "runpod-pod-whisper"}

# app/main.py
# CLIPFILE BACKEND — ASYNC
# Upload → background process → polling → download
# Whisper corre en RunPod (GPU). Render NO se bloquea.

import os
import uuid
import json
import shutil
import subprocess
import requests
import threading
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STORAGE_INPUT = os.path.join(BASE_DIR, "storage", "input")
STORAGE_OUTPUT = os.path.join(BASE_DIR, "storage", "output")
STORAGE_TMP = os.path.join(BASE_DIR, "storage", "tmp")

for p in [STORAGE_INPUT, STORAGE_OUTPUT, STORAGE_TMP]:
    os.makedirs(p, exist_ok=True)

RUNPOD_POD_URL = os.getenv("RUNPOD_POD_URL")
if not RUNPOD_POD_URL:
    raise RuntimeError("RUNPOD_POD_URL no definida")

FONT_NAME = "Poppins ExtraBold"
ASS_FONT_SIZE = 110
ASS_PRIMARY = "&H00FFFF00"
ASS_OUTLINE = "&H00000000"
ASS_OUTLINE_SIZE = 6
ASS_ALIGN = 5
MAX_WORDS_ON_SCREEN = 2

MIN_CLIP = 20
MAX_CLIP = 30

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="ClipFile Backend — ASYNC")

# ============================================================
# UTILS
# ============================================================

def run(cmd):
    subprocess.run(cmd, check=True)

def write_progress(job_id, percent):
    with open(os.path.join(STORAGE_TMP, f"{job_id}.progress.json"), "w") as f:
        json.dump({"percent": percent}, f)

def extract_audio(video_path, wav_path):
    run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ])

def ffprobe_duration(path):
    r = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(r.decode().strip())

# ============================================================
# RUNPOD
# ============================================================

def pod_transcribe_words(wav_path):
    with open(wav_path, "rb") as f:
        r = requests.post(
            RUNPOD_POD_URL,
            files={"file": f},
            timeout=1800
        )
    r.raise_for_status()
    data = r.json()
    return data["words"]

# ============================================================
# CLIP LOGIC
# ============================================================

def score_window(words, start_t, end_t):
    score = 0.0
    for w in words:
        if w["end"] < start_t or w["start"] > end_t:
            continue
        score += 1.0
    return score / max(end_t - start_t, 1.0)

def pick_best_window(words, total_dur):
    best = (0, MIN_CLIP)
    best_score = -1
    for dur in range(MIN_CLIP, MAX_CLIP + 1, 2):
        t = 0.0
        while t + dur <= total_dur:
            s = score_window(words, t, t + dur)
            if s > best_score:
                best_score = s
                best = (t, t + dur)
            t += 1.0
    return best

def ts_ass(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def build_ass(words, clip_start, clip_end, ass_path):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, Outline, Alignment
Style: Default,{FONT_NAME},{ASS_FONT_SIZE},{ASS_PRIMARY},{ASS_OUTLINE},{ASS_OUTLINE_SIZE},{ASS_ALIGN}

[Events]
Format: Start, End, Style, Text
"""
    lines = [header]
    buf = []
    buf_start = None

    for w in words:
        if w["end"] < clip_start or w["start"] > clip_end:
            continue
        t0 = max(w["start"], clip_start) - clip_start
        t1 = min(w["end"], clip_end) - clip_start

        if buf_start is None:
            buf_start = t0
        buf.append(w["word"])

        if len(buf) >= MAX_WORDS_ON_SCREEN:
            lines.append(
                f"Dialogue: {ts_ass(buf_start)},{ts_ass(t1)},Default,{' '.join(buf)}\n"
            )
            buf = []
            buf_start = None

    if buf and buf_start is not None:
        lines.append(
            f"Dialogue: {ts_ass(buf_start)},{ts_ass(clip_end-clip_start)},Default,{' '.join(buf)}\n"
        )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

# ============================================================
# BACKGROUND WORKER
# ============================================================

def process_job(job_id):
    try:
        write_progress(job_id, 5)

        input_video = os.path.join(STORAGE_INPUT, f"{job_id}.mp4")
        wav_path = os.path.join(STORAGE_TMP, f"{job_id}.wav")

        extract_audio(input_video, wav_path)
        write_progress(job_id, 20)

        words = pod_transcribe_words(wav_path)
        write_progress(job_id, 50)

        total_dur = ffprobe_duration(input_video)
        clip_start, clip_end = pick_best_window(words, total_dur)
        write_progress(job_id, 65)

        clip_video = os.path.join(STORAGE_TMP, f"{job_id}_clip.mp4")
        run([
            "ffmpeg", "-y",
            "-ss", str(clip_start),
            "-to", str(clip_end),
            "-i", input_video,
            "-c", "copy",
            clip_video
        ])

        ass_path = os.path.join(STORAGE_TMP, f"{job_id}.ass")
        build_ass(words, clip_start, clip_end, ass_path)
        write_progress(job_id, 80)

        final_out = os.path.join(STORAGE_OUTPUT, f"{job_id}.mp4")
        run([
            "ffmpeg", "-y",
            "-i", clip_video,
            "-vf", f"ass={ass_path}:fontsdir=/app/fonts",
            "-c:a", "copy",
            final_out
        ])

        write_progress(job_id, 100)

    except Exception as e:
        write_progress(job_id, -1)
        print("JOB ERROR:", job_id, repr(e))

# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    input_video = os.path.join(STORAGE_INPUT, f"{job_id}.mp4")
    with open(input_video, "wb") as f:
        shutil.copyfileobj(file.file, f)

    write_progress(job_id, 1)

    t = threading.Thread(target=process_job, args=(job_id,))
    t.start()

    return {"job_id": job_id}

@app.get("/progress/{job_id}")
def progress(job_id: str):
    p = os.path.join(STORAGE_TMP, f"{job_id}.progress.json")
    if not os.path.exists(p):
        return {"percent": 0}
    with open(p) as f:
        return json.load(f)

@app.get("/download/{job_id}")
def download(job_id: str):
    path = os.path.join(STORAGE_OUTPUT, f"{job_id}.mp4")
    if not os.path.exists(path):
        return JSONResponse({"error": "clip no listo"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

@app.get("/")
def root():
    return {"status": "ok", "mode": "async"}

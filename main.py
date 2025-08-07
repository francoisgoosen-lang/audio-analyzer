from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
import tempfile

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Upload Audio File</h2>
        <form action="/analyze/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".wav,.mp3" required>
            <input type="submit" value="Analyze">
        </form>
    </body>
    </html>
    """

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

        return {"estimated_bpm": round(float(tempo[0]), 2)}
    except Exception as e:
        return {"error": str(e)}

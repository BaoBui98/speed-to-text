from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        segments, info = model.transcribe(
            temp_path,
            language="vi",
            beam_size=5
        )

        text = " ".join([segment.text for segment in segments])
        return {
            "text": text,
            "language": info.language
        }

    finally:
        os.remove(temp_path)
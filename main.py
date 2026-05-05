from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# Khởi tạo model turbo (Rất nhanh, chính xác cao)
model = WhisperModel(
    "turbo",
    device="cpu",
    compute_type="int8"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    def generate_segments():
        try:
            segments, info = model.transcribe(
                temp_path,
                language="vi",
                beam_size=1,        # Giảm beam_size để chạy cực nhanh
                vad_filter=True,    # Lọc bỏ đoạn im lặng để đỡ tốn thời gian xử lý
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            for segment in segments:
                yield f"{segment.text} "
        
        except Exception as e:
            yield f"Error: {str(e)}"
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(generate_segments(), media_type="text/plain")
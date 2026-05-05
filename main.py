import os
import sys
import queue
import threading
import numpy as np
import asyncio
import noisereduce as nr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# --- FIX LỖI DLL GPU ---
def fix_gpu_dlls():
    if os.name == "nt":
        nvidia_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
        if os.path.exists(nvidia_dir):
            for root, dirs, files in os.walk(nvidia_dir):
                if "bin" in dirs:
                    bin_path = os.path.abspath(os.path.join(root, "bin"))
                    try: os.add_dll_directory(bin_path)
                    except: pass
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

fix_gpu_dlls()

# --- SETTINGS ---
samplerate = 16000
chunk_duration = 2
frames_per_chunk = int(samplerate * chunk_duration)
audio_queue = queue.Queue()

# --- INIT MODEL ---
def init_model():
    try:
        # Thử test GPU thực tế
        m = WhisperModel("large-v3", device="cuda", compute_type="float16")
        list(m.transcribe(np.zeros(16000, dtype=np.float32)))
        return m, "🚀 GPU MODE (CUDA - float16)"
    except:
        return WhisperModel("small", device="cpu", compute_type="int8"), "🐌 CPU FALLBACK (int8)"

model, device_info = init_model()
print(f"INITIALIZED: {device_info}")

app = FastAPI()

# --- WEBSOCKET ---
@app.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket):
    await websocket.accept()
    # Thông báo cho bạn biết đang dùng gì ngay khi connect
    print(f"New client connected. Using: {device_info}")
    app.current_ws = websocket 
    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_queue.put(chunk)
    except WebSocketDisconnect:
        app.current_ws = None

# --- TRANSCRIBER ---
def transcriber_loop():
    audio_buffer = []
    last_text = ""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            overlap_frames = int(samplerate * 0.5)
            audio_buffer = [audio_data[-overlap_frames:]]

            # --- NOISE REDUCTION ---
            try:
                audio_data = nr.reduce_noise(y=audio_data, sr=samplerate, stationary=True)
            except Exception as nr_e:
                print(f"Noise reduction error: {nr_e}")

            try:
                segments, _ = model.transcribe(
                    audio_data.flatten().astype(np.float32),
                    language="vi",
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False, # Chống ảo tưởng
                    no_speech_threshold=0.6,          # Chống ảo tưởng
                    initial_prompt="Tôi đang nói chuyện bình thường."
                )

                for segment in segments:
                    text = segment.text.strip()
                    # Lọc bỏ các câu chào hỏi YouTube mặc định của Whisper
                    garbage_phrases = ["cảm ơn", "hẹn gặp lại", "subscribe", "đăng ký", "video tiếp theo"]
                    is_garbage = any(phrase in text.lower() for phrase in garbage_phrases)

                    if text and not is_garbage and text != last_text:
                        print(f"🗨️ {text}")
                        if hasattr(app, 'current_ws') and app.current_ws:
                            try:
                                loop.run_until_complete(app.current_ws.send_text(text))
                            except: pass
                        last_text = text
            except Exception as e:
                print(f"Error: {e}")

threading.Thread(target=transcriber_loop, daemon=True).start()
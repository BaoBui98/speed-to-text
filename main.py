import os
import sys
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

def fix_gpu_dlls():
    """Make NVIDIA wheel DLLs discoverable on Windows before CUDA initializes."""
    if os.name != "nt":
        return

    nvidia_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.exists(nvidia_dir):
        return

    for root, dirs, _files in os.walk(nvidia_dir):
        if "bin" in dirs:
            bin_path = os.path.abspath(os.path.join(root, "bin"))
            try:
                os.add_dll_directory(bin_path)
            except OSError:
                pass
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

fix_gpu_dlls()

SAMPLE_RATE = 16000
CHANNELS = 1
GARBAGE_PHRASES = ("cảm ơn", "hẹn gặp lại", "subscribe", "đăng ký", "video tiếp theo")

app = FastAPI()

# Thêm CORS để tránh lỗi kết nối từ các port khác nhau
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from faster_whisper import WhisperModel

# KHỞI TẠO MODEL LARGE_V3_TURBO TOÀN CỤC (Chỉ tải 1 lần duy nhất lúc khởi động server)
print("--- ĐANG KHỞI TẠO MODEL LARGE_V3_TURBO TOÀN CỤC (Vui lòng đợi) ---")
GLOBAL_WHISPER_MODEL = WhisperModel(
    Model.LARGE_V3_TURBO.value,
    device="cuda",
    compute_type="float16",
    # language="vi"
)
print("--- ĐÃ TẢI XONG MODEL LARGE_V3_TURBO VÀO BỘ NHỚ ---")

class ReusableWhisperSTTService(WhisperSTTService):
    def _load(self):
        # Tái sử dụng model đã tải sẵn toàn cục, tránh khởi tạo lại nhiều lần gây tốn tài nguyên và chậm kết nối
        self._model = GLOBAL_WHISPER_MODEL

    async def run_stt(self, audio: bytes):
        # Lọc các đoạn âm thanh quá ngắn (dưới 0.3 giây tương đương 9600 bytes ở 16kHz 16-bit PCM)
        # để tránh gây lỗi đổ vỡ (crash) thư viện CTranslate2 của faster-whisper.
        if len(audio) < 9600:
            print(f"Bỏ qua phân đoạn âm thanh quá ngắn ({len(audio)} bytes) để bảo vệ hệ thống.")
            return

        try:
            async for frame in super().run_stt(audio):
                yield frame
        except Exception as e:
            print(f"Lỗi khi nhận diện giọng nói: {e}")

class BrowserFloat32PCMSerializer(FrameSerializer):
    """Deserialize raw Float32 browser audio into Pipecat's 16-bit PCM frames."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, num_channels: int = CHANNELS):
        super().__init__()
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def serialize(self, frame: Frame) -> str | bytes | None:
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if not isinstance(data, bytes):
            return None

        audio = np.frombuffer(data, dtype=np.float32).copy()
        if audio.size == 0:
            return None

        audio = np.nan_to_num(audio, copy=False)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return InputAudioRawFrame(
            audio=pcm16.tobytes(),
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
        )


class WebSocketTranscriptionSender(FrameProcessor):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket
        self._last_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            is_garbage = any(phrase in text.lower() for phrase in GARBAGE_PHRASES)

            if text and not is_garbage and text != self._last_text:
                print(f"STT: {text}")
                await self._websocket.send_text(text)
                self._last_text = text

        elif isinstance(frame, ErrorFrame):
            print(f"Pipecat error: {frame.error}")
            await self._websocket.send_text(f"[error] {frame.error}")

        await self.push_frame(frame, direction)


@app.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket):
    await websocket.accept()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_in_channels=CHANNELS,
            serializer=BrowserFloat32PCMSerializer(),
            session_timeout=None,
        ),
    )
    
    # Khởi tạo VAD analyzer riêng biệt cho kết nối này để tránh lỗi trạng thái khi ngắt kết nối
    local_vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.45,      # Độ nhạy phát hiện giọng nói (mặc định 0.5, giảm nhẹ để nhận diện từ ngắn dễ hơn)
            start_secs=0.12,      # Nói siêu ngắn (0.12s) cũng bắt đầu bắt giọng để tránh mất chữ đầu câu
            stop_secs=1.0,        # Đợi 1.0 giây im lặng mới dừng nói, giúp câu nói ngắn không bị cắt cụt quá sớm
            min_volume=0.02,      # Nhạy bén hơn với các từ nói nhỏ
        )
    )
    vad = VADProcessor(vad_analyzer=local_vad_analyzer)
    
    # Khởi tạo STT service riêng biệt bằng cách tái sử dụng mô hình WhisperModel đã được tải toàn cục
    local_stt_service = ReusableWhisperSTTService(
        settings=ReusableWhisperSTTService.Settings(
            model=Model.LARGE_V3_TURBO.value,
            language=Language.VI,
            no_speech_prob=0.4,  # Nghiêm ngặt hơn để lọc tạp âm/im lặng tốt hơn
        ),
        device="cuda",
        compute_type="float16",
    )
    
    sender = WebSocketTranscriptionSender(websocket)

    pipeline = Pipeline([transport.input(), vad, local_stt_service, sender])
    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _websocket):
        print("New Pipecat STT client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _websocket):
        print("Pipecat STT client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
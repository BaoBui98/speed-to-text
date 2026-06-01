import asyncio
import os
import re
import sys
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, Frame, InputAudioRawFrame, StartFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.utils.time import time_now_iso8601

def fix_gpu_dlls():
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

# CẤU HÌNH HỆ THỐNG AUDIO ĐƯA VỀ CHUẨN AN TOÀN
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
MIN_STT_AUDIO_SECS = 0.3  
MIN_STT_AUDIO_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * MIN_STT_AUDIO_SECS)
STT_PRE_ROLL_SECS = 0.25   

# BỘ LỌC NĂNG LƯỢNG THÔ (Mở rộng tối đa để giao phó hoàn toàn cho Silero VAD)
MIN_STT_RMS = 0.001       
MIN_STT_PEAK = 0.005
MIN_VOICED_FRAME_RATIO = 0.02
VOICE_FRAME_SECS = 0.02

CONTEXT_MAX_ITEMS = 4
CONTEXT_MAX_CHARS = 400
GARBAGE_PHRASES = ("cảm ơn", "hẹn gặp lại", "subscribe", "đăng ký", "video tiếp theo", "tạm biệt")

def is_garbage_transcript(text: str) -> bool:
    return any(phrase in text.lower() for phrase in GARBAGE_PHRASES)

def normalize_repeated_transcript(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", normalized)
    deduped_sentences = []
    last_key = ""
    for sentence in sentences:
        sentence = sentence.strip()
        key = re.sub(r"[^\w\s]", "", sentence, flags=re.UNICODE).casefold().strip()
        if sentence and key and key != last_key:
            deduped_sentences.append(sentence)
            last_key = key
    return " ".join(deduped_sentences) or normalized

def has_enough_speech_energy(audio_float: np.ndarray) -> bool:
    if audio_float.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio_float))))
    return rms >= MIN_STT_RMS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- ĐANG KHỞI TẠO MODEL PHOWHISPER-LARGE TOÀN CỤC ---")
GLOBAL_WHISPER_MODEL = WhisperModel(
    "kiendt/PhoWhisper-large-ct2", 
    device="cuda",
    compute_type="float16",
)
print("--- ĐÃ TẢI XONG MODEL PHOWHISPER-LARGE VÀO BỘ NHỚ ---")

class SlidingTextContext:
    def __init__(self, max_items: int = CONTEXT_MAX_ITEMS, max_chars: int = CONTEXT_MAX_CHARS):
        self._items = deque(maxlen=max_items)
        self._max_chars = max_chars

    def add(self, text: str):
        normalized = " ".join(text.split())
        if normalized and len(normalized) > 1:
            self._items.append(normalized)

    def prompt(self) -> str:
        context = " ".join(self._items)
        if len(context) <= self._max_chars:
            return context
        return context[-self._max_chars:].lstrip()

class ReusableWhisperSTTService(WhisperSTTService):
    def __init__(self, *args, context_window: SlidingTextContext = None, pre_roll_secs: float = STT_PRE_ROLL_SECS, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_window = context_window or SlidingTextContext()
        self._pre_roll_secs = pre_roll_secs

    def _load(self):
        self._model = GLOBAL_WHISPER_MODEL

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._audio_buffer_size_1s = int(self.sample_rate * BYTES_PER_SAMPLE * self._pre_roll_secs)

    async def run_stt(self, audio: bytes):
        if len(audio) < MIN_STT_AUDIO_BYTES:
            return

        processing_started = False
        try:
            await self.start_processing_metrics()
            processing_started = True

            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            if not has_enough_speech_energy(audio_float):
                await self.stop_processing_metrics()
                return

            # Lấy prompt lịch sử viết hoa/từ vựng làm mồi (nếu có)
            history_prompt = self._context_window.prompt()

            transcribe_kwargs = {
                "language": 'vi',
                # CHỐNG ẢO GIÁC: Tắt kết nối chuỗi cũ để decoder không tự "bịa" chữ khi bạn dừng nói
                "condition_on_previous_text": False,  
                "initial_prompt": history_prompt if history_prompt else None,
                "no_speech_threshold": self._settings.no_speech_prob,
                "beam_size": 3,
                "temperature": 0.0, # Đưa về mặc định ổn định nhất, tránh sinh từ ngẫu nhiên
            }

            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio_float,
                **transcribe_kwargs,
            )

            # Đọc trọn vẹn kết quả từ các segment hợp lệ
            text = " ".join(segment.text.strip() for segment in segments).strip()
            text = normalize_repeated_transcript(text)

            await self.stop_processing_metrics()
            processing_started = False

            # CHỐNG BỊA CHỮ: Bỏ qua nếu text chỉ gồm các dấu câu hoặc ký tự rác cô đơn
            if text and len(re.sub(r'[^\w\s]', '', text).strip()) > 1 and not is_garbage_transcript(text):
                self._context_window.add(text)
                await self._handle_transcription(text, True, self._settings.language)
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), 'vi')
        except Exception as e:
            if processing_started:
                await self.stop_processing_metrics()
            print(f"Lỗi nhận diện giọng nói: {e}")

class BrowserFloat32PCMSerializer(FrameSerializer):
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
            if text and not is_garbage_transcript(text) and text != self._last_text:
                print(f"STT Output: {text}")
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
    
    # CẤU HÌNH VAD CHUẨN: Giữ câu liền mạch, không ngắt vội
    local_vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.40,   # Giảm xuống 0.40 để nhạy bén tối đa, không bao giờ nuốt chữ cuối câu
            start_secs=0.15,   # Bắt giọng cực nhanh ngay khi mở miệng
            stop_secs=0.85,    # Nới hẳn lên 0.85 giây giúp bạn thoải mái ngắt nhịp mà không sợ thiếu câu
            min_volume=0.01,   
        )
    )
    vad = VADProcessor(vad_analyzer=local_vad_analyzer)
    
    local_stt_service = ReusableWhisperSTTService(
        settings=ReusableWhisperSTTService.Settings(
            model="kiendt/PhoWhisper-large-ct2",
            language=Language.VI,
            no_speech_prob=0.35, # Ngưỡng tối ưu của thư viện gốc giúp nhận dạng nhạy bén
        ),
        context_window=SlidingTextContext(),
        pre_roll_secs=STT_PRE_ROLL_SECS,
        device="cuda",
        compute_type="float16",
    )
    
    sender = WebSocketTranscriptionSender(websocket)
    pipeline = Pipeline([transport.input(), vad, local_stt_service, sender])
    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _websocket):
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
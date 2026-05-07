import asyncio
import os
import re
import sys
from collections import deque
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pipecat.audio.vad.silero import SileroVADAnalyzer # loại bỏ tạp âm VAD = Voice Activity Detection.
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, Frame, InputAudioRawFrame, StartFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.utils.time import time_now_iso8601

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

fix_gpu_dlls() # sửa lỗi liên quan đến DLL của NVIDIA 

SAMPLE_RATE = 16000 # số lượng mẫu âm thanh được ghi lại trong mỗi giây
CHANNELS = 1 # số kênh âm thanh, 1 là mono (âm thanh một kênh), 2 là stereo (âm thanh hai kênh)
BYTES_PER_SAMPLE = 2
MIN_STT_AUDIO_SECS = 0.3
MIN_STT_AUDIO_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * MIN_STT_AUDIO_SECS)
STT_PRE_ROLL_SECS = 0.25
MIN_STT_RMS = 0.006
MIN_STT_PEAK = 0.03
MIN_VOICED_FRAME_RATIO = 0.08
VOICE_FRAME_SECS = 0.02
CONTEXT_MAX_ITEMS = 6
CONTEXT_MAX_CHARS = 700
GARBAGE_PHRASES = ("cảm ơn", "hẹn gặp lại", "subscribe", "đăng ký", "video tiếp theo")


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

    normalized = " ".join(deduped_sentences) or normalized
    words = normalized.split()

    for chunk_size in range(1, (len(words) // 2) + 1):
        if len(words) % chunk_size != 0:
            continue

        chunks = [
            " ".join(words[index : index + chunk_size]).casefold()
            for index in range(0, len(words), chunk_size)
        ]
        if len(chunks) > 1 and all(chunk == chunks[0] for chunk in chunks):
            return " ".join(words[:chunk_size])

    return normalized


def has_enough_speech_energy(audio_float: np.ndarray) -> bool:
    if audio_float.size == 0:
        return False

    rms = float(np.sqrt(np.mean(np.square(audio_float))))
    peak = float(np.max(np.abs(audio_float)))
    if rms < MIN_STT_RMS or peak < MIN_STT_PEAK:
        return False

    frame_size = max(1, int(SAMPLE_RATE * VOICE_FRAME_SECS))
    frame_count = audio_float.size // frame_size
    if frame_count == 0:
        return True

    framed_audio = audio_float[: frame_count * frame_size].reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(framed_audio), axis=1))
    voiced_ratio = float(np.mean(frame_rms >= MIN_STT_RMS))
    return voiced_ratio >= MIN_VOICED_FRAME_RATIO

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
    def __init__(
        self,
        *args,
        context_window: "SlidingTextContext | None" = None,
        pre_roll_secs: float = STT_PRE_ROLL_SECS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._context_window = context_window or SlidingTextContext()
        self._pre_roll_secs = pre_roll_secs

    def _load(self):
        # Tái sử dụng model đã tải sẵn toàn cục, tránh khởi tạo lại nhiều lần gây tốn tài nguyên và chậm kết nối
        self._model = GLOBAL_WHISPER_MODEL

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._audio_buffer_size_1s = int(self.sample_rate * BYTES_PER_SAMPLE * self._pre_roll_secs)

    async def run_stt(self, audio: bytes):
        # Lọc các đoạn âm thanh quá ngắn để tránh lỗi CTranslate2 và transcript rác.
        if len(audio) < MIN_STT_AUDIO_BYTES:
            print(f"Bỏ qua phân đoạn âm thanh quá ngắn ({len(audio)} bytes) để bảo vệ hệ thống.")
            return

        processing_started = False
        try:
            await self.start_processing_metrics()
            processing_started = True

            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            if not has_enough_speech_energy(audio_float):
                print("Bỏ qua phân đoạn im lặng/nhiễu thấp để tránh Whisper hallucination.")
                await self.stop_processing_metrics()
                processing_started = False
                return

            language = self._settings.language
            no_speech_prob_threshold = self._settings.no_speech_prob

            transcribe_kwargs = {
                "language": language,
                # Không truyền context vào decoder vì silence/noise dễ bị Whisper bịa câu cũ.
                "condition_on_previous_text": False,
                "no_speech_threshold": no_speech_prob_threshold,
            }

            segments, _ = await asyncio.to_thread(
                self._model.transcribe,
                audio_float,
                **transcribe_kwargs,
            )

            text = normalize_repeated_transcript(" ".join(
                segment.text.strip()
                for segment in segments
                if (
                    no_speech_prob_threshold is None
                    or segment.no_speech_prob < no_speech_prob_threshold
                )
            ).strip())

            await self.stop_processing_metrics()
            processing_started = False

            if text and not is_garbage_transcript(text):
                self._context_window.add(text)
                await self._handle_transcription(text, True, language)
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                )
        except Exception as e:
            if processing_started:
                await self.stop_processing_metrics()
            print(f"Lỗi khi nhận diện giọng nói: {e}")


class SlidingTextContext:
    """Giữ transcript gần nhất cho hậu xử lý mà không đưa vào Whisper decoder."""

    def __init__(self, max_items: int = CONTEXT_MAX_ITEMS, max_chars: int = CONTEXT_MAX_CHARS):
        self._items = deque(maxlen=max_items)
        self._max_chars = max_chars

    def add(self, text: str):
        normalized = " ".join(text.split())
        if normalized:
            self._items.append(normalized)

    def prompt(self) -> str:
        context = " ".join(self._items)
        if len(context) <= self._max_chars:
            return context
        return context[-self._max_chars:].lstrip()

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

            if text and not is_garbage_transcript(text) and text != self._last_text:
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
            stop_secs=0.55,       # Cắt segment nhanh hơn sau im lặng để lần nói tiếp không bị treo cảm giác realtime
            min_volume=0.02,      # Nhạy bén hơn với các từ nói nhỏ
        )
    )
    vad = VADProcessor(vad_analyzer=local_vad_analyzer)
    
    # Khởi tạo STT service riêng biệt bằng cách tái sử dụng mô hình WhisperModel đã được tải toàn cục
    local_stt_service = ReusableWhisperSTTService(
        settings=ReusableWhisperSTTService.Settings(
            model=Model.LARGE_V3_TURBO.value,
            language=Language.VI,
            no_speech_prob=0.35,  # Siết lọc silence/noise để tránh Whisper tự bịa câu khi không nói
        ),
        context_window=SlidingTextContext(),
        pre_roll_secs=STT_PRE_ROLL_SECS,
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
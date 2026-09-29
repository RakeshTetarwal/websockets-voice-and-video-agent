import asyncio
from typing import AsyncGenerator
import sounddevice as sd
from src.core.config import config


class AudioRecorder:
    """Async microphone audio recorder capturing 16kHz 16-bit PCM for Gemini Live."""

    def __init__(self, sample_rate: int = config.input_sample_rate, chunk_size: int = config.chunk_size):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.is_recording = False
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Hardware callback triggered when audio samples arrive from the microphone."""
        if status:
            pass  # Overflow / underflow handling if needed
        if self.is_recording and self._loop and self._loop.is_running():
            raw_bytes = bytes(indata)
            self._loop.call_soon_threadsafe(self.queue.put_nowait, raw_bytes)

    async def start(self) -> None:
        if self.is_recording:
            return
        self._loop = asyncio.get_running_loop()
        self.is_recording = True
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    async def read_chunk(self) -> bytes:
        """Reads the next 16kHz PCM audio chunk captured by the microphone."""
        return await self.queue.get()

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Yields audio chunks continuously while recording is active."""
        while self.is_recording:
            try:
                chunk = await self.queue.get()
                yield chunk
                self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Drain remaining items
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except (asyncio.QueueEmpty, ValueError):
                break

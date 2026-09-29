import asyncio
import sounddevice as sd
from src.core.config import config


class AudioPlayer:
    """Async speaker audio player for 24kHz 16-bit PCM output from Gemini Live."""

    def __init__(self, sample_rate: int = config.output_sample_rate):
        self.sample_rate = sample_rate
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.is_running = False
        self._stream: sd.RawOutputStream | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=2048,
        )
        self._stream.start()
        self._task = asyncio.create_task(self._play_loop())

    async def _play_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self.is_running:
            try:
                chunk = await self.queue.get()
                if chunk and self._stream:
                    # Write PCM chunk to the hardware speaker buffer
                    await loop.run_in_executor(None, self._stream.write, chunk)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[AudioPlayer] Playback error: {e}")

    def play_chunk(self, chunk: bytes) -> None:
        """Enqueues an incoming 24kHz audio chunk from Gemini."""
        if self.is_running:
            self.queue.put_nowait(chunk)

    def interrupt(self) -> None:
        """Instant interruption handling: Clears buffered audio so Gemini stops talking immediately."""
        # Drain all queued chunks awaiting playback
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

    async def stop(self) -> None:
        self.is_running = False
        self.interrupt()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

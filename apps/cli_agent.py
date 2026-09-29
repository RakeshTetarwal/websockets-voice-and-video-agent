import os
import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import config
from src.audio.recorder import AudioRecorder
from src.audio.player import AudioPlayer
from src.services.gemini_live_session import GeminiLiveSession


async def main():
    if not config.api_key:
        print("\n [ERROR] GEMINI_API_KEY is not set!")
        print("Please export GEMINI_API_KEY in your terminal or add it to .env:")
        print("   export GEMINI_API_KEY='AIzaSy...'")
        print("Get your free key at: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)

    print("=" * 65)
    print("🎙️  GEMINI LIVE REAL-TIME VOICE AGENT")
    print(f"• Model:      {config.model}")
    print(f"• Voice:      {config.voice_name}")
    print("• Audio:      16kHz Input (Mic) -> 24kHz Output (Speaker)")
    print("• Features:   Bidirectional streaming, Live VAD & Interruption, Tools")
    print("=" * 65)
    print("Connecting to Gemini Live WebSocket... Speak into your microphone!")
    print("Press Ctrl+C to exit.\n")

    player = AudioPlayer()
    recorder = AudioRecorder()

    def on_audio_chunk(chunk: bytes):
        player.play_chunk(chunk)

    def on_user_transcript(text: str):
        print(f"\r👤 You:    {text}")

    def on_agent_transcript(text: str):
        print(f"🤖 Gemini: {text}")

    def on_interrupted():
        # User spoke while Gemini was speaking -> immediately cut off audio playback
        player.interrupt()
        print("\n⚡ [Interrupted by User]")

    session = GeminiLiveSession(
        on_audio_chunk=on_audio_chunk,
        on_user_transcript=on_user_transcript,
        on_agent_transcript=on_agent_transcript,
        on_interrupted=on_interrupted,
    )

    try:
        await player.start()
        await recorder.start()
        await session.connect()
        print("🟢 LIVE SESSION CONNECTED! Listening...\n")

        # Stream microphone audio chunks directly to Gemini
        async for pcm_chunk in recorder.stream():
            await session.send_audio(pcm_chunk)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping voice agent...")
    finally:
        await recorder.stop()
        await player.stop()
        await session.close()
        print("Session closed gracefully. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

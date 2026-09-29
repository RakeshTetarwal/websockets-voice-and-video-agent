import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class VoiceAgentConfig:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
    voice_name: str = os.getenv("GEMINI_VOICE_NAME", "Puck")
    
    # Audio Specs (Gemini Live API native format)
    input_sample_rate: int = 16000     # 16kHz PCM
    output_sample_rate: int = 24000    # 24kHz PCM
    channels: int = 1                  # Mono
    chunk_size: int = 1024             # Samples per chunk
    dtype: str = "int16"               # 16-bit linear PCM

    @property
    def input_mime_type(self) -> str:
        return f"audio/pcm;rate={self.input_sample_rate}"


# Default singleton instance
config = VoiceAgentConfig()

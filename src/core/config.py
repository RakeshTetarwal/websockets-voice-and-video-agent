import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Ensure .env is always loaded from the project root regardless of CWD
ENV_PATH = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class VoiceAgentConfig:
    api_key: str = os.getenv('GEMINI_API_KEY', '')
    # Default to gemini-3.1-flash-live-preview for full real-time speech + vision capabilities
    model: str = os.getenv('GEMINI_LIVE_MODEL', 'gemini-3.1-flash-live-preview')
    voice_name: str = os.getenv('GEMINI_VOICE_NAME', 'Puck')
    enable_tools: bool = os.getenv('ENABLE_TOOLS', 'false').lower() == 'true'
    
    # Audio Specs (Gemini Live API native format)
    input_sample_rate: int = 16000     # 16kHz PCM
    output_sample_rate: int = 24000    # 24kHz PCM
    channels: int = 1                  # Mono
    chunk_size: int = 1024             # Samples per chunk
    dtype: str = 'int16'               # 16-bit linear PCM

    @property
    def input_mime_type(self) -> str:
        return f'audio/pcm;rate={self.input_sample_rate}'


# Default singleton instance
config = VoiceAgentConfig()

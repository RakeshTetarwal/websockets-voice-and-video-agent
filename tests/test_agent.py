import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import config, VoiceAgentConfig
from src.tools.system_tools import (
    get_current_time,
    get_weather,
    calculate,
    save_note,
    get_notes,
    AVAILABLE_TOOLS,
)
from src.services.gemini_live_session import GeminiLiveSession
from src.services.langchain_agent import LangChainVoiceAgent


class TestVoiceAgentCore(unittest.TestCase):
    def test_01_config_defaults(self):
        cfg = VoiceAgentConfig(api_key="mock_key")
        self.assertEqual(cfg.input_sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 24000)
        self.assertEqual(cfg.channels, 1)
        self.assertEqual(cfg.input_mime_type, "audio/pcm;rate=16000")

    def test_02_tools_execution(self):
        time_res = get_current_time()
        self.assertIn("UTC", time_res)

        weather_res = get_weather("London")
        self.assertIn("London", weather_res)

        calc_res = calculate("25 * 4")
        self.assertEqual(calc_res, "Result: 100")

        save_res = save_note("Buy groceries")
        self.assertIn("Buy groceries", save_res)
        notes = get_notes()
        self.assertIn("Buy groceries", notes)

    def test_03_live_session_config_builder(self):
        session = GeminiLiveSession(api_key="mock_key", model="gemini-3.1-flash-live-preview")
        live_cfg = session._build_config()
        self.assertIsNotNone(live_cfg.response_modalities)
        self.assertEqual(len(live_cfg.tools), len(AVAILABLE_TOOLS))

    def test_04_langchain_agent_initialization(self):
        agent = LangChainVoiceAgent(api_key="mock_key")
        self.assertIsNotNone(agent.llm)
        self.assertEqual(len(agent.history), 1)  # SystemMessage


if __name__ == "__main__":
    unittest.main(verbosity=2)

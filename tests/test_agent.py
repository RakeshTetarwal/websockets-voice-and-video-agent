import os
import sys
import unittest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

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
from src.services.image_processor import ImageProcessor, image_processor
from src.services.image_generator import ImageGenerator, image_generator


class TestVoiceAgentCore(unittest.TestCase):
    def test_01_config_defaults(self):
        cfg = VoiceAgentConfig(api_key="mock_key")
        self.assertEqual(cfg.input_sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 24000)
        self.assertEqual(cfg.channels, 1)
        self.assertEqual(cfg.input_mime_type, "audio/pcm;rate=16000")
        self.assertEqual(cfg.model, "gemini-3.1-flash-live-preview")

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
        session = GeminiLiveSession(api_key="mock_key", model="gemini-3.1-flash-live-preview", tools=AVAILABLE_TOOLS)
        live_cfg = session._build_config()
        self.assertIsNotNone(live_cfg.response_modalities)
        self.assertEqual(len(live_cfg.tools), len(AVAILABLE_TOOLS))
        self.assertTrue(hasattr(session, "send_image"))

    def test_04_langchain_agent_initialization(self):
        agent = LangChainVoiceAgent(api_key="mock_key")
        self.assertIsNotNone(agent.llm)
        self.assertEqual(len(agent.history), 1)  # SystemMessage

    def test_05_image_processor_initialization(self):
        self.assertIsNotNone(image_processor)
        self.assertTrue("gemini" in image_processor.model)
        self.assertTrue(hasattr(image_processor, "analyze_image"))
        self.assertTrue(hasattr(image_processor, "extract_text_ocr"))
        self.assertTrue(hasattr(image_processor, "visual_qa"))

    def test_06_image_generator_initialization(self):
        self.assertIsNotNone(image_generator)
        self.assertTrue(hasattr(image_generator, "generate_image"))

    def test_07_live_session_send_image(self):
        session = GeminiLiveSession(api_key="mock_key")
        session.is_connected = True
        session.session = AsyncMock()
        
        asyncio.run(session.send_image(b"fake_jpeg", "image/jpeg"))
        session.session.send_realtime_input.assert_called_once()

    def test_08_image_processor_mock_analysis(self):
        processor = ImageProcessor(api_key="mock_key")
        processor.client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Mocked visual description"
        processor.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        res = asyncio.run(processor.analyze_image(b"fake_bytes", "Describe"))
        self.assertEqual(res, "Mocked visual description")

    def test_09_image_generator_quota_handling(self):
        from google.genai.errors import ClientError
        generator = ImageGenerator(api_key="mock_key")
        generator.client = MagicMock()
        generator.client.aio.models.generate_content = AsyncMock(
            side_effect=ClientError(429, {"message": "RESOURCE_EXHAUSTED, limit: 0"}, None)
        )

        res = asyncio.run(generator.generate_image("A futuristic city"))
        self.assertFalse(res["success"])
        self.assertTrue(res.get("quota_exhausted"))
        self.assertIn("billing", res["error"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

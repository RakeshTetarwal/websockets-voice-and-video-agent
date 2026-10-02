import base64
import logging
from typing import Optional, List
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from src.core.config import config

logger = logging.getLogger('ImageProcessor')

CANDIDATE_VISION_MODELS = [
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemini-3.5-flash',
    'gemini-3.8-flash',
]


class ImageProcessor:
    """Handles multimodal image analysis, OCR, visual Q&A, and object inspection using Gemini vision models."""

    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-3.5-flash-lite'):
        self.api_key = api_key or config.api_key
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str = 'Describe this image in detail and highlight key visual elements.',
        mime_type: str = 'image/jpeg',
    ) -> str:
        """Performs visual analysis with automatic resilient fallback across models."""
        models_to_try = [self.model] + [m for m in CANDIDATE_VISION_MODELS if m != self.model]
        last_error = None

        for target_model in models_to_try:
            try:
                response = await self.client.aio.models.generate_content(
                    model=target_model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                )
                if response.text:
                    return response.text.strip()
                return 'No text generated.'
            except (ServerError, ClientError) as e:
                last_error = e
                logger.warning(f'Model {target_model} failed: {e}. Trying next candidate...')
                continue
            except Exception as e:
                logger.error(f'Unexpected error with {target_model}: {e}')
                raise

        raise last_error or RuntimeError('All vision models failed.')

    async def extract_text_ocr(
        self,
        image_bytes: bytes,
        mime_type: str = 'image/jpeg',
    ) -> str:
        """Extracts all readable text from an image with high fidelity."""
        prompt = (
            'Act as a high-precision OCR engine. Extract all readable text from this image line by line. '
            'Preserve original casing and layout where possible. Output ONLY the extracted text.'
        )
        return await self.analyze_image(image_bytes=image_bytes, prompt=prompt, mime_type=mime_type)

    async def visual_qa(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = 'image/jpeg',
    ) -> str:
        """Answers a specific question about an image."""
        prompt = f'Answer this question based on the image: {question}'
        return await self.analyze_image(image_bytes=image_bytes, prompt=prompt, mime_type=mime_type)


# Default singleton instance
image_processor = ImageProcessor()

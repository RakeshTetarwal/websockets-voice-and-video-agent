import base64
import logging
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError
from src.core.config import config

logger = logging.getLogger('ImageGenerator')


class ImageGenerator:
    """Handles AI image generation using Google Gemini and Imagen models."""

    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-2.5-flash-image'):
        self.api_key = api_key or config.api_key
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    async def generate_image(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generates an image from a natural language text prompt.
        
        Gracefully handles free-tier quota limits (limit: 0) and informs the user
        to link a billing account if quota is exhausted.
        """
        target_model = model or self.model
        try:
            response = await self.client.aio.models.generate_content(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE', 'TEXT']
                )
            )

            image_base64 = None
            mime_type = 'image/png'
            text_caption = ''

            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        image_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                        mime_type = part.inline_data.mime_type or 'image/png'
                    elif part.text:
                        text_caption += part.text

            if image_base64:
                return {
                    'success': True,
                    'image_base64': image_base64,
                    'mime_type': mime_type,
                    'caption': text_caption.strip(),
                    'model': target_model,
                }
            else:
                return {
                    'success': False,
                    'error': 'No image data returned from model.',
                    'caption': text_caption.strip(),
                }

        except (ClientError, APIError) as e:
            error_str = str(e)
            if 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str or 'limit: 0' in error_str:
                logger.warning(f'Image generation quota exceeded / limit 0: {error_str}')
                return {
                    'success': False,
                    'quota_exhausted': True,
                    'error': (
                        'Image generation requires an active Google Cloud / AI Studio billing account. '
                        'Free-tier API keys currently have a quota limit of 0 for image generation models.'
                    ),
                    'suggestion': 'To enable Imagen 3 and Gemini image generation, attach a billing account to your project at https://aistudio.google.com/.'
                }
            logger.error(f'Image generation error: {e}')
            return {
                'success': False,
                'error': f'Image generation failed: {error_str}'
            }
        except Exception as e:
            logger.error(f'Unexpected image generation error: {e}')
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
image_generator = ImageGenerator()

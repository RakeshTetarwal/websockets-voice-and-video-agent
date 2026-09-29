import asyncio
from typing import AsyncGenerator, Callable, Optional
from google import genai
from google.genai import types

from src.core.config import config
from src.tools.system_tools import AVAILABLE_TOOLS


class GeminiLiveSession:
    """Manages full-duplex real-time voice conversation with Gemini Live over WebSockets."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_user_transcript: Optional[Callable[[str], None]] = None,
        on_agent_transcript: Optional[Callable[[str], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
    ):
        self.api_key = api_key or config.api_key
        self.model = model or config.model
        self.voice_name = voice_name or config.voice_name
        self.system_prompt = system_prompt or (
            "You are a friendly, witty, and concise AI voice assistant. "
            "Keep your responses spoken, natural, conversational, and direct. "
            "Avoid markdown tables, bullet points, or complex formatting since your answers are spoken aloud. "
            "You have access to tools for current time, weather, math calculations, and saving notes."
        )
        self.tools = tools if tools is not None else AVAILABLE_TOOLS

        # Event Callbacks
        self.on_audio_chunk = on_audio_chunk
        self.on_user_transcript = on_user_transcript
        self.on_agent_transcript = on_agent_transcript
        self.on_interrupted = on_interrupted

        self.client = genai.Client(api_key=self.api_key)
        self.session = None
        self._receive_task: Optional[asyncio.Task] = None
        self.is_connected = False

    def _build_config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self.system_prompt)]
            ),
            tools=self.tools,
        )

    async def connect(self):
        """Opens the bidirectional WebSocket live session with Gemini."""
        live_config = self._build_config()
        self.session_ctx = self.client.aio.live.connect(
            model=self.model,
            config=live_config,
        )
        self.session = await self.session_ctx.__aenter__()
        self.is_connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, pcm_chunk: bytes):
        """Sends a 16kHz 16-bit PCM audio chunk to Gemini."""
        if not self.is_connected or not self.session:
            return
        await self.session.send_realtime_input(
            audio=types.Blob(
                data=pcm_chunk,
                mime_type=config.input_mime_type,
            )
        )

    async def send_text(self, text: str):
        """Sends text input into the live session."""
        if not self.is_connected or not self.session:
            return
        await self.session.send_realtime_input(text=text)

    async def _receive_loop(self):
        """Continuously listens for server events (audio, transcriptions, interruptions, tool calls)."""
        try:
            async for response in self.session.receive():
                server_content = response.server_content
                if server_content:
                    # 1. Voice Activity Detection / Interruption
                    if server_content.interrupted:
                        if self.on_interrupted:
                            self.on_interrupted()

                    # 2. Transcriptions
                    if server_content.input_transcription:
                        text = server_content.input_transcription.text
                        if text and self.on_user_transcript:
                            self.on_user_transcript(text)

                    if server_content.output_transcription:
                        text = server_content.output_transcription.text
                        if text and self.on_agent_transcript:
                            self.on_agent_transcript(text)

                    # 3. Native Spoken Audio Chunks
                    if server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if self.on_audio_chunk:
                                    self.on_audio_chunk(part.inline_data.data)

                # 4. Tool Calls
                tool_call = getattr(response, "tool_call", None)
                if tool_call and self.session:
                    await self._handle_tool_call(tool_call)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[GeminiLiveSession] Receive loop error: {e}")

    async def _handle_tool_call(self, tool_call):
        """Executes matching Python function and sends the result back to Gemini."""
        function_responses = []
        for call in tool_call.function_calls:
            func_name = call.name
            func_args = call.args or {}
            
            # Find matching tool in available tools
            target_func = next((t for t in self.tools if t.__name__ == func_name), None)
            if target_func:
                try:
                    result = target_func(**func_args)
                except Exception as ex:
                    result = f"Error executing tool {func_name}: {ex}"
            else:
                result = f"Tool {func_name} not found."

            function_responses.append(
                types.FunctionResponse(
                    id=call.id,
                    name=func_name,
                    response={"result": result},
                )
            )

        if function_responses:
            await self.session.send_tool_response(function_responses=function_responses)

    async def close(self):
        """Gracefully closes the WebSocket live session."""
        self.is_connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if hasattr(self, "session_ctx") and self.session_ctx:
            await self.session_ctx.__aexit__(None, None, None)
            self.session = None

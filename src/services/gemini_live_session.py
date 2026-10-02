import os
import sys
import asyncio
import logging
from typing import Optional, Callable, Set
from google import genai
from google.genai import types
from src.core.config import config
from src.tools.system_tools import AVAILABLE_TOOLS

logger = logging.getLogger('VoiceAgent')


class GeminiLiveSession:
    """Manages full-duplex real-time voice and vision conversation with Gemini Live over WebSockets."""

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
        on_interim_transcript: Optional[Callable[[str], None]] = None,
        on_interrupted: Optional[Callable[[], None]] = None,
        on_turn_complete: Optional[Callable[[], None]] = None,
        on_tool_call: Optional[Callable[[str, dict, str], None]] = None,
        on_tool_cancelled: Optional[Callable[[list], None]] = None,
        on_usage_metadata: Optional[Callable[[dict], None]] = None,
        on_grounding_metadata: Optional[Callable[[dict], None]] = None,
        on_setup_complete: Optional[Callable[[str], None]] = None,
        on_go_away: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.api_key = api_key or config.api_key
        self.model = model or config.model
        self.voice_name = voice_name or config.voice_name
        self.system_prompt = system_prompt or (
            "You are an observant, precise AI multimodal voice and vision assistant. "
            "You receive live camera video frames from the user's webcam alongside their spoken voice. "
            "When the user asks about an object, phone screen, wallpaper, document, or clothing they are showing you, examine the visual frames with high attention to detail. "
            "Report ONLY what is genuinely visible. Never guess, assume, or hallucinate popular apps (like YouTube, TikTok, or Netflix) unless their distinct UI is clearly shown. "
            "If an image, wallpaper, monument, or landmark is shown on a screen or held up, identify the subject, landmarks, architecture, or art accurately (such as monuments like the Taj Mahal, nature scenes, or text). "
            "Keep answers concise, natural, and direct in 1 to 2 spoken sentences."
        )
        self.tools = tools if tools is not None else (AVAILABLE_TOOLS if config.enable_tools else None)

        # Event Callbacks
        self.on_audio_chunk = on_audio_chunk
        self.on_user_transcript = on_user_transcript
        self.on_agent_transcript = on_agent_transcript
        self.on_interim_transcript = on_interim_transcript
        self.on_interrupted = on_interrupted
        self.on_turn_complete = on_turn_complete
        self.on_tool_call = on_tool_call
        self.on_tool_cancelled = on_tool_cancelled
        self.on_usage_metadata = on_usage_metadata
        self.on_grounding_metadata = on_grounding_metadata
        self.on_setup_complete = on_setup_complete
        self.on_go_away = on_go_away
        self.on_error = on_error

        self.client = genai.Client(api_key=self.api_key)
        self.session = None
        self._receive_task: Optional[asyncio.Task] = None
        self.is_connected = False
        self._cancelled_tool_call_ids: Set[str] = set()

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
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
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
        try:
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=pcm_chunk,
                    mime_type=config.input_mime_type,
                )
            )
        except Exception as e:
            if self.is_connected:
                logger.warning(f"[GeminiLiveSession] Error sending audio chunk: {e}")
                self.is_connected = False
                if self.on_error:
                    self.on_error(e)

    async def send_image(self, image_bytes: bytes, mime_type: str = "image/jpeg"):
        """Sends a camera video frame into the live session for continuous visual reasoning."""
        if not self.is_connected or not self.session:
            return
        try:
            await self.session.send_realtime_input(
                video=types.Blob(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            )
        except Exception as e:
            if self.is_connected:
                logger.warning(f"[GeminiLiveSession] Error sending image frame: {e}")
                self.is_connected = False
                if self.on_error:
                    self.on_error(e)

    async def send_text(self, text: str):
        """Sends text input into the live session."""
        if not self.is_connected or not self.session:
            return
        try:
            await self.session.send_realtime_input(text=text)
        except Exception as e:
            if self.is_connected:
                logger.warning(f"[GeminiLiveSession] Error sending text: {e}")
                self.is_connected = False
                if self.on_error:
                    self.on_error(e)

    async def _receive_loop(self):
        """Continuously listens for server events across all conversation turns."""
        while self.is_connected and self.session:
            try:
                async for response in self.session.receive():
                    # 1. Setup Complete (Handshake & Session ID)
                    setup_complete = getattr(response, "setup_complete", None)
                    if setup_complete:
                        session_id = getattr(setup_complete, "session_id", None)
                        if self.on_setup_complete and session_id:
                            self.on_setup_complete(session_id)

                    # 2. Token & Usage Metadata
                    usage_metadata = getattr(response, "usage_metadata", None)
                    if usage_metadata and self.on_usage_metadata:
                        self.on_usage_metadata({
                            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) or 0,
                            "response_tokens": getattr(usage_metadata, "response_token_count", 0) or 0,
                            "total_tokens": getattr(usage_metadata, "total_token_count", 0) or 0,
                        })

                    # 3. Tool Call Cancellation (User interrupted mid-tool execution)
                    tool_cancellation = getattr(response, "tool_call_cancellation", None)
                    if tool_cancellation:
                        cancelled_ids = getattr(tool_cancellation, "ids", []) or []
                        self._cancelled_tool_call_ids.update(cancelled_ids)
                        if self.on_tool_cancelled:
                            self.on_tool_cancelled(cancelled_ids)

                    # 4. Go Away (Session Expiration Warning from Google)
                    go_away = getattr(response, "go_away", None)
                    if go_away and self.on_go_away:
                        time_left = getattr(go_away, "time_left", "")
                        self.on_go_away(str(time_left))

                    # 5. Tool Calls (Function Calling)
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call and self.session:
                        await self._handle_tool_call(tool_call)

                    # 6. Server Content (Audio, Transcripts, VAD, Citations)
                    server_content = response.server_content
                    if server_content:
                        # Voice Activity Detection / Interruption
                        if server_content.interrupted:
                            if self.on_interrupted:
                                self.on_interrupted()

                        # Real-Time Interim Speech Transcript (streaming words as user speaks)
                        interim_tx = getattr(server_content, "interim_input_transcription", None)
                        if interim_tx and getattr(interim_tx, "text", None):
                            if self.on_interim_transcript:
                                self.on_interim_transcript(interim_tx.text)

                        # Final User Transcript (complete utterance)
                        if server_content.input_transcription:
                            text = server_content.input_transcription.text
                            if text and self.on_user_transcript:
                                self.on_user_transcript(text)

                        # Output Text Transcript (what Gemini speaks)
                        if server_content.output_transcription:
                            text = server_content.output_transcription.text
                            if text and self.on_agent_transcript:
                                self.on_agent_transcript(text)

                        # Model Turn (Spoken Audio Chunks or Direct Text parts)
                        if server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    if self.on_audio_chunk:
                                        self.on_audio_chunk(part.inline_data.data)
                                elif getattr(part, "text", None):
                                    if self.on_agent_transcript:
                                        self.on_agent_transcript(part.text)

                        # Grounding & Search Citations
                        grounding = getattr(server_content, "grounding_metadata", None)
                        if grounding and self.on_grounding_metadata:
                            sources = []
                            web_queries = getattr(grounding, "web_search_queries", None) or []
                            chunks = getattr(grounding, "grounding_chunks", None) or []
                            for c in chunks:
                                web = getattr(c, "web", None)
                                if web:
                                    sources.append({
                                        "title": getattr(web, "title", "Source"),
                                        "url": getattr(web, "uri", "")
                                    })
                            self.on_grounding_metadata({"queries": web_queries, "sources": sources})

                        # Turn Complete
                        if server_content.turn_complete:
                            if self.on_turn_complete:
                                self.on_turn_complete()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.is_connected = False
                logger.error(f"[GeminiLiveSession] Receive loop error: {e}")
                if self.on_error:
                    self.on_error(e)
                break

    async def _handle_tool_call(self, tool_call):
        """Executes matching Python function and sends the result back to Gemini."""
        function_responses = []
        for call in tool_call.function_calls:
            # If tool call was cancelled by user barge-in, abort execution
            if call.id in self._cancelled_tool_call_ids:
                logger.info(f"🛑 Skipping cancelled tool call {call.name} ({call.id})")
                continue

            func_name = call.name
            func_args = call.args or {}
            
            target_func = next((t for t in self.tools if t.__name__ == func_name), None)
            if target_func:
                try:
                    result = target_func(**func_args)
                except Exception as ex:
                    result = f"Error executing tool {func_name}: {ex}"
            else:
                result = f"Tool {func_name} not found."

            # Double-check if cancelled while running
            if call.id in self._cancelled_tool_call_ids:
                continue

            if self.on_tool_call:
                try:
                    self.on_tool_call(func_name, func_args, str(result))
                except Exception:
                    pass

            function_responses.append(
                types.FunctionResponse(
                    id=call.id,
                    name=func_name,
                    response={"result": result},
                )
            )

        if function_responses and self.session and self.is_connected:
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

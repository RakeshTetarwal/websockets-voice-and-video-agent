import os
import sys
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import config
from src.services.gemini_live_session import GeminiLiveSession

app = FastAPI(title="Gemini Live Voice Agent")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = PROJECT_ROOT / "web" / "index.html"
    return HTMLResponse(content=index_file.read_text())


@app.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    loop = asyncio.get_running_loop()

    def send_ws_json(payload: dict):
        if websocket.client_state.name == "CONNECTED":
            asyncio.run_coroutine_threadsafe(
                websocket.send_text(json.dumps(payload)), loop
            )

    def on_audio_chunk(chunk: bytes):
        if websocket.client_state.name == "CONNECTED":
            asyncio.run_coroutine_threadsafe(
                websocket.send_bytes(chunk), loop
            )

    def on_user_transcript(text: str):
        send_ws_json({"type": "user_transcript", "text": text})

    def on_agent_transcript(text: str):
        send_ws_json({"type": "agent_transcript", "text": text})

    def on_interrupted():
        send_ws_json({"type": "interrupted"})

    session = GeminiLiveSession(
        on_audio_chunk=on_audio_chunk,
        on_user_transcript=on_user_transcript,
        on_agent_transcript=on_agent_transcript,
        on_interrupted=on_interrupted,
    )

    try:
        await session.connect()
        while True:
            # Receive binary PCM audio chunk from browser microphone
            data = await websocket.receive_bytes()
            if data:
                await session.send_audio(data)

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await session.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.web_agent:app", host="0.0.0.0", port=8000, reload=True)

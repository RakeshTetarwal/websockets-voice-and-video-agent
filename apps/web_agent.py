import os
import sys
import json
import uuid
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logger for both console and file
LOG_FILE = PROJECT_ROOT / "voice_agent.log"
logger = logging.getLogger("VoiceAgent")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

from src.core.config import config
from src.services.gemini_live_session import GeminiLiveSession
from src.services.image_processor import image_processor
from src.services.image_generator import image_generator
from src.storage.postgres_db import postgres_db
from src.services.kafka_producer import kafka_producer
from src.services.kafka_consumer import kafka_consumer
from src.workers.image_storage_worker import image_storage_worker

app = FastAPI(title="Gemini Live Multimodal Voice & Vision Agent")

CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")



@app.on_event("startup")
async def on_startup():
    logger.info("Initializing PostgreSQL, dual Kafka streams, and Image Storage Worker...")
    await postgres_db.initialize()
    await kafka_producer.start()
    await kafka_consumer.start()
    await image_storage_worker.start()


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down Kafka streams, Image Worker, and PostgreSQL...")
    await image_storage_worker.stop()
    await kafka_consumer.stop()
    await kafka_producer.stop()
    await postgres_db.close()


class ImageProcessRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/jpeg"
    mode: str = "describe"  # "describe" | "ocr" | "qa"
    question: Optional[str] = None


class ImageGenerateRequest(BaseModel):
    prompt: str


@app.get("/api/logs", response_class=PlainTextResponse)
async def get_logs():
    if LOG_FILE.exists():
        return LOG_FILE.read_text(encoding="utf-8")
    return "No logs recorded yet."


@app.get("/api/config")
async def get_config():
    return {
        "model": config.model,
        "voice_name": config.voice_name,
        "has_api_key": bool(config.api_key),
        "input_sample_rate": config.input_sample_rate,
        "output_sample_rate": config.output_sample_rate,
        "kafka_connected": kafka_producer.is_connected,
        "postgres_connected": postgres_db.pool is not None,
    }


@app.get("/api/conversations")
async def list_conversations(limit: int = 50):
    """Returns stored conversation sessions from PostgreSQL."""
    sessions = await postgres_db.get_all_sessions(limit=limit)
    return {"success": True, "sessions": sessions}


@app.get("/api/conversations/{session_id}")
async def get_conversation_detail(session_id: str):
    """Returns all messages and stored images for a given session from PostgreSQL."""
    messages = await postgres_db.get_session_messages(session_id)
    images = await postgres_db.get_session_images(session_id)
    # If not in db yet, check disk as fallback
    if not images:
        for img_file in sorted(CAPTURES_DIR.glob(f"{session_id}_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True):
            images.append({
                "filename": img_file.name,
                "url": f"/captures/{img_file.name}",
                "size_bytes": img_file.stat().st_size,
                "timestamp": img_file.stat().st_mtime,
            })
    return {"success": True, "messages": messages, "images": images}


@app.get("/api/images")
async def list_all_captured_images(limit: int = 50):
    """Returns all captured images saved in PostgreSQL across all sessions."""
    images = await postgres_db.get_all_images(limit=limit)
    return {"success": True, "images": images}


@app.get("/api/latest-frame")
async def get_latest_frame():
    """Returns URL of the latest camera snapshot processed by Gemini."""
    frame_file = CAPTURES_DIR / "latest_camera_frame.jpg"
    if frame_file.exists():
        return {"has_frame": True, "url": f"/captures/latest_camera_frame.jpg?t={frame_file.stat().st_mtime}"}
    return {"has_frame": False}


@app.post("/api/process-image")
async def process_image_endpoint(req: ImageProcessRequest):
    """Performs OCR, visual Q&A, or in-depth image description using Gemini vision."""
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)

        if req.mode == "ocr":
            result = await image_processor.extract_text_ocr(img_bytes, req.mime_type)
        elif req.mode == "qa":
            question = req.question or "What is shown in this image?"
            result = await image_processor.visual_qa(img_bytes, question, req.mime_type)
        else:
            prompt = req.question or "Describe this image in detail and highlight key visual elements."
            result = await image_processor.analyze_image(img_bytes, prompt, req.mime_type)

        return {"success": True, "result": result, "mode": req.mode}
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/generate-image")
async def generate_image_endpoint(req: ImageGenerateRequest):
    """Generates images using Imagen / Gemini multimodal image models with quota handling."""
    if not req.prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty."}
    logger.info(f"🎨 Generating image for prompt: {req.prompt}")
    result = await image_generator.generate_image(req.prompt)
    return result


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = PROJECT_ROOT / "web" / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    print(f"\n🌐 [WebAgent] Browser WebSocket connected. Assigned Session ID: {session_id}")

    if not config.api_key:
        logger.error("❌ GEMINI_API_KEY is not set in .env!")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "GEMINI_API_KEY is missing. Please set it in .env!"
        }))
        await websocket.close()
        return

    outgoing_queue: asyncio.Queue = asyncio.Queue()
    agent_turn_text = []

    def send_ws_json(payload: dict):
        outgoing_queue.put_nowait(json.dumps(payload))

    def on_audio_chunk(chunk: bytes):
        outgoing_queue.put_nowait(chunk)

    active_camera_frame_url = None

    def on_user_transcript(text: str):
        logger.info(f"👤 [User]: {text}")
        img_url = active_camera_frame_url
        send_ws_json({"type": "user_transcript", "text": text, "image_url": img_url})
        # Emit to Kafka asynchronously with linked camera snapshot
        asyncio.create_task(kafka_producer.emit_event("message", session_id, {
            "role": "user",
            "content": text,
            "image_url": img_url
        }))

    def on_agent_transcript(text: str):
        logger.info(f"🤖 [Gemini]: {text}")
        send_ws_json({"type": "agent_transcript", "text": text})
        agent_turn_text.append(text)

    def on_interrupted():
        logger.info("⚡ [User Interrupted]")
        send_ws_json({"type": "interrupted"})

    def on_turn_complete():
        logger.info("✅ [Turn Complete]")
        send_ws_json({"type": "turn_complete"})
        if agent_turn_text:
            full_reply = "".join(agent_turn_text)
            asyncio.create_task(kafka_producer.emit_event("message", session_id, {
                "role": "gemini",
                "content": full_reply
            }))
            agent_turn_text.clear()

    def on_interim_transcript(text: str):
        send_ws_json({"type": "interim_transcript", "text": text})

    def on_usage_metadata(meta: dict):
        send_ws_json({"type": "usage_metadata", "data": meta})
        asyncio.create_task(kafka_producer.emit_event("usage_metrics", session_id, meta))

    def on_grounding_metadata(meta: dict):
        send_ws_json({"type": "grounding_metadata", "data": meta})

    def on_setup_complete(sid: str):
        logger.info(f"🔗 Gemini Live session ready: {sid}")
        send_ws_json({"type": "setup_complete", "session_id": sid})

    def on_go_away(time_left: str):
        logger.warning(f"⚠️ Gemini Live session expiration warning: {time_left}")
        send_ws_json({"type": "go_away", "time_left": time_left})

    def on_tool_cancelled(ids: list):
        logger.info(f"🛑 Tool calls cancelled: {ids}")
        send_ws_json({"type": "tool_cancelled", "ids": ids})

    def on_live_error(err: Exception):
        logger.error(f"❌ [GeminiLiveSession] Session error: {err}")
        send_ws_json({"type": "session_error", "message": f"Gemini connection issue: {err}"})

    session = GeminiLiveSession(
        on_audio_chunk=on_audio_chunk,
        on_user_transcript=on_user_transcript,
        on_agent_transcript=on_agent_transcript,
        on_interim_transcript=on_interim_transcript,
        on_interrupted=on_interrupted,
        on_turn_complete=on_turn_complete,
        on_usage_metadata=on_usage_metadata,
        on_grounding_metadata=on_grounding_metadata,
        on_setup_complete=on_setup_complete,
        on_go_away=on_go_away,
        on_tool_cancelled=on_tool_cancelled,
        on_error=on_live_error,
    )

    # Emit session start event to Kafka
    await kafka_producer.emit_event("session_started", session_id, {
        "model": config.model,
        "voice_name": config.voice_name,
        "metadata": {"user_agent": "browser"}
    })

    async def sender_worker():
        try:
            while True:
                item = await outgoing_queue.get()
                if item is None:
                    break
                if websocket.client_state.name == "CONNECTED":
                    if isinstance(item, bytes):
                        await websocket.send_bytes(item)
                    else:
                        await websocket.send_text(item)
                outgoing_queue.task_done()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.error(f"[Sender Error]: {e}")

    sender_task = asyncio.create_task(sender_worker())
    chunk_counter = 0
    frame_counter = 0
    image_send_lock = asyncio.Lock()
    last_video_send_time = 0.0

    async def dispatch_continuous_frame(raw_bytes: bytes, mime_type: str, f_num: int):
        nonlocal last_video_send_time
        # Drop frame immediately if previous upload is still in flight (non-blocking)
        if image_send_lock.locked():
            return
        async with image_send_lock:
            now = asyncio.get_event_loop().time()
            # Strictly enforce >= 1.5s interval (< 0.67 FPS) to prevent Gemini gateway 1011 drops
            if now - last_video_send_time < 1.45:
                return
            last_video_send_time = now
            if session.is_connected:
                await session.send_image(raw_bytes, mime_type)

    try:
        logger.info(f"⏳ Connecting to Gemini Live ({config.model})...")
        await session.connect()
        logger.info("🟢 Connected to Gemini Live! Listening for microphone and continuous camera stream...")
        send_ws_json({"type": "ready", "session_id": session_id})

        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                audio_bytes = msg["bytes"]
                chunk_counter += 1
                if chunk_counter % 20 == 1:
                    logger.info(f"🎙️ Mic audio stream (chunk #{chunk_counter}, {len(audio_bytes)} bytes)")
                await session.send_audio(audio_bytes)

            elif "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                    msg_type = payload.get("type")

                    if msg_type == "text":
                        user_text = payload.get("text", "")
                        logger.info(f"💬 User text: {user_text}")
                        await session.send_text(user_text)

                    elif msg_type == "image":
                        # Continuous live video frame from webcam
                        active_camera_frame_url = f"/captures/{session_id}_latest_camera_frame.jpg"
                        img_b64 = payload.get("data", "")
                        mime = payload.get("mime_type", "image/jpeg")
                        if img_b64:
                            raw_b64 = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
                            img_bytes = base64.b64decode(raw_b64)
                            frame_counter += 1
                            if frame_counter % 10 == 1:
                                # Stream rolling camera frame into dedicated 'voice.images' Kafka topic
                                asyncio.create_task(kafka_producer.emit_image_event(
                                    session_id=session_id,
                                    image_id="latest_camera_frame",
                                    image_b64=raw_b64,
                                    mime_type=mime
                                ))
                                logger.info(f"📹 Streaming camera frame #{frame_counter} ({len(img_bytes)} bytes)")
                            # Non-blocking async dispatch: Never stalls the microphone receive loop
                            asyncio.create_task(dispatch_continuous_frame(img_bytes, mime, frame_counter))

                    elif msg_type == "image_ask":
                        # Uploaded photo or manual snapshot with prompt
                        img_b64 = payload.get("data", "")
                        mime = payload.get("mime_type", "image/jpeg")
                        prompt = payload.get("prompt", "What do you see in this image?")
                        if img_b64:
                            raw_b64 = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
                            img_bytes = base64.b64decode(raw_b64)

                            img_id = uuid.uuid4().hex[:8]
                            img_url = f"/captures/{session_id}_{img_id}.jpg"

                            logger.info(f"📸 Visual request: '{prompt}' ({len(img_bytes)} bytes) -> streaming to Kafka voice.images")

                            # 1. Stream image binary to dedicated Kafka 'voice.images' topic (non-blocking)
                            asyncio.create_task(kafka_producer.emit_image_event(
                                session_id=session_id,
                                image_id=img_id,
                                image_b64=raw_b64,
                                mime_type=mime,
                                prompt=prompt
                            ))

                            # 2. Stream conversation message to Kafka 'voice.conversations' with image reference
                            asyncio.create_task(kafka_producer.emit_event("message", session_id, {
                                "role": "user",
                                "content": f"[Shared Image]: {prompt}",
                                "image_url": img_url
                            }))

                            send_ws_json({
                                "type": "user_transcript",
                                "text": f"[Shared Image]: {prompt}",
                                "image_url": img_url
                            })

                            # 3. Stream to Gemini Live with lock so it does not collide with continuous frames
                            async def dispatch_inspect(b: bytes, m: str, p: str):
                                nonlocal last_video_send_time
                                async with image_send_lock:
                                    last_video_send_time = asyncio.get_event_loop().time()
                                    if session.is_connected:
                                        await session.send_image(b, m)
                                        await asyncio.sleep(0.15)
                                        await session.send_text(p)

                            asyncio.create_task(dispatch_inspect(img_bytes, mime, prompt))

                except json.JSONDecodeError:
                    pass

            elif msg.get("type") == "websocket.disconnect":
                break

    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info("🔌 Browser disconnected.")
    except Exception as e:
        logger.error(f"❌ WebAgent error: {e}")
        send_ws_json({"type": "error", "message": str(e)})
    finally:
        await kafka_producer.emit_event("session_ended", session_id, {})
        outgoing_queue.put_nowait(None)
        sender_task.cancel()
        await session.close()
        logger.info(f"🔒 Session {session_id} closed.")


if __name__ == "__main__":
    import uvicorn
    sep = "=" * 65
    print("\n" + sep)
    print("🎙️   GEMINI LIVE MULTIMODAL VOICE & VISION AGENT")
    print(f"• Model:      {config.model}")
    print(f"• Voice:      {config.voice_name}")
    print("• Kafka:      localhost:9092 (topic: voice.conversations)")
    print("• PostgreSQL: localhost:5432 (db: voice_agent)")
    print("• Web UI:     http://localhost:8000")
    print(sep + "\n")
    uvicorn.run("apps.web_agent:app", host="0.0.0.0", port=8000, reload=True)

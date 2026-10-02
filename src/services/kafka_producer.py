import os
import json
import logging
import asyncio
from typing import Optional
from aiokafka import AIOKafkaProducer

logger = logging.getLogger("VoiceAgent")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CONVERSATION_TOPIC = os.getenv("KAFKA_CONVERSATION_TOPIC", "voice.conversations")
KAFKA_IMAGE_TOPIC = os.getenv("KAFKA_IMAGE_TOPIC", "voice.images")


class KafkaProducerService:
    """High-throughput, asynchronous, dual-stream Kafka event producer.
    
    Streams:
      1. voice.conversations: Lightweight dialogue, transcripts, turn signals & metrics.
      2. voice.images: Dedicated high-volume stream for video frames & camera captures.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        conversation_topic: str = KAFKA_CONVERSATION_TOPIC,
        image_topic: str = KAFKA_IMAGE_TOPIC,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.conversation_topic = conversation_topic
        self.image_topic = image_topic
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_connected = False

    async def start(self):
        """Starts the Kafka producer connection."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
                acks=1,
                linger_ms=5,
                max_request_size=10485760,  # 10MB limit for image payloads
            )
            await self.producer.start()
            self.is_connected = True
            logger.info(f"🚀 Kafka Producer connected! (Streams: '{self.conversation_topic}', '{self.image_topic}')")
        except Exception as e:
            logger.warning(f"⚠️ Kafka Producer connection skipped / unavailable: {e}")
            self.producer = None
            self.is_connected = False

    async def emit_event(self, event_type: str, session_id: str, data: dict):
        """Emits conversational transcripts and events to voice.conversations."""
        if not self.is_connected or not self.producer:
            return
        payload = {
            "event_type": event_type,
            "session_id": session_id,
            "data": data,
        }
        try:
            await self.producer.send(self.conversation_topic, key=session_id, value=payload)
        except Exception as e:
            logger.error(f"Failed to emit conversation event '{event_type}': {e}")

    async def emit_image_event(self, session_id: str, image_id: str, image_b64: str, mime_type: str = "image/jpeg", prompt: str = None):
        """Emits camera frames and captures to the dedicated voice.images stream."""
        if not self.is_connected or not self.producer:
            return
        payload = {
            "session_id": session_id,
            "image_id": image_id,
            "image_b64": image_b64,
            "mime_type": mime_type,
            "prompt": prompt,
        }
        try:
            # Partitioned by session_id so frames for each session stay ordered
            await self.producer.send(self.image_topic, key=session_id, value=payload)
        except Exception as e:
            logger.error(f"Failed to emit image event to '{self.image_topic}': {e}")

    async def stop(self):
        """Flushes and stops the Kafka producer."""
        if self.producer and self.is_connected:
            self.is_connected = False
            try:
                await self.producer.stop()
            except Exception:
                pass


kafka_producer = KafkaProducerService()

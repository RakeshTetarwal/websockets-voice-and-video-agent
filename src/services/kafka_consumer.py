import os
import json
import logging
import asyncio
from typing import Optional
from aiokafka import AIOKafkaConsumer
from src.storage.postgres_db import postgres_db

logger = logging.getLogger("VoiceAgent")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_CONVERSATION_TOPIC", "voice.conversations")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "voice_agent_postgres_sink")


class KafkaPostgresConsumer:
    """Consumes conversation events from Kafka and writes them into PostgreSQL."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS, topic: str = KAFKA_TOPIC, group_id: str = KAFKA_GROUP_ID):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_running = False
        self._consume_task: Optional[asyncio.Task] = None

    async def start(self):
        """Starts the consumer loop in the background."""
        try:
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            await self.consumer.start()
            self.is_running = True
            self._consume_task = asyncio.create_task(self._consume_loop())
            logger.info(f"📥 Kafka Postgres Consumer listening on '{self.topic}' (Group: {self.group_id})")
        except Exception as e:
            logger.warning(f"⚠️ Kafka Consumer skipped / unavailable: {e}")
            self.consumer = None
            self.is_running = False

    async def _consume_loop(self):
        """Polls Kafka in micro-batches and commits writes to PostgreSQL."""
        message_batch = []
        last_flush_time = asyncio.get_event_loop().time()
        BATCH_SIZE = 10
        FLUSH_INTERVAL = 1.5

        while self.is_running and self.consumer:
            try:
                # Fetch a batch of records from Kafka with timeout
                batch_data = await self.consumer.getmany(timeout_ms=1000, max_records=BATCH_SIZE)
                now = asyncio.get_event_loop().time()

                for tp, messages in batch_data.items():
                    for msg in messages:
                        event = msg.value
                        event_type = event.get("event_type")
                        session_id = event.get("session_id")
                        data = event.get("data", {})

                        if event_type == "session_started":
                            await postgres_db.save_session_start(
                                session_id=session_id,
                                model=data.get("model", ""),
                                voice_name=data.get("voice_name", ""),
                                metadata=data.get("metadata", {}),
                            )

                        elif event_type == "message":
                            message_batch.append((
                                session_id,
                                data.get("role", "user"),
                                data.get("content", ""),
                                data.get("image_url"),
                                json.dumps(data.get("metadata", {})),
                            ))

                        elif event_type == "usage_metrics":
                            await postgres_db.update_session_metrics(
                                session_id=session_id,
                                total_tokens=data.get("total_tokens", 0),
                                prompt_tokens=data.get("prompt_tokens", 0),
                                response_tokens=data.get("response_tokens", 0),
                            )

                        elif event_type == "session_ended":
                            await postgres_db.save_session_end(session_id)

                # Flush message batch if full or timer elapsed
                if len(message_batch) >= BATCH_SIZE or (message_batch and (now - last_flush_time) >= FLUSH_INTERVAL):
                    batch_to_commit = list(message_batch)
                    message_batch.clear()
                    last_flush_time = now
                    await postgres_db.save_messages_batch(batch_to_commit)
                    logger.info(f"📥 Committed batch of {len(batch_to_commit)} messages to PostgreSQL")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Kafka consumer batch error: {e}")
                await asyncio.sleep(1)

        # Flush any remaining on exit
        if message_batch:
            await postgres_db.save_messages_batch(message_batch)

    async def stop(self):
        """Stops the consumer loop."""
        self.is_running = False
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None


kafka_consumer = KafkaPostgresConsumer()

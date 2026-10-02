from src.storage.postgres_db import postgres_db
import os
import json
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from aiokafka import AIOKafkaConsumer

logger = logging.getLogger("VoiceAgent")

PROJECT_ROOT = Path(__file__).parent.parent.parent
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_IMAGE_TOPIC = os.getenv("KAFKA_IMAGE_TOPIC", "voice.images")
KAFKA_IMAGE_GROUP_ID = os.getenv("KAFKA_IMAGE_GROUP_ID", "voice_agent_image_storage_workers")


class ImageStorageWorker:
    """Dedicated background worker that consumes image frames from the 'voice.images'
    Kafka stream and flushes them to disk storage and PostgreSQL in high-throughput batches.
    
    Can run inside the FastAPI process or as an independent standalone worker daemon.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic: str = KAFKA_IMAGE_TOPIC,
        group_id: str = KAFKA_IMAGE_GROUP_ID,
        target_dir: Path = CAPTURES_DIR,
        batch_size: int = 10,
        flush_interval: float = 1.5,
        max_workers: int = 4,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.consumer: Optional[AIOKafkaConsumer] = None
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="img_saver")
        self._consume_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.total_saved = 0

    async def start(self):
        """Starts consuming from voice.images topic."""
        try:
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                max_partition_fetch_bytes=10485760,  # 10MB per batch
            )
            await self.consumer.start()
            self.is_running = True
            self._consume_task = asyncio.create_task(self._consume_loop())
            logger.info(f"📸 ImageStorageWorker active on dedicated stream '{self.topic}' (Group: {self.group_id})")
        except Exception as e:
            logger.warning(f"⚠️ ImageStorageWorker skipped / unavailable: {e}")
            self.consumer = None
            self.is_running = False

    async def _consume_loop(self):
        """Consumes image frames in batches and writes them to storage and PostgreSQL."""
        buffer: List[dict] = []
        last_flush = asyncio.get_event_loop().time()

        while self.is_running and self.consumer:
            try:
                # Fetch a batch of image messages from Kafka
                batch_records = await self.consumer.getmany(timeout_ms=1000, max_records=self.batch_size)
                now = asyncio.get_event_loop().time()

                for tp, messages in batch_records.items():
                    for msg in messages:
                        data = msg.value
                        image_id = data.get("image_id", "img")
                        image_b64 = data.get("image_b64", "")
                        session_id = data.get("session_id", "sess")
                        mime_type = data.get("mime_type", "image/jpeg")
                        prompt = data.get("prompt")

                        if image_b64:
                            if "," in image_b64:
                                image_b64 = image_b64.split(",", 1)[1]
                            try:
                                img_bytes = base64.b64decode(image_b64)
                                filename = f"{session_id}_{image_id}.jpg"
                                buffer.append({
                                    "filename": filename,
                                    "img_bytes": img_bytes,
                                    "session_id": session_id,
                                    "image_id": image_id,
                                    "mime_type": mime_type,
                                    "prompt": prompt,
                                    "size_bytes": len(img_bytes),
                                    "image_url": f"/captures/{filename}",
                                })
                            except Exception as ex:
                                logger.error(f"Failed to decode image frame {image_id}: {ex}")

                # Flush condition: batch size reached OR flush interval passed
                if len(buffer) >= self.batch_size or (buffer and (now - last_flush) >= self.flush_interval):
                    batch_to_write = list(buffer)
                    buffer.clear()
                    last_flush = now
                    await self._flush_images(batch_to_write)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ImageStorageWorker loop error: {e}")
                await asyncio.sleep(1)

        if buffer:
            await self._flush_images(buffer)

    async def _flush_images(self, batch: List[dict]):
        """Persists batch of images to both disk storage and PostgreSQL."""
        if not batch:
            return
        count = len(batch)
        try:
            # 1. Disk write via thread pool
            disk_items = [(item["filename"], item["img_bytes"]) for item in batch]
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._thread_pool, self._sync_write, disk_items)

            # 2. PostgreSQL batch metadata insert
            db_records = [
                (
                    item["session_id"],
                    item["image_id"],
                    item["filename"],
                    item["image_url"],
                    item["mime_type"],
                    item["size_bytes"],
                    item["prompt"]
                )
                for item in batch
            ]
            await postgres_db.save_images_batch(db_records)

            self.total_saved += count
            logger.info(f"📸 ImageStorageWorker flushed {count} images to disk & PostgreSQL (Total: {self.total_saved})")
        except Exception as e:
            logger.error(f"Failed to flush image batch: {e}")

    def _sync_write(self, batch: List[Tuple[str, bytes]]):
        """Threadpool disk write execution."""
        for filename, img_bytes in batch:
            target_path = self.target_dir / filename
            try:
                target_path.write_bytes(img_bytes)
            except Exception as e:
                logger.error(f"Disk write error for {filename}: {e}")

    async def stop(self):
        """Shuts down consumer and thread pool."""
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
        self._thread_pool.shutdown(wait=True)


image_storage_worker = ImageStorageWorker()

if __name__ == "__main__":
    # Can also be launched as a standalone CLI worker: python -m src.workers.image_storage_worker
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Starting standalone ImageStorageWorker daemon...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(image_storage_worker.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Stopping ImageStorageWorker...")
        loop.run_until_complete(image_storage_worker.stop())

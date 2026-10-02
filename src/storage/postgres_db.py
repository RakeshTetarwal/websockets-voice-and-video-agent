import os
import json
import logging
import asyncpg
from typing import Optional, List, Dict, Any

logger = logging.getLogger("VoiceAgent")

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgrespassword@localhost:5432/voice_agent")


class PostgresManager:
    """Manages PostgreSQL connection pool and queries for conversation persistence."""

    def __init__(self, dsn: str = POSTGRES_URL):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Creates connection pool and ensures database schema exists."""
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=2,
                max_size=10,
                command_timeout=10,
            )
            await self._create_tables()
            logger.info("🐘 PostgreSQL connection pool established & schema ready.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            self.pool = None

    async def _create_tables(self):
        if not self.pool:
            return
        query = """
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            id VARCHAR(64) PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMPTZ,
            model VARCHAR(64),
            voice_name VARCHAR(32),
            total_tokens INTEGER DEFAULT 0,
            prompt_tokens INTEGER DEFAULT 0,
            response_tokens INTEGER DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS conversation_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(64) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            image_url TEXT,
            tokens INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS conversation_images (
            id BIGSERIAL PRIMARY KEY,
            session_id VARCHAR(64) REFERENCES conversation_sessions(id) ON DELETE CASCADE,
            image_id VARCHAR(64) NOT NULL,
            filename VARCHAR(255) NOT NULL,
            image_url TEXT NOT NULL,
            mime_type VARCHAR(32) DEFAULT 'image/jpeg',
            size_bytes INTEGER DEFAULT 0,
            prompt TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_id ON conversation_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON conversation_messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON conversation_sessions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_images_session_id ON conversation_images(session_id);
        CREATE INDEX IF NOT EXISTS idx_images_created_at ON conversation_images(created_at DESC);
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query)

    async def save_session_start(self, session_id: str, model: str, voice_name: str, metadata: dict = None):
        if not self.pool:
            return
        q = """
        INSERT INTO conversation_sessions (id, model, voice_name, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (id) DO NOTHING;
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q, session_id, model, voice_name, json.dumps(metadata or {}))

    async def save_message(self, session_id: str, role: str, content: str, image_url: str = None, metadata: dict = None):
        if not self.pool:
            return
        q_msg = """
        INSERT INTO conversation_messages (session_id, role, content, image_url, metadata)
        VALUES ($1, $2, $3, $4, $5::jsonb);
        """
        q_update = """
        UPDATE conversation_sessions
        SET message_count = message_count + 1
        WHERE id = $1;
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q_msg, session_id, role, content, image_url, json.dumps(metadata or {}))
            await conn.execute(q_update, session_id)

    
    async def save_messages_batch(self, messages: List[Tuple[str, str, str, Optional[str], str]]):
        """Batch inserts multiple conversation messages into PostgreSQL in a single database transaction."""
        if not self.pool or not messages:
            return
        q_msg = """
        INSERT INTO conversation_messages (session_id, role, content, image_url, metadata)
        VALUES ($1, $2, $3, $4, $5::jsonb);
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(q_msg, messages)

    async def update_session_metrics(self, session_id: str, total_tokens: int, prompt_tokens: int, response_tokens: int):
        if not self.pool:
            return
        q = """
        UPDATE conversation_sessions
        SET total_tokens = $2, prompt_tokens = $3, response_tokens = $4
        WHERE id = $1;
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q, session_id, total_tokens, prompt_tokens, response_tokens)

    async def save_session_end(self, session_id: str):
        if not self.pool:
            return
        q = """
        UPDATE conversation_sessions
        SET ended_at = CURRENT_TIMESTAMP
        WHERE id = $1;
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q, session_id)

    async def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        q = """
        SELECT id, created_at, ended_at, model, voice_name, total_tokens, message_count
        FROM conversation_sessions
        ORDER BY created_at DESC
        LIMIT $1;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, limit)
            result = []
            for r in rows:
                d = dict(r)
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat()
                if d.get("ended_at"):
                    d["ended_at"] = d["ended_at"].isoformat()
                result.append(d)
            return result

    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        q = """
        SELECT id, session_id, role, content, timestamp, image_url, metadata
        FROM conversation_messages
        WHERE session_id = $1
        ORDER BY id ASC;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, session_id)
            result = []
            for r in rows:
                d = dict(r)
                if d.get("timestamp"):
                    d["timestamp"] = d["timestamp"].isoformat()
                if isinstance(d.get("metadata"), str):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except Exception:
                        pass
                result.append(d)
            return result

    async def save_images_batch(self, image_records: List[tuple]):
        """Batch inserts multiple image metadata records into PostgreSQL."""
        if not self.pool or not image_records:
            return
        q = """
        INSERT INTO conversation_images (session_id, image_id, filename, image_url, mime_type, size_bytes, prompt)
        VALUES ($1, $2, $3, $4, $5, $6, $7);
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(q, image_records)

    async def get_session_images(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns all captured images for a session directly from PostgreSQL."""
        if not self.pool:
            return []
        q = """
        SELECT id, session_id, image_id, filename, image_url, mime_type, size_bytes, prompt, created_at
        FROM conversation_images
        WHERE session_id = $1
        ORDER BY created_at DESC;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, session_id)
            result = []
            for r in rows:
                d = dict(r)
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat()
                result.append(d)
            return result

    async def get_all_images(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns all captured images across all sessions directly from PostgreSQL."""
        if not self.pool:
            return []
        q = """
        SELECT id, session_id, image_id, filename, image_url, mime_type, size_bytes, prompt, created_at
        FROM conversation_images
        ORDER BY created_at DESC
        LIMIT $1;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, limit)
            result = []
            for r in rows:
                d = dict(r)
                if d.get("created_at"):
                    d["created_at"] = d["created_at"].isoformat()
                result.append(d)
            return result

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None


postgres_db = PostgresManager()

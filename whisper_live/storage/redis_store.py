"""
Redis storage layer for WhisperLive sessions.
Handles hot data storage for active sessions with TTL-based expiration.
"""

import json
import logging
from typing import Dict, List, Optional, Set
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisSessionStore:
    """
    Redis-based storage for active session data.

    Key patterns:
        session:{session_id}:meta          - Hash: session metadata
        session:{session_id}:segments      - List: raw segments (JSON)
        session:{session_id}:consolidated  - List: consolidated blocks (JSON)
        session:{session_id}:speakers      - Set: unique speaker names
        session:{session_id}:context       - Hash: Whisper context (prompt, speaker)
    """

    def __init__(self, redis_url: str, default_ttl: int = 3600):
        """
        Initialize Redis session store.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
            default_ttl: Default session TTL in seconds (default 1 hour)
        """
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"[RedisStore] Connected to Redis at {redis_url}")
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to connect to Redis: {e}")
            raise

        self.default_ttl = default_ttl

    def _session_key(self, session_id: str, suffix: str) -> str:
        """Generate Redis key for session data."""
        return f"session:{session_id}:{suffix}"

    def save_session_metadata(self, session_id: str, metadata: Dict) -> bool:
        """
        Save session metadata to Redis.

        Args:
            session_id: Session ID
            metadata: Dictionary with session metadata (status, config, timestamps)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._session_key(session_id, "meta")
            # Convert all values to strings for Redis Hash
            string_metadata = {k: json.dumps(v) if not isinstance(v, str) else v
                             for k, v in metadata.items()}

            self.redis_client.hset(key, mapping=string_metadata)
            self.redis_client.expire(key, self.default_ttl)
            logger.debug(f"[RedisStore] Saved metadata for session {session_id}")
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to save metadata for {session_id}: {e}")
            return False

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """
        Get session metadata from Redis.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with metadata or None if not found
        """
        try:
            key = self._session_key(session_id, "meta")
            data = self.redis_client.hgetall(key)
            if not data:
                return None

            # Parse JSON values
            parsed = {}
            for k, v in data.items():
                try:
                    parsed[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed[k] = v

            return parsed
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to get metadata for {session_id}: {e}")
            return None

    def save_segment(self, session_id: str, segment: Dict) -> bool:
        """
        Append a segment to session's segment list.

        Args:
            session_id: Session ID
            segment: Segment dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._session_key(session_id, "segments")
            self.redis_client.rpush(key, json.dumps(segment))
            self.redis_client.expire(key, self.default_ttl)
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to save segment for {session_id}: {e}")
            return False

    def get_segments(self, session_id: str) -> List[Dict]:
        """
        Get all segments for a session.

        Args:
            session_id: Session ID

        Returns:
            List of segment dictionaries
        """
        try:
            key = self._session_key(session_id, "segments")
            segments_json = self.redis_client.lrange(key, 0, -1)
            return [json.loads(s) for s in segments_json]
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to get segments for {session_id}: {e}")
            return []

    def save_consolidated_block(self, session_id: str, block: Dict) -> bool:
        """
        Append a consolidated block to session's consolidated list.

        Args:
            session_id: Session ID
            block: Consolidated block dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._session_key(session_id, "consolidated")
            self.redis_client.rpush(key, json.dumps(block))
            self.redis_client.expire(key, self.default_ttl)
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to save consolidated block for {session_id}: {e}")
            return False

    def get_consolidated_blocks(self, session_id: str) -> List[Dict]:
        """
        Get all consolidated blocks for a session.

        Args:
            session_id: Session ID

        Returns:
            List of consolidated block dictionaries
        """
        try:
            key = self._session_key(session_id, "consolidated")
            blocks_json = self.redis_client.lrange(key, 0, -1)
            return [json.loads(b) for b in blocks_json]
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to get consolidated blocks for {session_id}: {e}")
            return []

    def add_speaker(self, session_id: str, speaker_name: str) -> bool:
        """
        Add a speaker to session's speaker set.

        Args:
            session_id: Session ID
            speaker_name: Speaker name

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._session_key(session_id, "speakers")
            self.redis_client.sadd(key, speaker_name)
            self.redis_client.expire(key, self.default_ttl)
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to add speaker for {session_id}: {e}")
            return False

    def get_speakers(self, session_id: str) -> Set[str]:
        """
        Get all speakers for a session.

        Args:
            session_id: Session ID

        Returns:
            Set of speaker names
        """
        try:
            key = self._session_key(session_id, "speakers")
            return self.redis_client.smembers(key)
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to get speakers for {session_id}: {e}")
            return set()

    def save_context(self, session_id: str, context: Dict) -> bool:
        """
        Save Whisper context for session (prompt, last_speaker, etc.).

        Args:
            session_id: Session ID
            context: Context dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._session_key(session_id, "context")
            string_context = {k: str(v) for k, v in context.items()}
            self.redis_client.hset(key, mapping=string_context)
            self.redis_client.expire(key, self.default_ttl)
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to save context for {session_id}: {e}")
            return False

    def get_context(self, session_id: str) -> Optional[Dict]:
        """
        Get Whisper context for session.

        Args:
            session_id: Session ID

        Returns:
            Context dictionary or None if not found
        """
        try:
            key = self._session_key(session_id, "context")
            return self.redis_client.hgetall(key)
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to get context for {session_id}: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete all data for a session.

        Args:
            session_id: Session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            keys = [
                self._session_key(session_id, "meta"),
                self._session_key(session_id, "segments"),
                self._session_key(session_id, "consolidated"),
                self._session_key(session_id, "speakers"),
                self._session_key(session_id, "context"),
            ]
            self.redis_client.delete(*keys)
            logger.info(f"[RedisStore] Deleted all data for session {session_id}")
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to delete session {session_id}: {e}")
            return False

    def set_ttl(self, session_id: str, ttl_seconds: int) -> bool:
        """
        Update TTL for all session keys.

        Args:
            session_id: Session ID
            ttl_seconds: New TTL in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            keys = [
                self._session_key(session_id, "meta"),
                self._session_key(session_id, "segments"),
                self._session_key(session_id, "consolidated"),
                self._session_key(session_id, "speakers"),
                self._session_key(session_id, "context"),
            ]
            for key in keys:
                self.redis_client.expire(key, ttl_seconds)
            return True
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to set TTL for {session_id}: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists in Redis.

        Args:
            session_id: Session ID

        Returns:
            True if exists, False otherwise
        """
        try:
            key = self._session_key(session_id, "meta")
            return self.redis_client.exists(key) > 0
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to check existence for {session_id}: {e}")
            return False

    def get_all_session_ids(self) -> List[str]:
        """
        Get all active session IDs.

        Returns:
            List of session IDs
        """
        try:
            pattern = "session:*:meta"
            keys = self.redis_client.keys(pattern)
            # Extract session_id from key pattern "session:{session_id}:meta"
            session_ids = [key.split(":")[1] for key in keys]
            return session_ids
        except RedisError as e:
            logger.error(f"[RedisStore] Failed to get all session IDs: {e}")
            return []

    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.info("[RedisStore] Connection closed")
        except RedisError as e:
            logger.error(f"[RedisStore] Error closing connection: {e}")

"""
Session management module for WhisperLive unified server.

This module provides session lifecycle management, including creation,
storage, retrieval, and TTL-based cleanup.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session lifecycle states."""
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class SessionConfig:
    """Immutable session configuration."""
    language: Optional[str] = None
    model: str = "small"
    task: str = "transcribe"
    use_vad: bool = True
    send_last_n_segments: int = 10
    no_speech_thresh: float = 0.45
    clip_audio: bool = False
    same_output_threshold: int = 10


@dataclass
class SessionData:
    """
    Session data container.

    Attributes:
        session_id: Unique session identifier (UUID v4)
        status: Current session status
        created_at: Unix timestamp of creation
        last_activity: Unix timestamp of last activity
        config: Session configuration
        segments: List of transcription segments
        summary: Optional LLM-generated summary
        client: Reference to transcription backend client
    """
    session_id: str
    status: SessionStatus
    created_at: float
    last_activity: float
    config: SessionConfig
    segments: List[Dict] = field(default_factory=list)
    summary: Optional[Dict] = None
    client: Optional[object] = None

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def add_segment(self, segment: Dict):
        """Add a transcription segment."""
        self.segments.append(segment)
        self.update_activity()

    def close(self):
        """Mark session as closed."""
        self.status = SessionStatus.CLOSED
        self.update_activity()


class SessionStore:
    """
    Thread-safe session storage with TTL-based cleanup.

    Manages session lifecycle, provides concurrent access control,
    and automatic cleanup of expired sessions.
    """

    def __init__(self, session_ttl: int = 3600, cleanup_interval: int = 60):
        """
        Initialize session store.

        Args:
            session_ttl: Session time-to-live in seconds (default: 3600 = 1 hour)
            cleanup_interval: Cleanup task interval in seconds (default: 60)
        """
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.RLock()
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

        logger.info(f"SessionStore initialized with TTL={session_ttl}s, cleanup_interval={cleanup_interval}s")

    def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
            logger.info("Session cleanup task started")

    def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("Session cleanup task stopped")

    def _cleanup_loop(self):
        """Background task to cleanup expired sessions."""
        while not self._stop_cleanup.is_set():
            try:
                self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

            self._stop_cleanup.wait(self.cleanup_interval)

    def cleanup_expired_sessions(self):
        """Remove sessions that exceeded TTL."""
        current_time = time.time()
        expired_sessions = []

        with self._lock:
            for session_id, session_data in list(self._sessions.items()):
                age = current_time - session_data.last_activity
                if age > self.session_ttl:
                    expired_sessions.append(session_id)
                    # Cleanup client resources if needed
                    if session_data.client:
                        try:
                            if hasattr(session_data.client, 'cleanup'):
                                session_data.client.cleanup()
                        except Exception as e:
                            logger.error(f"Error cleaning up client for session {session_id}: {e}")

                    del self._sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions: {expired_sessions}")

    def create_session(self, config: SessionConfig) -> str:
        """
        Create a new session.

        Args:
            config: Session configuration

        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        current_time = time.time()

        session_data = SessionData(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            created_at=current_time,
            last_activity=current_time,
            config=config
        )

        with self._lock:
            self._sessions[session_id] = session_data

        logger.info(f"Session created: {session_id}, config={vars(config)}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieve session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionData or None if not found
        """
        with self._lock:
            return self._sessions.get(session_id)

    def update_session(self, session_id: str, **updates):
        """
        Update session attributes.

        Args:
            session_id: Session identifier
            **updates: Attributes to update
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.update_activity()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]

                # Cleanup client resources
                if session.client:
                    try:
                        if hasattr(session.client, 'cleanup'):
                            session.client.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up client for session {session_id}: {e}")

                del self._sessions[session_id]
                logger.info(f"Session deleted: {session_id}")
                return True
            return False

    def list_sessions(self) -> List[Dict]:
        """
        List all sessions with basic info.

        Returns:
            List of session summaries
        """
        with self._lock:
            return [
                {
                    "session_id": session_id,
                    "status": session.status.value,
                    "created_at": session.created_at,
                    "last_activity": session.last_activity,
                    "segment_count": len(session.segments),
                    "has_summary": session.summary is not None
                }
                for session_id, session in self._sessions.items()
            ]

    def get_session_count(self) -> int:
        """Get total number of sessions."""
        with self._lock:
            return len(self._sessions)


# Global API key validation
_API_KEYS: Set[str] = set()


def set_api_keys(api_keys: List[str]):
    """
    Set valid API keys for authentication.

    Args:
        api_keys: List of valid API keys
    """
    global _API_KEYS
    _API_KEYS = set(api_keys)
    logger.info(f"API keys configured: {len(api_keys)} keys")


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key.

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    if not _API_KEYS:
        # No API keys configured - allow all requests
        logger.warning("No API keys configured - authentication disabled!")
        return True

    return api_key in _API_KEYS

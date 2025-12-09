"""
Enhanced transcription server with session support.

This module extends the original WhisperLive server with session-based
architecture, allowing REST API to create sessions and WebSocket to
connect to existing sessions.
"""

import asyncio
import json
import logging
import threading
from typing import Optional
from queue import Queue

from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
from whisper_live.session_manager import SessionData, SessionStatus

logger = logging.getLogger(__name__)


class MockWebSocket:
    """
    Mock WebSocket that stores messages in a queue instead of sending them.

    This allows the backend (running in a thread) to "send" messages synchronously,
    while the actual FastAPI WebSocket can consume them asynchronously.
    """

    def __init__(self):
        self.message_queue = Queue()
        self.closed = False

    def send(self, data):
        """Sync send method expected by backend - stores message in queue."""
        logger.info(f"[MockWebSocket] send() called, closed={self.closed}, data_type={type(data).__name__}, data_len={len(data) if isinstance(data, str) else 'N/A'}")

        if not self.closed:
            try:
                message = json.loads(data) if isinstance(data, str) else data
                self.message_queue.put(message)
                logger.info(f"[MockWebSocket] Queued message with {len(message.get('segments', []))} segments")
            except Exception as e:
                logger.error(f"[MockWebSocket] Error queuing message: {e}", exc_info=True)
        else:
            logger.warning(f"[MockWebSocket] send() called but websocket is closed!")

    def get_message(self, timeout=0.1):
        """Get next message from queue (non-blocking)."""
        try:
            return self.message_queue.get(timeout=timeout)
        except:
            return None

    def close(self):
        """Mark as closed."""
        self.closed = True


class SessionBackendWrapper:
    """
    Wrapper around WhisperLive backend that stores segments in SessionData.

    This class wraps the original backend client and intercepts segment
    updates to store them in the session manager.
    """

    def __init__(self, backend_client, session_data: SessionData, mock_websocket: MockWebSocket):
        """
        Initialize wrapper.

        Args:
            backend_client: Original backend client (e.g., ServeClientFasterWhisper)
            session_data: Session data container
            mock_websocket: MockWebSocket for message queuing
        """
        self.backend_client = backend_client
        self.session_data = session_data
        self.mock_websocket = mock_websocket
        self.original_send_method = backend_client.send_transcription_to_client

        # Replace send method with our interceptor
        backend_client.send_transcription_to_client = self.intercept_segments

        logger.info(f"SessionBackendWrapper initialized for session {session_data.session_id}")

    def intercept_segments(self, segments):
        """
        Intercept segments before sending to client.

        This method is called instead of the original send_transcription_to_client.
        It stores segments in session data and then calls original method.

        Args:
            segments: List of transcription segments
        """
        logger.info(f"[INTERCEPT] Received {len(segments)} segments for session {self.session_data.session_id}")

        # Store completed segments in session
        completed_count = 0
        for segment in segments:
            logger.debug(f"[INTERCEPT] Segment: completed={segment.get('completed')}, text={segment.get('text', '')[:50]}")
            if segment.get("completed", False):
                # Add segment to session
                self.session_data.add_segment(segment.copy())
                completed_count += 1

        logger.info(f"[INTERCEPT] Stored {completed_count} completed segments in session")

        # Call original send method (which now sends to MockWebSocket queue)
        self.original_send_method(segments)

    def get_message(self):
        """Get next message from mock websocket queue."""
        return self.mock_websocket.get_message()

    def cleanup(self):
        """Cleanup backend resources."""
        # Stop transcription thread
        if hasattr(self.backend_client, 'cleanup'):
            self.backend_client.cleanup()

        # Don't close MockWebSocket yet - let pending transcription finish
        # It will be closed automatically when sender_task is cancelled


class EnhancedTranscriptionServer:
    """
    Enhanced transcription server with session support.

    This server creates backend instances for sessions and manages
    their lifecycle separately from WebSocket connections.
    """

    def __init__(self, backend="faster_whisper", cache_path="~/.cache/whisper-live/"):
        """
        Initialize enhanced transcription server.

        Args:
            backend: Backend type (currently only "faster_whisper" supported)
            cache_path: Path for model cache
        """
        self.backend = backend
        self.cache_path = cache_path
        logger.info(f"EnhancedTranscriptionServer initialized: backend={backend}")

    def create_backend_for_session(
        self,
        session_data: SessionData
    ) -> SessionBackendWrapper:
        """
        Create a backend client for a session.

        Args:
            session_data: Session data container

        Returns:
            SessionBackendWrapper instance
        """
        config = session_data.config

        # For now, only faster_whisper is supported
        if self.backend != "faster_whisper":
            raise ValueError(f"Unsupported backend: {self.backend}")

        logger.info(
            f"Creating faster_whisper backend for session {session_data.session_id}, "
            f"model={config.model}, language={config.language}"
        )

        # Create mock websocket for message queuing
        mock_websocket = MockWebSocket()

        # Create faster-whisper client with mock websocket
        client = ServeClientFasterWhisper(
            websocket=mock_websocket,
            task=config.task,
            language=config.language,
            client_uid=session_data.session_id,
            model=config.model,
            use_vad=config.use_vad,
            send_last_n_segments=config.send_last_n_segments,
            no_speech_thresh=config.no_speech_thresh,
            clip_audio=config.clip_audio,
            same_output_threshold=config.same_output_threshold,
            cache_path=self.cache_path,
            single_model=False  # Each session gets its own model instance
        )

        # Wrap client with session backend wrapper
        wrapper = SessionBackendWrapper(client, session_data, mock_websocket)

        return wrapper

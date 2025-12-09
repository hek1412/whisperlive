"""
Enhanced transcription server with session support.

This module extends the original WhisperLive server with session-based
architecture, allowing REST API to create sessions and WebSocket to
connect to existing sessions.
"""

import json
import logging
import threading
from typing import Optional

from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
from whisper_live.session_manager import SessionData, SessionStatus

logger = logging.getLogger(__name__)


class SessionBackendWrapper:
    """
    Wrapper around WhisperLive backend that stores segments in SessionData.

    This class wraps the original backend client and intercepts segment
    updates to store them in the session manager.
    """

    def __init__(self, backend_client, session_data: SessionData):
        """
        Initialize wrapper.

        Args:
            backend_client: Original backend client (e.g., ServeClientFasterWhisper)
            session_data: Session data container
        """
        self.backend_client = backend_client
        self.session_data = session_data
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
        # Store completed segments in session
        for segment in segments:
            if segment.get("completed", False):
                # Add segment to session
                self.session_data.add_segment(segment.copy())

        # Call original send method to send via WebSocket
        self.original_send_method(segments)

    def cleanup(self):
        """Cleanup backend resources."""
        if hasattr(self.backend_client, 'cleanup'):
            self.backend_client.cleanup()


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
        session_data: SessionData,
        websocket
    ) -> SessionBackendWrapper:
        """
        Create a backend client for a session.

        Args:
            session_data: Session data container
            websocket: WebSocket connection

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

        # Create faster-whisper client
        client = ServeClientFasterWhisper(
            websocket=websocket,
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
        wrapper = SessionBackendWrapper(client, session_data)

        return wrapper

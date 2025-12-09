"""
Unified REST API and WebSocket server for WhisperLive.

This module provides a FastAPI application that combines REST API endpoints
for session management with WebSocket support for real-time transcription.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from whisper_live.session_manager import (
    SessionStore,
    SessionConfig,
    SessionStatus,
    validate_api_key
)
from whisper_live.server_enhanced import EnhancedTranscriptionServer
from whisper_live.summarizer import OllamaSummarizer

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""
    language: Optional[str] = Field(None, description="Language code (e.g., 'ru', 'en')")
    model: str = Field("small", description="Whisper model size")
    task: str = Field("transcribe", description="Task type: 'transcribe' or 'translate'")
    use_vad: bool = Field(True, description="Enable Voice Activity Detection")
    send_last_n_segments: int = Field(10, description="Number of recent segments to send")
    no_speech_thresh: float = Field(0.45, description="No speech probability threshold")
    clip_audio: bool = Field(False, description="Clip audio with no valid segments")
    same_output_threshold: int = Field(10, description="Threshold for repeated output")


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    websocket_url: str


class CloseSessionRequest(BaseModel):
    """Request model for closing a session."""
    generate_summary: bool = Field(False, description="Generate LLM summary")


class CloseSessionResponse(BaseModel):
    """Response model for session closure."""
    session_id: str
    status: str
    transcript_available: bool
    summary: Optional[Dict] = None


class SessionStatusResponse(BaseModel):
    """Response model for session status."""
    session_id: str
    status: str
    created_at: float
    last_activity: float
    duration: float
    total_segments: int


class TranscriptResponse(BaseModel):
    """Response model for transcript retrieval."""
    session_id: str
    status: str
    created_at: float
    last_activity: float
    duration: float
    total_segments: int
    transcript: List[Dict]


class SummaryResponse(BaseModel):
    """Response model for summary retrieval."""
    session_id: str
    status: str
    summary: Optional[Dict] = None


# ============================================================================
# API Key Dependency
# ============================================================================

async def verify_api_key(api_key: Optional[str] = Query(None, description="API key")):
    """
    Verify API key from query parameter.

    Args:
        api_key: API key from query string

    Raises:
        HTTPException: If API key is invalid
    """
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_unified_app(
    session_store: SessionStore,
    transcription_server: EnhancedTranscriptionServer,
    summarizer: Optional[OllamaSummarizer] = None,
    host: str = "0.0.0.0",
    port: int = 9090
) -> FastAPI:
    """
    Create FastAPI application with REST and WebSocket endpoints.

    Args:
        session_store: Session storage manager
        transcription_server: Enhanced transcription server
        summarizer: Optional LLM summarizer
        host: Server host
        port: Server port

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="WhisperLive Unified Server",
        description="Real-time audio transcription with REST API and WebSocket",
        version="1.0.0"
    )

    # Store dependencies in app state
    app.state.session_store = session_store
    app.state.transcription_server = transcription_server
    app.state.summarizer = summarizer
    app.state.host = host
    app.state.port = port

    # ========================================================================
    # REST API Endpoints
    # ========================================================================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sessions": session_store.get_session_count()
        }

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    async def create_session(
        request: CreateSessionRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """
        Create a new transcription session.

        Args:
            request: Session configuration
            api_key: Verified API key

        Returns:
            Session ID and WebSocket URL
        """
        # Create session config
        config = SessionConfig(
            language=request.language,
            model=request.model,
            task=request.task,
            use_vad=request.use_vad,
            send_last_n_segments=request.send_last_n_segments,
            no_speech_thresh=request.no_speech_thresh,
            clip_audio=request.clip_audio,
            same_output_threshold=request.same_output_threshold
        )

        # Create session
        session_id = session_store.create_session(config)

        # Generate WebSocket URL
        protocol = "wss" if "https" in str(app.state.host) else "ws"
        websocket_url = f"{protocol}://{app.state.host}:{app.state.port}/ws/transcribe/{session_id}?api_key={api_key}"

        logger.info(
            f"[SESSION_CREATED] session_id={session_id}, language={request.language}, "
            f"model={request.model}, task={request.task}"
        )

        return CreateSessionResponse(
            session_id=session_id,
            websocket_url=websocket_url
        )

    @app.get("/api/sessions/{session_id}", response_model=SessionStatusResponse)
    async def get_session_status(
        session_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """
        Get session status.

        Args:
            session_id: Session identifier
            api_key: Verified API key

        Returns:
            Session status information
        """
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        duration = session.last_activity - session.created_at

        return SessionStatusResponse(
            session_id=session.session_id,
            status=session.status.value,
            created_at=session.created_at,
            last_activity=session.last_activity,
            duration=duration,
            total_segments=len(session.segments)
        )

    @app.get("/api/sessions", response_model=List[Dict])
    async def list_sessions(api_key: str = Depends(verify_api_key)):
        """
        List all sessions.

        Args:
            api_key: Verified API key

        Returns:
            List of session summaries
        """
        return session_store.list_sessions()

    @app.get("/api/sessions/{session_id}/transcript", response_model=TranscriptResponse)
    async def get_transcript(
        session_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """
        Get session transcript.

        Args:
            session_id: Session identifier
            api_key: Verified API key

        Returns:
            Full transcript with consolidated blocks
        """
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Consolidate transcript
        if summarizer:
            consolidated = summarizer.consolidate_transcript(session.segments)
        else:
            consolidated = session.segments

        duration = session.last_activity - session.created_at

        logger.info(
            f"[TRANSCRIPT] session_id={session_id}, total={len(session.segments)}, "
            f"completed={sum(1 for s in session.segments if s.get('completed'))}, "
            f"blocks={len(consolidated)}"
        )

        return TranscriptResponse(
            session_id=session.session_id,
            status=session.status.value,
            created_at=session.created_at,
            last_activity=session.last_activity,
            duration=duration,
            total_segments=len(session.segments),
            transcript=consolidated
        )

    @app.post("/api/sessions/{session_id}/close", response_model=CloseSessionResponse)
    async def close_session(
        session_id: str,
        request: CloseSessionRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """
        Close a session and optionally generate summary.

        Args:
            session_id: Session identifier
            request: Close session options
            api_key: Verified API key

        Returns:
            Session closure status and optional summary
        """
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Close session
        session.close()

        # Cleanup backend if exists
        if session.client:
            try:
                session.client.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up client for session {session_id}: {e}")

        duration = session.last_activity - session.created_at

        logger.info(
            f"[SESSION_CLOSED] session_id={session_id}, duration={duration:.2f}s, "
            f"total_segments={len(session.segments)}"
        )

        # Generate summary if requested
        summary = None
        if request.generate_summary:
            if not summarizer:
                logger.warning("Summarization requested but summarizer not available")
                summary = {
                    "error": "Summarizer not configured",
                    "generated_at": datetime.utcnow().isoformat() + "Z"
                }
            elif not session.segments:
                logger.warning("Summarization requested but no segments available")
                summary = {
                    "error": "No transcript available for summarization",
                    "generated_at": datetime.utcnow().isoformat() + "Z"
                }
            else:
                logger.info(f"[SUMMARY] Generating summary for session {session_id}")
                try:
                    summary = await summarizer.summarize(session.segments)
                    session.summary = summary
                    logger.info(f"[SUMMARY] Summary generated for session {session_id}")
                except Exception as e:
                    logger.error(f"[SUMMARY] Error generating summary: {e}")
                    summary = {
                        "error": str(e),
                        "generated_at": datetime.utcnow().isoformat() + "Z"
                    }

        return CloseSessionResponse(
            session_id=session.session_id,
            status=session.status.value,
            transcript_available=len(session.segments) > 0,
            summary=summary
        )

    @app.get("/api/sessions/{session_id}/summary", response_model=SummaryResponse)
    async def get_summary(
        session_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """
        Get session summary.

        Args:
            session_id: Session identifier
            api_key: Verified API key

        Returns:
            Session summary if available
        """
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if not session.summary:
            raise HTTPException(status_code=404, detail="Summary not generated for this session")

        return SummaryResponse(
            session_id=session.session_id,
            status=session.status.value,
            summary=session.summary
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(
        session_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """
        Delete a session.

        Args:
            session_id: Session identifier
            api_key: Verified API key

        Returns:
            Deletion confirmation
        """
        deleted = session_store.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"session_id": session_id, "deleted": True}

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/transcribe/{session_id}")
    async def websocket_transcribe(websocket: WebSocket, session_id: str, api_key: Optional[str] = Query(None)):
        """
        WebSocket endpoint for real-time transcription.

        Args:
            websocket: WebSocket connection
            session_id: Session identifier
            api_key: API key for authentication
        """
        # Validate API key
        if not validate_api_key(api_key):
            await websocket.close(code=1008, reason="Invalid API key")
            return

        # Get session
        session = session_store.get_session(session_id)
        if not session:
            await websocket.close(code=1008, reason="Session not found")
            return

        # Accept WebSocket connection
        await websocket.accept()
        logger.info(f"[WS_CONNECTED] session_id={session_id}, status={session.status.value}")

        # Create backend client for session
        try:
            backend_wrapper = transcription_server.create_backend_for_session(session, websocket)
            session.client = backend_wrapper
            session_store.update_session(session_id, client=backend_wrapper)
        except Exception as e:
            logger.error(f"[WS_ERROR] Failed to create backend: {e}")
            await websocket.send_json({"error": str(e)})
            await websocket.close()
            return

        try:
            while True:
                # Receive message from client
                message = await websocket.receive_text()
                data = json.loads(message)

                message_type = data.get("type")

                if message_type == "audio_chunk":
                    # Decode base64 audio
                    audio_b64 = data.get("audio_data")
                    speaker = data.get("speaker", "Unknown")

                    if not audio_b64:
                        logger.warning(f"[WS_WARN] Empty audio data received for session {session_id}")
                        continue

                    # Decode base64 to bytes
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Convert to numpy float32 array
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        # Add speaker info to all segments
                        backend_wrapper.backend_client.current_speaker = speaker

                        # Add audio frames to backend
                        backend_wrapper.backend_client.add_frames(audio_np)

                    except Exception as e:
                        logger.error(f"[WS_ERROR] Error processing audio: {e}")
                        continue

                elif message_type == "end_of_stream":
                    logger.info(f"[WS_EOS] End of stream for session {session_id}")
                    break

                else:
                    logger.warning(f"[WS_WARN] Unknown message type: {message_type}")

        except WebSocketDisconnect:
            logger.info(f"[WS_DISCONNECTED] session_id={session_id}")
        except Exception as e:
            logger.error(f"[WS_ERROR] session_id={session_id}, error={e}")
        finally:
            # Update session status
            if session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.CLOSING

            logger.info(f"[WS_CLOSED] session_id={session_id}")

    return app

"""
Unified REST API and WebSocket server for WhisperLive.

This module provides a FastAPI application that combines REST API endpoints
for session management with WebSocket support for real-time transcription.
"""

import asyncio
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Header, Depends
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

async def verify_api_key(
    api_key: Optional[str] = Query(None, description="API key"),
    x_api_key: Optional[str] = Header(None, description="API key in header")
):
    """
    Verify API key from query parameter or header.

    Priority: query parameter > header

    Args:
        api_key: API key from query string
        x_api_key: API key from X-API-Key header

    Raises:
        HTTPException: If API key is invalid
    """
    # Check query parameter first, then header
    key = api_key or x_api_key

    if not validate_api_key(key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide via ?api_key=... or X-API-Key header"
        )
    return key


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

        # Send connection ready message
        await websocket.send_json({
            "type": "connection_ready",
            "message": "WebSocket connected. Loading model, please wait...",
            "session_id": session_id
        })

        # Create backend client for session (may take time for large models)
        backend_wrapper = None
        try:
            logger.info(f"[WS_BACKEND] Creating backend for session {session_id}...")
            backend_wrapper = transcription_server.create_backend_for_session(session)
            session.client = backend_wrapper
            session_store.update_session(session_id, client=backend_wrapper)

            # Notify client that backend is ready
            await websocket.send_json({
                "type": "backend_ready",
                "message": "Model loaded successfully. Ready to receive audio.",
                "session_id": session_id
            })
            logger.info(f"[WS_BACKEND] Backend ready for session {session_id}")

        except Exception as e:
            logger.error(f"[WS_ERROR] Failed to create backend: {e}")
            await websocket.send_json({"type": "error", "error": str(e)})
            await websocket.close()
            return

        # Background task to send transcription segments to client
        async def send_segments_task():
            """Continuously check for segments from backend and send to client."""
            try:
                while True:
                    # Check for messages from backend (non-blocking)
                    message = backend_wrapper.get_message()
                    if message:
                        # Backend sends: {"uid": "...", "segments": [...]}
                        # Transform to client format: one message per segment
                        segments = message.get("segments", [])
                        for segment in segments:
                            client_message = {
                                "type": "transcription",
                                "session_id": session_id,
                                "text": segment.get("text", ""),
                                "start": float(segment.get("start", 0.0)),
                                "end": float(segment.get("end", 0.0)),
                                "completed": segment.get("completed", False),
                                "speaker": segment.get("speaker", "Unknown")
                            }
                            await websocket.send_json(client_message)
                            # Log outgoing transcription
                            logger.info(
                                f"[WS_SENT] session={session_id}, speaker={client_message['speaker']}, "
                                f"start={client_message['start']}, end={client_message['end']}, "
                                f"completed={client_message['completed']}, text='{client_message['text'][:50]}...'"
                            )
                    await asyncio.sleep(0.01)  # 10ms polling interval
            except Exception as e:
                logger.error(f"[WS_SENDER_ERROR] {e}")

        # Start background task for sending segments
        sender_task = asyncio.create_task(send_segments_task())

        try:
            message_count = 0
            while True:
                # Receive message from client
                message = await websocket.receive_text()
                data = json.loads(message)

                message_type = data.get("type")
                message_count += 1

                # Log every message for debugging
                logger.debug(f"[WS_MSG] session={session_id}, count={message_count}, type={message_type}, keys={list(data.keys())}")

                if message_type == "audio_chunk":
                    # Check if backend is ready
                    if not backend_wrapper:
                        logger.warning(f"[WS_WARN] Backend not ready yet for session {session_id}")
                        await websocket.send_json({
                            "type": "warning",
                            "message": "Backend not ready yet, please wait..."
                        })
                        continue

                    # Extract speaker and timestamp from message
                    speaker = data.get("speaker", "Unknown")
                    client_timestamp = data.get("timestamp")

                    # Log incoming message with speaker info
                    logger.info(
                        f"[WS_RECEIVED] session={session_id}, type=audio_chunk, speaker={speaker}, "
                        f"timestamp={client_timestamp}, msg_count={message_count}"
                    )

                    # Decode base64 audio (support both 'audio_data' and 'audio' field names)
                    audio_b64 = data.get("audio_data") or data.get("audio")

                    # Handle case where audio is a dict (e.g., {"data": "base64_string"})
                    if isinstance(audio_b64, dict):
                        audio_b64 = audio_b64.get("data") or audio_b64.get("audio_data")

                    if not audio_b64:
                        logger.warning(
                            f"[WS_WARN] Empty audio data received for session {session_id}. "
                            f"Message keys: {list(data.keys())}, audio_data: {data.get('audio_data')}, "
                            f"audio: {data.get('audio')}"
                        )
                        continue

                    # Decode base64 to bytes
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Convert to numpy float32 array
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        # Add audio frames to backend with speaker and timestamp
                        backend_wrapper.backend_client.add_frames(
                            audio_np,
                            speaker=speaker,
                            client_timestamp=client_timestamp
                        )

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
            # Cancel sender task
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
            # Update session status
            if session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.CLOSING

            logger.info(f"[WS_CLOSED] session_id={session_id}")

    return app

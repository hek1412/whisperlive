# WhisperLive Project Context

## Project Overview

WhisperLive is a real-time audio transcription service with REST API and WebSocket support, featuring LLM-powered meeting summarization. Built on top of OpenAI's Whisper.

---

## Architecture

### Tech Stack

#### Backend Framework
- **FastAPI** (v0.104+)
  - **Justification**: Unified REST and WebSocket on single port, async support, automatic OpenAPI documentation, high performance
  - **Usage**: Single endpoint for both HTTP REST API and WebSocket connections

#### Transcription Engine
- **faster-whisper** (Primary backend)
  - **Justification**: GPU-accelerated, efficient memory usage, supports large-v3 model
  - **Alternatives**: TensorRT (for maximum performance), OpenVINO (CPU optimization)
  - **Model**: large-v3 (best accuracy for Russian language)

#### LLM Integration
- **vLLM / Ollama**
  - **Justification**: OpenAI-compatible API, supports local and remote deployment
  - **Models**: Qwen3-VL-32B-Instruct-AWQ, Llama-3.3-70B, gpt-oss:20b
  - **Usage**: Meeting summarization with context-aware error correction

#### Data Validation
- **Pydantic v2**
  - **Justification**: Type safety, automatic validation, FastAPI integration

#### Deployment
- **Docker + Docker Compose**
  - **Justification**: Reproducible environment, GPU passthrough, easy scaling
  - **Base Image**: NVIDIA CUDA for GPU support

### Folder Structure

```
Whisperlive/
├── whisper_live/                    # Core application package
│   ├── __init__.py
│   ├── __version__.py
│   ├── backend/                     # Transcription backends
│   │   ├── base.py                  # Base backend interface
│   │   ├── faster_whisper_backend.py
│   │   ├── trt_backend.py           # TensorRT backend
│   │   ├── openvino_backend.py      # OpenVINO backend
│   │   ├── session_backend_wrapper.py  # Session-specific wrapper
│   │   └── translation_backend.py   # Translation support
│   ├── transcriber/                 # Backend-specific transcribers
│   │   ├── transcriber_faster_whisper.py
│   │   ├── transcriber_tensorrt.py
│   │   └── transcriber_openvino.py
│   ├── server.py                    # Original WebSocket server
│   ├── server_enhanced.py           # Enhanced transcription server
│   ├── session_manager.py           # Session lifecycle management
│   ├── rest_api_unified.py          # Unified REST+WebSocket API
│   ├── summarizer.py                # LLM-based summarization
│   ├── vad.py                       # Voice Activity Detection
│   ├── utils.py                     # Utility functions
│   └── client.py                    # WebSocket client (reference)
├── docker/
│   └── Dockerfile.gpu               # GPU-enabled Docker image
├── run_server_unified.py            # Server entry point
├── docker-compose.yml               # Docker orchestration
├── prompt.yaml                      # LLM prompt configuration
├── .env                             # Environment variables (not in git)
├── .env.example                     # Environment template
├── requirements_api.txt             # REST API dependencies
├── README.md                        # User documentation
└── CLAUDE.md                        # This file - AI context
```

### Key Architectural Decisions

#### 1. Unified Port Architecture
**Decision**: Combine REST API and WebSocket on single FastAPI application (port 9090)
**Rationale**:
- Simplifies client connectivity (no separate ports)
- Reduces infrastructure complexity
- Shares authentication and session management

#### 2. Session-Based Design
**Decision**: REST API creates sessions, WebSocket connects to existing session
**Rationale**:
- Decouples session lifecycle from WebSocket connection
- Enables reconnection without losing transcription state
- Allows multiple clients to access same session transcript

#### 3. Thread-Safe Session Store
**Decision**: Centralized `SessionStore` with locking mechanism
**Rationale**:
- Shared state between REST and WebSocket handlers
- TTL-based automatic cleanup
- Thread-safe concurrent access

#### 4. Transcript Consolidation
**Decision**: Deduplication by timestamps + text, grouping by speaker with pause detection
**Rationale**:
- Whisper produces overlapping/duplicate segments
- Improves readability by grouping consecutive utterances
- Preserves chronological order
- Speaker information provided by client in WebSocket messages

#### 5. LLM Summarization as Optional Feature
**Decision**: Summarization triggered by `generate_summary` flag, separate from transcription
**Rationale**:
- Not all use cases need summarization
- Avoids dependency on external LLM service for core functionality
- Allows graceful degradation if LLM API unavailable

#### 6. Base64 Audio Transport
**Decision**: PCM audio encoded as base64 in JSON WebSocket messages
**Rationale**:
- JSON compatibility (WebSocket text frames)
- Simple client implementation
- Trade-off: ~33% overhead acceptable for real-time use

---

## Development Rules

### Code Standards

#### Python Style
- **PEP 8** compliance with 4-space indentation
- **Type hints** required for function signatures
- **Docstrings** required for modules, classes, and public functions (Google style)
- **Line length**: 120 characters max

#### Logging
- Use structured logging with logger name
- Format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Levels:
  - `INFO`: Session lifecycle, API calls, successful operations
  - `WARNING`: Recoverable errors, degraded functionality
  - `ERROR`: Critical failures, exceptions
  - `DEBUG`: Detailed traces (not in production)

#### Error Handling
- Use `HTTPException` for API errors with appropriate status codes
- Always log exceptions before raising
- Return structured error responses: `{"error": "message", "generated_at": "ISO8601"}`
- Never expose internal paths or sensitive data in error messages

### Naming Conventions

#### Files and Modules
- **snake_case** for Python files: `session_manager.py`
- **Descriptive names**: `rest_api_unified.py` not `api.py`

#### Classes
- **PascalCase**: `SessionStore`, `OllamaSummarizer`
- **Descriptive suffixes**: `...Backend`, `...Manager`, `...Validator`

#### Functions and Variables
- **snake_case**: `consolidate_transcript()`, `session_id`
- **Verb prefixes** for functions: `get_`, `create_`, `update_`, `delete_`
- **Boolean prefixes**: `is_`, `has_`, `should_`, `enable_`

#### Constants
- **UPPER_SNAKE_CASE**: `MAX_CLIENTS`, `DEFAULT_PORT`

#### API Endpoints
- **RESTful conventions**:
  - `POST /api/sessions` - Create resource
  - `GET /api/sessions/{id}` - Read resource
  - `POST /api/sessions/{id}/close` - Action
  - `DELETE /api/sessions/{id}` - Delete resource

### Design Patterns

#### Dependency Injection
- FastAPI `Depends()` for API key validation
- Constructor injection for services (`SessionStore`, `EnhancedTranscriptionServer`)

#### Factory Pattern
- `create_unified_app()` factory for FastAPI application
- Backend selection via `BackendType` enum

#### Data Classes
- Use `@dataclass` for data containers: `SessionData`, `SessionConfig`
- Immutable where possible (frozen=True for config)

#### Async/Await
- Use `async def` for FastAPI endpoints
- Use `await` for I/O operations (WebSocket, database)
- Keep synchronous code in separate threads (Whisper inference)

---

## Domain Context

### Terminology

#### Session
- **Definition**: A transcription job with unique ID, configuration, and lifecycle
- **Lifecycle**: `active` → `closing` → `closed`
- **Persistence**: In-memory with TTL-based cleanup (default 3600s)

#### Segment
- **Definition**: Single transcription output from Whisper (1-30 seconds of audio)
- **Attributes**: `text`, `start`, `end`, `speaker`, `completed`
- **Deduplication**: By `f"{start:.2f}:{end:.2f}:{text}"` key

#### Consolidated Block
- **Definition**: Grouped segments by speaker with pause detection
- **Purpose**: Improve readability, remove duplicates
- **Split Conditions**:
  - Speaker change
  - Pause > 3.0 seconds
  - Duration > 30.0 seconds

#### Speaker Attribution
- **Definition**: Speaker labels provided by client application
- **Format**: String field in WebSocket audio messages
- **Example**: `"speaker": "Егор Атанов"`
- **Note**: Server does not perform automatic speaker identification

#### Summarization
- **Definition**: LLM-generated structured meeting summary
- **Output Format**: JSON with fields:
  - `"Название встречи"` - Meeting title (2-7 words)
  - `"Дата и время"` - Timestamp
  - `"Участники"` - List of participants
  - `"Ключевые темы"` - Key topics array
  - `"Принятые решения на встрече и обсуждаемые вопросы"` - Decisions/questions
  - `"Поставленные задачи"` - Tasks with assignees

### Business Logic

#### Audio Format Requirements
- **Sample Rate**: 16 kHz
- **Channels**: Mono
- **Format**: PCM int16 (signed 16-bit)
- **Encoding**: Base64 for JSON transport
- **Chunk Size**: Variable (typically 1-5 seconds)

#### VAD (Voice Activity Detection)
- **Purpose**: Filter silence to reduce processing
- **Implementation**: Silero VAD
- **Threshold**: Configurable via `vad_parameters`

#### Transcript Consolidation Rules
1. **Deduplication**: Skip segments with identical `start:end:text`
2. **Speaker Grouping**: Merge consecutive segments from same speaker
3. **Pause Detection**: Split if gap > 3.0s
4. **Duration Limit**: Split blocks > 30.0s
5. **Order**: Preserve chronological sequence

#### Summarization Workflow
1. **Trigger**: `POST /api/sessions/{id}/close` with `"generate_summary": true`
2. **Prerequisites**: Session must have segments
3. **Process**:
   - Consolidate transcript
   - Generate full_text with speaker labels
   - Send to LLM with prompt from `prompt.yaml`
   - Parse JSON response
   - Store in `session.summary`
4. **Error Handling**: Return error object if LLM fails, transcript still available

### Constraints

#### Technical Constraints
- **GPU Memory**: Model size limited by VRAM (large-v3 requires ~8GB)
- **Concurrent Clients**: Default max 10 (configurable)
- **Session TTL**: 3600 seconds (1 hour) default
- **WebSocket Timeout**: 6000 seconds max connection time
- **LLM Timeout**: 180 seconds for summarization request

#### Data Constraints
- **Session ID**: UUID v4 format
- **Language Codes**: ISO 639-1 (ru, en, etc.)
- **API Keys**: Minimum 16 characters recommended
- **Audio Chunk**: No size limit, but recommend < 5MB per message

#### Runtime Constraints
- **Python**: 3.8+ required
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **Memory**: 32GB minimum recommended, 64GB limit

---

## Workflow

### Development Workflow

#### Local Testing
1. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Build Docker image**:
   ```bash
   docker-compose build
   ```

3. **Run server**:
   ```bash
   docker-compose up -d
   docker logs -f whisperlive-server
   ```

4. **Test endpoints**:
   ```bash
   # Health check
   curl http://localhost:5205/health

   # Create session
   curl -X POST "http://localhost:5205/api/sessions?api_key=your-secret-api-key-change-in-production" \
     -H "Content-Type: application/json" \
     -d '{"language": "ru", "model": "large-v3", "use_vad": true}'
   ```

### Testing Requirements

#### Manual Testing Checklist
- [ ] Create session with valid API key
- [ ] Reject session with invalid API key
- [ ] Connect WebSocket with session_id
- [ ] Send base64 audio chunks
- [ ] Receive transcription segments in real-time
- [ ] Verify speaker labels present
- [ ] Get transcript via GET /transcript
- [ ] Close session with `generate_summary: false`
- [ ] Close session with `generate_summary: true`
- [ ] Get summary via GET /summary
- [ ] List all sessions
- [ ] Delete session
- [ ] Verify session TTL cleanup (wait 1 hour or modify code)

#### Error Cases to Test
- [ ] WebSocket without API key → 1008 close
- [ ] Session not found → 404
- [ ] Invalid audio format → Log error, continue
- [ ] LLM API unavailable → Summary error, transcript OK
- [ ] Empty transcript summarization → "No consolidated transcript" error

### Commit Requirements

#### Commit Message Format
```
<type>(<scope>): <short description>

<detailed description if needed>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `docs`: Documentation update
- `perf`: Performance improvement
- `test`: Test additions

**Examples**:
- `feat(api): add summarization endpoint`
- `fix(consolidation): handle empty segments gracefully`
- `docs(readme): update LLM configuration section`

#### Pre-Commit Checklist
- [ ] Code follows PEP 8 style
- [ ] Type hints added to new functions
- [ ] Docstrings added to public APIs
- [ ] No hardcoded credentials or paths
- [ ] Logging added for significant operations
- [ ] Error handling for external dependencies (LLM, models)
- [ ] README updated if public API changed
- [ ] `.env.example` updated if new environment variables added

### Deployment Checklist

#### Production Deployment
- [ ] Change `API_KEY` from default value
- [ ] Configure `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` for summarization
- [ ] Set appropriate `session_ttl` and `cleanup_interval`
- [ ] Configure HTTPS proxy if needed (`https_proxy` env var)
- [ ] Set resource limits in `docker-compose.yml`
- [ ] Verify GPU accessible: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- [ ] Mount persistent volume for model cache (`./models:/root/.cache/huggingface/hub`)
- [ ] Configure log aggregation (e.g., ELK, Loki)
- [ ] Set up monitoring (GPU utilization, memory, API latency)
- [ ] Test failover behavior (LLM API down, GPU OOM)

---

## Key Files Reference

### Configuration Files
- **docker-compose.yml**: Service orchestration, environment variables, resource limits
- **.env**: Runtime configuration (API keys, LLM settings, proxy)
- **prompt.yaml**: LLM system/user prompts for summarization

### Core Application Files
- **run_server_unified.py**: Entry point, argument parsing, server initialization
- **whisper_live/rest_api_unified.py**: FastAPI app factory, REST endpoints, WebSocket handler
- **whisper_live/session_manager.py**: SessionStore, SessionData, API key validation
- **whisper_live/server_enhanced.py**: Enhanced transcription server with session support
- **whisper_live/summarizer.py**: Ollama/vLLM API client, prompt rendering

### API Models (Pydantic)
- **CreateSessionRequest**: Session creation parameters
- **CreateSessionResponse**: session_id, websocket_url
- **CloseSessionRequest**: generate_summary flag
- **CloseSessionResponse**: session_id, status, segments, summary
- **SessionStatusResponse**: Basic session info

---

## Common Issues & Solutions

### Duplicate Segments in Transcript
**Symptoms**: Same text appears multiple times with different timestamps
**Solutions**:
- Check `consolidate_transcript()` deduplication logic
- Verify `completed` flag set correctly in segments
- Adjust `max_pause` parameter (increase to merge more aggressively)

### LLM Returns Markdown Instead of JSON
**Symptoms**: Summary contains ` ```json` wrappers
**Solutions**:
- Update `prompt.yaml` to emphasize "NO markdown blocks"
- Parse and strip markdown in `summarizer.py`
- Use model with better instruction following (Qwen3, GPT-4)

### Speaker Labels Missing
**Symptoms**: All segments have speaker="Unknown"
**Solutions**:
- Verify client sends `"speaker"` field in WebSocket audio messages
- Check message format: `{"type": "audio_chunk", "audio_data": "...", "speaker": "Name"}`
- Ensure speaker name is not empty string
- Server does not perform automatic speaker identification - client must provide labels

### WebSocket Disconnects Immediately
**Symptoms**: 1008 close code, "Invalid API key" reason
**Solutions**:
- Include `api_key` query parameter in WebSocket URL
- Verify API_KEY environment variable set
- Check `x-api-key` header if using alternative auth

---

## Future Enhancements

### Planned Features
- [ ] Persistent storage (PostgreSQL, Redis) for sessions
- [ ] Real-time streaming summarization (incremental updates)
- [ ] Multi-language translation (not just transcription)
- [ ] WebRTC audio input (browser direct)
- [ ] Prometheus metrics export
- [ ] Webhook notifications (session closed, summary ready)

### Performance Optimizations
- [ ] Batch processing for multiple clients
- [ ] Model caching and warm-up
- [ ] TensorRT backend for production
- [ ] Distributed deployment (separate transcription/API services)

---

## Contact & Support

**Repository**: git@github.com:hek1412/whisperlive.git
**Branch**: main

For issues or questions, check logs with:
```bash
docker logs whisperlive-server-gpu --tail 100 --follow
```

Look for `[SUMMARY]`, `[TRANSCRIPT]`, `[SESSION_CLOSED]` prefixed messages for debugging.

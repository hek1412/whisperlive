#!/usr/bin/env python3
"""
WhisperLive Unified Server Entry Point.

This script starts the FastAPI server with REST and WebSocket support
for real-time audio transcription with session management.
"""

import argparse
import logging
import os
import sys

import uvicorn

from whisper_live.session_manager import SessionStore, set_api_keys
from whisper_live.server_enhanced import EnhancedTranscriptionServer
from whisper_live.summarizer import OllamaSummarizer
from whisper_live.rest_api_unified import create_unified_app
from whisper_live.rate_limiter import RateLimitConfig

# Optional storage imports
try:
    from whisper_live.storage import RedisSessionStore, MongoDBArchiveStore
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.warning("Storage modules not available. Install redis and pymongo packages to enable storage features.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress TensorRT-LLM warnings
logging.getLogger("tensorrt_llm").setLevel(logging.ERROR)
logging.getLogger("tensorrt").setLevel(logging.ERROR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperLive Unified Server - Real-time transcription with REST API and WebSocket"
    )

    # Server configuration
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Server host address (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=9090,
        help='Server port (default: 9090)'
    )

    # Backend configuration
    parser.add_argument(
        '--backend', '-b',
        type=str,
        default='faster_whisper',
        choices=['faster_whisper', 'tensorrt', 'openvino'],
        help='Transcription backend (default: faster_whisper)'
    )
    parser.add_argument(
        '--cache-path', '-c',
        type=str,
        default='~/.cache/whisper-live/',
        help='Model cache path (default: ~/.cache/whisper-live/)'
    )

    # TensorRT specific options
    parser.add_argument(
        '--trt-model-path',
        type=str,
        default=None,
        help='Path to TensorRT engine directory (required for tensorrt backend)'
    )
    parser.add_argument(
        '--trt-multilingual',
        action='store_true',
        help='Use multilingual TensorRT model'
    )
    parser.add_argument(
        '--trt-py-session',
        action='store_true',
        help='Use Python session for TensorRT (default: C++ session)'
    )

    # Session management
    parser.add_argument(
        '--session-ttl',
        type=int,
        default=3600,
        help='Session time-to-live in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--cleanup-interval',
        type=int,
        default=60,
        help='Cleanup task interval in seconds (default: 60)'
    )

    # Authentication
    parser.add_argument(
        '--api-keys',
        type=str,
        nargs='+',
        default=None,
        help='API keys for authentication (space-separated). If not provided, uses API_KEY env var'
    )

    # LLM configuration (from environment or args)
    parser.add_argument(
        '--ollama-base-url',
        type=str,
        default=None,
        help='Ollama/vLLM API base URL (default: from OLLAMA_BASE_URL env or http://localhost:11434)'
    )
    parser.add_argument(
        '--ollama-api-key',
        type=str,
        default=None,
        help='Ollama/vLLM API key (default: from OLLAMA_API_KEY env)'
    )
    parser.add_argument(
        '--ollama-model',
        type=str,
        default=None,
        help='Ollama model name (default: from OLLAMA_MODEL env or qwen2.5:latest)'
    )
    parser.add_argument(
        '--prompt-config',
        type=str,
        default='prompt.yaml',
        help='Path to prompt configuration YAML (default: prompt.yaml)'
    )

    # OpenMP threading
    parser.add_argument(
        '--omp-num-threads', '-omp',
        type=int,
        default=4,
        help='Number of OpenMP threads (default: 4)'
    )

    # Logging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Update logging level if debug flag is set (from CLI arg or env var)
    debug_mode = args.debug or os.environ.get('DEBUG', '0') == '1'
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    # Set OpenMP threads
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
        logger.info(f"Set OMP_NUM_THREADS={args.omp_num_threads}")

    # Configure API keys
    api_keys = args.api_keys
    if not api_keys:
        # Try to load from environment
        env_key = os.getenv("API_KEY")
        if env_key:
            api_keys = [env_key]
            logger.info("Using API_KEY from environment")
        else:
            logger.warning(
                "No API keys configured! Authentication is disabled. "
                "Set API_KEY environment variable or use --api-keys argument."
            )
            api_keys = []

    set_api_keys(api_keys)

    # Initialize session store
    logger.info(
        f"Initializing session store: TTL={args.session_ttl}s, "
        f"cleanup_interval={args.cleanup_interval}s"
    )
    session_store = SessionStore(
        session_ttl=args.session_ttl,
        cleanup_interval=args.cleanup_interval
    )
    session_store.start_cleanup_task()

    # Initialize transcription server
    logger.info(f"Initializing transcription server: backend={args.backend}")

    # Prepare backend-specific arguments
    server_kwargs = {
        "backend": args.backend,
        "cache_path": args.cache_path
    }

    # Add TensorRT-specific parameters if using tensorrt backend
    if args.backend == "tensorrt":
        trt_model_path = args.trt_model_path or os.getenv("TRT_MODEL_PATH", "./trt_engines/whisper_large_v3_float16")
        trt_multilingual = args.trt_multilingual or os.getenv("TRT_MULTILINGUAL", "true").lower() == "true"
        trt_py_session = args.trt_py_session or os.getenv("TRT_PY_SESSION", "false").lower() == "true"

        server_kwargs.update({
            "whisper_tensorrt_path": trt_model_path,
            "trt_multilingual": trt_multilingual,
            "trt_py_session": trt_py_session
        })

        logger.info(
            f"TensorRT backend configuration: "
            f"model_path={trt_model_path}, "
            f"multilingual={trt_multilingual}, "
            f"py_session={trt_py_session}"
        )

    transcription_server = EnhancedTranscriptionServer(**server_kwargs)

    # Initialize summarizer (optional)
    summarizer = None
    try:
        ollama_base_url = args.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
        ollama_api_key = args.ollama_api_key or os.getenv("OLLAMA_API_KEY")
        ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL")

        if ollama_base_url:
            logger.info(f"Initializing summarizer: base_url={ollama_base_url}, model={ollama_model}")
            summarizer = OllamaSummarizer(
                base_url=ollama_base_url,
                api_key=ollama_api_key,
                model=ollama_model,
                prompt_config_path=args.prompt_config
            )
        else:
            logger.warning("Summarizer not configured (OLLAMA_BASE_URL not set)")
    except Exception as e:
        logger.warning(f"Failed to initialize summarizer: {e}. Summarization will be unavailable.")

    # Initialize storage layers (optional)
    redis_store = None
    mongodb_store = None
    enable_storage = os.getenv("ENABLE_STORAGE", "false").lower() == "true"

    if STORAGE_AVAILABLE and enable_storage:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            logger.info(f"Initializing Redis storage: {redis_url}")
            redis_store = RedisSessionStore(
                redis_url=redis_url,
                default_ttl=args.session_ttl
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Redis storage: {e}")

        try:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            mongodb_database = os.getenv("MONGODB_DATABASE", "whisperlive")
            logger.info(f"Initializing MongoDB storage: {mongodb_url}/{mongodb_database}")
            mongodb_store = MongoDBArchiveStore(
                mongodb_url=mongodb_url,
                database_name=mongodb_database
            )
        except Exception as e:
            logger.warning(f"Failed to initialize MongoDB storage: {e}")
    else:
        if not STORAGE_AVAILABLE:
            logger.info("Storage packages not installed. Storage features disabled.")
        else:
            logger.info("Storage disabled via ENABLE_STORAGE=false")

    # Initialize rate limiting configuration
    rate_limit_config = None
    enable_rate_limit = os.getenv("ENABLE_RATE_LIMIT", "false").lower() == "true"

    if enable_rate_limit:
        rate_limit_redis_url = os.getenv("RATE_LIMIT_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/1"))
        logger.info(f"Rate limiting enabled with Redis: {rate_limit_redis_url}")
        rate_limit_config = RateLimitConfig(
            redis_url=rate_limit_redis_url,
            enabled=True
        )
    else:
        logger.info("Rate limiting disabled")

    # Create FastAPI app
    logger.info("Creating FastAPI application")
    app = create_unified_app(
        session_store=session_store,
        transcription_server=transcription_server,
        summarizer=summarizer,
        host=args.host,
        port=args.port,
        redis_store=redis_store,
        mongodb_store=mongodb_store,
        rate_limit_config=rate_limit_config
    )

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"API documentation available at http://{args.host}:{args.port}/docs")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        # Cleanup
        session_store.stop_cleanup_task()

        # Close storage connections
        if redis_store:
            try:
                redis_store.close()
            except Exception as e:
                logger.error(f"Error closing Redis store: {e}")

        if mongodb_store:
            try:
                mongodb_store.close()
            except Exception as e:
                logger.error(f"Error closing MongoDB store: {e}")

        logger.info("Server stopped")


if __name__ == "__main__":
    main()

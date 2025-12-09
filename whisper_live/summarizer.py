"""
LLM-based summarization module for WhisperLive.

Provides integration with Ollama and vLLM API for meeting summarization
with context-aware error correction.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


class OllamaSummarizer:
    """
    Ollama/vLLM API client for meeting summarization.

    Supports OpenAI-compatible chat completion API with custom prompt templates.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        prompt_config_path: str = "prompt.yaml",
        timeout: int = 180
    ):
        """
        Initialize Ollama summarizer.

        Args:
            base_url: Ollama/vLLM API base URL
            api_key: API key for authentication (optional for local Ollama)
            model: Model name to use
            prompt_config_path: Path to prompt configuration YAML
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
        self.timeout = timeout

        # Load prompt configuration
        self.prompt_config = self._load_prompt_config(prompt_config_path)

        # Ensure base URL doesn't end with slash
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]

        logger.info(
            f"OllamaSummarizer initialized: base_url={self.base_url}, "
            f"model={self.model}, has_api_key={bool(self.api_key)}"
        )

    def _load_prompt_config(self, config_path: str) -> Dict:
        """
        Load prompt configuration from YAML file.

        Args:
            config_path: Path to YAML file

        Returns:
            Prompt configuration dict
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Prompt configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Prompt config file not found: {config_path}, using defaults")
            return self._get_default_prompt_config()
        except Exception as e:
            logger.error(f"Error loading prompt config: {e}, using defaults")
            return self._get_default_prompt_config()

    def _get_default_prompt_config(self) -> Dict:
        """Get default prompt configuration."""
        return {
            "system_prompt": "Ты - ассистент для составления протоколов встреч. Твоя задача - анализировать транскрипты встреч и создавать структурированные протоколы.",
            "user_prompt_template": "Проанализируй следующий транскрипт встречи и составь протокол в формате JSON:\n\n{full_text}\n\nВерни ТОЛЬКО JSON без дополнительных комментариев."
        }

    def consolidate_transcript(self, segments: List[Dict], max_pause: float = 3.0, max_block_duration: float = 30.0) -> List[Dict]:
        """
        Consolidate transcript segments by speaker with pause detection.

        Args:
            segments: List of transcription segments
            max_pause: Maximum pause in seconds to group segments (default: 3.0)
            max_block_duration: Maximum block duration in seconds (default: 30.0)

        Returns:
            List of consolidated blocks
        """
        if not segments:
            return []

        # Deduplicate segments by (start, end, text)
        seen_keys = set()
        unique_segments = []

        for seg in segments:
            # Only process completed segments
            if not seg.get("completed", False):
                continue

            key = f"{float(seg['start']):.2f}:{float(seg['end']):.2f}:{seg['text']}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_segments.append(seg)

        # Sort by start time
        unique_segments.sort(key=lambda x: float(x['start']))

        if not unique_segments:
            return []

        # Group by speaker with pause and duration detection
        consolidated = []
        current_block = {
            "speaker": unique_segments[0].get("speaker", "Unknown"),
            "utterances": [{"text": unique_segments[0]["text"]}],
            "start": float(unique_segments[0]["start"]),
            "end": float(unique_segments[0]["end"])
        }

        for seg in unique_segments[1:]:
            speaker = seg.get("speaker", "Unknown")
            start = float(seg["start"])
            end = float(seg["end"])
            pause = start - current_block["end"]
            block_duration = current_block["end"] - current_block["start"]

            # Start new block if: speaker changed, pause too long, or block too long
            if (speaker != current_block["speaker"] or
                pause > max_pause or
                block_duration > max_block_duration):

                consolidated.append(current_block)
                current_block = {
                    "speaker": speaker,
                    "utterances": [{"text": seg["text"]}],
                    "start": start,
                    "end": end
                }
            else:
                # Continue current block
                current_block["utterances"].append({"text": seg["text"]})
                current_block["end"] = end

        # Add last block
        consolidated.append(current_block)

        logger.info(f"Consolidated {len(unique_segments)} segments into {len(consolidated)} blocks")
        return consolidated

    def generate_full_text(self, consolidated_blocks: List[Dict]) -> str:
        """
        Generate full text from consolidated blocks with speaker labels.

        Args:
            consolidated_blocks: List of consolidated transcript blocks

        Returns:
            Full text with speaker labels
        """
        lines = []
        for block in consolidated_blocks:
            speaker = block["speaker"]
            utterances = block["utterances"]
            text = " ".join(u["text"].strip() for u in utterances)
            lines.append(f"{speaker}: {text}")

        return "\n".join(lines)

    async def summarize(self, segments: List[Dict]) -> Dict:
        """
        Generate meeting summary from transcription segments.

        Args:
            segments: List of transcription segments

        Returns:
            Summary dict with meeting information
        """
        # Consolidate transcript
        consolidated = self.consolidate_transcript(segments)

        if not consolidated:
            logger.warning("No consolidated transcript available for summarization")
            return {
                "error": "No consolidated transcript available",
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }

        # Generate full text
        full_text = self.generate_full_text(consolidated)

        # Prepare messages
        system_prompt = self.prompt_config.get("system_prompt", "")
        user_prompt_template = self.prompt_config.get("user_prompt_template", "{full_text}")
        user_prompt = user_prompt_template.format(full_text=full_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM API
        try:
            summary_json = await self._call_llm_api(messages)
            summary_json["model"] = self.model
            summary_json["generated_at"] = datetime.utcnow().isoformat() + "Z"
            return summary_json

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }

    async def _call_llm_api(self, messages: List[Dict]) -> Dict:
        """
        Call Ollama/vLLM API with chat completion.

        Args:
            messages: List of chat messages

        Returns:
            Parsed JSON response

        Raises:
            Exception: If API call fails or response invalid
        """
        url = f"{self.base_url}/api/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000
        }

        logger.info(f"Calling LLM API: {url}, model={self.model}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()

            # Extract content from response
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
            else:
                raise ValueError("Invalid API response format")

            # Parse JSON from content
            summary_json = self._extract_json_from_content(content)
            logger.info("Summary generated successfully")
            return summary_json

    def _extract_json_from_content(self, content: str) -> Dict:
        """
        Extract JSON from LLM response content.

        Handles markdown code blocks and plain JSON.

        Args:
            content: LLM response content

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        # Strip markdown code blocks if present
        content = content.strip()

        # Remove ```json ... ``` or ``` ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()

        # Parse JSON
        return json.loads(content)

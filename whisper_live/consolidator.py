"""
Advanced transcript consolidation for WhisperLive.
Handles deduplication, speaker grouping, and incremental consolidation.
"""

import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class TranscriptConsolidator:
    """
    Advanced transcript consolidator with incremental processing.

    Features:
    - Deduplication by timestamp + text fingerprint
    - Speaker-based grouping with pause detection
    - Incremental consolidation (O(1) vs O(n²))
    - Duration-based block splitting
    """

    def __init__(
        self,
        max_pause: float = 3.0,
        max_block_duration: float = 30.0,
        dedup_time_threshold: float = 0.1
    ):
        """
        Initialize consolidator.

        Args:
            max_pause: Maximum pause in seconds before splitting blocks (default: 3.0)
            max_block_duration: Maximum block duration in seconds (default: 30.0)
            dedup_time_threshold: Time threshold for deduplication in seconds (default: 0.1)
        """
        self.max_pause = max_pause
        self.max_block_duration = max_block_duration
        self.dedup_time_threshold = dedup_time_threshold

        # State for incremental consolidation
        self.seen_segments: Set[str] = set()
        self.current_block: Optional[Dict] = None
        self.consolidated_blocks: List[Dict] = []

    def _make_segment_fingerprint(self, segment: Dict) -> str:
        """
        Create unique fingerprint for segment deduplication.

        Args:
            segment: Segment dictionary with start, end, text

        Returns:
            Fingerprint string
        """
        start = float(segment.get("start", 0))
        end = float(segment.get("end", 0))
        text = segment.get("text", "").strip()

        # Round to 2 decimal places to handle float precision issues
        return f"{start:.2f}:{end:.2f}:{text}"

    def _is_duplicate(self, segment: Dict) -> bool:
        """
        Check if segment is a duplicate.

        Args:
            segment: Segment to check

        Returns:
            True if duplicate, False otherwise
        """
        fingerprint = self._make_segment_fingerprint(segment)
        return fingerprint in self.seen_segments

    def _mark_seen(self, segment: Dict):
        """
        Mark segment as seen.

        Args:
            segment: Segment to mark
        """
        fingerprint = self._make_segment_fingerprint(segment)
        self.seen_segments.add(fingerprint)

    def _should_start_new_block(self, segment: Dict) -> bool:
        """
        Determine if a new block should be started.

        Args:
            segment: Current segment

        Returns:
            True if new block should be started
        """
        if self.current_block is None:
            return True

        current_speaker = segment.get("speaker", "Unknown")
        block_speaker = self.current_block.get("speaker", "Unknown")

        # Speaker changed
        if current_speaker != block_speaker:
            logger.debug(f"[CONSOLIDATOR] Speaker change: {block_speaker} -> {current_speaker}")
            return True

        # Calculate pause duration
        block_end = float(self.current_block.get("end", 0))
        segment_start = float(segment.get("start", 0))
        pause = segment_start - block_end

        # Long pause detected
        if pause > self.max_pause:
            logger.debug(f"[CONSOLIDATOR] Long pause detected: {pause:.2f}s")
            return True

        # Block too long
        block_start = float(self.current_block.get("start", 0))
        block_duration = segment_start - block_start
        if block_duration > self.max_block_duration:
            logger.debug(f"[CONSOLIDATOR] Block duration limit reached: {block_duration:.2f}s")
            return True

        return False

    def _finalize_current_block(self):
        """Finalize current block and add to consolidated list."""
        if self.current_block is not None:
            self.consolidated_blocks.append(self.current_block)
            logger.debug(
                f"[CONSOLIDATOR] Finalized block: speaker={self.current_block['speaker']}, "
                f"duration={float(self.current_block['end']) - float(self.current_block['start']):.2f}s, "
                f"text_length={len(self.current_block['text'])}"
            )
            self.current_block = None

    def add_segment(self, segment: Dict) -> Optional[Dict]:
        """
        Add a segment with incremental consolidation.

        Args:
            segment: Segment dictionary with start, end, text, speaker

        Returns:
            Finalized block if one was completed, None otherwise
        """
        # Skip duplicates
        if self._is_duplicate(segment):
            logger.debug(f"[CONSOLIDATOR] Skipping duplicate segment: {segment.get('text', '')[:50]}")
            return None

        # Skip empty text
        text = segment.get("text", "").strip()
        if not text:
            logger.debug("[CONSOLIDATOR] Skipping empty segment")
            return None

        # Mark as seen
        self._mark_seen(segment)

        # Check if new block should be started
        if self._should_start_new_block(segment):
            # Finalize current block
            finalized_block = None
            if self.current_block is not None:
                finalized_block = self.current_block.copy()
                self.consolidated_blocks.append(finalized_block)

            # Start new block
            self.current_block = {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
                "speaker": segment.get("speaker", "Unknown")
            }

            logger.debug(
                f"[CONSOLIDATOR] Started new block: speaker={self.current_block['speaker']}, "
                f"start={self.current_block['start']}"
            )

            return finalized_block

        else:
            # Append to current block
            self.current_block["end"] = segment.get("end")
            self.current_block["text"] += " " + text

            logger.debug(
                f"[CONSOLIDATOR] Extended current block: "
                f"duration={float(self.current_block['end']) - float(self.current_block['start']):.2f}s"
            )

            return None

    def get_consolidated_blocks(self, include_current: bool = True) -> List[Dict]:
        """
        Get all consolidated blocks.

        Args:
            include_current: If True, include current incomplete block

        Returns:
            List of consolidated blocks
        """
        blocks = self.consolidated_blocks.copy()

        if include_current and self.current_block is not None:
            blocks.append(self.current_block.copy())

        return blocks

    def finalize(self) -> List[Dict]:
        """
        Finalize consolidation and return all blocks.

        Returns:
            List of all consolidated blocks
        """
        self._finalize_current_block()
        return self.consolidated_blocks.copy()

    def reset(self):
        """Reset consolidator state."""
        self.seen_segments.clear()
        self.current_block = None
        self.consolidated_blocks.clear()
        logger.debug("[CONSOLIDATOR] State reset")

    def get_statistics(self) -> Dict:
        """
        Get consolidation statistics.

        Returns:
            Dictionary with statistics
        """
        total_blocks = len(self.consolidated_blocks)
        if self.current_block:
            total_blocks += 1

        # Calculate speaker distribution
        speaker_counts = defaultdict(int)
        for block in self.consolidated_blocks:
            speaker_counts[block.get("speaker", "Unknown")] += 1

        if self.current_block:
            speaker_counts[self.current_block.get("speaker", "Unknown")] += 1

        # Calculate total duration
        total_duration = 0.0
        if self.consolidated_blocks:
            first_start = float(self.consolidated_blocks[0].get("start", 0))
            last_end = float(self.consolidated_blocks[-1].get("end", 0))
            total_duration = last_end - first_start

        if self.current_block:
            last_end = float(self.current_block.get("end", 0))
            if self.consolidated_blocks:
                first_start = float(self.consolidated_blocks[0].get("start", 0))
            else:
                first_start = float(self.current_block.get("start", 0))
            total_duration = last_end - first_start

        return {
            "total_blocks": total_blocks,
            "finalized_blocks": len(self.consolidated_blocks),
            "unique_segments_seen": len(self.seen_segments),
            "speaker_distribution": dict(speaker_counts),
            "total_duration_seconds": total_duration
        }


def consolidate_transcript_batch(
    segments: List[Dict],
    max_pause: float = 3.0,
    max_block_duration: float = 30.0
) -> List[Dict]:
    """
    Consolidate a batch of segments (non-incremental, for backward compatibility).

    Args:
        segments: List of segments to consolidate
        max_pause: Maximum pause in seconds before splitting blocks
        max_block_duration: Maximum block duration in seconds

    Returns:
        List of consolidated blocks
    """
    consolidator = TranscriptConsolidator(
        max_pause=max_pause,
        max_block_duration=max_block_duration
    )

    for segment in segments:
        consolidator.add_segment(segment)

    return consolidator.finalize()

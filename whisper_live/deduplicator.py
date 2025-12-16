"""
Post-processing deduplicator for WhisperLive transcripts.
Handles text overlap removal at speaker boundaries.
"""

import logging
from typing import List, Dict, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class TranscriptDeduplicator:
    """
    Post-processing deduplicator for removing overlapping text between blocks.

    Use case: Whisper sometimes repeats the last few words of a block
    at the beginning of the next block, especially at speaker changes.
    """

    def __init__(
        self,
        min_overlap_words: int = 3,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize deduplicator.

        Args:
            min_overlap_words: Minimum number of words to consider as overlap (default: 3)
            similarity_threshold: Similarity threshold for fuzzy matching (default: 0.8)
        """
        self.min_overlap_words = min_overlap_words
        self.similarity_threshold = similarity_threshold

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text (lowercase, stripped, single spaces)
        """
        return " ".join(text.lower().strip().split())

    def _get_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _find_overlap(self, prev_text: str, curr_text: str) -> Optional[int]:
        """
        Find overlapping text between end of previous block and start of current block.

        Args:
            prev_text: Text from previous block
            curr_text: Text from current block

        Returns:
            Number of words to remove from current block, or None if no overlap
        """
        prev_words = prev_text.split()
        curr_words = curr_text.split()

        # Check different overlap lengths, starting from longest
        max_check = min(len(prev_words), len(curr_words), 15)  # Don't check more than 15 words

        for overlap_len in range(max_check, self.min_overlap_words - 1, -1):
            # Get last N words from previous block
            prev_tail = " ".join(prev_words[-overlap_len:])

            # Get first N words from current block
            curr_head = " ".join(curr_words[:overlap_len])

            # Check similarity
            similarity = self._get_text_similarity(prev_tail, curr_head)

            if similarity >= self.similarity_threshold:
                logger.debug(
                    f"[DEDUPLICATOR] Found overlap: {overlap_len} words, "
                    f"similarity={similarity:.2f}, text='{curr_head}'"
                )
                return overlap_len

        return None

    def deduplicate_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Remove overlapping text between consecutive blocks.

        Args:
            blocks: List of consolidated blocks

        Returns:
            List of deduplicated blocks
        """
        if len(blocks) <= 1:
            return blocks

        deduplicated = []
        prev_block = None

        for block in blocks:
            if prev_block is None:
                # First block - no deduplication needed
                deduplicated.append(block.copy())
                prev_block = block
                continue

            # Check for overlap
            prev_text = prev_block.get("text", "")
            curr_text = block.get("text", "")

            overlap_words = self._find_overlap(prev_text, curr_text)

            if overlap_words:
                # Remove overlapping words from current block
                curr_words = curr_text.split()
                new_text = " ".join(curr_words[overlap_words:])

                # Create deduplicated block
                dedup_block = block.copy()
                dedup_block["text"] = new_text

                logger.info(
                    f"[DEDUPLICATOR] Removed {overlap_words} overlapping words from block "
                    f"(speaker={block.get('speaker', 'Unknown')})"
                )

                deduplicated.append(dedup_block)
                prev_block = dedup_block
            else:
                # No overlap found
                deduplicated.append(block.copy())
                prev_block = block

        return deduplicated

    def deduplicate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Remove duplicate segments based on exact timestamp + text match.

        Args:
            segments: List of segments

        Returns:
            List of unique segments
        """
        seen = set()
        unique_segments = []

        for segment in segments:
            # Create fingerprint
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            text = segment.get("text", "").strip()
            fingerprint = f"{start:.2f}:{end:.2f}:{text}"

            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_segments.append(segment)
            else:
                logger.debug(f"[DEDUPLICATOR] Removed duplicate segment: {text[:50]}")

        removed_count = len(segments) - len(unique_segments)
        if removed_count > 0:
            logger.info(f"[DEDUPLICATOR] Removed {removed_count} duplicate segments")

        return unique_segments

    def get_statistics(self, original: List[Dict], deduplicated: List[Dict]) -> Dict:
        """
        Get deduplication statistics.

        Args:
            original: Original list of blocks/segments
            deduplicated: Deduplicated list

        Returns:
            Dictionary with statistics
        """
        original_word_count = sum(
            len(block.get("text", "").split())
            for block in original
        )

        dedup_word_count = sum(
            len(block.get("text", "").split())
            for block in deduplicated
        )

        removed_words = original_word_count - dedup_word_count

        return {
            "original_blocks": len(original),
            "deduplicated_blocks": len(deduplicated),
            "original_word_count": original_word_count,
            "deduplicated_word_count": dedup_word_count,
            "removed_words": removed_words,
            "compression_ratio": removed_words / original_word_count if original_word_count > 0 else 0.0
        }


def deduplicate_transcript(
    blocks: List[Dict],
    min_overlap_words: int = 3,
    similarity_threshold: float = 0.8
) -> List[Dict]:
    """
    Deduplicate a transcript (convenience function).

    Args:
        blocks: List of consolidated blocks
        min_overlap_words: Minimum number of words to consider as overlap
        similarity_threshold: Similarity threshold for fuzzy matching

    Returns:
        List of deduplicated blocks
    """
    deduplicator = TranscriptDeduplicator(
        min_overlap_words=min_overlap_words,
        similarity_threshold=similarity_threshold
    )

    return deduplicator.deduplicate_blocks(blocks)

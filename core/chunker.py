"""Sliding window text chunking algorithm for streaming TTS synthesis.

Implements:
- Sliding Window over word tokens while respecting sentence/clause punctuation boundaries.
"""

from typing import List

SENTENCE_ENDINGS = (".", "!", "?", ";", ":")


def chunk_text_sliding_window(
    text: str,
    window_size: int = 8,
    overlap: int = 0,
) -> List[str]:
    """Split cleaned text into word chunks using a sliding window, respecting punctuation.

    Algorithm: Sliding Window with Punctuation Lookback (O(W) time where W = number of words)
    - Why in TTS: Streaming TTS models generate audio chunk-by-chunk. Splitting mid-phrase
      degrades prosody, so if a punctuation boundary (`.`, `!`, `?`, `;`, `:`) appears inside
       the window `[left, right)`, we snap the right boundary to that punctuation token so
      each chunk ends on a natural acoustic boundary when possible.
    - Supports configurable `overlap` (e.g. for cross-fading context windows) or `overlap=0`
      for disjoint streaming segments.

    Args:
        text: Cleaned input text to split.
        window_size: Maximum number of words per chunk (must be >= 1).
        overlap: Number of words to overlap between consecutive chunks (0 <= overlap < window_size).

    Returns:
        Ordered list of text chunks.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    words = cleaned.split()
    n = len(words)
    if n <= window_size:
        return [" ".join(words)]

    chunks: List[str] = []
    left = 0

    while left < n:
        right = min(left + window_size, n)

        # If we are not at the end of the text, look back inside [left, right)
        # for the last word that ends with sentence/clause punctuation.
        snapped_to_punctuation = False
        if right < n:
            for idx in range(right - 1, left - 1, -1):
                if words[idx].endswith(SENTENCE_ENDINGS):
                    right = idx + 1
                    snapped_to_punctuation = True
                    break

        chunk = " ".join(words[left:right])
        chunks.append(chunk)

        if right >= n:
            break

        # Advance window pointer.
        # Note: When snapping early to a punctuation boundary, we advance cleanly to `right`
        # to avoid repeating across sentence boundaries; otherwise we apply `overlap`.
        if snapped_to_punctuation or overlap == 0:
            left = right
        else:
            left = max(left + 1, right - overlap)

    return chunks


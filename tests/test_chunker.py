"""Unit tests for the Sliding Window text chunker (`core/chunker.py`)."""

import pytest
from core.chunker import chunk_text_sliding_window


def test_chunk_empty_and_whitespace_text() -> None:
    assert chunk_text_sliding_window("") == []
    assert chunk_text_sliding_window("   \n\t  ") == []


def test_chunk_single_word() -> None:
    assert chunk_text_sliding_window("ElevenLabs", window_size=5) == ["ElevenLabs"]


def test_chunk_respects_sentence_punctuation_boundary() -> None:
    text = "Hello world. This is a streaming test for natural speech synthesis."
    # With window_size=5, the first window sees ["Hello", "world.", "This", "is", "a"]
    # and snaps early at "world." because it ends with a period.
    chunks = chunk_text_sliding_window(text, window_size=5, overlap=0)
    assert chunks[0] == "Hello world."
    assert chunks[1] == "This is a streaming test"
    assert chunks[2] == "for natural speech synthesis."


def test_chunk_sliding_window_with_overlap() -> None:
    text = "one two three four five six seven"
    chunks = chunk_text_sliding_window(text, window_size=4, overlap=2)
    assert chunks == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]


def test_chunk_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="window_size must be at least 1"):
        chunk_text_sliding_window("hello world", window_size=0)

    with pytest.raises(ValueError, match="overlap must satisfy"):
        chunk_text_sliding_window("hello world", window_size=4, overlap=4)


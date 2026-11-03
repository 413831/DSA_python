"""Unit tests for the SSML string parser (`core/parser.py`)."""

import pytest
from core.exceptions import SSMLParseError
from core.parser import parse_ssml_text


def test_parse_valid_ssml_tags() -> None:
    raw = 'Hello <break time="500ms"/> welcome to <emphasis level="strong">ElevenLabs</emphasis> API!'
    parsed = parse_ssml_text(raw)

    assert parsed.clean_text == "Hello welcome to ElevenLabs API!"
    assert len(parsed.annotations) == 2

    break_ann = parsed.annotations[0]
    assert break_ann.tag == "break"
    assert break_ann.attributes == {"time": "500ms"}

    emp_ann = parsed.annotations[1]
    assert emp_ann.tag == "emphasis"
    assert emp_ann.attributes == {"level": "strong"}
    assert emp_ann.content == "ElevenLabs"


def test_parse_default_emphasis_level() -> None:
    parsed = parse_ssml_text("Speak <emphasis>clearly</emphasis> now.")
    assert parsed.clean_text == "Speak clearly now."
    assert parsed.annotations[0].attributes == {"level": "moderate"}
    assert parsed.annotations[0].content == "clearly"


def test_parse_unclosed_emphasis_raises_error() -> None:
    with pytest.raises(SSMLParseError, match="Unclosed <emphasis>"):
        parse_ssml_text("Hello <emphasis level='strong'>world")


def test_parse_unmatched_closing_tag_raises_error() -> None:
    with pytest.raises(SSMLParseError, match="without matching opening tag"):
        parse_ssml_text("Hello world</emphasis>")


def test_parse_unsupported_or_malformed_tag_raises_error() -> None:
    with pytest.raises(SSMLParseError, match="Unsupported or malformed SSML tag"):
        parse_ssml_text("Hello <script>alert(1)</script> world")

    with pytest.raises(SSMLParseError, match="Unsupported or malformed SSML tag"):
        parse_ssml_text('Hello <break time="invalid"/> world')


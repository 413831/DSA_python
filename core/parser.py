"""String parsing and SSML tag extraction algorithms.

Implements:
- String Parsing / Stack-based validation + Regex scanning to validate and extract
  simplified SSML tags (<emphasis>, <break time="..."/>) and produce clean text.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from core.exceptions import SSMLParseError


@dataclass(frozen=True)
class StyleAnnotation:
    """Represents an extracted style or prosody instruction from SSML-like input."""

    tag: str
    attributes: Dict[str, str]
    position: int  # Character index in the cleaned text where the tag applies
    content: str = ""  # Inner text for paired tags like <emphasis>


@dataclass
class ParsedText:
    """Output of the SSML parser containing stripped text and extracted metadata."""

    clean_text: str
    annotations: List[StyleAnnotation] = field(default_factory=list)


# Regex pattern that matches:
# 1. Self-closing <break time="500ms"/> or <break time="1s" />
# 2. Paired <emphasis level="strong">...</emphasis> or <emphasis>...</emphasis>
# 3. Any other angle-bracket tag to detect unsupported/malformed tags
_TOKEN_RE = re.compile(r"(<[^>]+>)")
_BREAK_RE = re.compile(
    r'^<break\s+time=["\'](\d+(?:ms|s))["\']\s*/?>$'
)
_EMPHASIS_OPEN_RE = re.compile(
    r'^<emphasis(?:\s+level=["\'](strong|moderate|reduced)["\'])?\s*>$'
)
_EMPHASIS_CLOSE_RE = re.compile(r"^</emphasis>$")


def parse_ssml_text(raw_text: str) -> ParsedText:
    """Parse simplified SSML text, validating tags and extracting clean text + annotations.

    Algorithm: String Parsing & Stack Validation (O(N) time and space where N = len(raw_text))
    - Tokenizes the string into literal segments and XML-like tags.
    - Uses an explicit stack to track open `<emphasis>` tags, detecting unclosed or
      improperly nested tags (classic interview String/Stack pattern).
    - Collapses redundant whitespace while tracking exact character offsets in the clean output.

    Raises:
        SSMLParseError: If unsupported tags, malformed attributes, or unclosed tags are found.
    """
    if not raw_text or not raw_text.strip():
        return ParsedText(clean_text="", annotations=[])

    tokens = _TOKEN_RE.split(raw_text)
    clean_parts: List[str] = []
    annotations: List[StyleAnnotation] = []
    # Stack stores tuples of (level_attribute, start_offset_in_clean_text)
    emphasis_stack: List[tuple[str, int]] = []
    current_clean_len = 0

    for token in tokens:
        if not token:
            continue

        if token.startswith("<") and token.endswith(">"):
            # Case 1: <break time="..."/>
            break_match = _BREAK_RE.match(token)
            if break_match:
                annotations.append(
                    StyleAnnotation(
                        tag="break",
                        attributes={"time": break_match.group(1)},
                        position=current_clean_len,
                    )
                )
                continue

            # Case 2: <emphasis ...>
            emp_open = _EMPHASIS_OPEN_RE.match(token)
            if emp_open:
                if emphasis_stack:
                    raise SSMLParseError("Nested <emphasis> tags are not supported.")
                level = emp_open.group(1) or "moderate"
                emphasis_stack.append((level, current_clean_len))
                continue

            # Case 3: </emphasis>
            if _EMPHASIS_CLOSE_RE.match(token):
                if not emphasis_stack:
                    raise SSMLParseError("Closing </emphasis> tag without matching opening tag.")
                level, start_pos = emphasis_stack.pop()
                full_clean_so_far = "".join(clean_parts)
                inner_text = full_clean_so_far[start_pos:].strip()
                annotations.append(
                    StyleAnnotation(
                        tag="emphasis",
                        attributes={"level": level},
                        position=start_pos,
                        content=inner_text,
                    )
                )
                continue

            # Any other tag is invalid
            raise SSMLParseError(f"Unsupported or malformed SSML tag: {token}")
        else:
            # Check for stray '<' or '>' characters indicating broken tags
            if "<" in token or ">" in token:
                raise SSMLParseError(f"Malformed angle bracket in text segment: {token}")
            clean_parts.append(token)
            current_clean_len += len(token)

    if emphasis_stack:
        raise SSMLParseError("Unclosed <emphasis> tag detected at end of input.")

    # Normalize multiple spaces into single spaces
    normalized_text = re.sub(r"\s+", " ", "".join(clean_parts)).strip()
    return ParsedText(clean_text=normalized_text, annotations=annotations)


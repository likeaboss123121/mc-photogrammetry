"""Encoding-safe terminal output for the dataset CLI.

Save directory names and ``level.dat`` ``LevelName`` values are arbitrary user
data: Minecraft accepts world names a legacy console cannot represent. A bare
``print`` therefore dies with ``UnicodeEncodeError`` on, for example, a CJK world
name under a strict Windows cp1252 stdout, and every later record in the listing
is lost - the inventory looks like it found fewer saves than it did.

These helpers keep a listing complete and deterministic. Text is written
unchanged wherever the stream can represent it; only the characters the stream's
encoding cannot represent are replaced by their Python escapes (``\\u9ad8``,
``\\xXX``). Records are never dropped, reordered, or silently folded to ASCII
lookalikes, so one exotic world name cannot hide the worlds after it.
"""
from __future__ import annotations

from typing import TextIO

#: The codec error handler that renders an unrepresentable character as its
#: deterministic Python escape. The result is ASCII by construction, so the
#: escaped string is always writable to any ASCII-superset stream.
ESCAPE_ERRORS = "backslashreplace"


def stream_encoding(stream: TextIO) -> str | None:
    """The codec ``stream`` declares, or ``None`` when there is no byte codec.

    A ``StringIO`` or other test double has no ``encoding`` attribute; it carries
    Unicode directly, so nothing ever needs escaping for it.
    """
    encoding = getattr(stream, "encoding", None)
    return str(encoding) if encoding else None


def is_encodable(text: str, encoding: str | None) -> bool:
    """Whether ``encoding`` can represent every character of ``text`` exactly.

    An unknown codec is reported as encodable: the stream itself is then the only
    authority, and :func:`write_safe` still has a strict-error fallback.
    """
    if encoding is None:
        return True
    try:
        text.encode(encoding, "strict")
    except UnicodeEncodeError:
        return False
    except LookupError:
        return True
    return True


def _escape_ascii(text: str) -> str:
    return text.encode("ascii", ESCAPE_ERRORS).decode("ascii")


def escape_for_encoding(text: str, encoding: str | None = None) -> str:
    """Replace exactly the characters ``encoding`` cannot represent.

    Representable characters - including non-ASCII ones such as the e-acute
    character on cp1252 - are kept verbatim. Every unrepresentable character
    becomes its deterministic Python escape, so the same text always renders the
    same bytes for a given encoding, on every platform.
    """
    if encoding is None:
        return text
    escaped: list[str] = []
    for character in text:
        try:
            character.encode(encoding, "strict")
        except UnicodeEncodeError:
            escaped.append(_escape_ascii(character))
        except LookupError:
            return _escape_ascii(text)
        else:
            escaped.append(character)
    return "".join(escaped)


def write_safe(stream: TextIO, text: str) -> None:
    """Write ``text`` to ``stream`` without letting its encoding drop a record.

    The text is written verbatim when the stream's declared encoding can carry
    it. Otherwise - or if the stream turns out stricter than it declared - the
    unencodable characters are escaped and the whole string is written, so a
    caller iterating a record list always emits one entry per record.
    """
    encoding = stream_encoding(stream)
    if not is_encodable(text, encoding):
        text = escape_for_encoding(text, encoding)
    try:
        stream.write(text)
    except UnicodeError:
        # The declared codec lied, or a wrapper enforces strictness it did not
        # advertise. An ASCII-only rendering is always writable and still names
        # the record, so retry instead of aborting the listing.
        stream.write(_escape_ascii(text))


def write_line(stream: TextIO, text: str = "") -> None:
    """Like :func:`write_safe` but appends the newline that ends a record line."""
    write_safe(stream, text + "\n")

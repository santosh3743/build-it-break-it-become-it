"""
Structured JSON-lines logging with PII and secret redaction.

Logs are where privacy incidents hide. The model's input and output are the
most useful things to log when debugging, and they are exactly where users
paste their email address, their phone number, and occasionally an API key.
Logs are then copied to a log platform, retained for months, and read by
people who were never meant to see customer data. OWASP's LLM02:2025
(Sensitive Information Disclosure) covers the model leaking data; your own
telemetry pipeline is an equally good place to leak it.

So every string field is redacted *before* the line is written. Two layers:

1. `SECRET_PATTERNS` (this lab): credential formats -- `sk-...` style API keys,
   AWS access key ids, GitHub tokens, bearer tokens, PEM private-key headers.
2. Lab 01's `scrub_pii` (reused, not rewritten): emails, phone numbers, card
   numbers, and simple `key_...`/`token_...` keys.

Order matters, and finding that out was one of this lab's surprises. Lab 01's
PHONE pattern matches any run of 10-13 digits, and it runs before its APIKEY
pattern. Fed `sk-FAKE1234567890abcdEF` directly, Lab 01 turns it into
`sk-FAKE[PHONE]abcdEF`: the digits vanish but the key's prefix and suffix are
still in the log. Running the secret patterns first replaces the whole key
with `[SECRET]` before the phone pattern ever sees it.

The other half of the observability-privacy tension: over-redaction. The same
PHONE pattern would happily eat a 13-digit millisecond timestamp. So:

* only *string* values are redacted -- numbers (latency, token counts) pass
  through untouched;
* a short list of structural keys whose values this app generates itself
  (`trace_id`, `request_id`, `event` ...) is never redacted, so you can still
  join a log line to its trace.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, TextIO

try:  # loaded as part of a package (e.g. `lab06`) by a later lab
    from . import labs_path
except ImportError:  # run from inside this folder
    import labs_path

# Lab 01's PII scrubber, loaded by file path. One implementation, two labs.
scrub_pii = labs_path.lab01_pipeline().scrub_pii

# Credential formats Lab 01 does not cover, applied BEFORE scrub_pii.
SECRET_PATTERNS: dict[str, re.Pattern] = {
    # OpenAI-style secret keys: "sk-" followed by a long token (may contain - or _).
    "SECRET": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    # AWS access key id: "AKIA" + 16 upper-case letters/digits.
    "AWS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # GitHub tokens: ghp_/gho_/ghu_/ghs_/ghr_ prefixes + 36 characters.
    "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
    # HTTP bearer tokens in pasted headers.
    "BEARER": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}=*"),
    # The first line of a PEM private key.
    "PRIVATE_KEY": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def find_secrets(text: str) -> list[str]:
    """Names of the secret patterns present in `text` (used by the output filter)."""
    return [tag for tag, pat in SECRET_PATTERNS.items() if pat.search(text)]


def redact(text: str) -> tuple[str, int]:
    """Replace secrets, then PII, with [TAG] placeholders. Returns (text, count)."""
    count = 0
    for tag, pat in SECRET_PATTERNS.items():
        text, n = pat.subn(f"[{tag}]", text)
        count += n
    text, n = scrub_pii(text)  # Lab 01
    return text, count + n


def contains_sensitive(text: str) -> bool:
    """True if redaction would change anything: a cheap leak detector."""
    return redact(text)[1] > 0


# Values this app generates itself. Never user-controlled, never redacted.
STRUCTURAL_KEYS = frozenset({
    "ts", "level", "event", "trace_id", "span_id", "request_id", "guard",
    "status", "model", "prompt_version", "tool", "cost_usd",
})


class JsonLogger:
    """Writes one JSON object per line. Redacts string values unless told not to.

    `records` keeps every (already redacted) record in memory so tests and the
    demo can inspect them; `stream`, if given, also receives each line.
    `redact=False` exists only to show the "before" picture in the demo.
    """

    def __init__(self, clock: Any = None, stream: TextIO | None = None,
                 redact: bool = True):
        self.clock = clock
        self.stream = stream
        self.redact_enabled = redact
        self.records: list[dict[str, Any]] = []
        self.lines: list[str] = []
        self.redactions = 0

    def _clean(self, key: str, value: Any) -> Any:
        if key in STRUCTURAL_KEYS or not self.redact_enabled:
            return value
        if isinstance(value, str):
            cleaned, n = redact(value)
            self.redactions += n
            return cleaned
        if isinstance(value, dict):
            return {k: self._clean(k, v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(key, v) for v in value]
        return value  # int/float/bool/None: numbers are never redacted

    def log(self, event: str, level: str = "info", **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"ts": self.clock.now() if self.clock else None,
                                  "level": level, "event": event}
        for k, v in fields.items():
            record[k] = self._clean(k, v)
        line = json.dumps(record, default=str, ensure_ascii=False)
        self.records.append(record)
        self.lines.append(line)
        if self.stream is not None:
            print(line, file=self.stream)
        return record

    def text(self) -> str:
        """Everything this logger wrote, as one string (what lands on disk)."""
        return "\n".join(self.lines)

    def find(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["event"] == event]


def stdout_logger(clock: Any = None) -> JsonLogger:
    return JsonLogger(clock=clock, stream=sys.stdout)

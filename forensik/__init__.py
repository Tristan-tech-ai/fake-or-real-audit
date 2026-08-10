"""Audit of audio deepfake detection on the Fake-or-Real dataset."""

from __future__ import annotations

import sys

# Reports contain arrows, Greek letters and en dashes. The Windows console
# defaults to cp1252 and raises UnicodeEncodeError on all of them.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

__all__ = ["manifest", "metrics", "report", "results", "runs", "stats"]

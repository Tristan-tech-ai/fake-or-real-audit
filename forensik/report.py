"""Markdown report writer shared by every script under ``forensik.laporan``."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "hasil"


class Report:
    """Collects markdown lines, echoes them, and writes them to ``docs/hasil``."""

    def __init__(self, filename: str) -> None:
        self.path = DOCS / filename
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list]) -> None:
        self.add("| " + " | ".join(header) + " |")
        self.add("|" + "---|" * len(header))
        for row in rows:
            self.add("| " + " | ".join(str(cell) for cell in row) + " |")
        self.add("")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\n-> {self.path.relative_to(ROOT)}")

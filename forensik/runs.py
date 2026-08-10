"""Loading and grouping of training runs.

A run directory is named ``{model}_{split}_{aug}[_b{batch}e{epochs}]_s{seed}``
and holds ``test_scores.npy`` (labels, scores, logits) plus ``results.json``.

Grouping keys always include batch size and epoch count. Dropping them merges
runs that differ in more than the seed, which reports the spread between
configurations as if it were seed noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .metrics import full_metrics, prior_matched_threshold

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

TAG = re.compile(
    r"^(?P<model>.+?)"
    r"_(?P<split>official|random|clean_val|wavval)"
    r"_(?P<aug>[A-Za-z0-9.]+?)"
    r"(?:_b(?P<batch>\d+)e(?P<epochs>\d+))?"
    r"_s(?P<seed>\d+)$"
)


@dataclass(frozen=True, order=True)
class Config:
    """Everything about a run except its random seed."""

    model: str
    split: str
    aug: str
    batch: int | None = None
    epochs: int | None = None

    @property
    def suffix(self) -> str:
        return f"b{self.batch}e{self.epochs}" if self.batch else "legacy"

    def __str__(self) -> str:
        return f"{self.model}_{self.split}_{self.aug}_{self.suffix}"


@dataclass(frozen=True)
class Run:
    tag: str
    config: Config
    seed: int
    path: Path

    @cached_property
    def scores(self) -> tuple[np.ndarray, np.ndarray]:
        """True labels (1 = fake) and predicted fake probabilities."""
        labels, probs, _ = np.load(self.path / "test_scores.npy")
        return labels.astype(int), probs

    @cached_property
    def args(self) -> dict:
        path = self.path / "results.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def metrics(self, threshold: float | str = 0.5) -> dict:
        """Metrics at a fixed threshold, or at ``"prior"`` for prior matching."""
        labels, probs = self.scores
        if threshold == "prior":
            threshold = prior_matched_threshold(probs, 0.5)
        return full_metrics(labels, probs, threshold)


def parse(tag: str) -> tuple[Config, int] | None:
    m = TAG.match(tag)
    if not m:
        return None
    batch = int(m["batch"]) if m["batch"] else None
    epochs = int(m["epochs"]) if m["epochs"] else None
    config = Config(m["model"], m["split"], m["aug"], batch, epochs)
    return config, int(m["seed"])


def load(pattern: str = "*", root: Path = RUNS) -> list[Run]:
    """Every run matching a glob pattern that has scores on disk."""
    runs = []
    for path in sorted(root.glob(pattern)):
        parsed = parse(path.name)
        if parsed and (path / "test_scores.npy").exists():
            runs.append(Run(path.name, parsed[0], parsed[1], path))
    return runs


def group(runs: list[Run]) -> dict[Config, list[Run]]:
    """Runs bucketed by configuration, so each bucket differs only by seed."""
    buckets: dict[Config, list[Run]] = {}
    for run in runs:
        buckets.setdefault(run.config, []).append(run)
    return buckets


@dataclass(frozen=True)
class Summary:
    """One metric across the seeds of a single configuration."""

    values: np.ndarray
    metric: str

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def sd(self) -> float:
        return float(self.values.std(ddof=1)) if self.n > 1 else 0.0

    def __str__(self) -> str:
        return f"{self.mean:.2f}" if self.n < 2 else f"{self.mean:.2f} ({self.sd:.2f})"


def summarize(
    runs: list[Run], metric: str = "accuracy", threshold: float | str = "prior"
) -> Summary | None:
    """Collect one metric over a set of runs, as a percentage."""
    if not runs:
        return None
    values = [run.metrics(threshold)[metric] * 100 for run in runs]
    return Summary(np.array(values), metric)


def scores(
    pattern: str, metric: str = "accuracy", threshold: float | str = "prior"
) -> Summary | None:
    """Shorthand for ``summarize(load(pattern), ...)``."""
    return summarize(load(pattern), metric, threshold)

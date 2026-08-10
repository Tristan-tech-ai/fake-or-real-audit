"""Significance tests used to decide whether a difference beats seed noise.

Sample sizes here are small: at most eight seeds per cell. A large p-value
means "not shown to differ", never "shown to be equal".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Comparison:
    """Welch t-test between two sets of seed results."""

    name: str
    a: np.ndarray
    b: np.ndarray
    p_raw: float
    p_holm: float = float("nan")

    @property
    def diff(self) -> float:
        return float(self.a.mean() - self.b.mean())

    @property
    def ci95(self) -> float:
        """Half-width of the 95 percent interval around ``diff``."""
        va, vb = self.a.var(ddof=1) / len(self.a), self.b.var(ddof=1) / len(self.b)
        se = np.sqrt(va + vb)
        dof = se**4 / (va**2 / (len(self.a) - 1) + vb**2 / (len(self.b) - 1))
        return float(stats.t.ppf(0.975, dof) * se)

    @property
    def significant(self) -> bool:
        return self.p_holm < 0.05


def welch(name: str, a: np.ndarray, b: np.ndarray) -> Comparison:
    return Comparison(name, a, b, float(stats.ttest_ind(a, b, equal_var=False).pvalue))


def paired_t(differences: np.ndarray) -> float:
    """Two-sided p-value for paired measurements sharing the same seeds."""
    return float(stats.ttest_1samp(differences, 0.0).pvalue)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for step, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - step) * p_values[i]))
        adjusted[i] = running
    return adjusted


def correct(comparisons: list[Comparison]) -> list[Comparison]:
    """Attach Holm-corrected p-values to a family of comparisons."""
    adjusted = holm([c.p_raw for c in comparisons])
    return [
        Comparison(c.name, c.a, c.b, c.p_raw, p) for c, p in zip(comparisons, adjusted, strict=False)
    ]


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    scale = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / scale) if scale > 0 else 0.0


@dataclass(frozen=True)
class Correlation:
    r: float
    p: float
    permutations: int
    exact: bool
    ci95: tuple[float, float] | None = None


def permutation_correlation(
    x, y, trials: int = 20_000, seed: int = 0, exact_below: int = 9
) -> Correlation:
    """Pearson r with a two-sided permutation p-value.

    Up to eight points every permutation is enumerated, so the p-value is exact.
    Beyond that it is sampled. The analytic p-value is not used because these
    samples are far too small for the normal approximation to hold.
    """
    observed = abs(pearson(x, y))
    if len(x) < exact_below:
        null = [abs(pearson(x, list(p))) for p in permutations(y)]
        hits = float(np.mean([value >= observed - 1e-12 for value in null]))
        return Correlation(pearson(x, y), hits, len(null), True)

    rng = np.random.default_rng(seed)
    hits = sum(
        abs(pearson(x, rng.permutation(y))) >= observed - 1e-12 for _ in range(trials)
    )
    return Correlation(pearson(x, y), (hits + 1) / (trials + 1), trials, False)


def bootstrap_correlation_ci(
    x, y, trials: int = 20_000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for Pearson r."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    resampled = []
    for _ in range(trials):
        index = rng.integers(0, len(x), len(x))
        if len(set(index.tolist())) >= 3:
            resampled.append(pearson(x[index], y[index]))
    if len(resampled) < 100:
        return None
    return float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


def bootstrap_ci(
    values: np.ndarray, trials: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(trials)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

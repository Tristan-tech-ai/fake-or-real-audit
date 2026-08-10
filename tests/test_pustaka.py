"""Unit tests for the shared library."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from forensik import metrics, runs
from forensik import stats as fstats


class TestRunParsing:
    def test_modern_tag(self):
        config, seed = runs.parse("wavlm_official_full_b16e10_s42")
        assert (config.model, config.split, config.aug) == ("wavlm", "official", "full")
        assert (config.batch, config.epochs, seed) == (16, 10, 42)

    def test_legacy_tag_has_no_configuration(self):
        config, seed = runs.parse("cnn_asp_official_none_s42")
        assert (config.batch, config.epochs, seed) == (None, None, 42)
        assert config.suffix == "legacy"

    def test_configuration_is_part_of_the_key(self):
        """Two runs differing only in epochs must never share a group."""
        a, _ = runs.parse("wavlm_official_full_b16e10_s42")
        b, _ = runs.parse("wavlm_official_full_b16e20_s42")
        assert a != b

    def test_unparseable_tag(self):
        assert runs.parse("bukan_tag_yang_sah") is None


class TestThresholds:
    def test_prior_matched_splits_at_the_class_balance(self):
        scores = np.linspace(0, 1, 100)
        threshold = metrics.prior_matched_threshold(scores, 0.5)
        assert (scores >= threshold).sum() == pytest.approx(50, abs=1)

    def test_perfect_separation(self):
        labels = np.array([0] * 50 + [1] * 50)
        scores = np.array([0.1] * 50 + [0.9] * 50)
        result = metrics.full_metrics(labels, scores, 0.5)
        assert result["accuracy"] == 1.0
        assert result["auc"] == 1.0
        assert result["eer"] == 0.0

    def test_accuracy_can_collapse_while_auc_holds(self):
        """The central mechanism of the study, as an executable assertion."""
        labels = np.array([0] * 50 + [1] * 50)
        scores = np.array([0.90] * 50 + [0.95] * 50)
        assert metrics.full_metrics(labels, scores, 0.5)["accuracy"] == 0.5
        assert metrics.full_metrics(labels, scores, 0.5)["auc"] == 1.0
        prior = metrics.prior_matched_threshold(scores, 0.5)
        assert metrics.full_metrics(labels, scores, prior)["accuracy"] == 1.0


class TestStatistics:
    def test_welch_matches_scipy(self):
        a = np.array([98.4, 97.9, 98.8, 98.1, 98.6])
        b = np.array([63.1, 60.4, 67.2])
        assert fstats.welch("x", a, b).p_raw == pytest.approx(
            stats.ttest_ind(a, b, equal_var=False).pvalue
        )

    def test_holm_is_monotone_and_bounded(self):
        adjusted = fstats.holm([0.001, 0.02, 0.04, 0.9])
        assert adjusted == sorted(adjusted)
        assert all(0 <= p <= 1 for p in adjusted)

    def test_holm_multiplies_the_smallest_by_the_family_size(self):
        assert fstats.holm([0.01, 0.5])[0] == pytest.approx(0.02)

    def test_holm_preserves_input_order(self):
        assert fstats.holm([0.5, 0.01])[1] == pytest.approx(0.02)

    def test_permutation_is_exact_for_small_samples(self):
        x = [1.0, 2.0, 3.0]
        result = fstats.permutation_correlation(x, [1.0, 2.0, 3.0])
        assert result.exact
        assert result.permutations == 6
        # Three points allow six permutations, so p can never fall below 1/3.
        assert result.p >= 1 / 3

    def test_summary_reports_no_spread_for_a_single_seed(self):
        summary = runs.Summary(np.array([98.4]), "accuracy")
        assert summary.sd == 0.0
        assert str(summary) == "98.40"

"""Regression tests pinning the numbers the manuscript quotes.

If a refactor changes any of these, the manuscript changes with it, so they are
asserted here rather than trusted.
"""

from __future__ import annotations

import pytest

from forensik.manifest import codec_provenance
from forensik.runs import scores
from forensik.stats import correct, welch

TOLERANCE = 0.005


class TestCodecProvenance:
    """Read straight from the manifest, so these have no seed spread at all."""

    def test_training_fakes_are_almost_all_mp3(self):
        assert codec_provenance()[("training", "fake")].percent == pytest.approx(90.7, abs=0.1)

    def test_test_fakes_are_never_mp3(self):
        assert codec_provenance()[("testing", "fake")].percent == 0.0

    def test_no_real_sample_is_mp3_in_any_split(self):
        provenance = codec_provenance()
        assert all(
            value.from_mp3 == 0
            for (_, cls), value in provenance.items()
            if cls == "real"
        )


class TestCalibrationGap:
    """Accuracy collapses at a fixed threshold while AUC barely moves."""

    def test_fixed_threshold_collapses_to_chance(self):
        assert scores("cnn_asp_official_none_b32e10_s*", "accuracy", 0.5).mean == \
            pytest.approx(50.03, abs=TOLERANCE)

    def test_prior_matched_threshold_recovers_it(self):
        assert scores("cnn_asp_official_none_b32e10_s*", "accuracy", "prior").mean == \
            pytest.approx(92.56, abs=TOLERANCE)

    def test_discrimination_survives(self):
        assert scores("cnn_asp_official_none_b32e10_s*", "auc", 0.5).mean == \
            pytest.approx(97.56, abs=TOLERANCE)


class TestEncoderTreatment:
    """The direction of the effect differs per architecture."""

    CELLS = {
        "wavlm frozen": ("wavlm_official_full_b16e10_s*", 98.36, 5),
        "wavlm proposal": ("wavlm_official_proposalULRPK_b16e20_s*", 63.29, 5),
        "hubert tuned": ("hubert_official_fullUF_b32e10_s*", 96.19, 5),
        "hubert proposal": ("hubert_official_proposalULRPK_b32e20_s*", 51.64, 5),
        "ast frozen": ("ast_official_full_b32e10_s*", 89.15, 3),
        "ast tuned": ("ast_official_fullUF_b32e10_s*", 93.38, 3),
    }

    @pytest.mark.parametrize("name", list(CELLS))
    def test_cell(self, name):
        pattern, expected, seeds = self.CELLS[name]
        summary = scores(pattern, "accuracy", "prior")
        assert summary.n == seeds
        assert summary.mean == pytest.approx(expected, abs=TOLERANCE)

    def test_freezing_helps_wavlm_but_hurts_hubert(self):
        wavlm = (
            scores("wavlm_official_full_b16e10_s*", "accuracy", "prior").mean
            - scores("wavlm_official_fullUF_b16e10_s*", "accuracy", "prior").mean
        )
        hubert = (
            scores("hubert_official_full_b32e10_s*", "accuracy", "prior").mean
            - scores("hubert_official_fullUF_b32e10_s*", "accuracy", "prior").mean
        )
        assert wavlm > 0 > hubert


class TestSurvivingComparisons:
    """Only the two tens-of-points differences clear Holm correction."""

    def comparisons(self):
        pairs = [
            ("ast tuned vs frozen", "ast_official_fullUF_b32e10_s*",
             "ast_official_full_b32e10_s*"),
            ("ast tuned vs proposal", "ast_official_fullUF_b32e10_s*",
             "ast_official_proposalULRPK_b32e20_s*"),
            ("wavlm frozen vs tuned", "wavlm_official_full_b16e10_s*",
             "wavlm_official_fullUF_b16e10_s*"),
            ("hubert tuned vs frozen", "hubert_official_fullUF_b32e10_s*",
             "hubert_official_full_b32e10_s*"),
            ("wavlm engineered vs proposal", "wavlm_official_full_b16e10_s*",
             "wavlm_official_proposalULRPK_b16e20_s*"),
            ("hubert engineered vs proposal", "hubert_official_fullUF_b32e10_s*",
             "hubert_official_proposalULRPK_b32e20_s*"),
        ]
        return correct([
            welch(name, scores(a, "accuracy", "prior").values,
                  scores(b, "accuracy", "prior").values)
            for name, a, b in pairs
        ])

    def test_exactly_two_survive(self):
        assert sum(c.significant for c in self.comparisons()) == 2

    def test_survivors_are_the_proposal_comparisons(self):
        survivors = {c.name for c in self.comparisons() if c.significant}
        assert survivors == {"wavlm engineered vs proposal",
                             "hubert engineered vs proposal"}

    def test_encoder_treatment_differences_stay_unproven(self):
        unproven = [c for c in self.comparisons() if not c.significant]
        assert all(abs(c.diff) < 5 for c in unproven)

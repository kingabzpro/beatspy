"""Uncertainty: block bootstrap intervals and within-date permutation p-values.

These tests pin the two properties that make the numbers usable: the resampling
must treat a decision date as the unit (not an individual asset-day), and the
permutation must hold the date-level outcome counts fixed, because that is the
null a benchmark-relative label calls for. Everything is seeded, so the same
input must always produce the same output; replay depends on it.
"""

from __future__ import annotations

import random

import pytest

from beatspy.dcn.calibration import Observation, auc, build_report
from beatspy.dcn.statistics import (
    SEED,
    auc_statistic,
    block_bootstrap_excess,
    block_bootstrap_returns,
    brier_skill_statistic,
    permutation_p_value,
)


class TestBlockBootstrap:
    def test_interval_contains_the_compounded_total(self):
        blocks = [[0.01, -0.005], [0.02], [-0.01, 0.003, 0.001]]
        total = 1.0
        for block in blocks:
            for value in block:
                total *= 1.0 + value
        lower, upper = block_bootstrap_returns(blocks)
        assert lower <= total - 1.0 <= upper

    def test_is_deterministic_for_the_same_input(self):
        blocks = [[0.01, -0.02], [0.03], [0.0, 0.01]]
        assert block_bootstrap_returns(blocks) == block_bootstrap_returns(blocks)

    def test_no_dispersion_means_no_interval(self):
        """The bootstrap must not invent uncertainty that is not in the data."""
        blocks = [[0.01, 0.0]] * 20
        lower, upper = block_bootstrap_returns(blocks)
        assert lower == pytest.approx(upper)

    def test_dispersion_widens_the_interval(self):
        calm = [[0.001]] * 20
        wild = [[0.001 if index % 2 else -0.001] for index in range(20)]
        calm_low, calm_high = block_bootstrap_returns(calm)
        wild_low, wild_high = block_bootstrap_returns(wild)
        assert (wild_high - wild_low) > (calm_high - calm_low) == pytest.approx(0.0)

    def test_excess_interval_uses_paired_windows(self):
        portfolio = [[0.02], [0.01], [0.0]]
        spy = [[0.01], [0.01], [0.0]]
        lower, upper = block_bootstrap_excess(portfolio, spy)
        total = 1.0
        for block in portfolio:
            for value in block:
                total *= 1.0 + value
        spy_total = 1.0
        for block in spy:
            for value in block:
                spy_total *= 1.0 + value
        assert lower <= (total - spy_total) <= upper

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError):
            block_bootstrap_returns([])


class TestPermutation:
    def test_statistic_agrees_with_the_reference_auc(self):
        """The fast rank-based AUC must equal the documented Mann-Whitney one."""
        rng = random.Random(11)
        for _ in range(20):
            pairs = []
            for _ in range(60):
                probability = round(rng.random(), 1)  # forces many ties
                pairs.append((probability, int(rng.random() < probability)))
            probabilities = [probability for probability, _ in pairs]
            outcomes = [outcome for _, outcome in pairs]
            assert auc_statistic(probabilities, outcomes) == pytest.approx(auc(pairs))

    def test_no_signal_gives_a_large_p_value(self):
        """A model answering a constant cannot be distinguished from the null."""
        rng = random.Random(3)
        probabilities = [0.5] * 200
        outcomes = [int(rng.random() < 0.5) for _ in range(200)]
        clusters = [f"week-{index // 10}" for index in range(200)]
        p_value = permutation_p_value(probabilities, outcomes, clusters, brier_skill_statistic)
        assert p_value is not None and p_value > 0.2

    def test_a_real_edge_gives_a_small_p_value(self):
        """When the ranking carries the answer, the test must say so."""
        rng = random.Random(5)
        probabilities = []
        outcomes = []
        clusters = []
        for week in range(40):
            for _ in range(5):
                probability = rng.uniform(0.0, 1.0)
                probabilities.append(probability)
                outcomes.append(int(rng.random() < probability))
                clusters.append(f"week-{week}")
        p_value = permutation_p_value(probabilities, outcomes, clusters, auc_statistic)
        assert p_value is not None and p_value < 0.01

    def test_shuffle_happens_within_clusters(self):
        """The null keeps each date's outcome count, so a date-fixed effect is
        not evidence of skill."""
        probabilities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] * 5
        # Every cluster is all-winners or all-losers, so within-cluster shuffling
        # cannot change anything, and the observed ranking has no within-week
        # information to find.
        outcomes = [1] * 4 + [0] * 4
        outcomes = outcomes * 5
        clusters = [f"week-{index}" for index in range(10) for _ in range(4)]
        p_value = permutation_p_value(probabilities, outcomes, clusters, auc_statistic)
        assert p_value is not None and p_value > 0.5

    def test_is_deterministic(self):
        probabilities = [0.2, 0.4, 0.6, 0.8] * 10
        outcomes = ([1, 0, 1, 0] * 10)
        clusters = [f"week-{index}" for index in range(40)]
        first = permutation_p_value(probabilities, outcomes, clusters, auc_statistic, draws=200)
        second = permutation_p_value(probabilities, outcomes, clusters, auc_statistic, draws=200)
        assert first == second

    def test_degenerate_inputs_return_none(self):
        probabilities = [0.5] * 10
        outcomes = [1] * 10
        clusters = ["a"] * 10
        assert permutation_p_value(probabilities, outcomes, clusters, auc_statistic) is None
        assert permutation_p_value(
            probabilities, [1] * 5 + [0] * 5, ["a"] * 10, auc_statistic
        ) is None

    def test_alignment_is_required(self):
        with pytest.raises(ValueError):
            permutation_p_value([0.5], [1, 0], ["a", "b"], auc_statistic)


class TestReportIntegration:
    def test_report_carries_p_values_when_clusters_are_given(self):
        observations = []
        for index in range(30):
            observations.append(Observation(ticker=f"T{index}", noul=0.2 + 0.02 * index, outcome=index % 2))
        report = build_report(
            observations,
            coverage=1.0,
            horizons=3,
            clusters=[f"week-{index // 10}" for index in range(30)],
        )
        stats = report.signals["noul"]
        assert stats.brier_skill_p_value is not None
        assert stats.auc_p_value is not None

    def test_report_without_clusters_has_no_p_values(self):
        observations = [Observation(ticker="AAPL", noul=0.6, outcome=1)]
        report = build_report(observations, coverage=1.0, horizons=1)
        assert report.signals["noul"].brier_skill_p_value is None

    def test_rebuild_matches_the_original_run(self, dcn_run_dir):
        """Replay recomputes the same p-values from the recorded decisions."""
        from beatspy.dcn.runner import calibration_from_decisions, read_run

        _, metrics, lines = read_run(dcn_run_dir)
        rebuilt = calibration_from_decisions(lines)
        assert rebuilt.signals["noul"].brier_skill_p_value == metrics["calibration"]["signals"]["noul"]["brier_skill_p_value"]
        assert rebuilt.signals["noul"].auc_p_value == metrics["calibration"]["signals"]["noul"]["auc_p_value"]

    def test_run_metrics_carry_return_intervals(self, dcn_run_dir):
        from beatspy.dcn.runner import read_run

        _, metrics, _ = read_run(dcn_run_dir)
        for key in ("return_ci", "excess_return_ci"):
            interval = metrics[key]
            assert interval["lower"] <= interval["upper"]
            assert interval["draws"] > 0
            assert interval["level"] == pytest.approx(0.95)


def test_seed_is_stable():
    """The published seed is part of reproducibility; changing it changes runs."""
    assert SEED == 20261011

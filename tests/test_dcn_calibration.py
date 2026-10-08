"""Calibration and discrimination, checked against hand-computed values.

These are the numbers the whole benchmark rests on, so they are verified against
closed-form answers rather than snapshot strings.
"""

from __future__ import annotations

import math

import pytest

from beatspy.dcn.calibration import (
    Observation,
    auc,
    brier,
    build_report,
    choice_probability,
    log_loss,
    reliability,
    score_to_probability,
    signal_stats,
)


def test_brier_matches_closed_form():
    # p=0.7 with exactly 70% positives gives (0.3^2 * 0.7 + 0.7^2 * 0.3) = 0.21
    pairs = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    assert brier(pairs) == pytest.approx(0.21, abs=1e-12)


def test_log_loss_matches_closed_form():
    pairs = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    expected = -(0.7 * math.log(0.7) + 0.3 * math.log(0.3))
    assert log_loss(pairs) == pytest.approx(expected, abs=1e-12)


def test_log_loss_is_clipped_not_infinite():
    assert math.isfinite(log_loss([(0.0, 1)]))
    assert math.isfinite(log_loss([(1.0, 0)]))


def test_auc_perfect_and_inverted():
    perfect = [(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]
    assert auc(perfect) == pytest.approx(1.0)
    assert auc([(p, 1 - o) for p, o in perfect]) == pytest.approx(0.0)


def test_auc_counts_ties_as_a_half():
    # Two positives and two negatives, every prediction identical: no information.
    assert auc([(0.5, 1), (0.5, 1), (0.5, 0), (0.5, 0)]) == pytest.approx(0.5)
    # One clean win, one tie, out of four pairs.
    assert auc([(0.6, 1), (0.5, 0)]) == pytest.approx(1.0)
    assert auc([(0.5, 1), (0.5, 0)]) == pytest.approx(0.5)


def test_auc_undefined_with_a_single_class():
    assert auc([(0.9, 1), (0.8, 1)]) is None
    assert auc([(0.9, 0), (0.8, 0)]) is None


def test_reliability_bins_and_errors():
    # Everything lands in the 0.8-0.9 bin and half of it happens.
    pairs = [(0.85, 1)] * 5 + [(0.85, 0)] * 5
    bins, ece, mce = reliability(pairs)
    assert len(bins) == 10
    target = next(b for b in bins if b.lower == pytest.approx(0.8))
    assert target.count == 10
    assert target.mean_predicted == pytest.approx(0.85)
    assert target.observed_rate == pytest.approx(0.5)
    assert ece == pytest.approx(0.35)
    assert mce == pytest.approx(0.35)


def test_reliability_treats_probability_one_as_the_last_bin():
    bins, _, _ = reliability([(1.0, 1)] * 3)
    assert bins[-1].count == 3
    assert sum(b.count for b in bins) == 3


def test_reliability_is_zero_when_perfectly_calibrated():
    pairs = [(0.5, 1)] * 5 + [(0.5, 0)] * 5
    _, ece, mce = reliability(pairs)
    assert ece == pytest.approx(0.0)
    assert mce == pytest.approx(0.0)


def test_signal_stats_reports_skill_against_the_base_rate():
    # A constant predictor at the sample rate must score exactly zero skill.
    pairs = [(0.7, 1)] * 70 + [(0.7, 0)] * 30
    stats = signal_stats("noul", pairs)
    assert stats.n == 100
    assert stats.base_rate == pytest.approx(0.7)
    assert stats.brier_skill_score == pytest.approx(0.0, abs=1e-12)


def test_skill_catches_a_predictor_that_is_sharp_and_wrong():
    """Skill is what separates a confident miss from an honest constant.

    At a 0.1 base rate the constant predictor's Brier is 0.09. A predictor that
    says ~0.29 for everything is overconfident — it is wrong about its own
    certainty — and still scores positive skill, because it sits on the right side
    of 0.5. Sharpening the same predictor to 0.9 collapses Brier to 0.73 with
    negative skill. That is the trap a "same question, sharper answer" comparison
    falls into: the two forms can be compared only against the base rate.
    """
    pairs = [(0.2887, 1)] * 10 + [(0.2887, 0)] * 90
    stats = signal_stats("noul", pairs)
    # A constant predictor at the sample rate is the reference the skill score
    # divides by; at 0.1 positives that reference scores exactly 0.09.
    base = brier([(0.1, outcome) for _, outcome in pairs])
    assert base == pytest.approx(0.09)
    assert stats.brier == pytest.approx(0.125608, abs=1e-6)
    assert stats.brier_skill_score == pytest.approx(1 - stats.brier / base, abs=1e-5)
    # Overconfident constant, so worse than knowing the base rate: negative skill.
    assert stats.brier_skill_score < 0
    # Constant predictions discriminate nothing, which shows up as chance-level
    # AUC rather than as None: None is reserved for a single-class sample.
    assert stats.auc == pytest.approx(0.5)
    assert stats.mean_predicted == pytest.approx(0.2887)
    assert stats.expected_calibration_error == pytest.approx(0.1887, abs=1e-4)

    # An actually informative predictor: every outcome that happened was given the
    # high probability, so it separates the classes perfectly and beats the base rate.
    informative = [(0.9, 1)] * 90 + [(0.1, 0)] * 10
    good = signal_stats("noul", informative)
    assert good.base_rate == pytest.approx(0.9)
    assert good.brier == pytest.approx(0.01)  # 0.9*0.01 + 0.1*0.01
    assert good.brier_skill_score > 0
    assert good.auc == pytest.approx(1.0)  # every positive outranks every negative
    assert good.auc > stats.auc

    sharper = signal_stats("noul", [(0.9, 1)] * 10 + [(0.9, 0)] * 90)
    assert sharper.brier == pytest.approx(0.73)
    assert sharper.brier_skill_score < 0
    assert sharper.brier > stats.brier

    # Brier is invariant under flipping both the probability and the outcome
    # distribution (p -> 1-p with positives -> negatives), which is why a single
    # Brier number cannot say which way a model is wrong; accuracy and the bins can.
    mostly_up = signal_stats("noul", [(0.9, 1)] * 90 + [(0.1, 0)] * 10)
    mostly_down = signal_stats("noul", [(0.1, 0)] * 90 + [(0.9, 1)] * 10)
    assert mostly_up.brier == pytest.approx(mostly_down.brier)
    assert mostly_up.accuracy == pytest.approx(1.0)
    assert mostly_down.accuracy == pytest.approx(1.0)  # both get the side of 0.5 right


def test_signal_stats_of_nothing_is_empty():
    stats = signal_stats("noul", [])
    assert stats.n == 0
    assert stats.brier is None


def test_score_and_choice_conversions():
    # Five levels: index 0 -> 0.0, index 4 -> 1.0, index 2 -> 0.5.
    assert score_to_probability(0.0, 5) == pytest.approx(0.0)
    assert score_to_probability(4.0, 5) == pytest.approx(1.0)
    assert score_to_probability(2.0, 5) == pytest.approx(0.5)
    assert score_to_probability(-3.0, 5) == pytest.approx(0.0)  # clamped
    assert score_to_probability(99.0, 5) == pytest.approx(1.0)  # clamped
    assert score_to_probability(1.0, 1) == pytest.approx(0.5)  # degenerate level count

    assert choice_probability({"outperform": 0.8, "underperform": 0.2}) == pytest.approx(0.8)
    # A distribution that does not sum to 1 is normalized rather than trusted.
    assert choice_probability({"outperform": 4.0, "underperform": 1.0}) == pytest.approx(0.8)
    assert choice_probability({}) is None
    assert choice_probability({"underperform": 1.0}) == pytest.approx(0.0)


def test_choice_form_can_be_measurably_worse_than_noul():
    """The motivating claim: the same question, sharper answers, no error signal.

    The noul form answers the base rate exactly (0.5) and lands at the 0.25 Brier
    of an uninformative forecast. The choice form is confident and inverted, which
    is the reported failure mode: sharp probabilities that carry no information
    cost far more than an honest constant.
    """
    observations = []
    for index in range(40):
        outcome = 1 if index < 20 else 0
        observations.append(
            Observation(
                ticker=f"T{index}",
                noul=0.5,
                choice=0.95 if outcome == 0 else 0.05,
                outcome=outcome,
            )
        )
    report = build_report(observations, coverage=1.0, horizons=1)
    assert report.signals["noul"].brier == pytest.approx(0.25)
    assert report.signals["choice"].brier == pytest.approx(0.9025)
    assert report.signals["choice"].brier > report.signals["noul"].brier
    assert report.signals["choice"].brier_skill_score < 0
    # Mean |noul - choice| = |0.5 - 0.95| on the winners, |0.5 - 0.95| on the rest
    assert report.miscalibration_gap == pytest.approx(0.45)


def test_report_counts_and_gap():
    observations = [
        Observation(ticker="AAPL", noul=0.6, choice=0.9, rank=0.75, outcome=1, excess_return=0.01),
        Observation(ticker="MSFT", noul=0.4, choice=0.1, rank=0.25, outcome=0, excess_return=-0.01),
        Observation(ticker="TLT", noul=0.5, choice=None, rank=0.5, outcome=None),  # unscored, excluded
    ]
    report = build_report(observations, coverage=0.9, horizons=2, note="x")
    assert report.observations == 2
    assert report.horizons == 2
    assert report.coverage == pytest.approx(0.9)
    assert report.miscalibration_gap == pytest.approx(0.3)  # mean(|0.6-0.9|, |0.4-0.1|)
    assert set(report.signals) == {"noul", "choice", "rank"}
    assert report.signals["noul"].n == 2


def test_report_with_no_observations_is_empty_not_an_error():
    report = build_report([], coverage=0.0, horizons=0)
    assert report.observations == 0
    assert report.signals == {}
    assert report.miscalibration_gap is None
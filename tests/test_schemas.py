"""Schema, lenient parsing, and decision-validation tests."""

from __future__ import annotations

import pytest

from beatspy.schemas import (
    PortfolioDecision,
    extract_json,
    parse_artifact,
    validate_decision,
)


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped(self):
        text = 'Here is my decision:\n{"allocations": [{"ticker": "SPY", "weight": 1.0}]} hope that helps'
        assert extract_json(text)["allocations"][0]["ticker"] == "SPY"

    def test_picks_first_object(self):
        text = 'note {"a": 1} then {"b": 2}'
        assert extract_json(text) == {"a": 1}

    def test_garbage_returns_none(self):
        assert extract_json("no json here at all") is None
        assert extract_json("") is None

    def test_unbalanced_returns_none(self):
        assert extract_json('{"a": [1, 2}') is None

    def test_malformed_outer_report_does_not_return_nested_fragment(self):
        text = '{"forecasts": [{"ticker": "SPY"}], "method_notes": "cut off</tool_call>'
        assert extract_json(text) is None
        assert extract_json(text, legacy=True) == {"ticker": "SPY"}


class TestParseArtifact:
    def test_research_ok(self):
        model, err = parse_artifact("research", '{"summary": "s", "sentiment": {"SPY": 0.1}}')
        assert err is None
        assert model.summary == "s"

    def test_unknown_role(self):
        model, err = parse_artifact("nope", "{}")
        assert model is None and "unknown artifact role" in err

    def test_validation_failure(self):
        model, err = parse_artifact("research", '{"sentiment": "not a dict"}')
        assert model is None and err and "schema validation" in err

    def test_unrelated_object_cannot_become_empty_report(self):
        for text in ('{"ticker": "SPY"}', '{"method_notes": "partial"}'):
            model, err = parse_artifact("forecaster", text)
            assert model is None and "missing report fields" in err
            model, err = parse_artifact("forecaster", text, legacy=True)
            assert err is None and model.forecasts == []


class TestValidateDecision:
    def test_clamps_position_weight(self):
        result = validate_decision({"allocations": [{"ticker": "SPY", "weight": 0.9}]}, ["SPY"], max_weight=0.35)
        assert result.weights["SPY"] == 0.35
        assert result.cash == pytest.approx(0.65)
        assert any("clamped" in v for v in result.violations)
        assert not result.invalid

    def test_drops_unknown_ticker(self):
        result = validate_decision(
            {"allocations": [{"ticker": "GME", "weight": 0.5}, {"ticker": "SPY", "weight": 0.3}]},
            ["SPY"],
            max_weight=0.5,
        )
        assert result.weights == {"SPY": 0.3}
        assert any("unknown ticker" in v for v in result.violations)

    def test_drops_negative_weight(self):
        result = validate_decision({"allocations": [{"ticker": "SPY", "weight": -0.2}]}, ["SPY"], max_weight=0.5)
        assert result.weights == {}
        assert result.cash == 1.0

    def test_scales_overexposure(self):
        result = validate_decision(
            {
                "allocations": [{"ticker": "SPY", "weight": 0.5}, {"ticker": "TLT", "weight": 0.5}],
                "cash_weight": 0.4,
            },
            ["SPY", "TLT"],
            max_weight=0.6,
        )
        assert sum(result.weights.values()) == pytest.approx(0.6)
        assert any("scaled exposure" in v for v in result.violations)

    def test_cash_implicit_when_missing(self):
        result = validate_decision({"allocations": [{"ticker": "SPY", "weight": 0.4}]}, ["SPY"], max_weight=0.5)
        assert result.cash == pytest.approx(0.6)

    def test_non_dict_invalid(self):
        result = validate_decision("hello", ["SPY"], 0.35)
        assert result.invalid and result.weights == {}

    def test_wrong_schema_invalid(self):
        result = validate_decision({"allocations": "all of it"}, ["SPY"], 0.35)
        assert result.invalid

    def test_missing_allocations_is_invalid_instead_of_liquidating(self):
        assert validate_decision({"ticker": "SPY", "weight": 0.3}, ["SPY"], 0.35).invalid
        assert not validate_decision({"ticker": "SPY"}, ["SPY"], 0.35, legacy=True).invalid

    def test_pydantic_model_accepted(self):
        decision = PortfolioDecision(allocations=[{"ticker": "SPY", "weight": 0.2}], expected_direction="up")
        result = validate_decision(decision.model_dump(), ["SPY"], max_weight=0.35)
        assert result.weights == {"SPY": pytest.approx(0.2)}

    def test_cash_weight_is_clamped(self):
        result = validate_decision({"allocations": [], "cash_weight": 1.5}, ["SPY"], max_weight=0.35)
        assert result.weights == {}
        assert result.cash == 1.0
        assert not result.invalid


class TestDiversificationFloor:
    """The floor is measured, never enforced by rejecting a portfolio."""

    def _stocks(self, n: int, weight: float = 0.04) -> dict:
        return {
            "allocations": [{"ticker": f"S{i}", "weight": weight} for i in range(n)],
            "cash_weight": max(0.0, 1.0 - n * weight),
        }

    def test_meeting_the_floor_is_clean(self):
        tickers = [f"S{i}" for i in range(22)]
        result = validate_decision(self._stocks(22), tickers, max_weight=0.04, min_names=22)
        assert len(result.weights) == 22
        assert result.violations == []
        assert not result.invalid

    def test_shortfall_is_recorded_but_not_rejected(self):
        tickers = [f"S{i}" for i in range(22)]
        result = validate_decision(self._stocks(4), tickers, max_weight=0.04, min_names=22)
        assert not result.invalid
        assert len(result.weights) == 4
        assert any("diversification floor is 22" in v for v in result.violations)

    def test_floor_ignores_dust_weights(self):
        tickers = [f"S{i}" for i in range(22)]
        payload = self._stocks(2)
        payload["allocations"] += [{"ticker": t, "weight": 0.0} for t in tickers[2:]]
        result = validate_decision(payload, tickers, max_weight=0.04, min_names=22)
        assert any("only 2 funded names" in v for v in result.violations)

    def test_default_floor_permits_concentration(self):
        result = validate_decision({"allocations": [{"ticker": "SPY", "weight": 0.35}]}, ["SPY"], max_weight=0.35)
        assert not any("diversification floor" in v for v in result.violations)


class TestScenarioBenchmarkFamilies:
    def test_six_month_fields_default_for_legacy_scenarios(self):
        from beatspy.schemas import Scenario

        s = Scenario(name="x", title="x", start="2022-01-03", end="2022-02-01", tradable=["AAPL"])
        assert s.protocol_version is None  # keeps the historical derivation
        assert s.reference_kind == "momentum"
        assert s.first_only is False
        assert s.require_min_names == 1
        assert s.universe_groups == {}
        assert s.score_assets == []

    def test_group_tickers_must_be_in_universe(self):
        from pydantic import ValidationError

        from beatspy.schemas import Scenario

        with pytest.raises(ValidationError, match="outside the universe"):
            Scenario(
                name="x",
                title="x",
                start="2022-01-03",
                end="2022-02-01",
                tradable=["AAPL"],
                universe_groups={"stocks": ["TSLA"]},
            )

    def test_score_assets_must_be_in_universe(self):
        from pydantic import ValidationError

        from beatspy.schemas import Scenario

        with pytest.raises(ValidationError, match="score_assets"):
            Scenario(
                name="x",
                title="x",
                start="2022-01-03",
                end="2022-02-01",
                tradable=["AAPL"],
                score_assets=["USO"],
            )

    def test_floor_cannot_exceed_listed_stocks(self):
        from pydantic import ValidationError

        from beatspy.schemas import Scenario

        with pytest.raises(ValidationError, match="require_min_names exceeds"):
            Scenario(
                name="x",
                title="x",
                start="2022-01-03",
                end="2022-02-01",
                tradable=["AAPL", "MSFT"],
                universe_groups={"stocks": ["AAPL", "MSFT"]},
                require_min_names=5,
            )


class TestScenarioUniverse:
    def test_benchmark_appended(self):
        from beatspy.schemas import Scenario

        s = Scenario(name="x", title="x", start="2022-01-03", end="2022-02-01", tradable=["AAPL"])
        assert s.universe == ["AAPL", "SPY"]

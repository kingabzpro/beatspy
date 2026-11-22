import asyncio

from beatspy.config import DECISION_MODELS, DecisionConfig
from beatspy.dcn.questions import build_questions


class FakeLlm:
    """Stand-in for a kaggle_benchmarks llm: sync prompt(), scripted replies."""

    model = "google/fake-3.5"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.calls = []

    def prompt(self, text, **kwargs):
        self.prompts.append(text)
        self.calls.append(kwargs)
        return self.replies.pop(0)


def questions():
    return build_questions("per_asset", ["AAPL", "MSFT"], "SPY", 5).questions


def test_decide_parses_json_answers_for_every_question():
    from beatspy.dcn.kbench import KbenchClient

    llm = FakeLlm(
        ['{"answers": {"noul_AAPL": {"type": "noul", "noul": 0.62},'
        ' "noul_MSFT": {"type": "noul", "noul": 0.4}}}']
    )
    result = asyncio.run(KbenchClient(llm).decide({"as_of": "2026-01-05"}, questions()))
    assert result.missing_ids == []
    assert result.answers["noul_AAPL"].noul == 0.62
    assert result.answers["noul_MSFT"].noul == 0.4
    assert result.requests == 1
    assert '"as_of": "2026-01-05"' in llm.prompts[0]
    assert "noul_AAPL" in llm.prompts[0] and "noul_MSFT" in llm.prompts[0]


def test_decide_retries_until_every_question_is_answered():
    from beatspy.dcn.kbench import KbenchClient

    llm = FakeLlm([
        "not json at all",
        '```json\n{"answers": {"noul_AAPL": {"type": "noul", "noul": 1.0}}}\n```',
        '{"answers": {"noul_AAPL": {"type": "noul", "noul": 0.5},'
        ' "noul_MSFT": {"type": "noul", "noul": 0.9}}}',
    ])
    result = asyncio.run(KbenchClient(llm).decide({}, questions()))
    assert result.missing_ids == []
    assert result.answers["noul_MSFT"].noul == 0.9
    assert result.attempts == 3
    assert "previous" in llm.prompts[1].lower()


def test_decide_reports_missing_after_exhausting_retries():
    from beatspy.dcn.kbench import KbenchClient

    llm = FakeLlm(["nope", "still nope"])
    result = asyncio.run(KbenchClient(llm, max_retries=1).decide({}, questions()))
    assert result.answers == {}
    assert set(result.missing_ids) == {"noul_AAPL", "noul_MSFT"}
    assert result.attempts == 2


def test_decide_caps_output_tokens_to_lower_the_quota_reservation():
    from beatspy.dcn.kbench import KbenchClient

    ok = (
        '{"answers": {"noul_AAPL": {"type": "noul", "noul": 0.5},'
        ' "noul_MSFT": {"type": "noul", "noul": 0.5}}}'
    )
    llm = FakeLlm([ok])
    asyncio.run(KbenchClient(llm).decide({}, questions()))
    assert llm.calls[0] == {"extra_api_params": {"max_completion_tokens": 1024}}
    llm = FakeLlm([ok])
    asyncio.run(KbenchClient(llm, max_output_tokens=512).decide({}, questions()))
    assert llm.calls[0]["extra_api_params"]["max_completion_tokens"] == 512


def test_register_makes_a_proxy_slug_usable_as_decision_model():
    from beatspy.dcn.kbench import register

    try:
        register("gemini-3.5-flash")
        config = DecisionConfig(decision_model="gemini-3.5-flash")
        assert config.label.startswith("Kaggle")
        assert config.category == "ai"
    finally:
        del DECISION_MODELS["gemini-3.5-flash"]


def test_model_names_are_canonicalized_to_bare_slugs():
    from beatspy.dcn.kbench import KbenchClient, canonical_model, register

    assert canonical_model("google/gemini-3.1-flash-lite-preview") == "gemini-3.1-flash-lite-preview"
    assert canonical_model("anthropic/claude-sonnet-5@default") == "claude-sonnet-5-default"
    assert canonical_model("gpt-oss-120b") == "gpt-oss-120b"
    assert KbenchClient(FakeLlm(["{}"])).model == "fake-3.5"
    try:
        assert register("google/gemini-3.1-flash-lite-preview") == "gemini-3.1-flash-lite-preview"
        assert "gemini-3.1-flash-lite-preview" in DECISION_MODELS
        assert "google/gemini-3.1-flash-lite-preview" not in DECISION_MODELS
    finally:
        del DECISION_MODELS["gemini-3.1-flash-lite-preview"]

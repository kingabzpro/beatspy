"""Kaggle Model Proxy client: any kaggle_benchmarks ``llm`` answering BeatSPY's questions."""

from __future__ import annotations

import asyncio
import json
import time

from ..config import DECISION_MODELS
from ..schemas import Question, parse_answers
from .clients import DcnResult, _questions_payload

INSTRUCTIONS = (
    "You are a decision model in the BeatSPY benchmark. The JSON request below holds a market "
    "state and one question per ticker; judge every question on the state alone. Reply with only "
    'a JSON object of the form {"answers": {"<question_id>": {"type": "noul", "noul": '
    "<probability>}}} where probability is your calibrated chance, between 0 and 1, that the "
    "question's answer is yes. Answer every question id, no extra text.\n\n"
)

RETRY_NOTE = (
    "\n\nYour previous reply was not a valid answer for every question id. Reply again with only "
    "the JSON object, one entry per question id."
)


def canonical_model(raw: str) -> str:
    """Proxy names arrive prefixed (google/...) and suffixed (@default); bare slugs key everything."""
    return raw.split("/")[-1].replace("@", "-")


def register(model: str) -> str:
    """Make a Model Proxy slug usable as a decision model for this process."""
    model = canonical_model(model)
    DECISION_MODELS.setdefault(
        model,
        {
            "provider": "kaggle",
            "model": model,
            "label": f"Kaggle Model Proxy ({model})",
            "base_url": "kaggle://model-proxy",
            "api_key_env": None,
            "cost_per_m_input": 0.0,
            "cost_per_m_output": 0.0,
            "context_window": 0,
            "max_questions": 64,
            "state_note": "Answers the BeatSPY decision questions through Kaggle's Model Proxy.",
            "category": "ai",
        },
    )
    return model


def _extract_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


class KbenchClient:
    """A DcnClient that sends the systemone request as one prompt and parses the reply."""

    provider = "kaggle"
    endpoint = "kaggle://model-proxy"

    def __init__(self, llm, *, model: str | None = None, max_retries: int = 2, max_output_tokens: int = 1024):
        self.llm = llm
        raw = model or getattr(llm, "model", None) or getattr(llm, "name", None) or "kaggle-llm"
        self.model = canonical_model(raw)
        self.max_retries = max_retries
        # The proxy pre-reserves quota from the max output size; without a cap the
        # reservation ($1.4+ on premium models) exceeds the account quota with 403s.
        self.max_output_tokens = max_output_tokens

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        ids = list(questions)
        request = json.dumps(
            {"model": self.model, "state": state, "questions": _questions_payload(questions)}
        )
        started = time.perf_counter()
        answers: dict = {}
        missing = list(ids)
        text = ""
        for attempt in range(1, self.max_retries + 2):
            note = "" if attempt == 1 else RETRY_NOTE
            text = await asyncio.to_thread(
                self.llm.prompt,
                INSTRUCTIONS + request + note,
                extra_api_params={"max_completion_tokens": self.max_output_tokens},
            )
            answers, missing = parse_answers(_extract_json(text), ids)
            if not missing:
                break
        return DcnResult(
            answers=answers,
            missing_ids=missing,
            latency_s=round(time.perf_counter() - started, 3),
            attempts=attempt,
            requests=attempt,
            raw={"response": text[:2000]},
        )

    async def close(self) -> None:
        pass

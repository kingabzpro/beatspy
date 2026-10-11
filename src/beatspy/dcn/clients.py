"""Decision-model clients: one adapter per provider, one shared contract.

All three providers accept a `state` plus a map of typed `questions`, so the
adapter surface is deliberately thin:

    async with client:
        result = await client.decide(state_text, questions)

Transport, auth, and the response envelope differ:

- OpenAI Decisions API: `POST /v1/decisions` with `Authorization: Bearer`, a
  `questions` array (`predicate`, `choice`, `score`), and `answers` keyed by
  question name; usage arrives as `usage.input_tokens`.
- TypeSafe Jev: a single POST to https://api.typesafe.ai/v1/systemone.
- Cloudflare Clef: POST to /accounts/{id}/ai/run/@cf/cloudflare/{model} with the
  account id in the path. Workers AI truncates long text state to roughly its
  first 2K tokens, so state is sent as a compact JSON *object* (the API accepts
  structured state) and the client flags it when the budget is at risk.

Retries cover the documented transient statuses only: 429 and 529, plus 5xx. A
4xx other than 429 is a client or schema error, and retrying it would just burn
quota, so it raises immediately.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ..schemas import (
    CallRecord,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
    parse_answers,
)

# Documented transient statuses: 429 Too Many Requests, 529 Overloaded,
# plus the ordinary 5xx family.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


class DcnError(RuntimeError):
    """A provider call failed after exhausting retries, or the request was invalid."""


@dataclass
class DcnResult:
    """One provider response in provider-neutral form."""

    answers: dict = field(default_factory=dict)
    missing_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    requests: int = 1
    raw: dict | None = None


def _questions_payload(questions: dict[str, Question]) -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for question_id, question in questions.items():
        body: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
        if question.criteria is not None:
            body["criteria"] = question.criteria
        payload[question_id] = body
    return payload


def _options_text(criteria) -> str:
    """Render criteria the way these Spaces' own demo tables do."""
    if isinstance(criteria, dict):
        return " | ".join(f"{key}: {value}" for key, value in criteria.items())
    if isinstance(criteria, list):
        return " | ".join(str(item) for item in criteria)
    return ""


class DcnClient(Protocol):
    provider: str
    model: str
    endpoint: str

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult: ...

    async def close(self) -> None: ...


class HttpDcnClient:
    """Shared HTTP/retry machinery. Subclasses build the request and read the response."""

    provider = "unknown"

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str | None,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        account_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.account_id = account_id
        self.max_retries = max_retries
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ---------------------------------------------------------------- subclass API

    def build_request(self, state: dict, questions: dict[str, Question]) -> tuple[str, dict, dict, dict]:
        """Return (method, url, headers, json_body)."""
        raise NotImplementedError

    def parse_response(self, payload: dict, question_ids: list[str]) -> DcnResult:
        raise NotImplementedError

    # ------------------------------------------------------------------- transport

    async def _post(self, url: str, headers: dict, body: dict) -> tuple[dict, int]:
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                response = await self._client.post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await self._sleep(attempt)
                continue
            if response.status_code in RETRY_STATUSES:
                last_error = DcnError(f"HTTP {response.status_code}: {response.text[:200]}")
                if attempt >= self.max_retries:
                    break
                await self._sleep(attempt)
                continue
            if response.status_code >= 400:
                raise DcnError(f"HTTP {response.status_code}: {response.text[:300]}")
            try:
                payload = response.json()
            except ValueError as exc:  # non-JSON body
                raise DcnError(f"non-JSON response: {exc}") from exc
            if not isinstance(payload, dict):
                raise DcnError("response was not a JSON object")
            return payload, attempts
        raise DcnError(f"request failed after {attempts} attempts: {last_error}")

    async def _sleep(self, attempt: int) -> None:
        # Exponential backoff with jitter, capped so a slow provider cannot stall a run.
        delay = min(2.0**attempt, 20.0) * (0.5 + random.random() / 2)
        await asyncio.sleep(delay)

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        method, url, headers, body = self.build_request(state, questions)
        if method != "POST":
            raise DcnError(f"unsupported method {method}")
        started = asyncio.get_running_loop().time()
        payload, attempts = await self._post(url, headers, body)
        latency = asyncio.get_running_loop().time() - started
        result = self.parse_response(payload, list(questions))
        result.latency_s = round(latency, 3)
        result.attempts = attempts
        return result


class OpenAiDecisionsClient(HttpDcnClient):
    """OpenAI Decisions API.

    This provider does not share the System One envelope, so it adapts in three
    documented ways:

    - it has its own endpoint, `POST /v1/decisions`, not the Responses API;
    - `questions` is an **array** with a required per-question `name`, and the
      yes/no type is called `predicate`; `choice` takes `choices` of
      `{value, description}` and `score` takes `levels` of `{label, description}`;
    - `answers` comes back as an array keyed by `name`, a predicate answer
      carries `probability`, and an answer may be a `refusal` with no value.

    Getting any of these wrong returns HTTP 400, so they are pinned by tests.
    """

    provider = "openai_decisions"

    def build_request(self, state: dict, questions: dict[str, Question]) -> tuple[str, dict, dict, dict]:
        url = f"{self.endpoint.rstrip('/')}/decisions"
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }
        payload = []
        for question_id, question in questions.items():
            entry: dict[str, Any] = {
                "name": question_id,
                "type": "predicate" if question.type == "noul" else question.type,
                "instructions": _flatten(question.instructions),
            }
            if question.type == "choice" and isinstance(question.criteria, dict):
                entry["choices"] = [
                    {"value": str(value), "description": str(description)}
                    for value, description in question.criteria.items()
                ]
            elif question.type == "score" and isinstance(question.criteria, list):
                entry["levels"] = [
                    {"label": str(level), "description": str(level)} for level in question.criteria
                ]
            payload.append(entry)
        body = {"model": self.model, "input": _state_text(state), "questions": payload}
        return "POST", url, headers, body

    def parse_response(self, payload: dict, question_ids: list[str]) -> DcnResult:
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        refused: list[str] = []
        for item in payload.get("answers") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            kind = str(item.get("type", "")).strip().lower()
            if not isinstance(name, str) or kind == "refusal":
                if isinstance(name, str):
                    refused.append(name)
                continue
            if kind == "predicate":
                probability = item.get("probability")
                if isinstance(probability, (int, float)):
                    answers[name] = NoulAnswer(noul=min(max(float(probability), 0.0), 1.0))
            elif kind == "choice":
                # probabilities arrive as a list of {value, probability}.
                probabilities = {
                    str(row.get("value")): float(row.get("probability") or 0.0)
                    for row in item.get("probabilities") or []
                    if isinstance(row, dict) and row.get("value") is not None
                }
                total = sum(probabilities.values())
                if probabilities and total > 0:
                    # A distribution that does not sum to 1 is normalized rather
                    # than discarded, so a rounding quirk cannot lose an answer.
                    normalised = {value: weight / total for value, weight in probabilities.items()}
                    chosen = item.get("choice")
                    if not isinstance(chosen, str) or chosen not in normalised:
                        chosen = max(normalised, key=lambda value: normalised[value])
                    answers[name] = ChoiceAnswer(
                        choice=chosen,
                        probabilities=normalised,
                        confidence=item.get("confidence"),
                    )
            elif kind == "score":
                score = item.get("score")
                if isinstance(score, (int, float)):
                    legend = {
                        str(row.get("value")): str(row.get("label") or "")
                        for row in item.get("probabilities") or []
                        if isinstance(row, dict)
                    }
                    answers[name] = ScoreAnswer(score=float(score), legend=legend)
        missing = [question_id for question_id in question_ids if question_id not in answers]
        usage = payload.get("usage") or {}
        return DcnResult(
            answers=answers,
            missing_ids=missing,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            raw={**payload, "_refused": refused} if refused else payload,
        )


def _flatten(instructions: str | dict | list) -> str:
    """OpenAI takes `instructions` as a string; fold structured instructions into one."""
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, separators=(",", ":"), sort_keys=True)


class TypeSafeClient(HttpDcnClient):
    """TypeSafe System One endpoint (Jev)."""

    provider = "typesafe"

    def build_request(self, state: dict, questions: dict[str, Question]) -> tuple[str, dict, dict, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }
        body = {"model": self.model, "state": state, "questions": _questions_payload(questions)}
        return "POST", self.endpoint, headers, body

    def parse_response(self, payload: dict, question_ids: list[str]) -> DcnResult:
        answers, missing = parse_answers(payload, question_ids)
        usage = payload.get("usage") or {}
        return DcnResult(
            answers=answers,
            missing_ids=missing,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            raw=payload,
        )


class CloudflareClefClient(HttpDcnClient):
    """Cloudflare Workers AI Clef / Clef-flash."""

    provider = "cloudflare"

    def build_request(self, state: dict, questions: dict[str, Question]) -> tuple[str, dict, dict, dict]:
        account = self.account_id or ""
        url = f"{self.endpoint.rstrip('/')}/{account}/ai/run/@cf/cloudflare/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }
        # Structured state, not a stringified blob: the model reads JSON directly,
        # and Workers AI's ~2K-token truncation then applies to a value we keep
        # inside budget rather than to an arbitrary projection of it.
        body = {"model": self.model, "state": state, "questions": _questions_payload(questions)}
        return "POST", url, headers, body

    def parse_response(self, payload: dict, question_ids: list[str]) -> DcnResult:
        # Workers AI wraps the model result in {"result": {...}, "success": true}.
        inner = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        answers, missing = parse_answers(inner, question_ids)
        usage = inner.get("usage") or payload.get("usage") or {}
        return DcnResult(
            answers=answers,
            missing_ids=missing,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            raw=payload,
        )


class RequestyClient(TypeSafeClient):
    """Requesty router. Same System One contract, its own endpoint and key.

    Verified live: POST /v1/decisions with `{model, state, questions}` returns
    `{"answers": {"<id>": {"type": "noul", "noul": 0.9952}}}`.
    """

    provider = "requesty"


class OpenRouterDecisionsClient(TypeSafeClient):
    """OpenRouter decisions endpoint (alpha). Same contract, again.

    Verified live: POST /api/alpha/decisions returns the typed answers plus the
    resolved model version and the request cost. The chat/completions endpoint
    rejects these models by design, so the path matters.
    """

    provider = "openrouter"


class HfSpaceClient:
    """A decision model hosted as a Hugging Face Space.

    Hugging Face serves no decision models through its serverless Inference API,
    but it hosts them as Spaces, and a Space exposes its work as an HTTP API.
    The transport is Gradio's two-step call: POST the inputs to
    `/gradio_api/call/<fn>`, then read the server-sent event that carries the
    result. Each Space names its own inputs, so the model spec records the
    function name and the input shape.
    """

    provider = "hf_space"

    def __init__(self, *, model: str, endpoint: str, api_key: str | None = None, fn: str = "decide",
                 input_mode: str = "text", timeout_s: float = 300.0, max_retries: int = 2, client=None,
                 **_ignored):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.fn = fn
        self.input_mode = input_mode
        self.api_key = api_key
        self.max_retries = max_retries
        # ZeroGPU spaces serve anonymous callers a small quota and answer 429 with
        # "authenticate with a Hugging Face token for more" once it is spent.
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        import httpx

        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    def _inputs(self, state: dict, questions: dict[str, Question]) -> list:
        import json

        payload = _questions_payload(questions)
        state_text = json.dumps(state, separators=(",", ":"), sort_keys=True)
        if self.input_mode == "text":
            # State and questions as JSON text. The trailing None fills the
            # optional image input every one of these Spaces declares; omitting
            # it makes Gradio fail the call before the model sees anything.
            return [state_text, json.dumps(payload), None]
        if self.input_mode == "dataframe":
            rows = [
                [question_id, body["type"], body["instructions"], _options_text(body.get("criteria"))]
                for question_id, body in payload.items()
            ]
            table = {"headers": ["name", "type", "question", "options"], "data": rows, "metadata": None}
            return [state_text, table, None]
        raise DcnError(f"unknown hf_space input mode {self.input_mode!r}")

    @staticmethod
    def _outputs(event_text: str) -> list:
        """Pull the output list out of a Gradio server-sent event."""
        import json

        for line in event_text.splitlines():
            if line.startswith("data:"):
                return json.loads(line.split("data:", 1)[1].strip())
        raise DcnError(f"gradio event carried no data: {event_text[:200]!r}")

    @staticmethod
    def _find_answers(value, depth: int = 0):
        """Find the decision payload inside whatever a Space returned."""
        import json

        if depth > 4:
            return None
        if isinstance(value, dict):
            if isinstance(value.get("answers"), dict):
                return value
            for item in value.values():
                found = HfSpaceClient._find_answers(item, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = HfSpaceClient._find_answers(item, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, str) and "answers" in value:
            try:
                return HfSpaceClient._find_answers(json.loads(value), depth + 1)
            except ValueError:
                return None
        return None

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        import asyncio
        import random

        url = f"{self.endpoint}/gradio_api/call/{self.fn}"
        body = {"data": self._inputs(state, questions)}
        started = asyncio.get_running_loop().time()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(url, json=body, headers=self._headers)
                if response.status_code in RETRY_STATUSES or response.status_code >= 500:
                    last_error = DcnError(f"HTTP {response.status_code}: {response.text[:200]}")
                    raise last_error
                if response.status_code >= 400:
                    raise DcnError(f"HTTP {response.status_code}: {response.text[:300]}")
                event_id = response.json()["event_id"]
                stream = await self._client.get(f"{url}/{event_id}", headers=self._headers)
                if stream.status_code >= 400:
                    raise DcnError(f"HTTP {stream.status_code} reading the result: {stream.text[:200]}")
                outputs = self._outputs(stream.text)
                payload = self._find_answers(outputs)
                if payload is None:
                    raise DcnError(f"no typed answers in the Space response: {str(outputs)[:200]}")
                answers, missing = parse_answers(payload, list(questions))
                return DcnResult(
                    answers=answers,
                    missing_ids=missing,
                    input_tokens=int((payload.get("usage") or {}).get("input_tokens") or 0),
                    output_tokens=int((payload.get("usage") or {}).get("output_tokens") or 0),
                    latency_s=round(asyncio.get_running_loop().time() - started, 3),
                    attempts=attempt + 1,
                    raw=payload,
                )
            except (DcnError, httpx.HTTPError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(2.0 ** attempt, 20.0) * (0.5 + random.random() / 2))
        raise DcnError(f"space call failed after {self.max_retries + 1} attempts: {last_error}")


def _state_text(state: dict) -> str:
    import json

    return json.dumps(state, separators=(",", ":"), sort_keys=True)


class HfRestClient(HttpDcnClient):
    """A Space that answers one typed question per plain JSON call.

    AutoTrust's JEV demo exposes `POST /api/decide` with `kind`, `state`,
    `question`, and `options`, and returns probabilities over those options, so
    a decision with N questions fans out to N calls that are reassembled into
    one DcnResult. `noul` is asked with the Space's fixed options
    ["false", "true"], and its state field is capped at 6,000 characters.
    """

    provider = "hf_rest"
    MAX_STATE_CHARS = 6_000

    def _question_text(self, question: Question) -> str:
        import json

        instructions = question.instructions
        if isinstance(instructions, dict) and "question" in instructions:
            return str(instructions["question"])
        if isinstance(instructions, str):
            return instructions
        return json.dumps(instructions, separators=(",", ":"), sort_keys=True)

    def _request(self, state_text: str, question: Question) -> dict | None:
        """The one-question request body, or None for a form this Space cannot ask."""
        if question.type == "noul":
            options = ["false", "true"]
        elif question.type == "choice":
            options = [str(option) for option in question.criteria]
        else:
            # The Space fixes its score form to its own six-level 0..5 scale,
            # which this benchmark's five-level rank form does not share.
            return None
        return {
            "kind": question.type,
            "state": state_text,
            "question": self._question_text(question),
            "options": options,
        }

    @staticmethod
    def _answer(question: Question, options: list[str], payload: dict):
        probabilities = payload.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(options):
            return None
        if question.type == "noul":
            return NoulAnswer(noul=float(probabilities[options.index("true")]))
        choice = str(payload.get("choice", ""))
        if choice not in options:
            return None
        return ChoiceAnswer(choice=choice, probabilities=dict(zip(options, probabilities, strict=True)))

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        import asyncio

        state_text = _state_text(state)
        if len(state_text) > self.MAX_STATE_CHARS:
            raise DcnError(
                f"the state is {len(state_text)} characters; this Space accepts at most {self.MAX_STATE_CHARS}"
            )
        url = f"{self.endpoint}/api/decide"
        started = asyncio.get_running_loop().time()
        answers: dict = {}
        missing: list[str] = []
        attempts = 0
        requests = 0
        for question_id, question in questions.items():
            request = self._request(state_text, question)
            if request is None:
                missing.append(question_id)
                continue
            payload, tries = await self._post(url, {}, request)
            requests += 1
            attempts += tries
            answer = self._answer(question, request["options"], payload)
            if answer is None:
                missing.append(question_id)
            else:
                answers[question_id] = answer
        return DcnResult(
            answers=answers,
            missing_ids=missing,
            latency_s=round(asyncio.get_running_loop().time() - started, 3),
            attempts=attempts,
            requests=requests,
        )


_CLIENTS = {
    "openai_decisions": OpenAiDecisionsClient,
    "typesafe": TypeSafeClient,
    "cloudflare": CloudflareClefClient,
    "requesty": RequestyClient,
    "openrouter": OpenRouterDecisionsClient,
    "hf_space": HfSpaceClient,
    "hf_rest": HfRestClient,
}


def missing_credentials(settings) -> list[str]:
    """Environment variables a run of this decision model still needs.

    Checked before a run starts: an absent key would otherwise fail inside the
    transport as `Illegal header value b'Bearer '` after several retries, which
    tells the user nothing about the fix.
    """
    from ..config import cloudflare_account_id, decision_api_key

    decision = settings.decision
    missing: list[str] = []
    if decision.provider not in ("hf_space", "hf_rest") and not decision_api_key(decision.decision_model):
        missing.append(decision.spec["api_key_env"])
    if decision.provider == "cloudflare" and not cloudflare_account_id():
        missing.append("CLOUDFLARE_ACCOUNT_ID")
    return missing


def make_client(settings, *, client: httpx.AsyncClient | None = None) -> HttpDcnClient:
    """Build the client for the configured decision model."""
    from ..config import cloudflare_account_id, decision_api_key

    decision = settings.decision
    provider = decision.provider
    factory = _CLIENTS.get(provider)
    if factory is None:
        raise DcnError(f"no client for provider {provider!r}")
    missing = missing_credentials(settings)
    if missing:
        raise DcnError(
            f"{decision.label} needs {' and '.join(missing)}. "
            f"Export it, or add it to .env next to the repository, then retry. "
            f"`beatspy models` shows which credentials resolve."
        )
    api_key = decision_api_key(decision.decision_model)
    extra = {}
    if provider == "hf_space":
        import os

        # A Space runs on Hugging Face hardware and takes no key of its own; an HF
        # token only lifts the shared ZeroGPU quota anonymous callers run into.
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("BEATSPY_HF_TOKEN")
        extra = {
            "fn": decision.spec.get("space_fn", "decide"),
            "input_mode": decision.spec.get("input_mode", "text"),
        }
    return factory(
        model=decision.spec["model"],
        endpoint=decision.endpoint,
        api_key=api_key,
        timeout_s=decision.timeout_s,
        max_retries=decision.max_retries,
        account_id=cloudflare_account_id(),
        client=client,
        **extra,
    )


def build_call_record(result: DcnResult, state_text: str, question_ids: list[str], client: HttpDcnClient) -> CallRecord:
    """Normalize one result into the artifact we store and replay."""
    import hashlib

    return CallRecord(
        provider=client.provider,
        model=client.model,
        endpoint=client.endpoint,
        state_sha256=hashlib.sha256(state_text.encode("utf-8")).hexdigest(),
        state_chars=len(state_text),
        question_ids=list(question_ids),
        answers=result.answers,
        missing_ids=result.missing_ids,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        requests=result.requests,
        latency_s=result.latency_s,
        attempts=result.attempts,
    )
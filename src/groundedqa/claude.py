"""Answering with Claude through the Messages API, with citations enabled."""

import anthropic

from .answers import MODE_CLAUDE, NOT_COVERED, Answer, AnswerPart
from .citations import answer_from_response
from .config import DEFAULT_MODEL
from .retrieval import Hit

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You answer questions using only the documents in the user's message. They are \
excerpts from a company's own manuals, terms and policies.

- Use only what the documents say. Do not fill gaps with general knowledge, even \
when you are confident it is right.
- Keep answers short: the direct answer first, then only the conditions or \
exceptions that change it.
- If the documents do not contain the answer, start your reply with exactly \
"{not_covered}" Then, in one sentence, suggest who to ask: {contact}, or a \
contact named in the documents if one fits better. Do not guess.
- If the documents answer only part of the question, answer that part and say \
what is missing.
- Do not add reference markers or a list of sources; citations are attached to \
your answer automatically."""


class ClaudeUnavailable(Exception):
    """No usable credentials. The caller should switch to the offline answerer."""


class ClaudeError(Exception):
    """The request failed for a reason the user needs to see."""


def make_client() -> anthropic.Anthropic | None:
    """Return a client if the SDK can resolve credentials, otherwise None.

    The SDK looks for ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN and `ant auth login`
    profiles, so a missing environment variable alone does not mean offline.
    """
    try:
        client = anthropic.Anthropic()
    except anthropic.CredentialsError:
        return None
    if client.api_key is None and client.auth_token is None and client.credentials is None:
        return None
    return client


class ClaudeAnswerer:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = DEFAULT_MODEL,
        contact: str = "the team that maintains these documents",
        effort: str = "low",
        max_tokens: int = 2000,
    ):
        self.client = client
        self.model = model
        self.contact = contact
        self.effort = effort
        self.max_tokens = max_tokens

    def build_request(self, question: str, hits: list[Hit]) -> dict:
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
                "title": hit.chunk.label,
                "citations": {"enabled": True},
            }
            for hit in hits
        ]
        # No `thinking` parameter: this model thinks adaptively by default, and an
        # explicit "disabled" or budget is rejected. Effort is the control instead.
        # Citations cannot be combined with output_config.format, so no structured output here.
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT.format(not_covered=NOT_COVERED, contact=self.contact),
            "messages": [{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        request = self.build_request(question, hits)
        try:
            response = self.client.beta.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise ClaudeUnavailable(f"The Anthropic API rejected the credentials ({exc.status_code}).") from exc
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after")
            hint = f" Retry after {retry_after} s." if retry_after else ""
            raise ClaudeError(f"Rate limited by the Anthropic API.{hint}") from exc
        except anthropic.APIStatusError as exc:
            raise ClaudeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ClaudeError(f"Could not reach the Anthropic API: {exc}") from exc

        return self.to_answer(question, response, [hit.chunk for hit in hits])

    def to_answer(self, question: str, response, chunks) -> Answer:
        stop_reason = response.stop_reason
        if stop_reason == "refusal":
            # Any partial output before a refusal is discarded, not shown.
            category = getattr(response.stop_details, "category", None)
            reason = f" (category: {category})" if category else ""
            return Answer(
                question=question,
                parts=[AnswerPart("Claude declined to answer this question.")],
                sources=[],
                mode=MODE_CLAUDE,
                refused=True,
                model=response.model,
                note=f"The request was declined by Claude's safety checks{reason}.",
            )
        if stop_reason not in ("end_turn", "stop_sequence", "max_tokens", "model_context_window_exceeded"):
            raise ClaudeError(f"Unexpected stop reason from the API: {stop_reason!r}")

        answer = answer_from_response(question, response, chunks)
        if stop_reason in ("max_tokens", "model_context_window_exceeded"):
            answer.note = "The answer was cut off at the token limit and may be incomplete."
        elif _used_fallback(response):
            answer.note = f"Answered by {response.model} after a server-side fallback."
        return answer


def _used_fallback(response) -> bool:
    usage = getattr(response, "usage", None)
    iterations = getattr(usage, "iterations", None) or []
    return any(getattr(item, "type", None) == "fallback_message" for item in iterations)

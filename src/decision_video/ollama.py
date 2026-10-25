"""Ollama Decision API client."""

from __future__ import annotations

import logging
from typing import Any

import requests

from decision_video.models import (
    OllamaDecisionRequest,
    OllamaDecisionResponse,
    NoulAnswer,
    AnswerUnion,
)

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama API returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def build_request(
    model: str,
    state: str | dict[str, Any],
    questions: list[dict[str, Any]],
) -> OllamaDecisionRequest:
    """Build a validated Ollama Decision API request from config data."""
    parsed_questions: dict[str, Any] = {}
    for q in questions:
        key = q["key"]
        qtype = q.get("type", "noul")
        if qtype == "noul":
            parsed_questions[key] = {
                "type": "noul",
                "instructions": q["instructions"],
                "criteria": {
                    "false": q.get("criteria_false", "Does not match"),
                    "true": q.get("criteria_true", "Matches"),
                },
            }
        elif qtype == "choice":
            parsed_questions[key] = {
                "type": "choice",
                "instructions": q["instructions"],
                "criteria": q["criteria"],
            }
        elif qtype == "score":
            parsed_questions[key] = {
                "type": "score",
                "instructions": q["instructions"],
                "criteria": q["criteria"],
            }
        else:
            raise ValueError(f"Unknown question type: {qtype}")

    return OllamaDecisionRequest(
        model=model,
        state=state,
        questions=parsed_questions,
    )


def send_request(
    request: OllamaDecisionRequest,
    api_url: str,
    timeout: int = 120,
) -> OllamaDecisionResponse:
    """Send a request to the Ollama Decision API and return the parsed response."""
    body = request.model_dump(mode="json", exclude_none=True)

    logger.debug("Sending request to %s: %s", api_url, body)

    try:
        response = requests.post(
            api_url,
            json=body,
            timeout=timeout,
        )
    except requests.ConnectionError as e:
        raise OllamaError(
            f"Cannot connect to Ollama at {api_url}. Is Ollama running?"
        ) from e
    except requests.Timeout as e:
        raise OllamaError(
            f"Request to Ollama timed out after {timeout}s"
        ) from e

    if response.status_code != 200:
        raise OllamaError(
            f"Ollama API returned {response.status_code}: {response.text}",
            status_code=response.status_code,
        )

    try:
        return OllamaDecisionResponse.model_validate(response.json())
    except Exception as e:
        raise OllamaError(f"Failed to parse Ollama response: {e}") from e


def get_noul_probability(
    response: OllamaDecisionResponse,
    question_key: str,
) -> float:
    """Extract the noul probability (probability of true) from a response."""
    answer = response.answers.get(question_key)
    if answer is None:
        raise KeyError(f"No answer for question key: {question_key}")

    # The answer is a discriminated union instance
    if not isinstance(answer, NoulAnswer):
        raise TypeError(
            f"Expected NoulAnswer for '{question_key}', got {type(answer).__name__}"
        )

    return answer.noul

"""Pydantic models for Ollama Decision API request/response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel


class NoulQuestion(BaseModel):
    """A noul-type question: returns the probability of true (0 to 1)."""

    type: str = "noul"
    instructions: str
    criteria: dict[str, str] = Field(
        default_factory=lambda: {
            "false": "Does not match",
            "true": "Matches",
        }
    )


class ChoiceQuestion(BaseModel):
    """A choice-type question: pick one of 2–26 labeled options."""

    type: str = "choice"
    instructions: str
    criteria: dict[str, str]


class ScoreQuestion(BaseModel):
    """A score-type question: pick from 2–10 ordered levels."""

    type: str = "score"
    instructions: str
    criteria: list[str]


# Union type for questions — used directly in dict fields
QuestionUnion = NoulQuestion | ChoiceQuestion | ScoreQuestion


class OllamaDecisionRequest(BaseModel):
    """Request body for POST /v1/systemone."""

    model: str
    state: str | dict[str, Any]
    questions: dict[str, QuestionUnion]
    images: list[str] | None = None


class NoulAnswer(BaseModel):
    """The result of a noul-type question."""

    type: str = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    """The result of a choice-type question."""

    type: str = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    """The result of a score-type question."""

    type: str = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float


# Union type for answers — used directly in dict fields
AnswerUnion = NoulAnswer | ChoiceAnswer | ScoreAnswer


class OllamaDecisionResponse(BaseModel):
    """Response from the Ollama Decision API."""

    model: str
    answers: dict[str, AnswerUnion]
    usage: dict[str, int] | None = None


class QuestionList(RootModel[list[QuestionUnion]]):
    """Root model for a list of questions (used when serializing)."""

    root: list[QuestionUnion]


class AnswerDict(RootModel[dict[str, AnswerUnion]]):
    """Root model for answers dict (used when serializing)."""

    root: dict[str, AnswerUnion]

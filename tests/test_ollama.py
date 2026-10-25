"""Tests for the Ollama client and request building."""

from decision_video.ollama import build_request, get_noul_probability
from decision_video.models import (
    OllamaDecisionResponse,
    NoulAnswer,
    ChoiceAnswer,
)


def test_build_request_with_noul_question():
    """build_request correctly formats a noul question."""
    questions = [
        {
            "key": "moment_found",
            "type": "noul",
            "instructions": "Does this frame contain the target?",
            "criteria_false": "No match",
            "criteria_true": "Match found",
        }
    ]
    req = build_request(model="d3:9b", state="test state", questions=questions)

    assert req.model == "d3:9b"
    assert req.state == "test state"
    assert "moment_found" in req.questions
    q = req.questions["moment_found"]
    assert q.type == "noul"
    assert q.instructions == "Does this frame contain the target?"
    assert q.criteria == {"false": "No match", "true": "Match found"}


def test_build_request_with_choice_question():
    """build_request correctly formats a choice question."""
    questions = [
        {
            "key": "label",
            "type": "choice",
            "instructions": "Which label fits?",
            "criteria": {"a": "Option A", "b": "Option B"},
        }
    ]
    req = build_request(model="d3:9b", state="test state", questions=questions)

    q = req.questions["label"]
    assert q.type == "choice"
    assert q.criteria == {"a": "Option A", "b": "Option B"}


def test_build_request_with_score_question():
    """build_request correctly formats a score question."""
    questions = [
        {
            "key": "urgency",
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["Low", "Medium", "High"],
        }
    ]
    req = build_request(model="d3:9b", state="test state", questions=questions)

    q = req.questions["urgency"]
    assert q.type == "score"
    assert q.criteria == ["Low", "Medium", "High"]


def test_build_request_unknown_type_raises():
    """build_request raises ValueError for unknown question types."""
    questions = [
        {
            "key": "bad",
            "type": "unknown_type",
            "instructions": "test",
        }
    ]
    try:
        build_request(model="d3:9b", state="test state", questions=questions)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "unknown_type" in str(e)


def test_get_noul_probability():
    """get_noul_probability extracts the noul value from a response."""
    answer = NoulAnswer(noul=0.95)
    resp = OllamaDecisionResponse(
        model="d3:9b",
        answers={"moment_found": answer},
    )
    prob = get_noul_probability(resp, "moment_found")
    assert prob == 0.95


def test_get_noul_probability_missing_key():
    """get_noul_probability raises KeyError for missing question key."""
    answer = NoulAnswer(noul=0.95)
    resp = OllamaDecisionResponse(
        model="d3:9b",
        answers={"other_key": answer},
    )
    try:
        get_noul_probability(resp, "moment_found")
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_get_noul_probability_wrong_type():
    """get_noul_probability raises TypeError for non-noul answer."""
    answer = ChoiceAnswer(
        type="choice",
        choice="a",
        probabilities={"a": 0.9, "b": 0.1},
        confidence=0.9,
    )
    resp = OllamaDecisionResponse(
        model="d3:9b",
        answers={"label": answer},
    )
    try:
        get_noul_probability(resp, "label")
        assert False, "Expected TypeError"
    except TypeError:
        pass

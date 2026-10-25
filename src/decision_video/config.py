"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import tomli
from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    model: str = "d3:9b"
    api_url: str = "http://localhost:11434/v1/systemone"
    timeout: int = 120

    @field_validator("timeout")
    @classmethod
    def timeout_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be a positive integer")
        return v


class VideoConfig(BaseModel):
    input_path: str = ""
    frame_interval: float = 1.0
    frames_dir: str = ""

    @field_validator("frame_interval")
    @classmethod
    def interval_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("frame_interval must be positive")
        return v


class AnalysisConfig(BaseModel):
    threshold: float = 0.7
    output: str = ""
    questions: list[dict[str, object]] = Field(default_factory=list)

    @field_validator("threshold")
    @classmethod
    def threshold_must_be_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class StateConfig(BaseModel):
    context: str = ""


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    state: StateConfig = Field(default_factory=StateConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate configuration from a TOML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw = tomli.load(f)

    # Flatten nested sections into the AppConfig structure
    ollama_raw = raw.get("ollama", {})
    video_raw = raw.get("video", {})
    analysis_raw = raw.get("analysis", {})
    state_raw = raw.get("state", {})

    # Parse questions from the [[analysis.questions]] array-of-tables format
    questions_raw = analysis_raw.pop("questions", [])

    ollama = OllamaConfig(**ollama_raw)
    video = VideoConfig(**video_raw)
    analysis = AnalysisConfig(questions=questions_raw, **analysis_raw)
    state = StateConfig(**state_raw)

    return AppConfig(
        ollama=ollama,
        video=video,
        analysis=analysis,
        state=state,
    )

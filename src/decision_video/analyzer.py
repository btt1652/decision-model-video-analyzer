"""Core analysis logic: extract frames, query Ollama, collect timestamps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from decision_video.config import AppConfig
from decision_video.ollama import (
    OllamaError,
    build_request,
    get_noul_probability,
    send_request,
)
from decision_video.video import extract_frames, frame_to_base64

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds as hh:mm:ss."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class AnalysisResult:
    """A single matched timestamp from the analysis."""

    def __init__(
        self,
        timestamp: float,
        frame_number: int,
        probability: float,
        question_key: str,
    ):
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.probability = probability
        self.question_key = question_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "probability": self.probability,
            "question_key": self.question_key,
        }

    def to_text_line(self) -> str:
        return f"{format_timestamp(self.timestamp)}  (frame {self.frame_number}, prob={self.probability:.4f})"


def analyze_video(
    config: AppConfig,
    video_path: str,
    output_path: str | None = None,
    output_format: str = "text",
) -> list[AnalysisResult]:
    """Run the full analysis pipeline on a video.

    1. Extract frames at the configured interval
    2. For each frame, send the frame image + state context to Ollama
    3. Collect frames where the noul probability exceeds the threshold

    If output_path is provided, matches are written to the file
    incrementally as they are found.
    """
    if not video_path:
        raise ValueError("No video path provided")

    if not config.analysis.questions:
        raise ValueError("No analysis questions configured")

    # Open output file for live range writing if requested
    out_file: Any = None
    in_range = False
    if output_path:
        out_path = Path(output_path)
        out_file = out_path.open("w", encoding="utf-8")

    # Build the base state context
    state_context = config.state.context or ""

    results: list[AnalysisResult] = []
    match_count = 0

    for frame in extract_frames(
        video_path=video_path,
        interval_seconds=config.video.frame_interval,
        output_dir=config.video.frames_dir or None,
    ):
        logger.debug(
            "Analyzing frame %d at %s",
            frame["frame_number"],
            format_timestamp(frame["timestamp"]),
        )

        # Build per-frame state: combine context with timestamp info
        frame_state = state_context
        if state_context:
            frame_state += "\n"
        frame_state += f"Frame timestamp: {frame['timestamp']}s, frame number: {frame['frame_number']}"

        # Prepare the request with the frame image as base64
        request = build_request(
            model=config.ollama.model,
            state=frame_state,
            questions=config.analysis.questions,
        )

        # Add the image if available
        if frame.get("image_bytes"):
            request.images = [frame_to_base64(frame["image_bytes"])]

        try:
            response = send_request(
                request,
                api_url=config.ollama.api_url,
                timeout=config.ollama.timeout,
            )
        except OllamaError as e:
            logger.warning("Ollama request failed for frame %d: %s", frame["frame_number"], e.message)
            continue

        frame_matched = False

        # Check each noul question for matches
        for q in config.analysis.questions:
            key = q["key"]
            try:
                prob = get_noul_probability(response, key)
            except (KeyError, TypeError) as e:
                logger.warning("Skipping question '%s' for frame %d: %s", key, frame["frame_number"], e)
                continue

            if prob >= config.analysis.threshold:
                result = AnalysisResult(
                    timestamp=frame["timestamp"],
                    frame_number=frame["frame_number"],
                    probability=prob,
                    question_key=key,
                )
                results.append(result)
                match_count += 1
                frame_matched = True
                logger.debug(
                    "  MATCH at %s (probability: %.4f)",
                    format_timestamp(result.timestamp),
                    result.probability,
                )

        # Live write: start or end of range
        if out_file is not None:
            if frame_matched and not in_range:
                out_file.write(f"{format_timestamp(frame['timestamp'])}  true\n")
                out_file.flush()
                in_range = True
            elif not frame_matched and in_range:
                out_file.write(f"{format_timestamp(frame['timestamp'])}  false\n")
                out_file.flush()
                in_range = False

    # Close any open range at end of video
    if out_file is not None and in_range:
        out_file.write(f"{format_timestamp(frame['timestamp'])}  false\n")
        out_file.close()

    logger.info("Analysis complete: %d matches found", len(results))
    return results

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
from decision_video.video import extract_frame_batches, frame_to_base64

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
    start_time: float | None = None,
    end_time: float | None = None,
    batch_size: int = 2,
) -> list[AnalysisResult]:
    """Run the full analysis pipeline on a video.

    1. Extract frames in batches at the configured interval
    2. For each batch, send all frames + state context to Ollama
    3. Collect timestamps where the noul probability exceeds the threshold

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

    for batch in extract_frame_batches(
        video_path=video_path,
        interval_seconds=config.video.frame_interval,
        batch_size=batch_size,
        output_dir=config.video.frames_dir or None,
        start_time=start_time,
        end_time=end_time,
    ):
        logger.debug(
            "Analyzing batch of %d frames at %s",
            len(batch),
            format_timestamp(batch[0]["timestamp"]),
        )

        # Build batch state: include all frame timestamps
        batch_state = state_context
        if state_context:
            batch_state += "\n"
        timestamps = ", ".join(
            f"{f['timestamp']}s (frame {f['frame_number']})" for f in batch
        )
        batch_state += f"Frames: {timestamps}"

        # Prepare the request with all batch frames as images
        request = build_request(
            model=config.ollama.model,
            state=batch_state,
            questions=config.analysis.questions,
        )

        # Add all images from the batch
        batch_images = [
            frame_to_base64(f["image_bytes"])
            for f in batch
            if f.get("image_bytes")
        ]
        if batch_images:
            request.images = batch_images

        try:
            response = send_request(
                request,
                api_url=config.ollama.api_url,
                timeout=config.ollama.timeout,
            )
        except OllamaError as e:
            logger.warning(
                "Ollama request failed for batch starting at %s: %s",
                format_timestamp(batch[0]["timestamp"]),
                e.message,
            )
            continue

        # Process each question in the batch
        for q in config.analysis.questions:
            key = q["key"]
            try:
                prob = get_noul_probability(response, key)
            except (KeyError, TypeError) as e:
                logger.warning(
                    "Skipping question '%s' for batch starting at %s: %s",
                    key,
                    format_timestamp(batch[0]["timestamp"]),
                    e,
                )
                continue

            if prob >= config.analysis.threshold:
                # Record match for the first frame in the batch
                first_frame = batch[0]
                result = AnalysisResult(
                    timestamp=first_frame["timestamp"],
                    frame_number=first_frame["frame_number"],
                    probability=prob,
                    question_key=key,
                )
                results.append(result)
                match_count += 1
                logger.debug(
                    "  BATCH MATCH at %s (probability: %.4f)",
                    format_timestamp(result.timestamp),
                    result.probability,
                )

                # Live write: start or end of range
                if out_file is not None:
                    if not in_range:
                        out_file.write(f"{format_timestamp(result.timestamp)}  true\n")
                        out_file.flush()
                        in_range = True
            else:
                # No match in this batch - close any open range
                if out_file is not None and in_range:
                    last_ts = batch[-1]["timestamp"] + config.video.frame_interval
                    out_file.write(f"{format_timestamp(last_ts)}  false\n")
                    out_file.flush()
                    in_range = False

    # Close any open range at end of video
    if out_file is not None and in_range and batch:
        last_ts = batch[-1]["timestamp"] + config.video.frame_interval
        out_file.write(f"{format_timestamp(last_ts)}  false\n")
        out_file.close()

    logger.info("Analysis complete: %d matches found", len(results))
    return results

"""Video frame extraction utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    interval_seconds: float = 1.0,
    output_dir: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield frames from a video at a given interval, optionally within a time range.

    Each yielded dict has keys:
      - timestamp: float (seconds)
      - frame_number: int
      - image_path: str | None (if output_dir is set)
      - image_bytes: bytes | None (raw JPEG bytes for API submission)

    Frames are yielded one at a time instead of being collected into a list,
    so memory usage stays low for large videos and downstream processing can
    start immediately.

    If start_time is set, frames before that timestamp are skipped.
    If end_time is set, frames after that timestamp are not yielded.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    logger.info(
        "Video: %s | FPS: %.2f | Frames: %d | Duration: %.1fs",
        video_path,
        fps,
        total_frames,
        duration,
    )

    if start_time is not None:
        logger.info("Start time: %.1fs", start_time)
    if end_time is not None:
        logger.info("End time: %.1fs", end_time)

    interval_frames = max(1, int(round(fps * interval_seconds)))
    frame_idx = 0
    count = 0

    output_path = Path(output_dir) if output_dir else None
    if output_path and not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0

                # Skip frames before start_time
                if start_time is not None and timestamp < start_time:
                    frame_idx += 1
                    continue

                # Stop yielding frames after end_time
                if end_time is not None and timestamp > end_time:
                    cap.release()
                    logger.info("Extracted %d frames from video", count)
                    return

                entry: dict[str, Any] = {
                    "timestamp": round(timestamp, 3),
                    "frame_number": frame_idx,
                    "image_path": None,
                    "image_bytes": None,
                }

                if output_path is not None:
                    img_path = output_path / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    entry["image_path"] = str(img_path)

                # Encode frame as JPEG bytes for Ollama API
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    entry["image_bytes"] = buffer.tobytes()

                count += 1
                yield entry

            frame_idx += 1
    finally:
        cap.release()

    logger.info("Extracted %d frames from video", count)


def frame_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes as a base64 data URL for the Ollama API."""
    import base64

    return base64.b64encode(image_bytes).decode("ascii")

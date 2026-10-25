"""CLI entry point for decision-video-analyzer."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Literal, Optional

import typer

from decision_video import __version__
from decision_video.config import load_config
from decision_video.analyzer import analyze_video, AnalysisResult

app = typer.Typer(
    name="dv-analyze",
    help="Find video frame timestamps using a local Ollama decision model",
    add_completion=False,
)


def set_verbosity(verbose: bool):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress urllib3 HTTP debug logs (connection pooling, request details)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
):
    set_verbosity(verbose)
    if version:
        typer.echo(f"dv-analyze {__version__}")
        raise typer.Exit()


@app.command()
def analyze(
    video_path: str = typer.Argument(..., help="Path to the video file to analyze"),
    config: str = typer.Option(
        "config.toml",
        "--config",
        "-c",
        help="Path to the TOML configuration file",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Override output file path (overrides auto-derived name)",
    ),
    output_format: Literal["json", "text"] = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format (only used when writing to a file)",
    ),
    start: Optional[float] = typer.Option(
        None,
        "--start",
        help="Start timestamp in seconds (skip frames before this)",
    ),
    end: Optional[float] = typer.Option(
        None,
        "--end",
        help="End timestamp in seconds (stop analyzing after this)",
    ),
    batch_size: int = typer.Option(
        2,
        "--batch-size",
        help="Number of frames per batch (1-4, default 2). Higher = fewer API calls.",
    ),
):
    """Analyze a video and find frame timestamps matching the decision criteria."""
    cfg = load_config(config)

    # Determine output path: CLI flag > auto-derive from video filename
    if output:
        cfg.analysis.output = output
    else:
        video_stem = Path(video_path).stem
        cfg.analysis.output = f"{video_stem}_ts.txt"

    typer.echo(f"Using model: {cfg.ollama.model}")
    typer.echo(f"Video: {video_path}")
    typer.echo(f"Frame interval: {cfg.video.frame_interval}s")
    typer.echo(f"Batch size: {batch_size} frames")
    typer.echo(f"Threshold: {cfg.analysis.threshold}")
    typer.echo(f"Questions: {[q['key'] for q in cfg.analysis.questions]}")
    typer.echo(f"Output: {cfg.analysis.output}")
    typer.echo("")

    results = analyze_video(
        cfg,
        video_path=video_path,
        output_path=cfg.analysis.output,
        output_format=output_format,
        start_time=start,
        end_time=end,
        batch_size=batch_size,
    )

    typer.echo(f"Results written to {cfg.analysis.output}")


@app.command()
def extract_frames(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    config: str = typer.Option(
        "config.toml",
        "--config",
        "-c",
        help="Path to the TOML configuration file",
    ),
    output_dir: str = typer.Option(
        "frames",
        "--output-dir",
        "-o",
        help="Directory to save extracted frames",
    ),
):
    """Extract frames from a video at the configured interval."""
    cfg = load_config(config)

    from decision_video.video import extract_frames

    frames = extract_frames(
        video_path=video_path,
        interval_seconds=cfg.video.frame_interval,
        output_dir=output_dir,
    )

    typer.echo(f"Extracted {len(frames)} frames to {output_dir}")


@app.command()
def query(
    image: str = typer.Argument(..., help="Path to the image file to analyze"),
    config: str = typer.Option(
        "config.toml",
        "--config",
        "-c",
        help="Path to the TOML configuration file",
    ),
    state: str = typer.Option(
        "",
        "--state",
        "-s",
        help="State/context description for the decision model",
    ),
):
    """Query the Ollama decision model with a single image."""
    cfg = load_config(config)

    if not cfg.analysis.questions:
        typer.echo("Error: No analysis questions configured", err=True)
        raise typer.Exit(code=1)

    from decision_video.ollama import (
        build_request,
        send_request,
        get_noul_probability,
    )
    from decision_video.video import frame_to_base64
    from PIL import Image

    # Read and encode the image (convert RGBA to RGB for JPEG)
    img = Image.open(image)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    image_bytes = buffer.getvalue()

    # Build request
    request = build_request(
        model=cfg.ollama.model,
        state=state or cfg.state.context or "Analyze this image",
        questions=cfg.analysis.questions,
    )
    request.images = [frame_to_base64(image_bytes)]

    # Send request
    response = send_request(
        request,
        api_url=cfg.ollama.api_url,
        timeout=cfg.ollama.timeout,
    )

    # Print results
    typer.echo(f"Model: {response.model}")
    for q in cfg.analysis.questions:
        key = q["key"]
        try:
            prob = get_noul_probability(response, key)
        except (KeyError, TypeError):
            typer.echo(f"  {key}: (no noul answer returned)")
            continue

        qtype = next(
            (qq.get("type", "noul") for qq in cfg.analysis.questions if qq.get("key") == key),
            "noul",
        )
        if qtype == "noul":
            typer.echo(f"  {key}: {prob:.4f} (noul probability of true)")
        else:
            typer.echo(f"  {key}: {prob:.4f}")


@app.command()
def config_cmd(
    config: str = typer.Option(
        "config.toml",
        "--config",
        "-c",
        help="Path to the TOML configuration file",
    ),
    show: bool = typer.Option(False, "--show", "-s", help="Show the loaded configuration"),
):
    """Validate and optionally display the configuration file."""
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Config error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Configuration is valid.")

    if show:
        typer.echo(f"  Model: {cfg.ollama.model}")
        typer.echo(f"  API URL: {cfg.ollama.api_url}")
        typer.echo(f"  Video: {cfg.video.input_path}")
        typer.echo(f"  Frame interval: {cfg.video.frame_interval}s")
        typer.echo(f"  Threshold: {cfg.analysis.threshold}")
        typer.echo(f"  Questions: {[q['key'] for q in cfg.analysis.questions]}")
        typer.echo(f"  State context: {cfg.state.context[:80]}...")


if __name__ == "__main__":
    app()

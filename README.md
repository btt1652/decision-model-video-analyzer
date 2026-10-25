# decision-video-analyzer

CLI app for finding video frame timestamps using a local Ollama decision model.

Sends frame images and context to the [Ollama Decision API](https://docs.ollama.com/capabilities/decision) using `noul`-type questions, and collects timestamps where the model returns a probability above the configured threshold.

## Requirements

- Python 3.10+
- Ollama v0.35.0+ with a decision model pulled locally (e.g. `d3:9b`)
- A video file to analyze

## Installation

```bash
pip install -e .
```

## Configuration

Copy the default config and edit it:

```bash
cp config/default.toml config.toml
```

Edit `config.toml` to set:

- **ollama.model** — the decision model name (e.g. `d3:9b`)
- **ollama.api_url** — Ollama Decision API endpoint
- **video.input_path** — path to your video file
- **video.frame_interval** — seconds between frames to analyze
- **analysis.threshold** — minimum noul probability to count as a match (0.0–1.0)
- **analysis.questions** — the noul/choice/score questions to ask per frame
- **state.context** — additional context about what to look for

## Usage

### Full analysis

```bash
dv-analyze analyze --config config.toml
```

Extract frames, send each to Ollama, and output matching timestamps.

### Extract frames only

```bash
dv-analyze extract-frames --config config.toml --output-dir frames/
```

### Query a single image

```bash
dv-analyze query screenshot.png --config config.toml --state "Is this the moment we are looking for?"
```

### Validate config

```bash
dv-analyze config --config config.toml --show
```

## Output

Results are printed to stdout (JSON or text format) or written to a file if `analysis.output` is set in the config.

JSON output example:

```json
[
  {
    "timestamp": 12.500,
    "frame_number": 375,
    "probability": 0.9421,
    "question_key": "moment_found"
  }
]
```

## Ollama Decision API

This tool uses the [Ollama Decision API](https://docs.ollama.com/capabilities/decision) endpoint `POST /v1/systemone`. The `noul` question type returns a probability of `true` (0 to 1), which is compared against the threshold to determine matches.
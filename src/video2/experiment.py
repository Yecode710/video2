"""Execute and persist a zero-shot experiment, including partial failures."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from .backend import ModelConfig, QwenBackend
from .data import load_dataset
from .evaluation import evaluate
from .prompts import make_prompt
from .video import sample_video


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_experiment(dataset: Path, output: Path, model_config: ModelConfig, *,
                   language: str = "en", max_words: int = 100,
                   num_frames: int = 32, max_frame_size: int = 448,
                   backend_factory=QwenBackend, sampler=sample_video) -> dict:
    examples = load_dataset(dataset)
    prompt = make_prompt(language, max_words)
    # Exclusive creation protects previous results and partial failed runs.
    output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("numpy", "opencv-python-headless", "Pillow", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        code_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).parent,
            text=True, capture_output=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        code_revision, code_dirty = None, None
    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "model": asdict(model_config), "prompt": prompt, "language": language,
        "max_words_or_zh_characters": max_words,
        "sampling": {"num_frames": num_frames, "max_frame_size": max_frame_size,
                     "strategy": "uniform full-video frames including endpoints; no audio"},
        "generation": {"do_sample": False, "num_beams": 1},
        "python_version": platform.python_version(), "dependencies": versions,
        "code_revision": code_revision, "code_dirty": code_dirty,
        "state": "loading_model",
    }
    write_json(output / "config.json", config)
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as stream:
        for example in examples:
            stat = example.video.stat()
            stream.write(json.dumps({
                "id": example.id, "video": str(example.video), "references": example.references,
                "video_size_bytes": stat.st_size, "video_mtime_ns": stat.st_mtime_ns,
            }, ensure_ascii=False) + "\n")
    try:
        backend = backend_factory(model_config)
    except Exception as error:
        config.update(state="model_load_failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "config.json", config)
        raise
    config.update(state="running", runtime=backend.metadata)
    write_json(output / "config.json", config)
    predictions = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, example in enumerate(examples, 1):
            start = time.perf_counter()
            record = {"id": example.id}
            try:
                video = sampler(example.video, num_frames=num_frames, max_frame_size=max_frame_size)
                summary = backend.summarize(video, prompt)
                if not isinstance(summary, str) or not summary.strip():
                    raise RuntimeError("Model returned an empty or invalid summary")
                record.update(
                    status="ok", summary=summary.strip(),
                    video={"duration_seconds": video.duration_seconds,
                           "source_fps": video.source_fps, "effective_fps": video.effective_fps,
                           "sampled_timestamps": video.timestamps, "num_frames": len(video.frames)},
                )
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                # Failed generation can leave temporary tensors in the CUDA cache.
                if hasattr(backend, "torch") and backend.torch.cuda.is_available():
                    backend.torch.cuda.empty_cache()
            record["elapsed_seconds"] = time.perf_counter() - start
            predictions.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(examples)}] {example.id}: {record['status']}", flush=True)
    metrics = evaluate(examples, predictions, language)
    write_json(output / "metrics.json", metrics)
    config.update(state="completed_with_errors" if metrics["counts"]["failed"] else "completed",
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "config.json", config)
    return metrics

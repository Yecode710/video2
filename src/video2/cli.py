import argparse
import json
from pathlib import Path
import sys

from .backend import ModelConfig
from .data import load_dataset, read_jsonl
from .evaluation import evaluate
from .experiment import run_experiment
from .video import sample_video


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Zero-shot Qwen2.5-VL video summarization")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate summaries and evaluate against references")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="New directory; never overwrites a run")
    run.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    run.add_argument("--revision", default="main", help="HF revision; pin a commit for reproducibility")
    run.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    run.add_argument("--language", choices=["en", "zh"], default="en")
    run.add_argument("--max-words", type=positive, default=100, help="Prompt word budget (zh: characters); not a hard cutoff")
    run.add_argument("--max-new-tokens", type=positive, default=256)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--min-pixels", type=positive, default=128 * 28 * 28)
    run.add_argument("--max-pixels", type=positive, default=256 * 28 * 28)
    validate = commands.add_parser("validate", help="Check dataset and actually decode sampled video frames; no model")
    validate.add_argument("--dataset", type=Path, required=True)
    for subcommand in (run, validate):
        subcommand.add_argument("--num-frames", type=positive, default=32)
        subcommand.add_argument("--max-frame-size", type=positive, default=448)
    score = commands.add_parser("evaluate", help="Score saved predictions without model/video dependencies")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--language", choices=["en", "zh"], default="en")
    score.add_argument("--output", type=Path, help="Optional new metrics JSON file")
    return root


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command in ("run", "validate"):
            if args.num_frames < 2 or args.num_frames % 2:
                raise ValueError("--num-frames must be even and >=2 (Qwen temporal patches)")
            if args.max_frame_size < 28:
                raise ValueError("--max-frame-size must be >=28")
        if args.command == "run":
            if args.min_pixels > args.max_pixels:
                raise ValueError("--min-pixels must be <= --max-pixels")
            config = ModelConfig(**{key: getattr(args, key) for key in (
                "model", "revision", "device", "dtype", "max_new_tokens", "seed", "min_pixels", "max_pixels"
            )})
            metrics = run_experiment(
                args.dataset, args.output, config, language=args.language,
                max_words=args.max_words, num_frames=args.num_frames, max_frame_size=args.max_frame_size,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 1 if metrics["counts"]["failed"] else 0
        if args.command == "validate":
            examples = load_dataset(args.dataset)
            for example in examples:
                video = sample_video(example.video, args.num_frames, args.max_frame_size)
                print(json.dumps({"id": example.id, "duration_seconds": video.duration_seconds,
                                  "num_frames": len(video.frames), "timestamps": video.timestamps,
                                  "references": len(example.references)}, ensure_ascii=False))
            return 0
        metrics = evaluate(load_dataset(args.dataset, check_videos=False),
                           read_jsonl(args.predictions, allow_empty=True), args.language)
        if args.output:
            # Exclusive creation protects existing reports.
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(metrics, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        counts = metrics["counts"]
        return 1 if counts["missing"] or counts["failed"] or not counts["scored"] else 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

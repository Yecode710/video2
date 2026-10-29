"""Strict JSONL input; references are kept separate from model input."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    video: Path
    references: list[str]


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: each record must be an object")
            rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"{path}: no records")
    return rows


def load_dataset(path: Path, *, check_videos: bool = True) -> list[Example]:
    examples = []
    seen = set()
    for number, row in enumerate(read_jsonl(path), 1):
        label = f"{path}, record {number}"
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label}: id must be a nonempty string")
        if identifier in seen:
            raise ValueError(f"{label}: duplicate id {identifier!r}")
        seen.add(identifier)
        video = row.get("video")
        if not isinstance(video, str) or not video.strip():
            raise ValueError(f"{label}: video must be a nonempty local path")
        video_path = Path(video).expanduser()
        if not video_path.is_absolute():
            video_path = path.resolve().parent / video_path
        video_path = video_path.resolve()
        if check_videos and not video_path.is_file():
            raise ValueError(f"{label}: video not found: {video_path}")
        refs = row.get("references", [])
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or not ref.strip() for ref in refs
        ):
            raise ValueError(f"{label}: references must be a list of nonempty strings")
        examples.append(Example(identifier, video_path, refs))
    return examples

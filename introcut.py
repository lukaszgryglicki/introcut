#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find a majority's shared video opening and remove it without re-encoding."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import glob
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

__version__ = "0.1.0"
WIDTH, HEIGHT = 16, 9
PIXELS = WIDTH * HEIGHT
FRAME_BYTES = PIXELS * 3
CONFIRM_FRAMES = 3


class IntrocutError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Frame:
    rgb: bytes
    luma: bytes
    mean: float
    variance: float

    @classmethod
    def from_rgb(cls, rgb: bytes) -> Frame:
        if len(rgb) != FRAME_BYTES:
            raise IntrocutError("Incomplete thumbnail frame")
        luma = bytes(
            (77 * rgb[i] + 150 * rgb[i + 1] + 29 * rgb[i + 2]) >> 8
            for i in range(0, FRAME_BYTES, 3)
        )
        mean = sum(luma) / PIXELS
        variance = sum(value * value for value in luma) / PIXELS - mean * mean
        return cls(rgb, luma, mean, variance)


@dataclass(frozen=True)
class Chapter:
    start: float
    end: float
    tags: dict[str, str]


@dataclass(frozen=True)
class Movie:
    path: Path
    video_index: int
    video_start: float
    frames: tuple[Frame, ...]
    snapshot: tuple[int, int, int, int]
    chapters: tuple[Chapter, ...] = ()
    chapter_streams: tuple[int, ...] = ()


@dataclass(frozen=True)
class Detection:
    members: tuple[int, ...] = ()
    seconds: float | None = None
    boundary_min: float | None = None
    boundary_max: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Keyframe:
    pts: float
    dts: float | None
    data_hash: str


@dataclass
class FilePlan:
    path: str
    status: str = "unmatched"
    matched: bool = False
    cut_seconds: float | None = None
    keyframe_pts_seconds: float | None = None
    estimated_intro_remaining_seconds: float | None = None
    estimated_content_removed_seconds: float | None = None
    output: str | None = None
    reason: str = ""


def snapshot(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def run_tool(command: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, check=False
        )
    except OSError as exc:
        raise IntrocutError(f"Cannot run {command[0]}: {exc}") from exc
    error = result.stderr.decode("utf-8", errors="replace").strip()
    if result.returncode or error:
        detail = error or f"exit status {result.returncode}"
        raise IntrocutError(f"{Path(command[0]).name}: {detail}")
    return result.stdout


def probe_json(arguments: Sequence[str], path: Path) -> dict:
    raw = run_tool(["ffprobe", "-v", "error", *arguments, "-of", "json", str(path)])
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise IntrocutError(f"Invalid ffprobe JSON for {path}") from exc
    if not isinstance(result, dict):
        raise IntrocutError(f"Unexpected ffprobe response for {path}")
    return result


def finite_number(value: object, label: str) -> float:
    try:
        number = float(str(value))
    except ValueError as exc:
        raise IntrocutError(f"Missing or invalid {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise IntrocutError(f"Non-finite {label}: {value!r}")
    return number


def analyze(path: Path, scan_seconds: float, sample_rate: float) -> Movie:
    before = snapshot(path)
    metadata = probe_json(
        [
            "-show_streams", "-show_chapters",
            "-show_entries",
            "stream=index,codec_type,codec_tag_string,start_time:"
            "stream_disposition=attached_pic:chapter=start_time,end_time:chapter_tags",
        ],
        path,
    )
    videos = [
        stream
        for stream in metadata.get("streams", [])
        if stream.get("codec_type") == "video"
        and not stream.get("disposition", {}).get("attached_pic", 0)
    ]
    if len(videos) != 1:
        raise IntrocutError("Expected exactly one video track (cover art is ignored)")
    video = videos[0]
    start = finite_number(video.get("start_time"), "video start timestamp")
    index = int(video["index"])
    chapters = tuple(
        Chapter(
            finite_number(chapter.get("start_time"), "chapter start"),
            finite_number(chapter.get("end_time"), "chapter end"),
            chapter.get("tags", {}),
        )
        for chapter in metadata.get("chapters", [])
    )
    if any(chapter.end <= chapter.start for chapter in chapters):
        raise IntrocutError("Input has an invalid chapter time range")
    chapter_streams = tuple(
        int(stream["index"]) for stream in metadata.get("streams", [])
        if chapters and stream.get("codec_type") == "data"
        and stream.get("codec_tag_string") == "text"
    )
    filters = (
        f"setpts=PTS-STARTPTS,fps=fps={sample_rate}:round=up,"
        f"scale={WIDTH}:{HEIGHT}:flags=area,format=rgb24"
    )
    raw = run_tool(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "1", "-filter_threads", "1",
            "-t", f"{scan_seconds:.9f}", "-i", str(path),
            "-map", f"0:{index}", "-an", "-sn", "-dn", "-vf", filters,
            "-frames:v", str(math.ceil(scan_seconds * sample_rate)),
            "-threads:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
    )
    if not raw or len(raw) % FRAME_BYTES:
        raise IntrocutError("Could not decode complete video thumbnails")
    frames = tuple(
        Frame.from_rgb(raw[offset:offset + FRAME_BYTES])
        for offset in range(0, len(raw), FRAME_BYTES)
    )
    if snapshot(path) != before:
        raise IntrocutError("Input changed while it was being analyzed")
    return Movie(path, index, start, frames, before, chapters, chapter_streams)


def similar(left: Frame, right: Frame, threshold: float) -> bool:
    if left is right:
        return True
    color_error = sum(abs(a - b) for a, b in zip(left.rgb, right.rgb))
    if color_error > threshold * 255 * FRAME_BYTES:
        return False
    covariance = (
        sum(a * b for a, b in zip(left.luma, right.luma)) / PIXELS
        - left.mean * right.mean
    )
    stabilizer = (0.03 * 255) ** 2
    structure = (2 * covariance + stabilizer) / (
        left.variance + right.variance + stabilizer
    )
    return (1 - structure) / 2 <= threshold


def majority_frame(frames: Sequence[Frame], threshold: float) -> Frame:
    candidate = frames[0]
    votes = 0
    for frame in frames:
        if votes == 0:
            candidate, votes = frame, 1
        elif similar(candidate, frame, threshold):
            votes += 1
        else:
            votes -= 1
    return candidate


def detect(
    movies: Sequence[Movie | None],
    required: int,
    sample_rate: float,
    min_intro: float,
    threshold: float,
) -> Detection:
    active = {
        index: movie for index, movie in enumerate(movies) if movie is not None
    }
    if len(active) < required:
        return Detection(reason="Not enough readable videos to reach the required majority")
    misses = {index: 0 for index in active}
    runs = dict(misses)
    failed_start = 0
    failed_count = 0
    informative = 0
    steps = max(len(movie.frames) for movie in active.values())
    for step in range(steps):
        available = {
            index: movie.frames[step]
            for index, movie in active.items()
            if step < len(movie.frames)
        }
        if len(available) < required:
            return Detection(
                reason="Common opening reaches the scan limit or a file's end; "
                "increase --scan-seconds or provide more varied, longer videos"
            )
        candidate = majority_frame(list(available.values()), threshold)
        matching = {
            index for index, frame in available.items()
            if similar(candidate, frame, threshold)
        }
        new_misses = {
            index: misses[index] + (index not in matching) for index in active
        }
        new_runs = {
            index: 0 if index in matching else runs[index] + 1 for index in active
        }
        budget = max(1, int((step + 1) * 0.05))
        survivors = {
            index: movie for index, movie in active.items()
            if index in available and new_misses[index] <= budget
            and new_runs[index] < 2
        }
        if len(matching) >= required and len(survivors) >= required:
            active, misses, runs = survivors, new_misses, new_runs
            failed_count = 0
            informative += candidate.variance >= 64
            continue
        if failed_count == 0:
            failed_start = step
        failed_count += 1
        if failed_count < CONFIRM_FRAMES:
            continue
        lower = max(0.0, (failed_start - 1) / sample_rate)
        if lower < min_intro:
            return Detection(reason="No sufficiently long common opening found")
        if informative < math.ceil(min_intro * sample_rate / 2):
            return Detection(
                reason="Only blank/solid or insufficiently distinctive opening frames matched"
            )
        return Detection(
            members=tuple(sorted(active)),
            seconds=failed_start / sample_rate,
            boundary_min=lower,
            boundary_max=(failed_start + 1) / sample_rate,
            reason="Shared visual opening found",
        )
    return Detection(
        reason="No ending observed before the scan limit or end of the videos; "
        "increase --scan-seconds or provide more varied, longer videos"
    )


def video_packets(path: Path, index: int, seconds: float) -> list[dict]:
    data = probe_json(
        [
            "-select_streams", str(index),
            "-read_intervals", f"%+{seconds:.9f}",
            "-show_packets", "-show_data_hash", "sha256",
            "-show_entries", "packet=pts_time,dts_time,flags,data_hash",
        ],
        path,
    )
    return data.get("packets", [])


def find_keyframe(
    movie: Movie, detection: Detection, policy: str, lookahead: float
) -> Keyframe | None:
    assert detection.boundary_min is not None and detection.boundary_max is not None
    limit = detection.boundary_max + (lookahead if policy == "next" else 1)
    packets = video_packets(movie.path, movie.video_index, limit)
    keys = []
    for packet in packets:
        if "K" not in packet.get("flags", "") or "D" in packet.get("flags", ""):
            continue
        pts = finite_number(packet.get("pts_time"), "keyframe PTS")
        raw_dts = packet.get("dts_time")
        dts = None if raw_dts in (None, "N/A") else finite_number(raw_dts, "keyframe DTS")
        data_hash = packet.get("data_hash")
        if not data_hash:
            raise IntrocutError("ffprobe did not return a keyframe payload hash")
        keys.append(Keyframe(pts, dts, data_hash))
    if policy == "previous":
        eligible = [
            key for key in keys
            if key.pts - movie.video_start <= detection.boundary_min + 0.000001
        ]
        chosen = max(eligible, key=lambda key: key.pts, default=None)
    else:
        eligible = [
            key for key in keys
            if key.pts - movie.video_start >= detection.boundary_max - 0.000001
            and key.pts - movie.video_start <= limit + 0.000001
        ]
        chosen = min(eligible, key=lambda key: key.pts, default=None)
    if chosen is None or chosen.pts - movie.video_start <= 0.000001:
        return None
    if chosen.dts is None:
        raise IntrocutError("Selected keyframe has no DTS; cannot safely plan a copy cut")
    if snapshot(movie.path) != movie.snapshot:
        raise IntrocutError("Input changed after analysis")
    return chosen


def chapter_metadata(movie: Movie) -> str:
    def escape(value: str) -> str:
        return "".join("\\" + char if char in "\\=;#\n\r" else char for char in value)

    lines = [";FFMETADATA1"]
    for chapter in movie.chapters:
        lines.extend([
            "[CHAPTER]", "TIMEBASE=1/1000000",
            f"START={round((chapter.start + movie.video_start) * 1000000)}",
            f"END={round((chapter.end + movie.video_start) * 1000000)}",
        ])
        lines.extend(f"{escape(key)}={escape(value)}" for key, value in chapter.tags.items())
    return "\n".join(lines) + "\n"


def copy_movie(
    movie: Movie, keyframe: Keyframe, destination: Path, overwrite: bool = False
) -> None:
    if snapshot(movie.path) != movie.snapshot:
        raise IntrocutError("Input changed after the cut was planned")
    if keyframe.dts is None:
        raise IntrocutError("Cannot cut a keyframe without a decode timestamp")
    with tempfile.TemporaryDirectory(prefix=".introcut-", dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        chapter_input = []
        chapter_map = "-1"
        if movie.chapters:
            chapter_file = Path(folder) / "chapters.ffmetadata"
            chapter_file.write_text(chapter_metadata(movie), encoding="utf-8")
            chapter_input = ["-f", "ffmetadata", "-i", str(chapter_file)]
            chapter_map = "1"
        stream_maps = ["-map", "0"]
        for index in movie.chapter_streams:
            stream_maps.extend(["-map", f"-0:{index}"])
        # Copy seeking must use DTS at the output, not PTS (B-frames/open GOPs).
        run_tool(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-copyts", "-seek_timestamp", "1", "-ss", f"{keyframe.pts:.9f}",
                "-i", str(movie.path), *chapter_input,
                "-ss", f"{keyframe.dts - 0.000001:.9f}",
                *stream_maps, "-map_metadata", "0", "-map_chapters", chapter_map,
                "-c", "copy", "-avoid_negative_ts", "make_zero", str(temporary),
            ]
        )
        output_index = movie.video_index - sum(
            index < movie.video_index for index in movie.chapter_streams
        )
        packets = video_packets(temporary, output_index, 2)
        if (
            not packets or "K" not in packets[0].get("flags", "")
            or "D" in packets[0].get("flags", "")
            or packets[0].get("data_hash") != keyframe.data_hash
        ):
            raise IntrocutError(
                "Remux did not start with the planned keyframe's unchanged payload; "
                "output was not published"
            )
        if snapshot(movie.path) != movie.snapshot:
            raise IntrocutError("Input changed while copying; output was not published")
        if overwrite:
            if destination != movie.path:
                raise IntrocutError("In-place output must be the analyzed input path")
            shutil.copystat(movie.path, temporary)
            os.replace(temporary, destination)
        else:
            # A same-filesystem hard link publishes atomically without replacing anything.
            os.link(temporary, destination)


def collect_inputs(paths: Sequence[str], files_from: str | None, null: bool) -> list[Path]:
    candidates = []
    for value in paths:
        path = Path(value).expanduser()
        if path.exists() or not glob.has_magic(str(path)):
            candidates.append(path)
            continue
        matches = sorted(glob.iglob(str(path)))
        if not matches:
            raise IntrocutError(f"No files match pattern: {value}")
        candidates.extend(Path(match) for match in matches)
    if files_from is not None:
        if files_from == "-":
            raw, base = sys.stdin.buffer.read(), Path.cwd()
        else:
            listing = Path(files_from).resolve(strict=True)
            raw, base = listing.read_bytes(), listing.parent
        entries = raw.split(b"\0") if null else raw.splitlines()
        candidates.extend(base / os.fsdecode(entry) for entry in entries if entry)
    result = []
    seen = set()
    for candidate in candidates:
        path = candidate.resolve(strict=True)
        if not path.is_file():
            raise IntrocutError(f"Not a regular file: {path}")
        info = path.stat()
        identity = info.st_dev, info.st_ino
        if identity not in seen:
            seen.add(identity)
            result.append(path)
    if len(result) < 2:
        raise IntrocutError("Provide at least two distinct video files")
    return result


def preflight_outputs(plans: Sequence[FilePlan], inputs: Sequence[Path], folder: Path) -> None:
    if folder.exists() and not folder.is_dir():
        raise IntrocutError(f"Output directory is not a directory: {folder}")
    destinations = set()
    sources = set(inputs)
    for plan in plans:
        if plan.status != "would_cut":
            continue
        destination = folder / Path(plan.path).name
        if not destination.suffix:
            raise IntrocutError(f"Output needs a container filename extension: {destination}")
        if destination in sources or os.path.lexists(destination):
            raise IntrocutError(f"Refusing to overwrite an input or existing output: {destination}")
        if destination in destinations:
            raise IntrocutError(f"Inputs have colliding output filenames: {destination.name}")
        destinations.add(destination)
        plan.output = str(destination)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a majority's common video intro and cut at keyframes, without re-encoding.",
        epilog="Default: dry run. Use --output-dir or --overwrite to apply; --dry-run overrides both.",
    )
    parser.add_argument("videos", nargs="*", help="input paths or quoted glob patterns")
    parser.add_argument("--version", action="version", version=f"introcut {__version__}")
    parser.add_argument("--files-from", metavar="FILE", help="read paths from a list; - means stdin")
    parser.add_argument("--null", action="store_true", help="the file list is NUL-delimited")
    parser.add_argument("--dry-run", action="store_true", help="report only; create no output files")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path, help="write matching videos here, keeping originals")
    output.add_argument(
        "--overwrite", "-overwrite", action="store_true",
        help="replace matching originals atomically; temporary files stay beside them, not in /tmp",
    )
    parser.add_argument("--json", action="store_true", help="emit one machine-readable JSON report")
    parser.add_argument("--min-fraction", type=float, default=0.9, help="required majority (default: 0.9)")
    parser.add_argument("--scan-seconds", type=float, default=30, help="opening scan limit (default: 30)")
    parser.add_argument("--sample-rate", type=float, default=8, help="thumbnail samples/sec (default: 8)")
    parser.add_argument("--min-intro", type=float, default=1, help="minimum intro seconds (default: 1)")
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="visual difference tolerance; lower is stricter (default: 0.10)",
    )
    parser.add_argument("--workers", type=int, default=2, help="parallel analysis workers, 1-8 (default: 2)")
    parser.add_argument(
        "--keyframe", choices=("previous", "next"), default="previous",
        help="previous preserves content; next removes the full intro but can lose content",
    )
    parser.add_argument(
        "--keyframe-lookahead", type=float, default=30,
        help="seconds beyond the intro to search with --keyframe next (default: 30)",
    )
    return parser


def validate_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    bounds = (
        ("min_fraction", 0.5, 1),
        ("sample_rate", 0, 60),
        ("threshold", 0, 0.5),
        ("scan_seconds", 0, 3600),
        ("min_intro", 0, 3600),
        ("keyframe_lookahead", 0, 3600),
    )
    for name, lower, upper in bounds:
        value = getattr(args, name)
        if not math.isfinite(value) or not lower < value <= upper:
            parser.error(f"--{name.replace('_', '-')} must be > {lower} and <= {upper}")
    if args.scan_seconds < args.min_intro + (CONFIRM_FRAMES + 1) / args.sample_rate:
        parser.error("--scan-seconds must leave at least four samples after --min-intro")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    if args.null and args.files_from is None:
        parser.error("--null requires --files-from")


def report(
    plans: Sequence[FilePlan], detection: Detection, required: int,
    args: argparse.Namespace, dry_run: bool,
) -> None:
    if args.json:
        intro = None
        if detection.seconds is not None:
            intro = {
                "seconds": detection.seconds,
                "boundary_min_seconds": detection.boundary_min,
                "boundary_max_seconds": detection.boundary_max,
                "matching_files": len(detection.members),
            }
        print(json.dumps(
            {
                "schema_version": 1, "dry_run": dry_run, "overwrite": args.overwrite,
                "input_count": len(plans), "required_matches": required,
                "sample_step_seconds": 1 / args.sample_rate,
                "keyframe_policy": args.keyframe,
                "intro": intro, "message": detection.reason,
                "files": [asdict(plan) for plan in plans],
            },
            indent=2, allow_nan=False,
        ))
        return
    if detection.seconds is None:
        print(detection.reason)
    else:
        print(
            f"Common intro: ~{detection.seconds:.3f}s "
            f"(boundary {detection.boundary_min:.3f}..{detection.boundary_max:.3f}s), "
            f"{len(detection.members)}/{len(plans)} files; required {required}."
        )
    mode = "DRY RUN" if dry_run else ("IN-PLACE COPY" if args.overwrite else "STREAM COPY")
    print(f"{mode}; {args.keyframe} keyframe.")
    for plan in plans:
        details = plan.reason
        if plan.cut_seconds is not None:
            details = (
                f"video cut {plan.cut_seconds:.3f}s; "
                f"intro left ~{plan.estimated_intro_remaining_seconds:.3f}s; "
                f"extra removed ~{plan.estimated_content_removed_seconds:.3f}s"
            )
            if plan.reason:
                details += f"; {plan.reason}"
        target = f" -> {plan.output!r}" if plan.output else ""
        print(f"{plan.status.upper():12} {plan.path!r}{target}: {details}")


def execute(args: argparse.Namespace) -> int:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise IntrocutError(f"{tool} is required on PATH")
    inputs = collect_inputs(args.videos, args.files_from, args.null)
    required = max(2, math.ceil(len(inputs) * args.min_fraction - 1e-12))
    dry_run = args.dry_run or (args.output_dir is None and not args.overwrite)
    plans = [FilePlan(str(path)) for path in inputs]
    movies: list[Movie | None] = [None] * len(inputs)
    if not args.json:
        print(f"Analyzing {len(inputs)} videos; {required} must share the opening.", file=sys.stderr)

    def scan(path: Path) -> Movie | str:
        try:
            return analyze(path, args.scan_seconds, args.sample_rate)
        except (IntrocutError, OSError) as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(scan, inputs)):
            if isinstance(result, Movie):
                movies[index] = result
            else:
                plans[index].status, plans[index].reason = "error", result
            if not args.json and ((index + 1) % 100 == 0 or index + 1 == len(inputs)):
                print(f"Analyzed {index + 1}/{len(inputs)} videos.", file=sys.stderr)
    detection = detect(
        movies, required, args.sample_rate, args.min_intro, args.threshold
    )
    keys: dict[int, Keyframe] = {}
    members = set(detection.members)
    for index, plan in enumerate(plans):
        if plan.status == "error":
            continue
        if index not in members:
            plan.reason = (
                "Does not share the detected opening"
                if detection.members else detection.reason
            )
            continue
        plan.matched = True
        movie = movies[index]
        assert movie is not None and detection.seconds is not None
        try:
            key = find_keyframe(movie, detection, args.keyframe, args.keyframe_lookahead)
            if key is None:
                plan.status = "no_keyframe"
                plan.reason = (
                    "No usable keyframe before the boundary; try --keyframe next"
                    if args.keyframe == "previous"
                    else "No keyframe within the lookahead; increase --keyframe-lookahead"
                )
                continue
            keys[index] = key
            plan.status = "would_cut"
            plan.cut_seconds = round(key.pts - movie.video_start, 6)
            plan.keyframe_pts_seconds = key.pts
            plan.estimated_intro_remaining_seconds = round(
                max(0.0, detection.seconds - plan.cut_seconds), 6
            )
            plan.estimated_content_removed_seconds = round(
                max(0.0, plan.cut_seconds - detection.seconds), 6
            )
        except (IntrocutError, OSError) as exc:
            plan.status, plan.reason = "error", str(exc)
    if args.output_dir is not None:
        folder = args.output_dir.resolve()
        preflight_outputs(plans, inputs, folder)
        if not dry_run and keys:
            folder.mkdir(parents=True, exist_ok=True)
    elif args.overwrite:
        for index in keys:
            plans[index].output = plans[index].path
    interrupted = False
    if not dry_run:
        for index, key in keys.items():
            plan, movie = plans[index], movies[index]
            assert movie is not None and plan.output is not None
            try:
                copy_movie(movie, key, Path(plan.output), overwrite=args.overwrite)
                plan.status = "cut"
            except (IntrocutError, OSError) as exc:
                plan.status, plan.reason = "error", str(exc)
            except KeyboardInterrupt:
                plan.status = "interrupted"
                plan.reason = "Interrupted during this copy; inspect this file before retrying"
                for pending in plans:
                    if pending.status == "would_cut":
                        pending.reason = "Not processed after interruption"
                interrupted = True
                break
    report(plans, detection, required, args, dry_run)
    if interrupted:
        return 130
    if any(plan.status == "error" for plan in plans):
        return 1
    return 0 if any(plan.status in ("would_cut", "cut") for plan in plans) else 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_options(parser, args)
    try:
        return execute(args)
    except (IntrocutError, OSError) as exc:
        print(f"introcut: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("introcut: interrupted; completed outputs are kept", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

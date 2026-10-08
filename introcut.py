#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find a majority's shared video opening and remove it without re-encoding."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from threading import Lock
from typing import Sequence
import zlib

__version__ = "0.1.0"
WIDTH, HEIGHT = 16, 9
PIXELS = WIDTH * HEIGHT
FRAME_BYTES = PIXELS * 3
CONFIRM_FRAMES = 3
PLANNING_BOUNDS = {
    "min_fraction": (0.5, 1),
    "sample_rate": (0, 60),
    "threshold": (0, 0.5),
    "min_intro": (0, 3600),
    "keyframe_lookahead": (0, 3600),
}
PLANNING_OPTIONS = (*PLANNING_BOUNDS, "scan_seconds", "keyframe")


class IntrocutError(Exception):
    pass


class PlanningOption(argparse.Action):
    """Track explicit overrides without changing the ordinary CLI defaults."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.planning_options = (*getattr(namespace, "planning_options", ()), self.dest)


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
    audio: tuple[Frame, ...] = ()


@dataclass(frozen=True)
class Detection:
    members: tuple[int, ...] = ()
    seconds: float | None = None
    boundary_min: float | None = None
    boundary_max: float | None = None
    reason: str = ""
    evidence: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class RunSummary:
    total_files: int
    matched_files: int
    would_cut: int
    cut: int
    unmatched: int
    no_keyframe: int
    errors: int
    interrupted: int
    pending: int
    cut_seconds_total: float
    cut_seconds_min: float | None
    cut_seconds_max: float | None
    estimated_intro_remaining_seconds_total: float
    estimated_content_removed_seconds_total: float


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


def audio_fingerprints(
    path: Path, index: int, start: float, scan_seconds: float, sample_rate: float,
) -> tuple[Frame, ...]:
    count = math.ceil(scan_seconds * sample_rate)
    duration = count / sample_rate
    filters = (
        f"[0:{index}]aformat=channel_layouts=mono,"
        f"asetpts=PTS-({start:.9f})/TB,aresample=8000:async=1:first_pts=0,"
        f"apad=whole_dur={duration:.9f},atrim=duration={duration:.9f},"
        f"showspectrumpic=s={count}x{PIXELS}:legend=0:scale=log:"
        "fscale=log:color=channel:saturation=0:drange=80,format=gray[out]"
    )
    raw = run_tool(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "1", "-filter_complex_threads", "1", "-copyts",
            "-t", f"{scan_seconds:.9f}", "-i", str(path),
            "-filter_complex", filters, "-map", "[out]", "-frames:v", "1",
            "-threads:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
    )
    if len(raw) != count * PIXELS:
        raise IntrocutError("Could not decode a complete audio spectrum")
    return tuple(
        Frame.from_rgb(bytes(value for value in raw[column::count] for _ in range(3)))
        for column in range(count)
    )


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
    input_limit = ["-t", f"{scan_seconds:.9f}"] if scan_seconds else []
    frame_limit = ["-frames:v", str(math.ceil(scan_seconds * sample_rate))] if scan_seconds else []
    raw = run_tool(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "1", "-filter_threads", "1",
            *input_limit, "-i", str(path),
            "-map", f"0:{index}", "-an", "-sn", "-dn", "-vf", filters,
            *frame_limit,
            "-threads:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
    )
    if not raw or len(raw) % FRAME_BYTES:
        raise IntrocutError("Could not decode complete video thumbnails")
    frames = tuple(
        Frame.from_rgb(raw[offset:offset + FRAME_BYTES])
        for offset in range(0, len(raw), FRAME_BYTES)
    )
    audio_stream = next(
        (stream for stream in metadata.get("streams", [])
         if stream.get("codec_type") == "audio"),
        None,
    )
    audio = ()
    if audio_stream is not None:
        audio = audio_fingerprints(
            path, int(audio_stream["index"]), start,
            scan_seconds or len(frames) / sample_rate, sample_rate,
        )[:len(frames)]
    if snapshot(path) != before:
        raise IntrocutError("Input changed while it was being analyzed")
    return Movie(path, index, start, frames, before, chapters, chapter_streams, audio)


class AnalysisCache:
    """Checkpoint decoded fingerprints, never executable copy decisions."""

    def __init__(self, path: Path):
        path = path.expanduser().absolute()
        if path.is_symlink():
            raise IntrocutError("Analysis cache must not be a symlink")
        created = False
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if not path.is_file():
                raise IntrocutError("Analysis cache must be a regular file")
        else:
            os.close(descriptor)
            created = True
        self.lock = Lock()
        self.connection = sqlite3.connect(
            path.as_uri() + "?mode=rw", uri=True, check_same_thread=False, isolation_level=None,
        )
        try:
            if created:
                self.connection.execute(
                    "CREATE TABLE fingerprints (key TEXT PRIMARY KEY, metadata TEXT NOT NULL, "
                    "frames BLOB NOT NULL, audio BLOB NOT NULL) WITHOUT ROWID"
                )
                self.connection.execute("PRAGMA application_id=1229148756")
                self.connection.execute("PRAGMA user_version=1")
            if (
                self.connection.execute("PRAGMA application_id").fetchone()[0] != 1229148756
                or self.connection.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise IntrocutError("Unrecognized analysis cache; use a new cache filename")
        except (sqlite3.Error, IntrocutError):
            self.connection.close()
            raise

    def __enter__(self) -> AnalysisCache:
        return self

    def __exit__(self, *_args) -> None:
        self.connection.close()

    @staticmethod
    def key(path: Path, state: tuple[int, int, int, int], seconds: float, rate: float) -> str:
        settings = repr((state, float(seconds), float(rate))).encode()
        return hashlib.sha256(os.fsencode(path) + b"\0" + settings).hexdigest()

    def load(self, path: Path, seconds: float, rate: float) -> Movie | None:
        state = snapshot(path)
        with self.lock:
            row = self.connection.execute(
                "SELECT metadata, frames, audio FROM fingerprints WHERE key=?",
                (self.key(path, state, seconds, rate),),
            ).fetchone()
        if row is None:
            return None
        try:
            movie = plan_movie(json.loads(row[0], object_pairs_hook=unique_plan_fields), path, state)
            decoded = []
            for data in row[1:]:
                raw = zlib.decompress(data)
                if len(raw) % FRAME_BYTES or (
                    seconds > 0 and len(raw) > math.ceil(seconds * rate) * FRAME_BYTES
                ):
                    raise IntrocutError("Invalid cached fingerprint dimensions")
                decoded.append(tuple(
                    Frame.from_rgb(raw[offset:offset + FRAME_BYTES])
                    for offset in range(0, len(raw), FRAME_BYTES)
                ))
            if not decoded[0] or len(decoded[1]) > len(decoded[0]):
                raise IntrocutError("Invalid cached video/audio lengths")
        except (ValueError, TypeError, zlib.error, IntrocutError) as exc:
            raise IntrocutError(f"Invalid analysis cache entry: {exc}") from exc
        if snapshot(path) != state:
            raise IntrocutError("Input changed while loading cached analysis")
        return replace(movie, frames=decoded[0], audio=decoded[1])

    def store(self, movie: Movie, seconds: float, rate: float) -> None:
        if snapshot(movie.path) != movie.snapshot:
            raise IntrocutError("Input changed before analysis could be checkpointed")
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?, ?)",
                (
                    self.key(movie.path, movie.snapshot, seconds, rate),
                    json.dumps(movie_metadata(movie), allow_nan=False),
                    zlib.compress(b"".join(frame.rgb for frame in movie.frames)),
                    zlib.compress(b"".join(frame.rgb for frame in movie.audio)),
                ),
            )


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


def similar_audio(left: Frame, right: Frame, threshold: float) -> bool:
    if left.variance < 64 or right.variance < 64:
        return left.variance < 64 and right.variance < 64
    covariance = (
        sum(a * b for a, b in zip(left.luma, right.luma)) / PIXELS
        - left.mean * right.mean
    )
    correlation = covariance / math.sqrt(left.variance * right.variance)
    return (1 - correlation) / 2 <= threshold


def majority_frame(
    frames: Sequence[Frame], threshold: float, *, audio: bool = False,
) -> Frame:
    compare = similar_audio if audio else similar
    candidate = frames[0]
    votes = 0
    for frame in frames:
        if votes == 0:
            candidate, votes = frame, 1
        elif compare(candidate, frame, threshold):
            votes += 1
        else:
            votes -= 1
    return candidate


def changing_audio(frames: Sequence[Frame], threshold: float) -> bool:
    previous = None
    changes = 0
    for frame in frames:
        if frame.variance >= 64 and (previous is None or not similar_audio(previous, frame, threshold)):
            changes += previous is not None
            previous = frame
            if changes >= 2:
                return True
    return False


def detect(
    movies: Sequence[Movie | None],
    required: int,
    sample_rate: float,
    min_intro: float,
    threshold: float,
    *,
    audio: bool = False,
) -> Detection:
    active = {
        index: movie.audio[:len(movie.frames)] if audio else movie.frames
        for index, movie in enumerate(movies)
        if movie is not None and (not audio or movie.audio)
    }
    if len(active) < required:
        kind = "audio tracks" if audio else "videos"
        return Detection(reason=f"Not enough readable {kind} to reach the required majority")
    compare = similar_audio if audio else similar
    misses = {index: 0 for index in active}
    runs = dict(misses)
    failed_start = 0
    failed_count = 0
    informative = 0
    quiet_start = None

    def finish(boundary: int) -> Detection:
        lower = max(0.0, (boundary - 1) / sample_rate)
        if lower < min_intro:
            return Detection(reason="No sufficiently long common opening found")
        # Vote winners can change recordings between timestamps; use each actual soundtrack.
        members = tuple(sorted(
            index for index, frames in active.items()
            if not audio or changing_audio(frames[:boundary], threshold)
        ))
        if informative < math.ceil(min_intro * sample_rate / 2) or len(members) < required:
            return Detection(
                reason=(
                    "Only silence, a steady tone, or insufficiently distinctive audio matched"
                    if audio else
                    "Only blank/solid or insufficiently distinctive opening frames matched"
                )
            )
        return Detection(
            members=members,
            seconds=boundary / sample_rate,
            boundary_min=lower,
            boundary_max=(boundary + 1) / sample_rate,
            reason="Shared audio opening found" if audio else "Shared visual opening found",
            evidence=("audio",) if audio else ("video",),
        )

    steps = max(len(frames) for frames in active.values())
    for step in range(steps):
        available = {
            index: frames[step]
            for index, frames in active.items()
            if step < len(frames)
        }
        if len(available) < required:
            if audio and quiet_start is not None and step - quiet_start >= CONFIRM_FRAMES:
                return finish(quiet_start)
            return Detection(
                reason="Common opening reaches the scan limit or a file's end; "
                "increase --detection-length or provide more varied, longer videos",
            )
        candidate = majority_frame(list(available.values()), threshold, audio=audio)
        matching = {
            index for index, frame in available.items()
            if compare(candidate, frame, threshold)
        }
        new_misses = {
            index: misses[index] + (index not in matching) for index in active
        }
        new_runs = {
            index: 0 if index in matching else runs[index] + 1 for index in active
        }
        budget = max(1, int((step + 1) * 0.05))
        survivors = {
            index: frames for index, frames in active.items()
            if index in available and new_misses[index] <= budget
            and new_runs[index] < 2
        }
        if len(matching) >= required and len(survivors) >= required:
            active, misses, runs = survivors, new_misses, new_runs
            failed_count = 0
            if candidate.variance >= 64:
                informative += 1
                quiet_start = None
            elif audio and quiet_start is None:
                quiet_start = step
            continue
        if failed_count == 0:
            failed_start = step
        failed_count += 1
        if failed_count < CONFIRM_FRAMES:
            continue
        return finish(quiet_start if audio and quiet_start is not None else failed_start)
    if audio and quiet_start is not None and steps - quiet_start >= CONFIRM_FRAMES:
        return finish(quiet_start)
    return Detection(
        reason="No ending observed before the scan limit or end of the videos; "
        "increase --detection-length or provide more varied, longer videos",
    )


def combine_detections(video: Detection, audio: Detection, required: int) -> Detection:
    if audio.seconds is None:
        if video.seconds is None:
            return Detection(reason=f"Video: {video.reason}. Audio: {audio.reason}.")
        return video
    if video.seconds is None:
        return audio
    assert video.boundary_min is not None and video.boundary_max is not None
    assert audio.boundary_min is not None and audio.boundary_max is not None
    shared = set(video.members) & set(audio.members)
    if (
        len(shared) >= required
        and video.boundary_min <= audio.boundary_max
        and audio.boundary_min <= video.boundary_max
    ):
        return Detection(
            members=tuple(sorted(set(video.members) | set(audio.members))),
            seconds=min(video.seconds, audio.seconds),
            boundary_min=min(video.boundary_min, audio.boundary_min),
            boundary_max=max(video.boundary_max, audio.boundary_max),
            reason="Shared visual and audio opening found",
            evidence=("video", "audio"),
        )
    # A shared background soundtrack must not extend an established visual intro.
    return video


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


def set_cut_details(plan: FilePlan, movie: Movie, key: Keyframe, detection: Detection) -> None:
    assert detection.seconds is not None
    plan.status = "would_cut"
    plan.cut_seconds = round(key.pts - movie.video_start, 6)
    plan.keyframe_pts_seconds = key.pts
    plan.estimated_intro_remaining_seconds = round(
        max(0.0, detection.seconds - plan.cut_seconds), 6
    )
    plan.estimated_content_removed_seconds = round(
        max(0.0, plan.cut_seconds - detection.seconds), 6
    )


def new_plan_path(path: Path) -> Path:
    path = path.expanduser()
    if os.path.lexists(path):
        raise IntrocutError(f"Refusing to overwrite an existing plan path: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise IntrocutError(f"Plan parent is not a directory: {parent}")
    return parent / path.name


def movie_metadata(movie: Movie) -> dict:
    return {
        "video_index": movie.video_index, "video_start": movie.video_start,
        "chapters": [asdict(chapter) for chapter in movie.chapters],
        "chapter_streams": list(movie.chapter_streams),
    }


def save_plan(
    path: Path, plans: Sequence[FilePlan], movies: Sequence[Movie | None],
    keys: dict[int, Keyframe], detection: Detection, required: int,
    args: argparse.Namespace, snapshots: Sequence[tuple[int, int, int, int]],
) -> None:
    entries = []
    for index, plan in enumerate(plans):
        movie, state = movies[index], snapshots[index]
        if snapshot(Path(plan.path)) != state or (movie is not None and movie.snapshot != state):
            raise IntrocutError(f"Input changed before the plan was saved: {plan.path}")
        media = None
        if index in keys:
            assert movie is not None
            media = movie_metadata(movie)
        entries.append({
            "path": plan.path, "snapshot": state, "status": plan.status, "reason": plan.reason,
            "movie": media, "keyframe": asdict(keys[index]) if index in keys else None,
        })
    data = {
        "format": "introcut-plan", "schema_version": 1,
        "settings": {name: getattr(args, name) for name in PLANNING_OPTIONS},
        "required_matches": required, "detection": asdict(detection), "files": entries,
    }
    try:
        text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise IntrocutError(f"Cannot serialize saved plan: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix=".introcut-", dir=path.parent) as folder:
        temporary = Path(folder) / "plan.json"
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(0o600)
        os.link(temporary, path)


def check_plan(condition: bool, detail: str) -> None:
    if not condition:
        raise IntrocutError(f"Invalid saved plan: {detail}; generate a new plan with --save-plan")


def plan_object(value: object, names: Sequence[str], label: str) -> dict:
    check_plan(isinstance(value, dict) and set(value) == set(names), f"{label} fields")
    assert isinstance(value, dict)
    return value


def plan_number(value: object, label: str) -> float:
    check_plan(type(value) in (int, float), f"{label} must be a number")
    return finite_number(value, f"saved plan {label}")


def plan_integers(value: object, label: str) -> tuple[int, ...]:
    check_plan(
        isinstance(value, list) and all(type(item) is int for item in value),
        f"{label} must be an integer array",
    )
    assert isinstance(value, list)
    return tuple(value)


def unique_plan_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        check_plan(name not in result, f"duplicate field {name!r}")
        result[name] = value
    return result


def plan_movie(
    raw: object, path: Path, state: tuple[int, int, int, int],
) -> Movie:
    data = plan_object(raw, ("video_index", "video_start", "chapters", "chapter_streams"), "movie")
    index = data["video_index"]
    check_plan(type(index) is int and index >= 0, "video stream index")
    start = plan_number(data["video_start"], "video start")
    streams = plan_integers(data["chapter_streams"], "chapter streams")
    check_plan(
        len(set(streams)) == len(streams) and all(item >= 0 and item != index for item in streams),
        "chapter stream indices",
    )
    check_plan(isinstance(data["chapters"], list), "chapters must be an array")
    chapters = []
    for raw_chapter in data["chapters"]:
        chapter = plan_object(raw_chapter, ("start", "end", "tags"), "chapter")
        begin = plan_number(chapter["start"], "chapter start")
        end = plan_number(chapter["end"], "chapter end")
        tags = chapter["tags"]
        check_plan(end > begin, "chapter time range")
        check_plan(
            isinstance(tags, dict)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in tags.items()),
            "chapter tags",
        )
        chapters.append(Chapter(begin, end, tags))
    check_plan(not streams or bool(chapters), "chapter streams without chapters")
    return Movie(path, index, start, (), state, tuple(chapters), streams)


def load_plan(
    path: Path, args: argparse.Namespace,
) -> tuple[list[FilePlan], list[Movie | None], dict[int, Keyframe], Detection, int]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise IntrocutError(f"Plan is not a regular file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_plan_fields)
    except (ValueError, UnicodeDecodeError) as exc:
        raise IntrocutError(f"Invalid saved plan JSON: {exc}") from exc
    data = plan_object(
        raw, ("format", "schema_version", "settings", "required_matches", "detection", "files"), "root",
    )
    check_plan(data["format"] == "introcut-plan", "not an introcut plan (JSON reports are not plans)")
    check_plan(
        type(data["schema_version"]) is int and data["schema_version"] == 1,
        "unsupported schema version",
    )
    settings = plan_object(data["settings"], PLANNING_OPTIONS, "settings")
    for name, (lower, upper) in PLANNING_BOUNDS.items():
        settings[name] = plan_number(settings[name], name)
        check_plan(lower < settings[name] <= upper, f"{name} is out of range")
    settings["scan_seconds"] = plan_number(settings["scan_seconds"], "scan_seconds")
    check_plan(settings["scan_seconds"] >= 0, "scan_seconds is out of range")
    check_plan(settings["keyframe"] in ("previous", "next"), "keyframe policy")
    for name in getattr(args, "planning_options", ()):
        if getattr(args, name) != settings[name]:
            raise IntrocutError(
                f"Cannot change --{name.replace('_', '-')} with --apply-plan; generate a new plan"
            )
    entries = data["files"]
    check_plan(isinstance(entries, list) and len(entries) >= 2, "at least two input files are required")
    required = data["required_matches"]
    check_plan(
        type(required) is int
        and required == max(2, math.ceil(len(entries) * settings["min_fraction"] - 1e-12)),
        "required match count",
    )
    found = plan_object(
        data["detection"], ("members", "seconds", "boundary_min", "boundary_max", "reason", "evidence"),
        "detection",
    )
    members = plan_integers(found["members"], "detection members")
    check_plan(
        len(set(members)) == len(members) and all(0 <= index < len(entries) for index in members),
        "detection member indices",
    )
    check_plan(isinstance(found["reason"], str), "detection reason")
    evidence = found["evidence"]
    check_plan(evidence in ([], ["video"], ["audio"], ["video", "audio"]), "detection evidence")
    seconds = lower = upper = None
    if found["seconds"] is None:
        check_plan(
            not members and not evidence
            and found["boundary_min"] is None and found["boundary_max"] is None,
            "absent intro has members, evidence, or a boundary",
        )
    else:
        seconds = plan_number(found["seconds"], "intro seconds")
        lower = plan_number(found["boundary_min"], "boundary minimum")
        upper = plan_number(found["boundary_max"], "boundary maximum")
        check_plan(
            settings["min_intro"] <= lower <= seconds <= upper
            and (settings["scan_seconds"] == 0 or upper <= settings["scan_seconds"])
            and seconds > 0 and len(members) >= required,
            "intro boundary/quorum",
        )
    detection = Detection(members, seconds, lower, upper, found["reason"], tuple(evidence))
    member_set = set(members)
    plans, movies, keys, states = [], [], {}, []
    for index, raw_entry in enumerate(entries):
        entry = plan_object(
            raw_entry, ("path", "snapshot", "status", "reason", "movie", "keyframe"), f"file {index}",
        )
        check_plan(isinstance(entry["path"], str) and "\0" not in entry["path"], "input path")
        source = Path(entry["path"])
        check_plan(source.is_absolute(), "input paths must be absolute")
        state_values = plan_integers(entry["snapshot"], "input snapshot")
        check_plan(len(state_values) == 4 and all(value >= 0 for value in state_values[:3]), "input snapshot")
        state = (state_values[0], state_values[1], state_values[2], state_values[3])
        status, reason = entry["status"], entry["reason"]
        check_plan(
            status in ("unmatched", "would_cut", "no_keyframe", "error")
            and isinstance(reason, str), "file status/reason",
        )
        matched = index in member_set
        check_plan(
            (matched and status in ("would_cut", "no_keyframe", "error"))
            or (not matched and status in ("unmatched", "error")), "file status disagrees with detection",
        )
        plan = FilePlan(str(source), status=status, matched=matched, reason=reason)
        movie = None
        if status == "would_cut":
            movie = plan_movie(entry["movie"], source, state)
            key_data = plan_object(entry["keyframe"], ("pts", "dts", "data_hash"), "keyframe")
            pts = plan_number(key_data["pts"], "keyframe PTS")
            dts = plan_number(key_data["dts"], "keyframe DTS")
            data_hash = key_data["data_hash"]
            check_plan(
                isinstance(data_hash, str)
                and re.fullmatch(r"SHA256:[0-9a-fA-F]{64}", data_hash) is not None,
                "keyframe hash",
            )
            key = Keyframe(pts, dts, data_hash)
            cut = pts - movie.video_start
            assert lower is not None and upper is not None
            check_plan(cut > 0.000001, "cut must remove a positive duration")
            check_plan(
                cut <= lower + 0.000001 if settings["keyframe"] == "previous" else
                upper - 0.000001 <= cut <= upper + settings["keyframe_lookahead"] + 0.000001,
                "keyframe is outside the saved policy boundary",
            )
            keys[index] = key
            set_cut_details(plan, movie, key, detection)
        else:
            check_plan(
                entry["movie"] is None and entry["keyframe"] is None,
                "non-cut file has copy instructions",
            )
        plans.append(plan)
        movies.append(movie)
        states.append(state)
    inputs = [Path(plan.path) for plan in plans]
    check_plan(
        len(set(inputs)) == len(inputs) and len({state[:2] for state in states}) == len(inputs),
        "duplicate input paths or identities",
    )
    if args.videos or args.files_from is not None:
        provided = collect_inputs(args.videos, args.files_from, args.null)
        if set(provided) != set(inputs):
            raise IntrocutError("Input set differs from the saved plan; generate a new plan")
    for source, state in zip(inputs, states):
        if source.resolve(strict=True) != source or not source.is_file() or snapshot(source) != state:
            raise IntrocutError(f"Input changed since the plan was saved: {source}; generate a new plan")
    for name, value in settings.items():
        setattr(args, name, value)
    return plans, movies, keys, detection, required


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a majority's common video intro and cut at keyframes, without re-encoding.",
        epilog="Default: dry run. Use --output-dir or --overwrite to apply; --dry-run overrides both.",
    )
    parser.add_argument("videos", nargs="*", help="input paths or quoted glob patterns")
    parser.add_argument("--version", action="version", version=f"introcut {__version__}")
    parser.add_argument("--files-from", metavar="FILE", help="read paths from a list; - means stdin")
    parser.add_argument("--null", action="store_true", help="the file list is NUL-delimited")
    parser.add_argument("--dry-run", action="store_true", help="report cuts without copying/replacing videos")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path, help="write matching videos here, keeping originals")
    output.add_argument(
        "--overwrite", "-overwrite", action="store_true",
        help="replace matching originals atomically; temporary files stay beside them, not in /tmp",
    )
    saved = parser.add_mutually_exclusive_group()
    saved.add_argument(
        "--save-plan", type=Path, metavar="FILE",
        help="save a reusable dry-run plan; FILE must not exist",
    )
    saved.add_argument(
        "--apply-plan", type=Path, metavar="FILE",
        help="reuse a saved plan without detection; inputs must be unchanged",
    )
    parser.add_argument(
        "--analysis-cache", type=Path, metavar="FILE",
        help="checkpoint analysis for retries; defaults to FILE.analysis.sqlite3 with --save-plan",
    )
    parser.add_argument("--json", action="store_true", help="emit one machine-readable JSON report")
    parser.add_argument(
        "--min-fraction", type=float, action=PlanningOption, default=0.7,
        help="required majority (default: 0.7)",
    )
    parser.add_argument(
        "--detection-length", "-detection-length", "--scan-seconds", dest="scan_seconds",
        type=float, action=PlanningOption, default=30, metavar="SECONDS",
        help="seconds to analyze per video: >=1, or 0 for the full video (default: 30)",
    )
    parser.add_argument(
        "--sample-rate", type=float, action=PlanningOption, default=8,
        help="video/audio samples/sec (default: 8)",
    )
    parser.add_argument(
        "--min-intro", type=float, action=PlanningOption, default=1,
        help="minimum intro seconds (default: 1)",
    )
    parser.add_argument(
        "--threshold", type=float, action=PlanningOption, default=0.10,
        help="picture/sound difference tolerance; lower is stricter (default: 0.10)",
    )
    worker_limit = os.cpu_count() or 16
    worker_default = min(4, worker_limit)
    parser.add_argument(
        "--workers", type=int, default=worker_default,
        help=f"parallel per-file analysis workers, 1-{worker_limit} (default: {worker_default})",
    )
    parser.add_argument(
        "--keyframe", choices=("next", "prev", "previous"), action=PlanningOption, default="next",
        type=lambda value: "previous" if value == "prev" else value,
        help="next (default) removes the full intro but can lose content; prev/previous preserves content",
    )
    parser.add_argument(
        "--keyframe-lookahead", type=float, action=PlanningOption, default=30,
        help="seconds beyond the intro to search with --keyframe next (default: 30)",
    )
    return parser


def validate_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name, (lower, upper) in PLANNING_BOUNDS.items():
        value = getattr(args, name)
        if not math.isfinite(value) or not lower < value <= upper:
            parser.error(f"--{name.replace('_', '-')} must be > {lower} and <= {upper}")
    if not math.isfinite(args.scan_seconds) or not (args.scan_seconds == 0 or args.scan_seconds >= 1):
        parser.error("--detection-length must be 0 (full video) or at least 1 second")
    if not math.isfinite(args.scan_seconds * args.sample_rate):
        parser.error("--detection-length and --sample-rate produce an unrepresentable sample count")
    worker_limit = os.cpu_count() or 16
    if not 1 <= args.workers <= worker_limit:
        parser.error(f"--workers must be between 1 and {worker_limit}")
    if args.null and args.files_from is None:
        parser.error("--null requires --files-from")
    if (
        args.save_plan is not None and not args.dry_run
        and (args.overwrite or args.output_dir is not None)
    ):
        parser.error("--save-plan requires a dry run; add --dry-run")
    if args.analysis_cache is not None and args.apply_plan is not None:
        parser.error("--analysis-cache is for detection, not --apply-plan")


def summarize(plans: Sequence[FilePlan], dry_run: bool) -> RunSummary:
    counts = Counter(plan.status for plan in plans)
    cuts: list[float] = []
    remaining: list[float] = []
    extra: list[float] = []
    for plan in plans:
        if plan.status != ("would_cut" if dry_run else "cut"):
            continue
        assert plan.cut_seconds is not None
        assert plan.estimated_intro_remaining_seconds is not None
        assert plan.estimated_content_removed_seconds is not None
        cuts.append(plan.cut_seconds)
        remaining.append(plan.estimated_intro_remaining_seconds)
        extra.append(plan.estimated_content_removed_seconds)
    return RunSummary(
        total_files=len(plans),
        matched_files=sum(plan.matched for plan in plans),
        would_cut=counts["would_cut"] if dry_run else 0,
        cut=counts["cut"],
        unmatched=counts["unmatched"],
        no_keyframe=counts["no_keyframe"],
        errors=counts["error"],
        interrupted=counts["interrupted"],
        pending=0 if dry_run else counts["would_cut"],
        cut_seconds_total=round(math.fsum(cuts), 6),
        cut_seconds_min=min(cuts, default=None),
        cut_seconds_max=max(cuts, default=None),
        estimated_intro_remaining_seconds_total=round(math.fsum(remaining), 6),
        estimated_content_removed_seconds_total=round(math.fsum(extra), 6),
    )


def report(
    plans: Sequence[FilePlan], detection: Detection, required: int,
    args: argparse.Namespace, dry_run: bool,
) -> None:
    summary = summarize(plans, dry_run)
    if args.json:
        intro = None
        if detection.seconds is not None:
            intro = {
                "seconds": detection.seconds,
                "boundary_min_seconds": detection.boundary_min,
                "boundary_max_seconds": detection.boundary_max,
                "matching_files": len(detection.members),
                "evidence": list(detection.evidence),
            }
        print(json.dumps(
            {
                "schema_version": 1, "dry_run": dry_run, "overwrite": args.overwrite,
                "input_count": len(plans), "required_matches": required,
                "sample_step_seconds": 1 / args.sample_rate,
                "keyframe_policy": args.keyframe,
                "intro": intro, "message": detection.reason,
                "files": [asdict(plan) for plan in plans],
                "summary": asdict(summary),
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
        if detection.evidence:
            print(f"Evidence: {' + '.join(detection.evidence)}.")
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
    intro_text = f"~{detection.seconds:.3f}s" if detection.seconds is not None else "not detected"
    print(f"\nSummary: {mode}; {args.keyframe} keyframe; intro {intro_text}.")
    print(
        f"Files: {summary.total_files} total; {summary.matched_files} matched; "
        f"{summary.unmatched} unmatched; {summary.no_keyframe} no keyframe; {summary.errors} errors."
    )
    print(
        f"Results: {summary.would_cut} would cut; {summary.cut} cut; "
        f"{summary.pending} pending; {summary.interrupted} interrupted."
    )
    if summary.cut_seconds_min is not None:
        assert summary.cut_seconds_max is not None
        label = "Planned" if dry_run else "Completed"
        print(
            f"{label} video cuts: {summary.cut_seconds_total:.3f}s total; "
            f"{summary.cut_seconds_min:.3f}..{summary.cut_seconds_max:.3f}s per file."
        )
        print(
            f"Estimated intro left: {summary.estimated_intro_remaining_seconds_total:.3f}s total; "
            f"extra content removed: {summary.estimated_content_removed_seconds_total:.3f}s total."
        )


def execute(args: argparse.Namespace) -> int:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise IntrocutError(f"{tool} is required on PATH")
    dry_run = args.dry_run or (args.output_dir is None and not args.overwrite)
    save_target = new_plan_path(args.save_plan) if args.save_plan is not None else None
    if args.apply_plan is not None:
        plans, movies, keys, detection, required = load_plan(args.apply_plan, args)
        inputs = [Path(plan.path) for plan in plans]
        if not args.json:
            print(f"Using saved plan for {len(inputs)} videos; detection skipped.", file=sys.stderr)
    else:
        inputs = collect_inputs(args.videos, args.files_from, args.null)
        output_paths = {
            args.output_dir.resolve() / path.name for path in inputs
        } if args.output_dir is not None else set()
        if save_target in output_paths:
            raise IntrocutError("Plan path conflicts with a video output path")
        snapshots = [snapshot(path) for path in inputs] if save_target is not None else []
        required = max(2, math.ceil(len(inputs) * args.min_fraction - 1e-12))
        plans = [FilePlan(str(path)) for path in inputs]
        movies: list[Movie | None] = [None] * len(inputs)
        cache_path = args.analysis_cache
        if cache_path is None and save_target is not None:
            cache_path = Path(str(save_target) + ".analysis.sqlite3")
        if cache_path is not None:
            cache_path = cache_path.expanduser().absolute()
            if cache_path.resolve() in {*inputs, *output_paths, save_target} or (
                cache_path.is_file()
                and snapshot(cache_path)[:2] in {snapshot(path)[:2] for path in inputs}
            ):
                raise IntrocutError("Analysis cache conflicts with an input or plan path")
        with AnalysisCache(cache_path) if cache_path is not None else nullcontext() as cache:
            if not args.json:
                window = f"up to {args.scan_seconds:g}s" if args.scan_seconds else "full videos"
                print(
                    f"Analyzing {len(inputs)} videos; {required} must share the opening "
                    f"(scan {window}).",
                    file=sys.stderr,
                )

            def scan(path: Path) -> tuple[Movie | str, bool]:
                if cache is not None:
                    cached = cache.load(path, args.scan_seconds, args.sample_rate)
                    if cached is not None:
                        return cached, True
                try:
                    return analyze(path, args.scan_seconds, args.sample_rate), False
                except (IntrocutError, OSError) as exc:
                    return str(exc), False

            cached_count = 0
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                for index, (result, cached) in enumerate(pool.map(scan, inputs)):
                    cached_count += cached
                    if isinstance(result, Movie):
                        movies[index] = result
                        plans[index].status, plans[index].reason = "unmatched", ""
                        if cache is not None and not cached:
                            cache.store(result, args.scan_seconds, args.sample_rate)
                    else:
                        movies[index] = None
                        plans[index].status, plans[index].reason = "error", result
                    if not args.json and ((index + 1) % 100 == 0 or index + 1 == len(inputs)):
                        print(
                            f"Analyzed {index + 1}/{len(inputs)} videos; {cached_count} cached.",
                            file=sys.stderr,
                        )
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        video_detection = detect(
            movies, required, args.sample_rate, args.min_intro, args.threshold
        )
        audio_detection = detect(
            movies, required, args.sample_rate, args.min_intro, args.threshold, audio=True
        )
        detection = combine_detections(video_detection, audio_detection, required)
        keys: dict[int, Keyframe] = {}
        members = set(detection.members)
        if not args.json and members:
            print(f"Planning keyframe cuts for {len(members)} matching videos.", file=sys.stderr)
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
                set_cut_details(plan, movie, key, detection)
            except (IntrocutError, OSError) as exc:
                plan.status, plan.reason = "error", str(exc)
        if not args.json and members:
            print(f"Planned {len(keys)} usable cuts.", file=sys.stderr)
    if args.output_dir is not None:
        folder = args.output_dir.resolve()
        preflight_outputs(plans, inputs, folder)
        if not dry_run and keys:
            folder.mkdir(parents=True, exist_ok=True)
    elif args.overwrite:
        for index in keys:
            plans[index].output = plans[index].path
    if save_target is not None:
        if any(plan.output is not None and Path(plan.output) == save_target for plan in plans):
            raise IntrocutError("Plan path conflicts with a video output path")
        save_plan(save_target, plans, movies, keys, detection, required, args, snapshots)
        if not args.json:
            print(f"Saved plan: {save_target}", file=sys.stderr)
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
    except (IntrocutError, OSError, sqlite3.Error) as exc:
        print(f"introcut: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("introcut: interrupted; completed outputs are kept", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import introcut as app


ROOT = Path(__file__).resolve().parents[1]


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, check=False)
    if result.returncode or result.stderr:
        raise AssertionError(
            f"Command failed ({result.returncode}): {arguments!r}\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packets(path):
    return json.loads(command([
        "ffprobe", "-v", "error", "-show_packets", "-show_data_hash", "sha256",
        "-show_entries", "packet=stream_index,pts_time,dts_time,flags,data_hash",
        "-of", "json", str(path),
    ]))["packets"]


def metadata(path):
    return json.loads(command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-show_chapters",
        "-of", "json", str(path),
    ]))


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("ffmpeg and ffprobe are required for media tests")
        cls.temporary = tempfile.TemporaryDirectory(prefix=".introcut-tests-", dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        chapters = cls.root / "chapters.txt"
        chapters.write_text(
            ";FFMETADATA1\ntitle=introcut fixture\n"
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=3000\ntitle=Intro\n"
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART=3000\nEND=7000\ntitle=Body\n",
            encoding="utf-8",
        )
        rates = ("24", "25", "30", "30000/1001", "60", "15", "24000/1001", "30", "25")
        sizes = ("160:90", "256:144", "320:180", "480:270", "640:360")
        colors = ("red", "blue", "green", "yellow", "magenta", "cyan", "white", "black", "orange")
        cls.matches = []
        for index in range(9):
            suffix = ".mp4" if index % 2 == 0 else ".mkv"
            name = f"match {index}{suffix}" if index != 5 else f"match\n{index}{suffix}"
            path = cls.root / name
            fps = rates[index]
            numerator, _, denominator = fps.partition("/")
            rate = float(numerator) / float(denominator or "1")
            sample_rate = (48000, 44100, 32000)[index % 3]
            channels = (1, 2, 6)[index % 3]
            arguments = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-threads", "1", "-filter_complex_threads", "1",
                "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=60:duration=3",
                "-f", "lavfi", "-i", f"color=c={colors[index]}:s=320x180:r=60:d=4",
                "-f", "lavfi", "-i", f"sine=frequency={400 + index * 37}:sample_rate={sample_rate}:duration=7",
                "-f", "ffmetadata", "-i", str(chapters),
                "-filter_complex",
                f"[0:v][1:v]concat=n=2:v=1:a=0,fps={fps},scale={sizes[index % len(sizes)]}[v]",
                "-map", "[v]", "-map", "2:a", "-map_metadata", "3", "-map_chapters", "3",
                "-c:v", "libx265", "-preset", "ultrafast",
                "-x265-params",
                f"pools=none:frame-threads=1:keyint={round(rate * 2)}:min-keyint={round(rate * 2)}:"
                f"open-gop={0 if index in (2, 6) else 1}:scenecut=0:log-level=error",
                "-c:a", "aac", "-b:a", "96k", "-ar", str(sample_rate), "-ac", str(channels),
                "-t", "7",
            ]
            if index in (3, 4):
                arguments.extend(["-output_ts_offset", "10"])
            arguments.append(str(path))
            command(arguments)
            cls.matches.append(path)
        cls.outlier = cls.root / "outlier.mp4"
        command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-threads", "1", "-filter_threads", "1",
            "-f", "lavfi", "-i", "color=c=gray:s=160x90:r=25:d=7",
            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=7",
            "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", "pools=none:frame-threads=1:keyint=50:scenecut=0:log-level=error",
            "-c:a", "aac", "-t", "7", str(cls.outlier),
        ])
        cls.inputs = [cls.outlier, *cls.matches]
        cls.hashes = {path: digest(path) for path in cls.inputs}
        cls.listing = cls.root / "inputs.list"
        cls.listing.write_bytes(b"\0".join(os.fsencode(path.name) for path in cls.inputs) + b"\0")

    def cli(self, extra, inputs=None, expected=0):
        arguments = [
            sys.executable, str(ROOT / "introcut.py"),
            "--scan-seconds", "6", "--json", *extra,
        ]
        if inputs is None:
            arguments += ["--files-from", str(self.listing), "--null"]
        else:
            arguments += [str(path) for path in inputs]
        env = dict(os.environ, TMPDIR=str(self.root / "must-not-use-tmp"))
        result = subprocess.run(arguments, capture_output=True, text=True, env=env)
        self.assertEqual(
            result.returncode, expected,
            f"{result.stderr}\n{result.stdout}",
        )
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def assert_originals_unchanged(self):
        self.assertEqual({path: digest(path) for path in self.inputs}, self.hashes)

    def assert_packet_preserving_cut(self, original, output, plan):
        source_packets, output_packets = packets(original), packets(output)
        before, after = metadata(original), metadata(output)
        self.assertEqual(len(before["streams"]), len(after["streams"]))
        for left, right in zip(before["streams"], after["streams"]):
            for name in (
                "codec_name", "codec_type", "profile", "width", "height",
                "pix_fmt", "sample_rate", "channels", "channel_layout",
            ):
                self.assertEqual(left.get(name), right.get(name), (output, name))
        offsets = []
        for index in (0, 1):
            source = [p for p in source_packets if p["stream_index"] == index]
            copied = [p for p in output_packets if p["stream_index"] == index]
            self.assertTrue(copied, (output, index))
            self.assertLess(len(copied), len(source))
            self.assertEqual(
                [p["data_hash"] for p in copied],
                [p["data_hash"] for p in source[-len(copied):]],
                (output, index),
            )
            first_source = source[-len(copied)]
            offsets.append(float(first_source["pts_time"]) - float(copied[0]["pts_time"]))
            if index == 0:
                self.assertIn("K", copied[0]["flags"])
                self.assertAlmostEqual(
                    float(first_source["pts_time"]), plan["keyframe_pts_seconds"], places=5
                )
                self.assertAlmostEqual(
                    float(first_source["pts_time"]) - float(before["streams"][0]["start_time"]),
                    plan["cut_seconds"], places=5,
                )
        self.assertAlmostEqual(offsets[0], offsets[1], delta=0.003)
        command([
            "ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
            "-i", str(output), "-f", "null", "-",
        ])
        self.assertEqual(
            after.get("format", {}).get("tags", {}).get("title"),
            before.get("format", {}).get("tags", {}).get("title"),
        )
        shift = plan["cut_seconds"] - float(after["streams"][0]["start_time"])
        surviving = [
            chapter for chapter in before["chapters"]
            if float(chapter["end_time"]) > shift
        ]
        self.assertEqual(len(after["chapters"]), len(surviving))
        for original_chapter, chapter in zip(surviving, after["chapters"]):
            self.assertEqual(chapter["tags"], original_chapter["tags"])
            self.assertAlmostEqual(
                float(chapter["start_time"]),
                max(0, float(original_chapter["start_time"]) - shift), delta=0.002,
            )
            self.assertAlmostEqual(
                float(chapter["end_time"]),
                float(original_chapter["end_time"]) - shift, delta=0.002,
            )

    def test_dry_run_ninety_percent_diverse_media_and_json(self):
        folder = self.root / "dry-output"
        report = self.cli(["--dry-run", "--output-dir", str(folder)])
        self.assertFalse(folder.exists())
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["required_matches"], 9)
        self.assertEqual(report["intro"]["matching_files"], 9)
        self.assertAlmostEqual(report["intro"]["seconds"], 3, delta=0.25)
        self.assertEqual(report["files"][0]["status"], "unmatched")
        self.assertFalse(report["files"][0]["matched"])
        for plan in report["files"][1:]:
            self.assertEqual(plan["status"], "would_cut")
            self.assertTrue(plan["matched"])
            self.assertGreater(plan["cut_seconds"], 0)
            self.assertLessEqual(plan["cut_seconds"], report["intro"]["boundary_min_seconds"])
            self.assertEqual(plan["estimated_content_removed_seconds"], 0)
        self.assert_originals_unchanged()

    def test_default_mode_is_dry_run(self):
        report = self.cli([])
        self.assertTrue(report["dry_run"])
        self.assertTrue(all(plan["output"] is None for plan in report["files"]))
        self.assert_originals_unchanged()

    def test_next_keyframe_reports_content_loss_and_honors_lookahead(self):
        report = self.cli(["--keyframe", "next", "--dry-run"])
        for plan in report["files"][1:]:
            self.assertGreaterEqual(plan["cut_seconds"], report["intro"]["boundary_max_seconds"])
            self.assertGreater(plan["estimated_content_removed_seconds"], 0)
            self.assertEqual(plan["estimated_intro_remaining_seconds"], 0)
        bounded = self.cli(["--keyframe", "next", "--keyframe-lookahead", "0.1"], expected=3)
        self.assertTrue(all(plan["status"] == "no_keyframe" for plan in bounded["files"][1:]))
        self.assert_originals_unchanged()

    def test_real_copies_preserve_packets_codecs_timestamps_and_originals(self):
        folder = self.root / "copied"
        report = self.cli(["--output-dir", str(folder)])
        self.assertFalse(report["dry_run"])
        self.assertEqual(report["files"][0]["status"], "unmatched")
        self.assertFalse((folder / self.outlier.name).exists())
        for plan in report["files"][1:]:
            with self.subTest(path=plan["path"]):
                self.assertEqual(plan["status"], "cut", plan)
                self.assert_packet_preserving_cut(Path(plan["path"]), Path(plan["output"]), plan)
        self.assertEqual(len(list(folder.iterdir())), 9)
        self.assert_originals_unchanged()

    def test_next_keyframe_copies_drop_old_chapters_and_preserve_packets(self):
        folder = self.root / "next-keyframe"
        report = self.cli(
            ["--keyframe", "next", "--output-dir", str(folder)], self.matches[:2]
        )
        for plan in report["files"]:
            self.assertEqual(plan["status"], "cut")
            self.assert_packet_preserving_cut(Path(plan["path"]), Path(plan["output"]), plan)
            self.assertEqual(
                [chapter["tags"]["title"] for chapter in metadata(Path(plan["output"]))["chapters"]],
                ["Body"],
            )
        self.assert_originals_unchanged()

    def test_overwrite_and_overwrite_dry_run_with_small_unusable_tmp(self):
        folder = self.root / "in-place"
        folder.mkdir()
        selected = [self.matches[0], self.matches[1], self.outlier]
        inputs = []
        for source in selected:
            target = folder / source.name
            shutil.copy2(source, target)
            target.chmod(0o640)
            inputs.append(target)
        before = {path: (app.snapshot(path), digest(path), path.stat().st_mode) for path in inputs}
        dry = self.cli(["--overwrite", "--dry-run", "--min-fraction", "0.6"], inputs)
        self.assertTrue(dry["dry_run"])
        self.assertTrue(dry["overwrite"])
        self.assertEqual(
            {path: (app.snapshot(path), digest(path), path.stat().st_mode) for path in inputs}, before
        )
        report = self.cli(["-overwrite", "--min-fraction", "0.6"], inputs)
        self.assertFalse(report["dry_run"])
        self.assertTrue(report["overwrite"])
        for source, target, plan in zip(selected[:2], inputs[:2], report["files"][:2]):
            self.assertEqual(plan["status"], "cut")
            self.assertEqual(plan["path"], plan["output"])
            self.assertNotEqual(digest(target), before[target][1])
            self.assertEqual(target.stat().st_mode, before[target][2])
            self.assert_packet_preserving_cut(source, target, plan)
        self.assertEqual(report["files"][2]["status"], "unmatched")
        self.assertEqual(digest(inputs[2]), before[inputs[2]][1])
        self.assertEqual(sorted(folder.iterdir()), sorted(inputs))
        self.assert_originals_unchanged()

    def test_identical_full_videos_have_no_known_boundary(self):
        clone = self.root / "identical.mp4"
        shutil.copy2(self.matches[0], clone)
        folder = self.root / "must-not-create"
        result = self.cli(["--output-dir", str(folder)], [self.matches[0], clone], expected=3)
        self.assertIsNone(result["intro"])
        self.assertFalse(folder.exists())

    def test_corrupt_input_is_reported_and_still_counts_in_quorum(self):
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not a video")
        result = self.cli(["--dry-run"], [corrupt, *self.matches], expected=1)
        self.assertEqual(result["required_matches"], 9)
        self.assertEqual(result["intro"]["matching_files"], 9)
        self.assertEqual(result["files"][0]["status"], "error")
        self.assertTrue(result["files"][0]["reason"])
        self.assert_originals_unchanged()


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import introcut as app


def solid(value: int) -> app.Frame:
    return app.Frame.from_rgb(bytes([value]) * app.FRAME_BYTES)


def texture(offset: int = 0) -> app.Frame:
    return app.Frame.from_rgb(bytes(
        (index * 37 + offset) % 256 for index in range(app.FRAME_BYTES)
    ))


def movie(index: int, frames: list[app.Frame]) -> app.Movie:
    return app.Movie(Path(f"/video-{index}.mp4"), 0, 0, tuple(frames), (0, 0, 0, 0))


def batch(count: int = 9, prefix: list[app.Frame] | None = None) -> list[app.Movie]:
    if prefix is None:
        prefix = [texture()] * 24
    return [movie(i, prefix + [solid((i * 28) % 256)] * 24) for i in range(count)]


class DetectionTests(unittest.TestCase):
    def detect(self, movies, required=9, min_intro=1):
        return app.detect(movies, required, 8, min_intro, 0.1)

    def test_ninety_percent_and_first_input_outlier(self):
        movies = [movie(99, [solid(80)] * 48)] + batch()
        result = self.detect(movies)
        self.assertEqual(result.members, tuple(range(1, 10)))
        self.assertEqual(result.seconds, 3)
        self.assertEqual((result.boundary_min, result.boundary_max), (2.875, 3.125))

    def test_all_inputs_share_the_intro(self):
        result = self.detect(batch(10), required=10)
        self.assertEqual(result.members, tuple(range(10)))
        self.assertEqual(result.seconds, 3)

    def test_outlier_sharing_a_shorter_logo_does_not_truncate_majority(self):
        prefix = [texture()] * 8
        outlier = movie(99, prefix + [solid(90)] * 40)
        self.assertEqual(self.detect([outlier] + batch()).seconds, 3)

    def test_common_blank_leader_does_not_select_the_wrong_reference(self):
        prefix = [solid(0)] * 8 + [texture()] * 24
        outlier = movie(99, [solid(0)] * 8 + [solid(90)] * 48)
        result = self.detect([outlier] + batch(prefix=prefix))
        self.assertEqual(result.seconds, 4)
        self.assertNotIn(0, result.members)

    def test_isolated_mismatch_is_tolerated(self):
        movies = batch()
        frames = list(movies[0].frames)
        frames[10] = solid(250)
        movies[0] = movie(0, frames)
        self.assertEqual(self.detect(movies).seconds, 3)

    def test_failed_input_does_not_shrink_the_denominator(self):
        result = self.detect([None] * 2 + batch(8))
        self.assertIsNone(result.seconds)
        self.assertIn("Not enough", result.reason)

    def test_one_failed_input_can_coexist_with_a_ninety_percent_majority(self):
        result = self.detect([None] + batch())
        self.assertEqual(result.members, tuple(range(1, 10)))

    def test_blank_or_solid_only_prefix_is_not_evidence(self):
        for value in (0, 127, 255):
            with self.subTest(value=value):
                result = self.detect(batch(prefix=[solid(value)] * 24))
                self.assertIsNone(result.seconds)

    def test_small_logo_is_not_equivalent_to_blank(self):
        pixels = bytearray(app.FRAME_BYTES)
        pixels[90:120] = bytes([230]) * 30
        logo = app.Frame.from_rgb(bytes(pixels))
        self.assertFalse(app.similar(logo, solid(0), 0.1))

    def test_minor_encoding_changes_are_similar(self):
        raw = bytes(min(255, value + 2) for value in texture().rgb)
        self.assertTrue(app.similar(texture(), app.Frame.from_rgb(raw), 0.1))

    def test_custom_tolerance_handles_differently_letterboxed_intros(self):
        def logo(rows):
            pixels = bytearray(app.FRAME_BYTES)
            for row in rows:
                for column in range(4, 12):
                    value = 190 if column % 2 else 120
                    offset = (row * app.WIDTH + column) * 3
                    pixels[offset:offset + 3] = bytes((value, value // 2, 0))
            return app.Frame.from_rgb(bytes(pixels))

        tall, thin = logo((3, 4, 5)), logo((4,))
        movies = [
            movie(index, [tall if index < 6 else thin] * 24 + [solid(index * 28)] * 24)
            for index in range(9)
        ] + [movie(99, [solid(100)] * 48)]
        self.assertIsNone(app.detect(movies, 9, 8, 1, 0.1).seconds)
        result = app.detect(movies, 9, 8, 1, 0.25)
        self.assertEqual(result.seconds, 3)
        self.assertEqual(result.members, tuple(range(9)))

    def test_identical_entire_samples_have_no_known_intro_boundary(self):
        movies = [movie(i, [texture()] * 48) for i in range(10)]
        result = self.detect(movies)
        self.assertIsNone(result.seconds)
        self.assertIn("No ending", result.reason)

    def test_end_of_file_is_not_a_visual_disagreement(self):
        movies = batch()
        movies[0] = movie(0, [texture()] * 24)
        result = self.detect(movies)
        self.assertIsNone(result.seconds)
        self.assertIn("file's end", result.reason)

    def test_opening_must_be_long_enough(self):
        self.assertIsNone(self.detect(batch(prefix=[texture()] * 4)).seconds)

    def test_rotating_majorities_do_not_become_one_shared_intro(self):
        movies = []
        for index in range(10):
            frames = [
                solid(220) if step % 10 == index else texture()
                for step in range(48)
            ]
            movies.append(movie(index, frames))
        self.assertIsNone(self.detect(movies, min_intro=2).seconds)

    def test_linear_number_of_frame_comparisons(self):
        movies = batch(900) + [movie(i, [solid(30)] * 48) for i in range(100)]
        with patch.object(app, "similar", wraps=app.similar) as comparisons:
            result = self.detect(movies, required=900)
        self.assertEqual(len(result.members), 900)
        self.assertLessEqual(comparisons.call_count, 3 * 1000 * 48)

    def test_incomplete_frame_is_an_error(self):
        with self.assertRaises(app.IntrocutError):
            app.Frame.from_rgb(b"\0")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="introcut-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.mp4"
        self.other = self.root / "other.mp4"
        self.source.write_bytes(b"original-video")
        self.other.write_bytes(b"other-video")
        self.key = app.Keyframe(2, 1.88, "SHA256:expected")

    def input_movie(self):
        return app.Movie(self.source, 0, 0, (), app.snapshot(self.source))

    def remux(self, command):
        target = Path(command[-1])
        self.assertEqual(target.parent.parent, self.root)
        target.write_bytes(b"trimmed-video")
        self.assertEqual(command[command.index("-c") + 1], "copy")
        self.assertIn("-copyts", command)
        self.assertIn("-seek_timestamp", command)
        seeks = [command[i + 1] for i, value in enumerate(command) if value == "-ss"]
        self.assertEqual(seeks, ["2.000000000", "1.879999000"])
        return b""

    def valid_packets(self):
        return [{"flags": "K__", "data_hash": self.key.data_hash}]

    def assert_no_temporaries(self):
        self.assertEqual(list(self.root.glob(".introcut-*")), [])

    def test_paths_and_hardlinks_are_deduplicated(self):
        alias = self.root / "alias.mp4"
        os.link(self.source, alias)
        paths = app.collect_inputs([str(self.source), str(alias), str(self.other)], None, False)
        self.assertEqual(paths, [self.source, self.other])

    def test_quoted_glob_expands_inside_python(self):
        result = app.collect_inputs([str(self.root / "*.mp4")], None, False)
        self.assertEqual(result, [self.other, self.source])

    def test_literal_existing_filename_with_glob_characters_is_preserved(self):
        literal = self.root / "[video]*.mp4"
        literal.write_bytes(b"video")
        result = app.collect_inputs([str(literal), str(self.source)], None, False)
        self.assertEqual(result, [literal, self.source])

    def test_unmatched_glob_is_an_error(self):
        with self.assertRaisesRegex(app.IntrocutError, "No files match"):
            app.collect_inputs([str(self.root / "missing-*.mkv")], None, False)

    def test_relative_paths_in_list_are_relative_to_the_list(self):
        listing = self.root / "files.txt"
        listing.write_text("source.mp4\nother.mp4\n", encoding="utf-8")
        self.assertEqual(app.collect_inputs([], str(listing), False), [self.source, self.other])

    def test_nul_list_preserves_newlines_and_whitespace(self):
        strange = self.root / " a\nb .mkv"
        strange.write_bytes(b"video")
        listing = self.root / "files.list"
        listing.write_bytes(b"source.mp4\0 a\nb .mkv\0")
        self.assertEqual(app.collect_inputs([], str(listing), True), [self.source, strange])

    def test_stdin_list_uses_working_directory(self):
        data = os.fsencode(self.source) + b"\0" + os.fsencode(self.other) + b"\0"
        with patch.object(app.sys, "stdin", io.TextIOWrapper(io.BytesIO(data))):
            self.assertEqual(app.collect_inputs([], "-", True), [self.source, self.other])

    def test_missing_input_is_not_silently_ignored(self):
        with self.assertRaises(FileNotFoundError):
            app.collect_inputs([str(self.source), str(self.root / "absent.mp4")], None, False)

    def test_requires_two_distinct_files(self):
        with self.assertRaises(app.IntrocutError):
            app.collect_inputs([str(self.source), str(self.source)], None, False)

    def test_preflight_rejects_basename_collisions_before_creating_output_dir(self):
        folder = self.root / "out"
        plans = [
            app.FilePlan("/a/same.mp4", status="would_cut"),
            app.FilePlan("/b/same.mp4", status="would_cut"),
        ]
        with self.assertRaisesRegex(app.IntrocutError, "colliding"):
            app.preflight_outputs(plans, [], folder)
        self.assertFalse(folder.exists())

    def test_preflight_rejects_original_output_and_dangling_symlink(self):
        plans = [app.FilePlan(str(self.source), status="would_cut")]
        with self.assertRaisesRegex(app.IntrocutError, "overwrite"):
            app.preflight_outputs(plans, [self.source], self.root)
        folder = self.root / "out"
        folder.mkdir()
        (folder / self.source.name).symlink_to(folder / "absent")
        with self.assertRaisesRegex(app.IntrocutError, "overwrite"):
            app.preflight_outputs(plans, [self.source], folder)

    def test_atomic_new_file_copy_keeps_source(self):
        output = self.root / "trimmed.mp4"
        with patch.object(app, "run_tool", side_effect=self.remux), patch.object(
            app, "video_packets", return_value=self.valid_packets()
        ):
            app.copy_movie(self.input_movie(), self.key, output)
        self.assertEqual(output.read_bytes(), b"trimmed-video")
        self.assertEqual(self.source.read_bytes(), b"original-video")
        self.assert_no_temporaries()

    def test_overwrite_uses_adjacent_temp_and_preserves_attributes(self):
        self.source.chmod(0o640)
        before = self.source.stat()
        with patch.dict(os.environ, {"TMPDIR": str(self.root / "nonexistent-tmp")}), patch.object(
            app, "run_tool", side_effect=self.remux
        ), patch.object(app, "video_packets", return_value=self.valid_packets()):
            app.copy_movie(self.input_movie(), self.key, self.source, overwrite=True)
        self.assertEqual(self.source.read_bytes(), b"trimmed-video")
        self.assertEqual(self.source.stat().st_mode, before.st_mode)
        self.assertEqual(self.source.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertNotEqual(self.source.stat().st_ino, before.st_ino)
        self.assert_no_temporaries()

    def test_bad_keyframe_never_replaces_original(self):
        before = app.snapshot(self.source)
        with patch.object(app, "run_tool", side_effect=self.remux), patch.object(
            app, "video_packets", return_value=[{"flags": "K__", "data_hash": "wrong"}]
        ):
            with self.assertRaisesRegex(app.IntrocutError, "not published"):
                app.copy_movie(self.input_movie(), self.key, self.source, overwrite=True)
        self.assertEqual(app.snapshot(self.source), before)
        self.assertEqual(self.source.read_bytes(), b"original-video")
        self.assert_no_temporaries()

    def test_discarded_first_keyframe_never_replaces_original(self):
        with patch.object(app, "run_tool", side_effect=self.remux), patch.object(
            app, "video_packets", return_value=[{"flags": "KD_", "data_hash": self.key.data_hash}]
        ):
            with self.assertRaisesRegex(app.IntrocutError, "not published"):
                app.copy_movie(self.input_movie(), self.key, self.source, overwrite=True)
        self.assertEqual(self.source.read_bytes(), b"original-video")
        self.assert_no_temporaries()

    def test_chapter_metadata_normalizes_start_and_escapes_values(self):
        source = app.Movie(
            self.source, 0, 10, (), app.snapshot(self.source),
            (app.Chapter(0, 3, {"title": "a=b;#\\\nnext"}),), (2,),
        )
        text = app.chapter_metadata(source)
        self.assertIn("START=10000000\nEND=13000000\n", text)
        self.assertIn("title=a\\=b\\;\\#\\\\\\\nnext\n", text)

    def test_failed_copy_cleans_partial_file_and_preserves_original(self):
        def fail(command):
            self.remux(command)
            raise app.IntrocutError("disk full")

        with patch.object(app, "run_tool", side_effect=fail):
            with self.assertRaisesRegex(app.IntrocutError, "disk full"):
                app.copy_movie(self.input_movie(), self.key, self.source, overwrite=True)
        self.assertEqual(self.source.read_bytes(), b"original-video")
        self.assert_no_temporaries()

    def test_changed_input_is_not_replaced(self):
        source = self.input_movie()
        self.source.write_bytes(b"changed by another process")
        with patch.object(app, "run_tool") as tool:
            with self.assertRaisesRegex(app.IntrocutError, "changed"):
                app.copy_movie(source, self.key, self.source, overwrite=True)
        tool.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b"changed by another process")

    def test_change_during_copy_is_not_replaced(self):
        def change(command):
            self.remux(command)
            self.source.write_bytes(b"concurrent edit")
            return b""

        with patch.object(app, "run_tool", side_effect=change), patch.object(
            app, "video_packets", return_value=self.valid_packets()
        ):
            with self.assertRaisesRegex(app.IntrocutError, "changed"):
                app.copy_movie(self.input_movie(), self.key, self.source, overwrite=True)
        self.assertEqual(self.source.read_bytes(), b"concurrent edit")
        self.assert_no_temporaries()

    def test_concurrent_destination_creation_does_not_overwrite(self):
        output = self.root / "trimmed.mp4"
        output.write_bytes(b"someone else's output")
        with patch.object(app, "run_tool", side_effect=self.remux), patch.object(
            app, "video_packets", return_value=self.valid_packets()
        ):
            with self.assertRaises(FileExistsError):
                app.copy_movie(self.input_movie(), self.key, output)
        self.assertEqual(output.read_bytes(), b"someone else's output")
        self.assert_no_temporaries()

    def test_copy_interruption_cleans_temporary_without_replacement(self):
        def interrupt(command):
            self.remux(command)
            raise KeyboardInterrupt

        with patch.object(app, "run_tool", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                app.copy_movie(self.input_movie(), self.key, self.source, overwrite=True)
        self.assertEqual(self.source.read_bytes(), b"original-video")
        self.assert_no_temporaries()

    def test_interrupted_batch_reports_completed_current_and_pending_files(self):
        third = self.root / "third.mp4"
        third.write_bytes(b"third-video")
        output = io.StringIO()
        detection = app.Detection((0, 1, 2), 3, 2.875, 3.125, "found")

        def analyze(path, scan_seconds, sample_rate):
            return app.Movie(path, 0, 0, (), app.snapshot(path))

        with patch.object(app.shutil, "which", return_value="/tool"), patch.object(
            app, "analyze", side_effect=analyze
        ), patch.object(app, "detect", return_value=detection), patch.object(
            app, "find_keyframe", return_value=self.key
        ), patch.object(
            app, "copy_movie", side_effect=[None, KeyboardInterrupt]
        ) as copier, redirect_stdout(output):
            code = app.main([
                "--overwrite", "--json", str(self.source), str(self.other), str(third)
            ])
        self.assertEqual(code, 130)
        self.assertEqual(copier.call_count, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(
            [plan["status"] for plan in report["files"]], ["cut", "interrupted", "would_cut"]
        )
        self.assertIn("Not processed", report["files"][2]["reason"])

    def test_keyframe_policy_uses_boundary_and_absolute_timestamps(self):
        source = app.Movie(self.source, 0, 10, (), app.snapshot(self.source))
        detection = app.Detection((0, 1), 3, 2.875, 3.125)
        packets = [
            {"pts_time": str(pts), "dts_time": str(pts - 0.12),
             "flags": "K__", "data_hash": f"hash-{pts}"}
            for pts in (10, 12, 13, 14, 16)
        ]
        with patch.object(app, "video_packets", return_value=packets):
            self.assertEqual(app.find_keyframe(source, detection, "previous", 30).pts, 12)
            self.assertEqual(app.find_keyframe(source, detection, "next", 30).pts, 14)
            self.assertIsNone(app.find_keyframe(source, detection, "next", 0.5))

    def test_no_useful_keyframe_is_not_a_zero_second_cut(self):
        packets = [{"pts_time": "0", "flags": "K__", "data_hash": "initial"}]
        with patch.object(app, "video_packets", return_value=packets):
            self.assertIsNone(app.find_keyframe(
                self.input_movie(), app.Detection((0, 1), 3, 2.875, 3.125), "previous", 30
            ))


class CLITests(unittest.TestCase):
    def test_overwrite_alias_and_modes(self):
        parser = app.make_parser()
        self.assertTrue(parser.parse_args(["-overwrite"]).overwrite)
        self.assertTrue(parser.parse_args(["--overwrite", "--dry-run"]).dry_run)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            parser.parse_args(["--overwrite", "--output-dir", "out"])
        self.assertEqual(result.exception.code, 2)

    def test_invalid_numeric_options_and_null_without_list(self):
        for arguments in (
            ["--threshold", "nan"], ["--sample-rate", "inf"],
            ["--min-fraction", "0.5"], ["--workers", "0"],
            ["--scan-seconds", "1"], ["--null"],
        ):
            with self.subTest(arguments=arguments):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                    app.main(arguments)
                self.assertEqual(result.exception.code, 2)

    def test_json_report_shape(self):
        args = app.make_parser().parse_args(["--json", "--overwrite", "--dry-run"])
        plan = app.FilePlan("input.mp4", "would_cut", True, 2, 2, 1, 0, "input.mp4")
        result = io.StringIO()
        with redirect_stdout(result):
            app.report([plan], app.Detection((0,), 3, 2.875, 3.125, "found"), 1, args, True)
        data = json.loads(result.getvalue())
        self.assertEqual(set(data), {
            "schema_version", "dry_run", "overwrite", "input_count", "required_matches",
            "sample_step_seconds", "keyframe_policy", "intro", "message", "files",
        })
        self.assertTrue(data["dry_run"])
        self.assertTrue(data["overwrite"])
        self.assertEqual(data["files"][0]["cut_seconds"], 2)
        self.assertEqual(data["intro"]["boundary_min_seconds"], 2.875)

    def test_tool_errors_are_not_silenced_even_with_zero_exit_code(self):
        result = app.subprocess.CompletedProcess(["ffmpeg"], 0, b"", b"decode error")
        with patch.object(app.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(app.IntrocutError, "decode error"):
                app.run_tool(["ffmpeg"])


if __name__ == "__main__":
    unittest.main()

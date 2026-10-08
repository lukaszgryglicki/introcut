# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
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


def movie(
    index: int, frames: list[app.Frame], audio: list[app.Frame] | None = None,
) -> app.Movie:
    return app.Movie(
        Path(f"/video-{index}.mp4"), 0, 0, tuple(frames), (0, 0, 0, 0),
        audio=tuple(audio or ()),
    )


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


def sound(position: int) -> app.Frame:
    values = [
        200 if position <= index < position + 5
        else 60 if position + 20 <= index < position + 24 else 8
        for index in range(app.PIXELS)
    ]
    return app.Frame.from_rgb(bytes(value for value in values for _ in range(3)))


def audio_batch(count: int = 7, prefix: list[app.Frame] | None = None) -> list[app.Movie]:
    if prefix is None:
        prefix = [sound(12 + 12 * (step // 4)) for step in range(24)]
    return [
        movie(
            index, [solid(index * 25)] * (len(prefix) + 24),
            prefix + [sound(80 + index * 7)] * 24,
        )
        for index in range(count)
    ]


class AudioDetectionTests(unittest.TestCase):
    def detect(self, movies, required=7):
        return app.detect(movies, required, 8, 1, 0.1, audio=True)

    def test_seventy_percent_audio_majority_skips_outliers(self):
        outliers = [movie(i, [solid(0)] * 48, [sound(135)] * 48) for i in range(3)]
        result = self.detect(outliers + audio_batch())
        self.assertEqual(result.members, tuple(range(3, 10)))
        self.assertEqual(result.seconds, 3)
        self.assertEqual(result.evidence, ("audio",))

    def test_audio_gain_and_offset_do_not_change_the_fingerprint_match(self):
        original = sound(15)
        quieter = app.Frame.from_rgb(bytes(value // 2 + 20 for value in original.rgb))
        self.assertTrue(app.similar_audio(original, quieter, 0.1))
        self.assertFalse(app.similar_audio(original, sound(90), 0.1))
        self.assertFalse(app.similar_audio(original, solid(0), 0.1))

    def test_silence_and_steady_tones_are_not_distinctive(self):
        for frame in (solid(0), solid(100), sound(15)):
            with self.subTest(frame=frame.mean):
                self.assertIsNone(self.detect(audio_batch(prefix=[frame] * 24)).seconds)

    def test_internal_quiet_gap_does_not_truncate_the_intro(self):
        prefix = [sound(12 + 12 * (step // 4)) for step in range(24)]
        prefix[8:14] = [solid(0)] * 6
        self.assertEqual(self.detect(audio_batch(prefix=prefix)).seconds, 3)

    def test_trailing_shared_silence_does_not_extend_the_intro(self):
        movies = [
            movie(i, list(item.frames), list(item.audio[:24]) + [solid(0)] * 24)
            for i, item in enumerate(audio_batch())
        ]
        self.assertEqual(self.detect(movies).seconds, 3)

    def test_padding_past_video_eof_is_not_an_audio_boundary(self):
        movies = [
            movie(i, list(item.frames[:24]), list(item.audio[:24]) + [solid(0)] * 24)
            for i, item in enumerate(audio_batch())
        ]
        self.assertIsNone(self.detect(movies).seconds)

    def test_identical_full_soundtracks_have_no_observed_boundary(self):
        prefix = list(audio_batch()[0].audio[:24]) * 2
        movies = [movie(i, [solid(i * 25)] * 48, prefix) for i in range(7)]
        self.assertIsNone(self.detect(movies).seconds)

    def test_missing_audio_does_not_shrink_the_quorum(self):
        movies = audio_batch(6) + [movie(99, [solid(0)] * 48)] * 4
        self.assertIsNone(self.detect(movies).seconds)

    def test_shorter_audio_outlier_does_not_truncate_the_majority(self):
        prefix = list(audio_batch()[0].audio[:8])
        outlier = movie(99, [solid(0)] * 48, prefix + [sound(135)] * 40)
        result = self.detect([outlier] + audio_batch())
        self.assertEqual(result.members, tuple(range(1, 8)))
        self.assertEqual(result.seconds, 3)

    def test_audio_spectrum_columns_preserve_the_sampling_grid(self):
        raw = bytes(value for _ in range(app.PIXELS) for value in (20, 100))
        with patch.object(app, "run_tool", return_value=raw):
            frames = app.audio_fingerprints(Path("input.mp4"), 1, 0, 0.25, 8)
        self.assertEqual([frame.luma for frame in frames], [bytes([20]) * app.PIXELS, bytes([100]) * app.PIXELS])

    def test_fractional_scan_duration_pads_to_complete_sample_intervals(self):
        with patch.object(app, "run_tool", return_value=bytes(9 * app.PIXELS)) as tool:
            frames = app.audio_fingerprints(Path("input.mp4"), 2, 10.25, 1.03, 8)
        self.assertEqual(len(frames), 9)
        command = tool.call_args.args[0]
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("[0:2]", filters)
        self.assertIn("PTS-(10.250000000)/TB", filters)
        self.assertIn("apad=whole_dur=1.125000000", filters)
        self.assertIn("aresample=8000:async=1:first_pts=0", filters)
        self.assertIn("-copyts", command)
        self.assertNotIn("-ss", command)

    def test_incomplete_audio_spectrum_is_an_error(self):
        with patch.object(app, "run_tool", return_value=b""):
            with self.assertRaisesRegex(app.IntrocutError, "audio spectrum"):
                app.audio_fingerprints(Path("input.mp4"), 1, 0, 1, 8)


class CombinedDetectionTests(unittest.TestCase):
    def detection(self, members, seconds, cue):
        return app.Detection(
            tuple(members), seconds, seconds - 0.125, seconds + 0.125,
            evidence=(cue,),
        )

    def test_agreeing_cues_merge_members_and_cover_both_boundaries(self):
        video = self.detection(range(7), 3.125, "video")
        audio = self.detection(range(9), 3, "audio")
        result = app.combine_detections(video, audio, 7)
        self.assertEqual(result.members, tuple(range(9)))
        self.assertEqual(result.seconds, 3)
        self.assertEqual((result.boundary_min, result.boundary_max), (2.875, 3.25))
        self.assertEqual(result.evidence, ("video", "audio"))

    def test_shared_background_music_does_not_extend_a_visual_intro(self):
        video = self.detection(range(7), 3, "video")
        audio = self.detection(range(10), 8, "audio")
        self.assertIs(app.combine_detections(video, audio, 7), video)

    def test_unrelated_majorities_are_not_merged(self):
        video = self.detection(range(7), 3, "video")
        audio = self.detection(range(3, 10), 3, "audio")
        self.assertIs(app.combine_detections(video, audio, 7), video)

    def test_either_cue_can_work_without_a_reliable_other_cue(self):
        video = self.detection(range(7), 3, "video")
        audio = self.detection(range(7), 3, "audio")
        self.assertIs(app.combine_detections(video, app.Detection(), 7), video)
        self.assertIs(app.combine_detections(app.Detection(), audio, 7), audio)


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

    def test_video_without_audio_still_analyzes(self):
        metadata = {"streams": [{"index": 0, "codec_type": "video", "start_time": "0"}]}
        with patch.object(app, "probe_json", return_value=metadata), patch.object(
            app, "run_tool", return_value=texture().rgb * 8
        ) as tool:
            result = app.analyze(self.source, 1, 8)
        self.assertEqual(len(result.frames), 8)
        self.assertEqual(result.audio, ())
        self.assertEqual(tool.call_count, 1)

    def test_audio_decode_error_is_not_silently_ignored(self):
        metadata = {"streams": [
            {"index": 0, "codec_type": "video", "start_time": "0"},
            {"index": 1, "codec_type": "audio"},
        ]}
        with patch.object(app, "probe_json", return_value=metadata), patch.object(
            app, "run_tool", side_effect=[texture().rgb * 8, app.IntrocutError("audio decode error")]
        ):
            with self.assertRaisesRegex(app.IntrocutError, "audio decode error"):
                app.analyze(self.source, 1, 8)

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
        self.assertEqual(report["summary"]["cut"], 1)
        self.assertEqual(report["summary"]["interrupted"], 1)
        self.assertEqual(report["summary"]["pending"], 1)
        self.assertEqual(report["summary"]["would_cut"], 0)
        self.assertEqual(report["summary"]["cut_seconds_total"], 2)

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


class SavedPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="introcut-plan-unit-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = [self.root / f"input-{index}.mp4" for index in range(4)]
        for index, path in enumerate(self.inputs):
            path.write_bytes(f"original-{index}".encode())
        self.plan = self.root / "saved.json"
        self.key = app.Keyframe(14, 13.88, "SHA256:" + "a" * 64)
        self.detection = app.Detection((0, 1, 2), 3, 2.875, 3.125, "found", ("video",))

    def analyze(self, path, seconds, rate):
        return app.Movie(
            path, 0, 10, (texture(),), app.snapshot(path),
            (app.Chapter(0, 7, {"title": "a=b;#\\\nchapter"}),), (2,), (texture(),),
        )

    def invoke(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors), patch.object(
            app.shutil, "which", return_value="/tool"
        ):
            code = app.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def save(self, extra=(), *, detection=None, analyzer=None, keys=None, inputs=None, dry_run=True):
        with patch.object(app, "analyze", side_effect=analyzer or self.analyze), patch.object(
            app, "detect", return_value=detection if detection is not None else self.detection
        ), patch.object(app, "find_keyframe", return_value=self.key, side_effect=keys):
            return self.invoke([
                *(["--dry-run"] if dry_run else []),
                *(str(path) for path in (self.inputs if inputs is None else inputs)),
                "--save-plan", str(self.plan), "--json", *extra,
            ])

    def apply(self, extra=(), inputs=()):
        with patch.object(app, "analyze", side_effect=AssertionError("analysis repeated")), patch.object(
            app, "detect", side_effect=AssertionError("detection repeated")
        ), patch.object(app, "find_keyframe", side_effect=AssertionError("keyframe search repeated")), patch.object(
            app, "run_tool", side_effect=AssertionError("unexpected media tool")
        ):
            return self.invoke([
                "--apply-plan", str(self.plan), "--json", *extra, *(str(path) for path in inputs),
            ])

    def assert_rejected_before_copy(self, extra=(), inputs=(), message=None):
        folder = self.root / "must-not-create"
        with patch.object(app, "copy_movie") as copier:
            code, output, errors = self.apply(["--output-dir", str(folder), *extra], inputs)
        self.assertEqual(code, 1, (output, errors))
        self.assertEqual(output, "")
        self.assertTrue(errors)
        if message is not None:
            self.assertIn(message, errors)
        copier.assert_not_called()
        self.assertFalse(folder.exists())

    def test_round_trip_restores_settings_and_copy_metadata_without_fingerprints(self):
        self.key = app.Keyframe(12, 11.88, self.key.data_hash)
        before = [app.snapshot(path) for path in self.inputs]
        code, output, errors = self.save(["--sample-rate", "16", "--keyframe", "prev"])
        self.assertEqual((code, errors), (0, ""))
        saved = json.loads(self.plan.read_text())
        self.assertEqual(saved["format"], "introcut-plan")
        self.assertEqual(set(saved["files"][0]["movie"]), {
            "video_index", "video_start", "chapters", "chapter_streams",
        })
        self.assertNotIn("rgb", self.plan.read_text())
        self.assertIsNone(saved["files"][3]["movie"])
        self.assertTrue(all("output" not in entry for entry in saved["files"]))
        self.assertEqual(self.plan.stat().st_mode & 0o777, 0o600)
        code, replay, errors = self.apply(["--keyframe", "previous"])
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(replay), json.loads(output))
        args = app.make_parser().parse_args([])
        _, movies, keys, _, _ = app.load_plan(self.plan, args)
        self.assertEqual(movies[0], app.Movie(
            self.inputs[0], 0, 10, (), before[0],
            (app.Chapter(0, 7, {"title": "a=b;#\\\nchapter"}),), (2,),
        ))
        self.assertEqual(keys[0], self.key)
        self.assertEqual(before, [app.snapshot(path) for path in self.inputs])
        self.assertEqual(list(self.root.glob(".introcut-*")), [])

    def test_same_glob_applies_only_reviewed_matches_in_place(self):
        pattern = str(self.root / "input-*.mp4")
        self.assertEqual(self.save(inputs=[pattern])[0], 0)
        with patch.object(app, "copy_movie") as copier:
            code, output, errors = self.apply(["--overwrite"], [pattern])
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(copier.call_count, 3)
        report = json.loads(output)
        self.assertEqual(report["summary"]["cut"], 3)
        self.assertEqual(report["summary"]["unmatched"], 1)
        for call, source in zip(copier.call_args_list, self.inputs):
            movie, key, destination = call.args
            self.assertEqual((movie.path, key, destination), (source, self.key, source))
            self.assertEqual(call.kwargs, {"overwrite": True})

    def test_default_replay_and_explicit_dry_run_never_copy(self):
        self.assertEqual(self.save(dry_run=False)[0], 0)
        for options in ([], ["--overwrite", "--dry-run"], [
            "--output-dir", str(self.root / "dry-output"), "--dry-run",
        ]):
            with self.subTest(options=options), patch.object(app, "copy_movie") as copier:
                code, output, errors = self.apply(options)
                self.assertEqual((code, errors), (0, ""))
                self.assertTrue(json.loads(output)["dry_run"])
                copier.assert_not_called()
        self.assertFalse((self.root / "dry-output").exists())

    def test_output_mode_is_not_inherited_from_saved_dry_run(self):
        self.assertEqual(self.save(["--overwrite"])[0], 0)
        folder = self.root / "copies"
        with patch.object(app, "copy_movie") as copier:
            code, output, errors = self.apply(["--output-dir", str(folder)])
        self.assertEqual((code, errors), (0, ""))
        self.assertFalse(json.loads(output)["overwrite"])
        for call, source in zip(copier.call_args_list, self.inputs):
            self.assertEqual(call.args[2], folder / source.name)
            self.assertEqual(call.kwargs, {"overwrite": False})

    def test_reordered_inputs_and_nul_lists_keep_saved_order(self):
        unusual = self.root / " quoted '\ninput.mp4"
        self.inputs[1].rename(unusual)
        self.inputs[1] = unusual
        code, original, _ = self.save()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.apply(inputs=reversed(self.inputs))[1]), json.loads(original))
        listing = self.root / "inputs.list"
        listing.write_bytes(b"\0".join(os.fsencode(path) for path in reversed(self.inputs)) + b"\0")
        code, output, errors = self.apply(["--files-from", str(listing), "--null"])
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(json.loads(output), json.loads(original))

    def test_different_input_sets_are_rejected_before_output_creation(self):
        self.assertEqual(self.save()[0], 0)
        extra = self.root / "extra.mp4"
        extra.write_bytes(b"extra")
        for inputs in (self.inputs[:-1], [*self.inputs, extra]):
            with self.subTest(count=len(inputs)):
                self.assert_rejected_before_copy(inputs=inputs, message="Input set differs")

    def test_changed_matching_input_invalidates_entire_plan(self):
        self.assertEqual(self.save()[0], 0)
        self.inputs[2].write_bytes(b"changed")
        self.assert_rejected_before_copy(message="Input changed")

    def test_changed_outlier_also_invalidates_entire_plan(self):
        self.assertEqual(self.save()[0], 0)
        self.inputs[3].write_bytes(b"changed outlier")
        self.assert_rejected_before_copy(message="Input changed")

    def test_nanosecond_mtime_change_invalidates_plan(self):
        self.assertEqual(self.save()[0], 0)
        info = self.inputs[0].stat()
        os.utime(self.inputs[0], ns=(info.st_atime_ns, info.st_mtime_ns + 1))
        self.assert_rejected_before_copy(message="Input changed")

    def test_replaced_inode_is_stale_even_with_identical_bytes_and_mtime(self):
        self.assertEqual(self.save()[0], 0)
        original = self.inputs[0]
        info = original.stat()
        replacement = self.root / "replacement"
        replacement.write_bytes(original.read_bytes())
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        os.replace(replacement, original)
        self.assert_rejected_before_copy(message="Input changed")

    def test_missing_input_is_not_silently_skipped(self):
        self.assertEqual(self.save()[0], 0)
        self.inputs[3].unlink()
        self.assert_rejected_before_copy()

    def test_retargeted_source_symlink_is_rejected(self):
        self.assertEqual(self.save()[0], 0)
        self.inputs[0].unlink()
        self.inputs[0].symlink_to(self.inputs[1])
        self.assert_rejected_before_copy(message="Input changed")

    def test_conflicting_explicit_settings_and_abbreviations_are_rejected(self):
        self.assertEqual(self.save()[0], 0)
        for option, value in (
            ("--min-fraction", "0.8"), ("--sample-rate", "16"), ("--threshold", "0.2"),
            ("--scan-seconds", "60"), ("--min-intro", "2"), ("--keyframe-lookahead", "60"),
            ("--keyframe", "prev"), ("--thr", "0.2"),
        ):
            with self.subTest(option=option):
                self.assert_rejected_before_copy([option, value], message="Cannot change")
        code, _, errors = self.apply(["--min-fraction", "0.7", "--keyframe", "next"])
        self.assertEqual((code, errors), (0, ""))

    def test_saved_short_scan_does_not_conflict_with_implicit_default_min_intro(self):
        self.detection = app.Detection((0, 1, 2), 0.375, 0.25, 0.5, "found", ("video",))
        self.assertEqual(self.save(["--min-intro", "0.1", "--scan-seconds", "1"])[0], 0)
        code, _, errors = self.apply(["--scan-seconds", "1"])
        self.assertEqual((code, errors), (0, ""))

    def test_no_intro_plan_replays_without_copies(self):
        code, original, errors = self.save(detection=app.Detection(reason="No intro"))
        self.assertEqual((code, errors), (3, ""))
        with patch.object(app, "copy_movie") as copier:
            code, output, errors = self.apply(["--overwrite"])
        self.assertEqual((code, errors), (3, ""))
        self.assertEqual(json.loads(output)["files"], json.loads(original)["files"])
        copier.assert_not_called()

    def test_errors_and_no_keyframes_are_retained_without_retrying(self):
        def analyze(path, seconds, rate):
            if path == self.inputs[3]:
                raise app.IntrocutError("synthetic decode failure")
            return self.analyze(path, seconds, rate)

        code, original, errors = self.save(
            analyzer=analyze, keys=[None, self.key, app.IntrocutError("synthetic keyframe failure")],
        )
        self.assertEqual((code, errors), (1, ""))
        code, output, errors = self.apply()
        self.assertEqual((code, errors), (1, ""))
        self.assertEqual(json.loads(output), json.loads(original))
        with patch.object(app, "copy_movie") as copier:
            code, output, errors = self.apply(["--overwrite"])
        self.assertEqual((code, errors), (1, ""))
        self.assertEqual(copier.call_count, 1)
        self.assertEqual(json.loads(output)["summary"]["errors"], 2)

    def test_invalid_plan_values_never_reach_copy_or_media_tools(self):
        self.assertEqual(self.save()[0], 0)
        original = json.loads(self.plan.read_text())
        changes = (
            (("format",), "other"), (("schema_version",), 2), (("schema_version",), True),
            (("settings", "sample_rate"), 0), (("settings", "sample_rate"), float("inf")),
            (("settings", "min_fraction"), True), (("settings", "keyframe"), "prev"),
            (("settings", "scan_seconds"), 0.1), (("required_matches",), 2),
            (("files",), []), (("files", 1), original["files"][0]),
            (("files", 0, "path"), "relative.mp4"), (("files", 0, "path"), "bad\0path"),
            (("files", 0, "snapshot"), [1, 2, 3]), (("files", 0, "snapshot"), [1, 2, True, 4]),
            (("files", 0, "status"), "cut"), (("files", 0, "status"), "unmatched"),
            (("files", 3, "status"), "would_cut"), (("files", 0, "reason"), []),
            (("files", 0, "movie"), None), (("files", 0, "movie", "video_index"), True),
            (("files", 0, "movie", "video_start"), float("nan")),
            (("files", 0, "movie", "chapter_streams"), [2, 2]),
            (("files", 0, "movie", "chapter_streams"), [0]),
            (("files", 0, "movie", "chapters", 0, "end"), -1),
            (("files", 0, "movie", "chapters", 0, "tags"), {"title": 7}),
            (("files", 0, "keyframe", "dts"), None),
            (("files", 0, "keyframe", "data_hash"), "invalid"),
            (("files", 0, "keyframe", "pts"), 10), (("files", 0, "keyframe", "pts"), 12),
            (("files", 0, "keyframe", "pts"), 100),
            (("detection", "members"), [0, 0, 1]), (("detection", "members"), [0, 1, 4]),
            (("detection", "seconds"), None), (("detection", "boundary_max"), 2),
            (("detection", "boundary_min"), 0.5), (("detection", "boundary_max"), 31),
            (("detection", "evidence"), ["unknown"]),
        )
        for location, value in changes:
            with self.subTest(location=location, value=value):
                data = deepcopy(original)
                parent = data
                for component in location[:-1]:
                    parent = parent[component]
                parent[location[-1]] = value
                self.plan.write_text(json.dumps(data), encoding="utf-8")
                self.assert_rejected_before_copy()

    def test_truncated_duplicate_unknown_and_report_json_are_rejected(self):
        code, report, _ = self.save()
        self.assertEqual(code, 0)
        text = self.plan.read_text()
        unknown = json.loads(text)
        unknown["files"][0]["output"] = str(self.root / "unreviewed.mp4")
        for raw in (
            b"{", b"\xff", b"[]", report.encode(), json.dumps(unknown).encode(),
            text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1').encode(),
        ):
            with self.subTest(raw=raw[:30]):
                self.plan.write_bytes(raw)
                self.assert_rejected_before_copy()

    def test_save_rejects_existing_files_and_aliases_before_analysis(self):
        alias = self.root / "alias.json"
        os.link(self.inputs[0], alias)
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "absent")
        self.plan.write_text("existing plan", encoding="utf-8")
        before = {path: path.read_bytes() for path in self.inputs}
        for target in (self.inputs[0], alias, dangling, self.plan):
            with self.subTest(target=target.name), patch.object(
                app, "analyze", side_effect=AssertionError("analysis should not start")
            ):
                code, _, errors = self.invoke([
                    "--save-plan", str(target), *(str(path) for path in self.inputs),
                ])
                self.assertEqual(code, 1)
                self.assertIn("Refusing to overwrite", errors)
        self.assertEqual(before, {path: path.read_bytes() for path in self.inputs})
        self.assertEqual(self.plan.read_text(), "existing plan")

    def test_changed_input_during_analysis_prevents_plan_publication(self):
        def analyze(path, seconds, rate):
            if path == self.inputs[3]:
                path.write_bytes(b"changed while waiting")
            return self.analyze(path, seconds, rate)

        code, _, errors = self.save(analyzer=analyze)
        self.assertEqual(code, 1)
        self.assertIn("Input changed", errors)
        self.assertFalse(self.plan.exists())
        self.assertEqual(list(self.root.glob(".introcut-*")), [])

    def test_concurrent_plan_creation_is_not_overwritten(self):
        link = os.link

        def publish(source, destination):
            Path(destination).write_bytes(b"created concurrently")
            link(source, destination)

        with patch.object(app.os, "link", side_effect=publish):
            code, _, errors = self.save()
        self.assertEqual(code, 1)
        self.assertTrue(errors)
        self.assertEqual(self.plan.read_bytes(), b"created concurrently")
        self.assertEqual(list(self.root.glob(".introcut-*")), [])

    def test_failed_plan_write_cleans_temporary_files(self):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            code, _, errors = self.save()
        self.assertEqual(code, 1)
        self.assertIn("disk full", errors)
        self.assertFalse(self.plan.exists())
        self.assertEqual(list(self.root.glob(".introcut-*")), [])

    def test_plan_cannot_occupy_a_planned_video_output_path(self):
        folder = self.root / "outputs"
        folder.mkdir()
        self.plan = folder / self.inputs[0].name
        code, _, errors = self.save(["--output-dir", str(folder)])
        self.assertEqual(code, 1)
        self.assertIn("conflicts", errors)
        self.assertEqual(list(folder.iterdir()), [])

    def test_save_requires_dry_run_and_save_apply_are_mutually_exclusive(self):
        for arguments in (
            ["--save-plan", str(self.plan), "--overwrite"],
            ["--save-plan", str(self.plan), "--output-dir", str(self.root / "out")],
            ["--save-plan", str(self.plan), "--apply-plan", str(self.plan)],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as result:
                self.invoke(arguments)
            self.assertEqual(result.exception.code, 2)
        self.assertFalse(self.plan.exists())

    def test_apply_interruption_uses_existing_partial_summary(self):
        self.assertEqual(self.save()[0], 0)
        with patch.object(app, "copy_movie", side_effect=[None, KeyboardInterrupt]) as copier:
            code, output, errors = self.apply(["--overwrite"])
        self.assertEqual((code, errors), (130, ""))
        self.assertEqual(copier.call_count, 2)
        summary = json.loads(output)["summary"]
        self.assertEqual((summary["cut"], summary["interrupted"], summary["pending"]), (1, 1, 1))
        self.assertEqual(summary["cut_seconds_total"], 4)


class CLITests(unittest.TestCase):
    def test_next_keyframe_is_default_and_previous_spellings_are_compatible(self):
        parser = app.make_parser()
        self.assertEqual(parser.parse_args([]).keyframe, "next")
        self.assertEqual(parser.parse_args(["--keyframe", "next"]).keyframe, "next")
        for spelling in ("prev", "previous"):
            self.assertEqual(parser.parse_args(["--keyframe", spelling]).keyframe, "previous")

    def test_default_majority_is_seventy_percent_and_remains_configurable(self):
        parser = app.make_parser()
        self.assertEqual(parser.parse_args([]).min_fraction, 0.7)
        self.assertEqual(parser.parse_args(["--min-fraction", "0.9"]).min_fraction, 0.9)

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
        args = app.make_parser().parse_args([
            "--json", "--overwrite", "--dry-run", "--keyframe", "prev",
        ])
        plan = app.FilePlan("input.mp4", "would_cut", True, 2, 2, 1, 0, "input.mp4")
        result = io.StringIO()
        with redirect_stdout(result):
            app.report([plan], app.Detection((0,), 3, 2.875, 3.125, "found"), 1, args, True)
        data = json.loads(result.getvalue())
        self.assertEqual(set(data), {
            "schema_version", "dry_run", "overwrite", "input_count", "required_matches",
            "sample_step_seconds", "keyframe_policy", "intro", "message", "files", "summary",
        })
        self.assertTrue(data["dry_run"])
        self.assertTrue(data["overwrite"])
        self.assertEqual(data["files"][0]["cut_seconds"], 2)
        self.assertEqual(data["intro"]["boundary_min_seconds"], 2.875)
        self.assertEqual(data["keyframe_policy"], "previous")
        self.assertEqual(data["summary"]["would_cut"], 1)
        self.assertEqual(data["summary"]["cut"], 0)
        self.assertEqual(data["summary"]["cut_seconds_total"], 2)
        self.assertEqual(data["summary"]["estimated_intro_remaining_seconds_total"], 1)

    def test_tool_errors_are_not_silenced_even_with_zero_exit_code(self):
        result = app.subprocess.CompletedProcess(["ffmpeg"], 0, b"", b"decode error")
        with patch.object(app.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(app.IntrocutError, "decode error"):
                app.run_tool(["ffmpeg"])


class SummaryTests(unittest.TestCase):
    def planned(self, name, seconds, status="would_cut"):
        return app.FilePlan(
            name, status, True, seconds, seconds, max(0, 3 - seconds), max(0, seconds - 3),
        )

    def dry_plans(self):
        return [
            self.planned("a.mp4", 4),
            self.planned("b.mp4", 5.5),
            app.FilePlan("no-key.mp4", "no_keyframe", True),
            app.FilePlan("outlier.mp4"),
            app.FilePlan("bad.mp4", "error", reason="decode failed"),
        ]

    def test_dry_run_counts_and_fractional_cut_totals(self):
        summary = app.summarize(self.dry_plans(), True)
        self.assertEqual(summary.total_files, 5)
        self.assertEqual(summary.matched_files, 3)
        self.assertEqual((summary.would_cut, summary.cut, summary.pending), (2, 0, 0))
        self.assertEqual((summary.unmatched, summary.no_keyframe, summary.errors), (1, 1, 1))
        self.assertEqual(summary.interrupted, 0)
        self.assertEqual(summary.cut_seconds_total, 9.5)
        self.assertEqual((summary.cut_seconds_min, summary.cut_seconds_max), (4, 5.5))
        self.assertEqual(summary.estimated_intro_remaining_seconds_total, 0)
        self.assertEqual(summary.estimated_content_removed_seconds_total, 3.5)

    def test_apply_totals_exclude_failed_interrupted_and_pending_copies(self):
        plans = [
            self.planned("done.mp4", 4, "cut"),
            self.planned("pending.mp4", 9),
            self.planned("failed.mp4", 10, "error"),
            self.planned("interrupted.mp4", 8, "interrupted"),
            app.FilePlan("no-key.mp4", "no_keyframe", True),
            app.FilePlan("outlier.mp4"),
        ]
        summary = app.summarize(plans, False)
        self.assertEqual((summary.total_files, summary.matched_files), (6, 5))
        self.assertEqual((summary.cut, summary.pending, summary.would_cut), (1, 1, 0))
        self.assertEqual((summary.errors, summary.interrupted), (1, 1))
        self.assertEqual(summary.cut_seconds_total, 4)
        self.assertEqual((summary.cut_seconds_min, summary.cut_seconds_max), (4, 4))
        self.assertEqual(summary.estimated_content_removed_seconds_total, 1)

    def test_no_cuts_have_zero_totals_and_no_range(self):
        for plans in (
            [],
            [app.FilePlan("outlier.mp4")],
            [app.FilePlan("no-key.mp4", "no_keyframe", True)],
        ):
            with self.subTest(count=len(plans)):
                summary = app.summarize(plans, True)
                self.assertEqual(summary.cut_seconds_total, 0)
                self.assertIsNone(summary.cut_seconds_min)
                self.assertIsNone(summary.cut_seconds_max)
                self.assertEqual(summary.estimated_intro_remaining_seconds_total, 0)
                self.assertEqual(summary.estimated_content_removed_seconds_total, 0)

    def test_text_summary_follows_all_files_and_includes_the_intro(self):
        args = app.make_parser().parse_args(["--dry-run"])
        output = io.StringIO()
        with redirect_stdout(output):
            app.report(
                self.dry_plans(), app.Detection((0, 1, 2), 3, 2.875, 3.125), 3, args, True,
            )
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[-5:], [
            "Summary: DRY RUN; next keyframe; intro ~3.000s.",
            "Files: 5 total; 3 matched; 1 unmatched; 1 no keyframe; 1 errors.",
            "Results: 2 would cut; 0 cut; 0 pending; 0 interrupted.",
            "Planned video cuts: 9.500s total; 4.000..5.500s per file.",
            "Estimated intro left: 0.000s total; extra content removed: 3.500s total.",
        ])
        self.assertTrue(any(line.startswith("ERROR") for line in lines[:-5]))

    def test_apply_text_labels_only_completed_cut_statistics(self):
        args = app.make_parser().parse_args(["--overwrite"])
        output = io.StringIO()
        with redirect_stdout(output):
            app.report([
                self.planned("done.mp4", 4, "cut"),
                self.planned("pending.mp4", 9),
            ], app.Detection((0, 1), 3, 2.875, 3.125), 2, args, False)
        self.assertIn("Results: 0 would cut; 1 cut; 1 pending; 0 interrupted.", output.getvalue())
        self.assertIn("Completed video cuts: 4.000s total;", output.getvalue())
        self.assertNotIn("Planned video cuts:", output.getvalue())

    def test_no_intro_report_still_ends_with_a_summary(self):
        args = app.make_parser().parse_args([])
        output = io.StringIO()
        with redirect_stdout(output):
            app.report([app.FilePlan("outlier.mp4")], app.Detection(reason="not found"), 2, args, True)
        self.assertIn("Summary: DRY RUN; next keyframe; intro not detected.", output.getvalue())
        self.assertNotIn("video cuts:", output.getvalue())
        args.json = True
        output = io.StringIO()
        with redirect_stdout(output):
            app.report([app.FilePlan("outlier.mp4")], app.Detection(reason="not found"), 2, args, True)
        data = json.loads(output.getvalue())
        self.assertIsNone(data["intro"])
        self.assertIsNone(data["summary"]["cut_seconds_min"])
        self.assertEqual(data["summary"]["unmatched"], 1)


if __name__ == "__main__":
    unittest.main()

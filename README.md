# introcut

Find the shared introduction in a batch of videos, then cut matching files
without re-encoding. Designed for HEVC/H.265 + AAC in MP4 or Matroska, including
different resolutions, frame rates, audio sample rates, and channel counts.
Python 3.10+ and `ffmpeg` / `ffprobe` 5+ on `PATH`; no Python runtime dependencies.

## Use

Run directly from this directory (no installation needed):

```sh
python3 introcut.py --dry-run '/videos/*.mp4'
python3 introcut.py --dry-run --json --files-from videos.txt > plan.json
python3 introcut.py --files-from videos.txt --output-dir /videos/trimmed
```

Or replace matching originals in place:

```sh
python3 introcut.py --files-from videos.txt --overwrite --dry-run
python3 introcut.py --files-from videos.txt --overwrite
```

Or install with `python3 -m pip install .` and use `introcut` instead of
`python3 introcut.py`. Without `--output-dir` or `--overwrite`, the default is a
dry run. `--dry-run` always disables copying/replacement, even with either output
mode. `-overwrite` is also accepted. `--overwrite` and `--output-dir` cannot be
combined.

For very large batches, **quote the glob** so Python expands it, not the shell:

```sh
python3 introcut.py --dry-run '/videos/hevc_*'
python3 introcut.py --overwrite --dry-run '/videos/hevc_*'
```

An unquoted glob can fail with `Argument list too long` before Python starts.
Alternatively, stream a NUL-delimited list without expanding a shell glob:

```sh
find /videos -type f -name '*.mkv' -print0 |
  python3 introcut.py --files-from - --null --dry-run
```

Positional arguments accept files or quoted glob patterns. File lists contain
one literal path per line, without quoting, glob expansion, or comments. Relative
paths are relative to the list file's directory (or the working directory for
stdin). Use `--null` for filenames containing newlines. Duplicate paths, symlinks,
and hard links to the same file count only once. Separate copies of the same
video still count separately; use a diverse batch, not duplicates.

## Detection

By default, at least **90%** of inputs (rounded up, minimum two) must share a
visual opening of at least one second. Only the first 30 seconds are analyzed,
at 8 samples/second. Small color thumbnails and a luminance-structure comparison
make matching independent of resolution and encoding settings. Audio is not
used for recognition, so different audio tracks/settings do not interfere.

The detector votes for a majority at each timestamp, retains a consistent group
of matching files, tolerates isolated mismatches, and waits for three consecutive
disagreements to identify the ending. This avoids relying on the first input
being a good reference and avoids an all-pairs video comparison. Analysis uses
two workers by default and one FFmpeg decoder/filter thread per worker.

Useful adjustments:

```sh
python3 introcut.py --dry-run --scan-seconds 60 --min-fraction 0.8 /videos/*.mp4
python3 introcut.py --dry-run --sample-rate 16 --threshold 0.08 /videos/*.mp4
```

`--threshold` defaults to `0.10`; lower values are stricter. `--sample-rate`
controls temporal resolution (default 0.125s steps). Reports include an estimated
ending and a conservative interval extending one step either side.
`--min-intro` sets the minimum duration; `--workers` limits analysis concurrency.
Memory grows with the number of files and sampled seconds, not full video length.

Different letterboxing or differently sized copies of the same logo can need a
higher tolerance. For a large batch with a few-second intro, this is a useful
**dry-run** starting point:

```sh
python3 introcut.py --dry-run --scan-seconds 8 --threshold 0.25 '/videos/hevc_*'
```

The shorter scan reduces decoding and memory use; it must still include the
intro's ending and some following content. Raising the tolerance increases false
match risk, so review the reported cohort/boundary and preview representative
results before applying cuts. The default stays conservative.

This is **visual similarity, not semantic recognition**. Starts must be aligned;
different edits, playback speeds, crops, substantial overlays, or long fades can
need manual review or fail to match. A shared scene following the intro is
indistinguishable from part of the intro. Blank/solid-only openings are rejected
as insufficient evidence. No cut is proposed if a reliable ending is not observed
before the scan limit or end of the videos. Increase the scan limit in that case.
Always inspect a dry run before processing valuable files.

## Lossless cuts and their limits

The default `--keyframe previous` chooses the last usable keyframe **before the
lower end of the detected boundary interval**. It favors preserving following
content but can leave part of the intro. If only the initial keyframe qualifies,
the file is reported as `no_keyframe` and is not copied.

To remove the entire estimated intro, explicitly choose the first keyframe after
the upper end of the interval:

```sh
python3 introcut.py --keyframe next --dry-run /videos/*.mp4
python3 introcut.py --keyframe next --output-dir /videos/trimmed /videos/*.mp4
```

This can also remove following content, especially with long GOPs. Both modes
report the actual planned **video** cut timestamp, estimated remaining intro,
and estimated extra content removed for every match. Next-keyframe searching is
bounded by `--keyframe-lookahead` (default 30 seconds beyond the boundary).

All streams use `-c copy`: no video/audio re-encoding, resizing, frame-rate
conversion, or resampling. Metadata is retained; chapters are shifted/clipped to
the new timeline, and MP4 chapter carrier tracks are regenerated rather than
duplicated.
Only analysis thumbnails are decoded/resized. The new container's headers and
timestamps necessarily differ from the original file. AAC packet boundaries
rarely coincide with video keyframes; a short amount of audio/reordering preroll
can remain to preserve synchronization. The reported video cut is **not** a
promise of sample-exact audio removal or an exact container-duration difference.
Frame-exact cuts between keyframes would require re-encoding; this tool never
silently falls back to it.

The source must have one real video track (attached cover art is ignored for
detection), valid timestamps, and an FFmpeg-supported remuxable container.
HEVC open-GOP/B-frame seeking uses absolute PTS plus DTS-aware output trimming;
the first copied video packet's hash must match the planned keyframe before
an output is published.

In output-directory mode, originals and existing outputs are never overwritten.
Matching files keep their names; colliding basenames fail before any copies.
Completed outputs are published atomically using a same-filesystem hard link
(the output filesystem must support hard links).

In **overwrite mode**, a complete temporary file is created **beside the original**,
verified, and atomically replaces it only on success. File permissions and
filesystem timestamps are retained. There is no backup after a successful
replacement; use output-directory mode if you want to keep originals. Symlinks
are resolved to their targets; other hard links still refer to the old file.
There is no batch-wide rollback: previously completed replacements remain if a
later file fails. Ctrl-C during copying returns a partial report identifying
completed, interrupted, and not-yet-processed files; inspect the interrupted file
before retrying (the signal may arrive at the instant of atomic publication).

Both modes put temporary media on the destination filesystem, **never in `/tmp`**.
Allow free space for one trimmed video at a time, in addition to completed new
outputs. Temporary files are removed on ordinary errors or interruption; a forced
kill/power failure may leave a `.introcut-*` directory beside the destination.
Unmatched files are untouched. Per-file errors are reported and other eligible
files can still finish; inspect the exit status and report before treating a
batch as successful.

Text reports list every input. `--json` emits a single report with `schema_version`,
`dry_run`, `overwrite`, `intro` (or `null`), `required_matches`, and `files`; each file has a
`status`, `matched`, `cut_seconds`, estimated leftovers/over-cut, output path, and
reason. Statuses are `unmatched`, `no_keyframe`, `would_cut`, `cut`, `error`, or
`interrupted`; after interruption, remaining `would_cut` entries were not processed.
Exit codes: **0** cuts planned/completed, **1** file/tool/output error, **2** invalid
CLI options, **3** no reliable intro or no usable cut, **130** interruption.
Invalid input paths and output collisions abort the batch before copying.

## Development

```sh
python3 -m unittest discover -s tests -v
```

Tests include generated HEVC/AAC media and require FFmpeg with `libx265` and AAC
encoders. They exercise majority/outlier detection, varied encoding settings,
dry-run side effects, real packet-preserving cuts, and timestamp/keyframe traps.

## License

Apache License 2.0; see [LICENSE](LICENSE).

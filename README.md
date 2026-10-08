# introcut

Find the shared introduction in a batch of videos, then cut matching files
without re-encoding. Designed for HEVC/H.265 + AAC in MP4 or Matroska, including
different resolutions, frame rates, audio sample rates, and channel counts.
Python 3.10+ and `ffmpeg` / `ffprobe` 5+ on `PATH`; no Python runtime dependencies.

## Use

Run directly from this directory (no installation needed):

```sh
python3 introcut.py --dry-run '/videos/*.mp4'
python3 introcut.py --dry-run --json --files-from videos.txt > report.json
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
combined. The default cut uses the **next keyframe**, which can remove following
content; use `--keyframe prev` to favor preserving it instead.

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

## Reuse a dry-run plan

Save the reviewed cuts once, then apply them without repeating video/audio
analysis, intro detection, or keyframe searching:

```sh
time python3 introcut.py --dry-run '/videos/hevc_*' --save-plan plan.json
time python3 introcut.py --overwrite '/videos/hevc_*' --apply-plan plan.json
```

The plan contains absolute input paths, file identities, selected keyframes,
and the metadata needed for the existing stream-copy checks, not decoded
pictures or audio. Applying checks **every input** for changes in device, inode,
size, and nanosecond modification time before creating any outputs. The supplied
glob/list must expand to the same input set; added, removed, replaced, or changed
files require a new plan. Reordering the inputs is allowed. If no inputs are
given, the plan's saved paths are used.

Output mode is chosen when applying, never inherited from the plan:

```sh
python3 introcut.py --apply-plan plan.json --overwrite --dry-run
python3 introcut.py --apply-plan plan.json --output-dir /videos/trimmed
```

Without an output mode, applying is still a dry run. `--dry-run` always prevents
video writes. Saved detection/keyframe settings remain in force; explicit
conflicting settings are rejected. Unmatched files and saved errors/no-keyframe
outcomes are retained, not reanalyzed. The final reports and exit codes work as
usual. Copying still verifies the first output video packet against the saved
keyframe payload hash before publishing anything.

`--save-plan` also checkpoints decoded analysis in `plan.json.analysis.sqlite3`
beside the plan. If interrupted before the plan is written, rerun the same command:
unchanged files already checkpointed are not decoded again. The cache is separate
from the compact executable plan and is not needed for `--apply-plan`.

If a completed plan found no intro, its report is not permission to trim anything.
Use a fresh plan name and reuse the analysis while changing detection settings:

```sh
python3 introcut.py --dry-run '/videos/hevc_*' --save-plan retry.json \
  --analysis-cache plan.json.analysis.sqlite3 --min-fraction 0.65
```

`--analysis-cache FILE` also enables checkpoints without `--save-plan`. Cache
entries are tied to each input's path, identity, size, nanosecond mtime, sample rate,
and scan duration. Changed inputs are analyzed again; a corrupt cache is an error,
not silently trusted. Errors from media decoding are retried rather than cached.

`--save-plan` is dry-run-only, cannot be combined with `--apply-plan`, and refuses
to overwrite an existing path. Use a fresh plan filename for another scan.
Plans are written atomically beside their destination, not in `/tmp`; the parent
directory must already exist. Store plans outside the input glob. A plan is
**not a resume log**: successful in-place replacements make it stale, even though
file timestamps are preserved, so it cannot accidentally trim those files twice.
After a partial overwrite, inspect the results before making a new plan.

Plans and analysis caches are private: keep them out of version control and use
only ones you created and trust; do not edit them. `--json` is a report format,
not an executable plan; use `--save-plan` to create one. Both options can be used
together.

## Detection

By default, at least **70%** of inputs (rounded up, minimum two) must share a
distinctive opening of at least one second. Nonmatching files are skipped.
Analysis uses **both video and audio**, at 8 samples/second, over the first
**30 seconds** of each video by default. `--detection-length SECONDS` sets this
window: use any finite value of at least one second, or **0 to analyze the full
video through EOF**. Fractional seconds are accepted. `-detection-length` and
the older `--scan-seconds` are equivalent spellings. The requested window is
used directly, without an implicit shorter scan or automatic extension.
Picture matching uses small color thumbnails and luminance structure.
Sound matching uses normalized log-frequency spectra of the first audio track,
downmixed/resampled in memory, to tolerate different sample rates, channel
counts, and volume levels. No Python audio/image packages are required.

Each cue establishes its own majority and ending. When their boundary intervals
overlap and the required majority belongs to both groups, their matching files
are combined. This lets shared sound recognize portrait/landscape variants that
picture matching alone misses. A reliable visual ending takes precedence when
the cues disagree, so a longer shared background soundtrack does not extend it.
If only one cue establishes an ending, that cue can be used alone; absent or
different audio does not prevent a valid visual match. Reports identify the
evidence used (`video`, `audio`, or both); a combined result does not mean every
member matched both cues.

The detector votes for a majority at each timestamp, retains a consistent group
of matching files, tolerates isolated mismatches, and waits for three consecutive
disagreements to identify the ending. This avoids relying on the first input
being a good reference and avoids an all-pairs video comparison. Analysis uses
four workers by default and one FFmpeg decoder/filter thread per worker.
`--workers N` runs independent per-file analyses, up to Python's reported
logical CPU count (maximum 16 if unavailable). On machines with fewer than four
logical CPUs, the default is capped accordingly. More workers consume CPU,
memory, and storage bandwidth; decoding is not purely I/O-bound.

Useful adjustments:

```sh
python3 introcut.py --dry-run --detection-length 60 --min-fraction 0.8 '/videos/*.mp4'
python3 introcut.py --dry-run --sample-rate 16 --threshold 0.08 /videos/*.mp4
python3 introcut.py --dry-run --detection-length 0 --workers 4 '/videos/*.mp4'
```

`--threshold` defaults to `0.10`; lower values are stricter. `--sample-rate`
controls temporal resolution (default 0.125s steps). Reports include an estimated
ending and a conservative interval extending one step either side of each
supporting estimate.
`--min-intro` sets the minimum duration; `--workers` limits analysis concurrency.
Memory grows with the number of files and sampled seconds. Full-video mode can
be very expensive for large or long collections; it does not require duration
metadata, but it really decodes through the end of each video.

For a large batch with a few-second intro, a shorter scan reduces decoding and
memory use:

```sh
python3 introcut.py --dry-run --detection-length 6 --workers 4 '/videos/hevc_*'
```

The scan must still include the intro's ending and some following content.
A too-short window, including a valid one-second window, can return no intro;
it is not silently extended.
Raising `--threshold` increases false-match risk; prefer the defaults and review
the reported cohort/boundary before applying cuts.

This is **picture/sound similarity, not semantic recognition**. Starts must be aligned;
different edits, playback speeds, crops, substantial overlays, or long fades can
need manual review or fail to match. A shared scene following the intro is
indistinguishable from part of the intro, and shared music alone is not proof of
identical pictures. Blank/solid-only pictures, silence, and steady tones are
insufficient evidence. Audio must change within a majority of the actual recordings;
switching between per-timestamp vote winners is not evidence of temporal change.
Internal quiet gaps are allowed; trailing shared silence
does not extend an audio intro. No cut is proposed if a reliable ending is not observed
before the scan limit or end of the videos. Increase the scan limit in that case.
Always inspect a dry run before processing valuable files.

## Lossless cuts and their limits

The default `--keyframe next` chooses the first usable keyframe **after the
upper end of the detected boundary interval** to remove the entire estimated
intro. This can also remove following content, especially with long GOPs.
Next-keyframe searching is bounded by `--keyframe-lookahead` (default 30 seconds
beyond the boundary).

To favor preserving following content, explicitly choose `--keyframe prev`
(`--keyframe previous` remains an equivalent spelling):

```sh
python3 introcut.py --keyframe prev --dry-run '/videos/*.mp4'
python3 introcut.py --keyframe prev --output-dir /videos/trimmed '/videos/*.mp4'
```

This chooses the last usable keyframe **before the lower end of the interval**
and can leave part of the intro. If only the initial keyframe qualifies, the file
is reported as `no_keyframe` and is not copied. Both modes report the actual
planned **video** cut timestamp, estimated remaining intro, and estimated extra
content removed for every match.

All output streams use `-c copy`: no video/audio re-encoding, resizing, frame-rate
conversion, or resampling. Metadata is retained; chapters are shifted/clipped to
the new timeline, and MP4 chapter carrier tracks are regenerated rather than
duplicated.
Only in-memory analysis decodes/resizes pictures and resamples sound.
The new container's headers and timestamps necessarily differ from the original
file. AAC packet boundaries
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

Text reports list every input and finish with a summary of the intro, policy,
file/status counts, total/range of video cuts, and estimated remaining intro and
extra content removed. Cut statistics describe planned cuts in dry-run mode and
only completed cuts otherwise; failed, interrupted, and pending copies are not
counted as completed work.

`--json` emits a single report with `schema_version`, `dry_run`, `overwrite`,
`intro` (or `null`), `required_matches`, `files`, and a structured `summary`.
No prose is appended to JSON. Summary cut min/max values are `null` when there
are no applicable cuts. In apply mode, `summary.pending` counts unprocessed
`would_cut` files after interruption.
A detected `intro` includes its `evidence`. Each file has a
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
Saved-plan tests cover separate-process replay, unchanged-input preflight,
malformed plans, and real stream copies with analysis/keyframe search disabled.
They also cover checkpoint retries, explicit scan windows and full-video EOF,
CPU-aware workers, and audio variation that frame-wise voting would incorrectly
classify as a steady tone.

## License

Apache License 2.0; see [LICENSE](LICENSE).

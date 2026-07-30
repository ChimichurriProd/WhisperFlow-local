"""Transcribe audio FILES through Marvin (drag onto the pill, or the
right-click menu): decode -> chunk -> the same Transcriber dictation uses
(so Swedish gets KB-Whisper) -> one .txt next to the source.

Grouping is decided by the GESTURE: files dropped together belong together
and become ONE combined .txt with a header per file; files dropped one at a
time each get their own. No setting needed.

Deliberately NO LLM cleanup here, unlike dictation: the cleanup prompt is
tuned for short dictated utterances, and pushing a 20-minute interview
through gemma4 would be slow and could rewrite or truncate it. Recordings
get the faithful transcript, guarded only by the STT-artifact net
(collapse_repeats) and the user's vocabulary fixes.

Long files are split into ~60s chunks CUT AT THE QUIETEST POINT near each
boundary (a naive fixed cut lands mid-word), and each chunk is a separate
job on the shared MLX worker — so a dictation started mid-file waits a
couple of seconds for the current chunk, never for the whole file.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio import load_wav
from .cleanup import apply_vocabulary_fixes, collapse_repeats

# What afconvert (built into macOS — no ffmpeg dependency) reliably decodes.
AUDIO_EXTS = frozenset(
    {".wav", ".m4a", ".mp3", ".aac", ".aif", ".aiff", ".caf", ".flac"}
)

CHUNK_SECONDS = 60.0
SEAM_SEARCH_SECONDS = 3.0


def filter_audio_paths(paths):
    """Keep only the paths this module can transcribe (used by the pill's
    drag-and-drop to decide whether a drag is for us at all)."""
    # NSURL.path() can be None for exotic URLs — skip falsy entries.
    return [str(p) for p in (paths or [])
            if p and Path(p).suffix.lower() in AUDIO_EXTS]


def decode_audio(path, rate=16000):
    """Any supported audio file -> 16 kHz mono float32.

    WAV loads directly; everything else goes through afconvert, which ships
    with macOS — dropping an iPhone voice memo (.m4a) must Just Work without
    a brew install.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            return load_wav(p, target_rate=rate)
        except Exception:
            pass  # unusual WAV encoding: let afconvert have a go
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", f"LEI16@{int(rate)}",
             "-c", "1", str(p), tmp],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise ValueError(
                f"could not decode {p.name}: {detail[-1] if detail else 'afconvert failed'}"
            )
        return load_wav(tmp, target_rate=rate)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def seam_chunks(audio, rate, chunk_s=CHUNK_SECONDS, search_s=SEAM_SEARCH_SECONDS):
    """Split long audio at the quietest instant near each ~chunk_s boundary.

    A fixed cut lands mid-word and garbles both sides of the seam; searching
    ±search_s for the minimum-energy 20ms frame almost always finds a pause.
    """
    n = len(audio)
    step = int(chunk_s * rate)
    search = int(search_s * rate)
    if n == 0:
        return []
    if n <= step + search:
        return [audio]
    frame = max(1, int(0.02 * rate))
    chunks, i = [], 0
    while n - i > step + search:
        lo = i + step - search
        win = audio[lo:i + step + search]
        m = len(win) // frame
        energies = [float(np.mean(np.square(win[j * frame:(j + 1) * frame])))
                    for j in range(m)]
        cut = lo + (int(np.argmin(energies)) * frame if energies else search)
        chunks.append(audio[i:cut])
        i = cut
    chunks.append(audio[i:])
    return chunks


def format_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def format_paragraphs(segments, gap_seconds=1.2, timestamps=False):
    """Whisper segments [(abs_start, abs_end, text)] -> readable paragraphs.

    A 19-minute meeting used to come out as one 8 500-word wall of text. A
    speech gap longer than *gap_seconds* starts a new paragraph, optionally
    stamped with the paragraph's start time — the natural navigation unit for
    a recorded meeting ("hoppa till [12:40]").
    """
    paras, cur, cur_start, prev_end = [], [], 0.0, None
    for start, end, text in segments:
        if not text:
            continue
        if cur and prev_end is not None and start - prev_end > gap_seconds:
            paras.append((cur_start, " ".join(cur)))
            cur = []
        if not cur:
            cur_start = start
        cur.append(text)
        prev_end = end
    if cur:
        paras.append((cur_start, " ".join(cur)))
    if timestamps:
        return "\n\n".join(f"[{format_duration(s)}] {t}" for s, t in paras)
    return "\n\n".join(t for _, t in paras)


def _unique_path(path):
    """Never silently overwrite: 'namn.txt' -> 'namn 2.txt' -> 'namn 3.txt'."""
    path = Path(path)
    out, k = path, 2
    while out.exists():
        out = path.parent / f"{path.stem} {k}{path.suffix}"
        k += 1
    return out


def write_output(results):
    """Write the transcript(s) and return (path, clipboard_text).

    One file -> '<stem>.txt' with just the text. Several dropped together ->
    ONE combined '<first stem> +N filer.txt' with a '## name (m:ss)' header
    per recording, in drop order.
    """
    if len(results) == 1:
        src, text, dur = results[0]
        out = _unique_path(Path(src).with_suffix(".txt"))
        body = text + "\n"
    else:
        first = Path(results[0][0])
        out = _unique_path(
            first.parent / f"{first.stem} +{len(results) - 1} filer.txt")
        parts = [f"## {Path(src).name}  ({format_duration(dur)})\n\n{text}\n"
                 for src, text, dur in results]
        body = "\n".join(parts)
    out.write_text(body, encoding="utf-8")
    return out, body


def transcribe_files(paths, transcriber, config, on_progress=None,
                     on_fraction=None):
    """Transcribe *paths* and write the .txt. Returns a summary dict:
    {"out_path", "text", "n", "seconds"}. Raises on a file that can't be
    decoded — the caller reports rather than half-succeeding silently.

    on_fraction(0..1) drives the progress ring around Marvin. Files are
    weighted by their BYTE SIZE (a fair duration proxy that is known before
    anything is decoded), so one long file among short ones doesn't make the
    ring sprint and then stall.
    """
    def progress(msg):
        if on_progress:
            on_progress(msg)

    sizes = [max(1, os.path.getsize(p)) for p in paths]
    total_bytes = float(sum(sizes))
    done_bytes = 0.0

    def fraction(file_frac, size):
        if on_fraction:
            on_fraction(min(1.0, (done_bytes + size * file_frac) / total_bytes))

    rate = int(config.get("audio", {}).get("sample_rate", 16000))
    fixes = config.get("vocabulary", {}).get("fixes", {})
    fcfg = config.get("files", {})
    gap = float(fcfg.get("paragraph_gap_seconds", 1.2))
    want_stamps = bool(fcfg.get("timestamps", True))
    results, total_s = [], 0.0
    for idx, (path, size) in enumerate(zip(paths, sizes), 1):
        name = Path(path).name
        fraction(0.0, size)
        audio = decode_audio(path, rate=rate)
        dur = len(audio) / float(rate)
        total_s += dur
        chunks = seam_chunks(audio, rate)
        segs, offset = [], 0.0
        for ci, chunk in enumerate(chunks, 1):
            progress(f"({idx}/{len(paths)}) {name}: del {ci}/{len(chunks)}")
            piece = transcriber.transcribe(chunk)
            piece_segs = getattr(transcriber, "last_segments", None)
            if piece_segs:
                # Whisper's own segment times, shifted to absolute file time.
                segs += [(offset + s, offset + e, collapse_repeats(t))
                         for s, e, t in piece_segs]
            elif piece:
                # Engine without segments: the chunk is one block.
                segs.append((offset, offset + len(chunk) / rate,
                             collapse_repeats(piece.strip())))
            offset += len(chunk) / float(rate)
            fraction(ci / float(len(chunks)), size)
        # Timestamps only where they help navigate — a 30s memo needs none.
        text = format_paragraphs(segs, gap_seconds=gap,
                                 timestamps=want_stamps and dur >= 120.0)
        text = apply_vocabulary_fixes(text, fixes)
        results.append((path, text, dur))
        done_bytes += size
    out, body = write_output(results)
    if on_fraction:
        on_fraction(1.0)
    return {"out_path": str(out), "text": body,
            "n": len(results), "seconds": total_s}

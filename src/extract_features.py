"""
Feature extraction module — extracts a 36-dimensional music feature vector from a MIDI file.
"""

import numpy as np
from pathlib import Path

try:
    import miditoolkit
except ImportError:
    raise ImportError("miditoolkit is required: pip install miditoolkit")


def extract_midi_features(midi_path: str) -> list:
    """
    Extract a 36-dimensional feature vector from a MIDI file.

    Feature layout:
        [0:12]   Normalized pitch-class histogram (12 dims)
        [12:16]  Pitch statistics      (mean, std, min, max)
        [16:20]  Duration statistics   (mean, std, min, max)
        [20:22]  Velocity statistics   (mean, std)
        [22:27]  Interval features     (abs mean, std, stepwise ratio, skip ratio, leap ratio)
        [27:28]  Upward motion ratio
        [28:30]  Inter-onset interval  (mean, std)
        [30:31]  Polyphony level
        [31:32]  Note density          (notes per second)
        [32:36]  Reserved (zeros)

    Returns:
        A list of 36 floats. Returns all-zeros on empty or invalid files.
    """
    zero_vector = [0.0] * 36

    midi_path = Path(midi_path)
    if not midi_path.exists():
        return zero_vector

    try:
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return zero_vector

    # Collect notes from all non-drum tracks
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)

    if len(notes) == 0:
        return zero_vector

    # Sort by onset time
    notes.sort(key=lambda n: n.start)

    pitches    = np.array([n.pitch     for n in notes], dtype=float)
    durations  = np.array([n.end - n.start for n in notes], dtype=float)
    velocities = np.array([n.velocity  for n in notes], dtype=float)
    onsets     = np.array([n.start     for n in notes], dtype=float)

    # === [0:12] Normalized pitch-class histogram ===
    pitch_classes = (pitches % 12).astype(int)
    pch = np.zeros(12)
    for pc in pitch_classes:
        pch[pc] += 1
    total = pch.sum()
    if total > 0:
        pch /= total

    # === [12:16] Pitch statistics ===
    pitch_mean = float(np.mean(pitches))
    pitch_std  = float(np.std(pitches))
    pitch_min  = float(np.min(pitches))
    pitch_max  = float(np.max(pitches))

    # === [16:20] Duration statistics ===
    dur_mean = float(np.mean(durations))
    dur_std  = float(np.std(durations))
    dur_min  = float(np.min(durations))
    dur_max  = float(np.max(durations))

    # === [20:22] Velocity statistics ===
    vel_mean = float(np.mean(velocities))
    vel_std  = float(np.std(velocities))

    # === [22:28] Interval features ===
    if len(pitches) > 1:
        intervals     = np.diff(pitches)
        abs_intervals = np.abs(intervals)
        interval_abs_mean = float(np.mean(abs_intervals))
        interval_std      = float(np.std(abs_intervals))

        n_intervals = len(intervals)
        # Stepwise motion: 1-2 semitones
        stepwise_ratio = float(np.sum(abs_intervals <= 2) / n_intervals)
        # Skip: 3-7 semitones
        skip_ratio     = float(np.sum((abs_intervals >= 3) & (abs_intervals <= 7)) / n_intervals)
        # Leap: >= 8 semitones
        leap_ratio     = float(np.sum(abs_intervals >= 8) / n_intervals)
        # Upward motion
        upward_ratio   = float(np.sum(intervals > 0) / n_intervals)
    else:
        interval_abs_mean = interval_std = 0.0
        stepwise_ratio = skip_ratio = leap_ratio = upward_ratio = 0.0

    # === [28:30] Inter-onset interval (IOI) ===
    if len(onsets) > 1:
        ioi = np.diff(onsets)
        ioi_mean = float(np.mean(ioi))
        ioi_std  = float(np.std(ioi))
    else:
        ioi_mean = ioi_std = 0.0

    # === [30] Polyphony level: average number of simultaneously sounding notes ===
    polyphony_levels = []
    for note in notes:
        overlap = sum(1 for other in notes
                      if other.start < note.end and other.end > note.start)
        polyphony_levels.append(overlap)
    polyphony_mean = float(np.mean(polyphony_levels)) if polyphony_levels else 0.0

    # === [31] Note density (notes per second) ===
    if midi.ticks_per_beat > 0 and len(notes) > 0:
        # Use the first tempo event; default to 120 BPM if none present
        tempos = midi.tempo_changes
        bpm = tempos[0].tempo if tempos else 500000
        ticks_per_second = midi.ticks_per_beat * (1e6 / bpm)
        total_ticks   = onsets[-1] - onsets[0]
        total_seconds = total_ticks / ticks_per_second if ticks_per_second > 0 else 1.0
        note_density  = float(len(notes) / max(total_seconds, 1e-6))
    else:
        note_density = 0.0

    # === Assemble 36-dim vector ([32:36] reserved, filled with zeros) ===
    features  = list(pch)                                              # 12  [0:12]
    features += [pitch_mean, pitch_std, pitch_min, pitch_max]         #  4  [12:16]
    features += [dur_mean, dur_std, dur_min, dur_max]                 #  4  [16:20]
    features += [vel_mean, vel_std]                                   #  2  [20:22]
    features += [interval_abs_mean, interval_std,                     #  5  [22:27]
                 stepwise_ratio, skip_ratio, leap_ratio]
    features += [upward_ratio]                                        #  1  [27]
    features += [ioi_mean, ioi_std]                                   #  2  [28:30]
    features += [polyphony_mean]                                      #  1  [30]
    features += [note_density]                                        #  1  [31]
    features += [0.0, 0.0, 0.0, 0.0]                                 #  4  [32:36]

    assert len(features) == 36, f"Feature dimension mismatch: {len(features)}"
    return features


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.mid"
    feats = extract_midi_features(path)
    print(f"Extracted {len(feats)}-dim feature vector:")
    print(np.round(feats, 4))

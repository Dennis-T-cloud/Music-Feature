"""
Generative plugin — produces MIDI sequences from composer style features via a Markov chain.

Two modes:
    conditioned   : generation constrained by style parameters derived from Task 1
    unconditioned : unconstrained random walk (no style prior)
"""

import random
from typing import Optional

import numpy as np
from pathlib import Path

try:
    import miditoolkit
    from miditoolkit.midi.containers import Instrument, Note, TempoChange
except ImportError:
    raise ImportError("miditoolkit is required: pip install miditoolkit")


# ── Markov transition matrix ──────────────────────────────────────────────────

def _build_pitch_transition(pitches: list, semitone_range: int = 12) -> np.ndarray:
    """
    Build a first-order pitch-class Markov transition matrix (12x12) from a pitch sequence.
    Only transitions within semitone_range semitones are counted.
    Laplace smoothing (0.1) prevents zero-probability entries.
    """
    matrix = np.ones((12, 12)) * 0.1
    for i in range(len(pitches) - 1):
        pc_from = int(pitches[i]) % 12
        pc_to   = int(pitches[i + 1]) % 12
        if abs(int(pitches[i + 1]) - int(pitches[i])) <= semitone_range:
            matrix[pc_from, pc_to] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def _sample_next_pitch(current_pc: int, matrix: np.ndarray,
                       base_octave: int = 5) -> int:
    """Sample the next MIDI pitch (0-127) from the transition matrix."""
    pc_next = np.random.choice(12, p=matrix[current_pc])
    return base_octave * 12 + pc_next


# ── Style parameter extraction ────────────────────────────────────────────────

def style_params_from_features(feature_vector: list) -> dict:
    """
    Unpack a 36-dim feature vector into a readable style parameter dict.
    Index positions match the layout defined in extract_features.py.
    """
    f = feature_vector
    return {
        "pitch_mean":     f[12],
        "pitch_std":      f[13],
        "dur_mean":       f[16],   # mean note duration in MIDI ticks
        "vel_mean":       f[20],
        "vel_std":        f[21],
        "stepwise_ratio": f[24],   # high value -> smooth, step-wise melody
        "leap_ratio":     f[26],   # high value -> dramatic leaps
        "ioi_mean":       f[28],   # mean inter-onset interval
        "note_density":   f[31],
    }


# ── Core generation function ──────────────────────────────────────────────────

def generate_midi(
    output_path: str,
    style_params: Optional[dict] = None,
    seed_pitches: Optional[list] = None,
    n_notes: int = 64,
    bpm: int = 120,
    ticks_per_beat: int = 480,
    random_seed: Optional[int] = None,
) -> str:
    """
    Generate a MIDI sequence using a first-order Markov chain.

    Args:
        output_path:    Destination .mid file path.
        style_params:   Dict from style_params_from_features(); None for unconditioned mode.
        seed_pitches:   Pitch sequence used to initialize the transition matrix.
                        Defaults to a C-major scale run when None.
        n_notes:        Number of notes to generate.
        bpm:            Tempo in beats per minute.
        ticks_per_beat: MIDI resolution.
        random_seed:    Integer seed for reproducibility.

    Returns:
        Absolute path of the written file as a string.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Default style parameters (unconditioned baseline)
    params = {
        "pitch_mean": 60.0, "pitch_std": 8.0,
        "dur_mean": ticks_per_beat // 2, "vel_mean": 80.0, "vel_std": 10.0,
        "stepwise_ratio": 0.5, "leap_ratio": 0.1,
        "ioi_mean": ticks_per_beat // 2, "note_density": 4.0,
    }
    if style_params:
        params.update({k: v for k, v in style_params.items() if v != 0.0})

    if seed_pitches is None:
        seed_pitches = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]

    transition_matrix = _build_pitch_transition(seed_pitches)

    base_pitch  = int(np.clip(params["pitch_mean"], 36, 84))
    current_pc  = base_pitch % 12

    notes  = []
    cursor = 0  # current position in MIDI ticks
    for _ in range(n_notes):
        next_pc = np.random.choice(12, p=transition_matrix[current_pc])

        # Conditioned mode: resolve pitch class to octave near pitch_mean
        target_octave = int(params["pitch_mean"]) // 12
        pitch = int(np.clip(target_octave * 12 + next_pc, 21, 108))

        # Add a small Gaussian perturbation scaled by pitch_std
        pitch += int(np.random.normal(0, max(params["pitch_std"] * 0.2, 0.5)))
        pitch = int(np.clip(pitch, 21, 108))

        duration = max(
            int(np.random.normal(params["dur_mean"], params["dur_mean"] * 0.3)),
            ticks_per_beat // 8,
        )

        velocity = int(np.clip(
            np.random.normal(params["vel_mean"], max(params["vel_std"], 1)),
            20, 127,
        ))

        notes.append(Note(
            start=cursor,
            end=cursor + duration,
            pitch=pitch,
            velocity=velocity,
        ))

        ioi = max(
            int(np.random.normal(params["ioi_mean"], params["ioi_mean"] * 0.25)),
            ticks_per_beat // 8,
        )
        cursor    += ioi
        current_pc = pitch % 12

    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tempo_changes.append(TempoChange(tempo=int(6e7 / bpm), time=0))
    instrument = Instrument(program=0, is_drum=False, name="Piano")
    instrument.notes = notes
    midi.instruments.append(instrument)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    midi.dump(output_path)
    return str(Path(output_path).resolve())


# ── Public interface ──────────────────────────────────────────────────────────

def generate_conditioned(
    feature_vector: list,
    output_path: str = "outputs/symbolic_conditioned.mid",
    n_notes: int = 64,
    bpm: int = 120,
    random_seed: int = 42,
) -> str:
    """Conditioned generation: style constrained by a composer feature vector from Task 1."""
    params = style_params_from_features(feature_vector)
    return generate_midi(output_path, style_params=params, n_notes=n_notes,
                         bpm=bpm, random_seed=random_seed)


def generate_unconditioned(
    output_path: str = "outputs/symbolic_unconditioned.mid",
    n_notes: int = 64,
    bpm: int = 120,
    random_seed: int = 0,
) -> str:
    """Unconditioned generation: pure random walk with no style prior."""
    return generate_midi(output_path, style_params=None, n_notes=n_notes,
                         bpm=bpm, random_seed=random_seed)


if __name__ == "__main__":
    dummy_features = [0.0] * 36
    dummy_features[12] = 65.0   # pitch_mean
    dummy_features[16] = 240.0  # dur_mean
    dummy_features[20] = 85.0   # vel_mean
    dummy_features[24] = 0.6    # stepwise_ratio

    out1 = generate_conditioned(dummy_features)
    out2 = generate_unconditioned()
    print(f"Conditioned:   {out1}")
    print(f"Unconditioned: {out2}")

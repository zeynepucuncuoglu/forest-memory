"""
audio_signals.py — Forest Memory acoustic feature extractor

Uses scipy + numpy only (no librosa dependency) so it runs without numba.

IMPORTANT: All outputs are acoustic proxies only.
- "insect_activity_proxy" = high-frequency energy fraction (> 4 kHz)
- "human_noise_proxy"     = low-frequency energy fraction (< 500 Hz)
- "acoustic_richness_proxy" = proportion of 1-kHz bands that are non-silent
None of these are species counts or direct biodiversity measurements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft as scipy_stft

# Frequency band boundaries (Hz)
_LOW_FREQ_MAX  = 500      # anthropogenic / machinery proxy
_MID_FREQ_MIN  = 1_000   # lower bound of typical bird call range
_MID_FREQ_MAX  = 8_000   # upper bound of typical bird call range
_HIGH_FREQ_MIN = 4_000   # lower bound of many insect calls
_BAND_WIDTH    = 1_000   # width of each richness band (Hz)
_RICHNESS_THRESHOLD = 0.001  # band must hold > 0.1% of total power to count
_ROLLOFF_PERC  = 0.85    # spectral rolloff percentile


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Collapse stereo to mono by averaging channels."""
    return y.mean(axis=1) if y.ndim > 1 else y


def extract_features(wav_path: str | Path) -> dict:
    """Return a flat dict of acoustic features for one WAV file."""
    wav_path = Path(wav_path)
    y_raw, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    y = _to_mono(y_raw)

    duration = len(y) / sr

    # ── Time-domain ──────────────────────────────────────────────────────────
    rms = float(np.sqrt(np.mean(y ** 2)))
    # Zero crossing rate: fraction of adjacent-sample sign changes
    signs = np.sign(y)
    signs[signs == 0] = 1
    zcr = float(np.mean(np.abs(np.diff(signs)) / 2))

    # ── Full-signal FFT for energy bands ─────────────────────────────────────
    N = len(y)
    Y = rfft(y)
    freqs = rfftfreq(N, d=1.0 / sr)     # Hz per bin
    power = np.abs(Y) ** 2              # power spectrum

    total_power = float(power.sum()) + 1e-12

    low_mask  = freqs < _LOW_FREQ_MAX
    mid_mask  = (freqs >= _MID_FREQ_MIN) & (freqs <= _MID_FREQ_MAX)
    high_mask = freqs >= _HIGH_FREQ_MIN

    low_freq_energy  = float(power[low_mask].sum()  / total_power)
    mid_freq_energy  = float(power[mid_mask].sum()  / total_power)
    high_freq_energy = float(power[high_mask].sum() / total_power)

    # ── Spectral centroid ────────────────────────────────────────────────────
    spectral_centroid = float((freqs * power).sum() / (power.sum() + 1e-12))

    # ── Spectral bandwidth ───────────────────────────────────────────────────
    spectral_bandwidth = float(
        np.sqrt(((freqs - spectral_centroid) ** 2 * power).sum() / (power.sum() + 1e-12))
    )

    # ── Spectral rolloff: frequency below which ROLLOFF_PERC of power sits ──
    cumpower = np.cumsum(power)
    threshold = _ROLLOFF_PERC * cumpower[-1]
    rolloff_idx = int(np.searchsorted(cumpower, threshold))
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # ── Acoustic richness proxy (simplified ADI) ─────────────────────────────
    nyquist = sr // 2
    band_starts = list(range(0, int(nyquist), _BAND_WIDTH))
    occupied = sum(
        1
        for lo in band_starts
        if (mask := (freqs >= lo) & (freqs < lo + _BAND_WIDTH)).any()
        and power[mask].sum() / total_power > _RICHNESS_THRESHOLD
    )
    acoustic_richness_proxy = occupied / max(len(band_starts), 1)

    return {
        "duration": round(duration, 3),
        "sample_rate": int(sr),
        "rms": round(rms, 6),
        "zcr": round(zcr, 6),
        "spectral_centroid": round(spectral_centroid, 2),
        "spectral_bandwidth": round(spectral_bandwidth, 2),
        "spectral_rolloff": round(spectral_rolloff, 2),
        "low_freq_energy": round(low_freq_energy, 6),
        "mid_freq_energy": round(mid_freq_energy, 6),
        "high_freq_energy": round(high_freq_energy, 6),
        "acoustic_richness_proxy": round(acoustic_richness_proxy, 4),
        # Named aliases used by scoring.py
        "insect_activity_proxy": round(high_freq_energy, 6),
        "human_noise_proxy": round(low_freq_energy, 6),
    }


def discover_wavs(audio_root: str | Path) -> list[dict]:
    """Walk audio_root and return metadata dicts for each WAV.

    Expected layout:
        <audio_root>/<role>/<device>_<begin_date>_<datetime>.wav

    Returns list of dicts: role, site_id, filename, path
    """
    audio_root = Path(audio_root)
    records = []
    for wav in sorted(audio_root.rglob("*.wav")):
        role = wav.parent.name
        parts = wav.stem.split("_")
        site_id = "_".join(parts[:2]) if len(parts) >= 2 else wav.stem
        records.append({"role": role, "site_id": site_id,
                         "filename": wav.name, "path": wav})
    return records


def process_directory(audio_root: str | Path) -> list[dict]:
    """Discover and extract features for every WAV under audio_root."""
    rows = []
    for meta in discover_wavs(audio_root):
        features = extract_features(meta["path"])
        rows.append({"role": meta["role"], "site_id": meta["site_id"],
                     "filename": meta["filename"], **features})
    return rows

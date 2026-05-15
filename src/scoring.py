"""
scoring.py — Forest Memory acoustic proxy scorer

Converts raw audio features into normalised 0–100 proxy scores.

DISCLAIMER
----------
These scores are acoustic proxies for ecosystem vitality.
They do NOT represent:
  - species richness or species counts
  - confirmed bird / insect presence
  - validated biodiversity metrics
They are intended as relative comparison signals between sites
within the same recording campaign and equipment setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Scoring weights ──────────────────────────────────────────────────────────

_BIRD_WEIGHTS = {
    "mid_freq_energy": 0.50,   # 1–8 kHz carries most songbird calls
    "spectral_bandwidth": 0.25, # richer call repertoire → wider spread
    "zcr": 0.25,                # rapid zero crossings → tonal activity
}

_INSECT_WEIGHTS = {
    "insect_activity_proxy": 0.70,  # high-freq energy (> 4 kHz)
    "zcr": 0.30,                    # insect stridulation is high-ZCR
}

_HUMAN_WEIGHTS = {
    "human_noise_proxy": 0.70,  # low-freq energy (< 500 Hz)
    "rms": 0.30,                # loud broadband signal correlates with machinery
}

_RICHNESS_WEIGHTS = {
    "acoustic_richness_proxy": 0.60,
    "spectral_bandwidth": 0.25,
    "mid_freq_energy": 0.15,
}


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def _weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute a weighted normalised composite score (0–100)."""
    cols = list(weights.keys())
    normed = df[cols].apply(_minmax)
    w = np.array([weights[c] for c in cols])
    return (normed * w).sum(axis=1)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add proxy score columns to a features DataFrame.

    Parameters
    ----------
    df : DataFrame from audio_signals.process_directory()

    Returns
    -------
    DataFrame with added columns:
        bird_activity_proxy, insect_activity_proxy_score,
        human_disturbance_proxy, acoustic_richness_score,
        bioacoustic_vitality_score
    """
    out = df.copy()

    out["bird_activity_proxy"] = _weighted_score(out, _BIRD_WEIGHTS).round(2)

    # Insect proxy score is already a feature alias; normalise it to 0-100
    insect_raw = out[list(_INSECT_WEIGHTS.keys())].apply(_minmax)
    insect_w = np.array(list(_INSECT_WEIGHTS.values()))
    out["insect_activity_proxy_score"] = (insect_raw * insect_w).sum(axis=1).round(2)

    human_raw = out[list(_HUMAN_WEIGHTS.keys())].apply(_minmax)
    human_w = np.array(list(_HUMAN_WEIGHTS.values()))
    out["human_disturbance_proxy"] = (human_raw * human_w).sum(axis=1).round(2)

    out["acoustic_richness_score"] = _weighted_score(out, _RICHNESS_WEIGHTS).round(2)

    # Vitality = biological signal minus disturbance penalty
    # All components already on 0-100; clamp to [0, 100]
    bio = (
        0.40 * out["bird_activity_proxy"]
        + 0.30 * out["insect_activity_proxy_score"]
        + 0.30 * out["acoustic_richness_score"]
    )
    out["bioacoustic_vitality_score"] = (bio - 0.5 * out["human_disturbance_proxy"]).clip(0, 100).round(2)

    return out


# Output column order for the final CSV
OUTPUT_COLS = [
    "role", "site_id", "filename",
    "duration", "sample_rate",
    "rms", "spectral_centroid",
    "high_freq_energy", "low_freq_energy",
    "acoustic_richness_proxy",
    "bird_activity_proxy",
    "insect_activity_proxy_score",
    "human_disturbance_proxy",
    "acoustic_richness_score",
    "bioacoustic_vitality_score",
]

"""
build_forest_memory_cases.py — Forest Memory case builder

Aggregates per-WAV audio proxy scores by role+site_id, merges with NDVI
remote-sensing data and BioSCape site metadata, adds satellite asset paths,
and attaches interpretation flags.

OUTPUT
------
outputs/forest_memory_cases.json        (compact)
outputs/forest_memory_cases_pretty.json (indented, human-readable)

PROXY DISCLAIMER
----------------
All acoustic and spectral-index values are proxy signals derived from
passive acoustic monitoring and satellite imagery. They do NOT represent:
  - species richness or confirmed species detections
  - validated biodiversity loss metrics
  - wildfire predictions or fire-risk assessments
They are relative comparison signals between sites within the same
campaign and sensor setup only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """Coerce numpy scalars to native Python types for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
AUDIO_CSV    = ROOT / "outputs" / "audio_features.csv"
NDVI_CSV     = ROOT / "outputs" / "remote_sensing" / "ndvi_summary.csv"
META_CSV     = ROOT / "data" / "bioscape" / "metadata" / "BioSCape_acoustic_sites.csv"
SAT_DIR      = ROOT / "data" / "bioscape" / "sattelite"   # actual on-disk spelling
AUDIO_DIR    = ROOT / "data" / "bioscape" / "audio"
OUT_COMPACT  = ROOT / "outputs" / "forest_memory_cases.json"
OUT_PRETTY   = ROOT / "outputs" / "forest_memory_cases_pretty.json"

# ── Role name mapping: audio short-name → canonical long-name ─────────────────
# (audio_features.csv uses folder names; NDVI + TIFs use descriptive names)
ROLE_MAP: dict[str, str] = {
    "healthy":    "healthy_baseline",
    "recovering": "burned_recovering",
    "disturbed":  "invasive_disturbed",
    "wet_dry":    "wet_dry_pair_complement",
}

# ── Interpretation flag thresholds ────────────────────────────────────────────
# All thresholds chosen conservatively; see docstrings below for rationale.
_NDVI_GREEN_THRESHOLD      = 0.45   # >= this is "noticeably green"
_VITALITY_LOW_THRESHOLD    = 40.0   # <= this is "acoustically quiet"
_NDVI_STD_VARIABLE         = 0.10   # >= this suggests patchy vegetation
_HUMAN_NOISE_UNCERTAIN     = 45.0   # >= this risks masking bio-signals

# Score columns to aggregate
SCORE_COLS = [
    "bioacoustic_vitality_score",
    "acoustic_richness_score",
    "bird_activity_proxy",
    "insect_activity_proxy_score",
    "human_disturbance_proxy",
]

PROXY_DISCLAIMER = (
    "All acoustic and NDVI values are proxy signals only. "
    "Do not interpret as confirmed species counts, validated biodiversity "
    "measurements, or wildfire predictions. Values are relative within this "
    "sample of sites and recording sessions."
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and aggregate audio features
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map short role names to canonical names
    df["role_canonical"] = df["role"].map(ROLE_MAP).fillna(df["role"])
    return df


def _aggregate_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / min / max of each score column, grouped by canonical role + site_id."""
    avail = [c for c in SCORE_COLS if c in df.columns]
    agg_funcs = {c: ["mean", "min", "max"] for c in avail}
    agg_funcs["filename"] = "count"

    grouped = df.groupby(["role_canonical", "site_id"]).agg(agg_funcs)
    grouped.columns = [
        f"{col}_{stat}" if col != "filename" else "wav_file_count"
        for col, stat in grouped.columns
    ]
    grouped = grouped.rename(columns={"wav_file_count": "wav_file_count"})
    return grouped.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load NDVI summary
# ─────────────────────────────────────────────────────────────────────────────

def _load_ndvi(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: NDVI CSV not found at {path} — NDVI fields will be null")
        return pd.DataFrame()
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load BioSCape site metadata for selected site_ids
# ─────────────────────────────────────────────────────────────────────────────

_META_COLS = [
    "SiteID", "Latitude", "Longitude", "LandCoverClass", "FireClass",
    "FieldVeldAge", "FieldAliensWithin20m", "ElevationClass",
    "Campaign", "RecordingNum", "AVIRIS", "LVIS", "PairID",
]

def _load_metadata(path: Path, site_ids: list[str]) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: Metadata CSV not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=_META_COLS)
    df = df[df["SiteID"].isin(site_ids)].copy()
    # Keep one representative row per SiteID (prefer dry-season row)
    df["_cam_rank"] = df["Campaign"].apply(
        lambda c: 0 if "Dry" in str(c) else 1
    )
    df = (
        df.sort_values(["SiteID", "_cam_rank"])
        .groupby("SiteID", as_index=False)
        .first()
        .drop(columns=["_cam_rank"])
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build asset paths
# ─────────────────────────────────────────────────────────────────────────────

def _asset_paths(role_canonical: str, site_id: str) -> dict:
    """Return relative paths from project root for TIF + WAV assets."""
    assets: dict[str, object] = {}

    # TIF assets: keyed by <role_canonical>_rgb/ndvi_2023_dry.tif
    for kind in ("rgb", "ndvi"):
        tif = SAT_DIR / f"{role_canonical}_{kind}_2023_dry.tif"
        assets[f"{kind}_tif"] = (
            str(tif.relative_to(ROOT)) if tif.exists() else None
        )

    # WAV assets: find by site_id prefix under any role subfolder
    wav_files = sorted(AUDIO_DIR.rglob(f"{site_id}_*.wav"))
    assets["audio_files"] = [
        str(w.relative_to(ROOT)) for w in wav_files
    ]

    return assets


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interpretation flags
# ─────────────────────────────────────────────────────────────────────────────

def _flags(row: dict) -> dict[str, bool | str]:
    """
    Attach conservative interpretation flags.

    green_not_alive_signal
        NDVI suggests green vegetation (proxy for photosynthetic activity)
        but bioacoustic vitality is below threshold. Possible causes:
        invasive plant cover, vegetation recovering structurally but with
        reduced wildlife acoustic activity. NOT a confirmed biodiversity loss claim.

    spatially_variable_vegetation
        High temporal/spatial spread in NDVI values suggests heterogeneous
        vegetation patchiness — possibly a mosaic of recovering and intact areas.

    recent_fire_recovery_context
        FireClass metadata indicates 1-to-6 year post-fire window. Acoustic
        patterns should be interpreted in light of early recovery dynamics.
        NOT a wildfire prediction.

    acoustic_uncertainty
        High human_disturbance_proxy (low-frequency energy) suggests the
        soundscape may contain wind, equipment hum, or anthropogenic noise
        that could mask biological signals. Treat acoustic scores with caution.
    """
    ndvi_mean    = row.get("ndvi_mean")
    ndvi_std     = row.get("ndvi_std")
    vitality     = row.get("bioacoustic_vitality_score_mean")
    human_noise  = row.get("human_disturbance_proxy_mean")
    fire_class   = str(row.get("fire_class", ""))

    return {
        "green_not_alive_signal": bool(
            ndvi_mean is not None
            and vitality is not None
            and ndvi_mean >= _NDVI_GREEN_THRESHOLD
            and vitality <= _VITALITY_LOW_THRESHOLD
        ),
        "spatially_variable_vegetation": bool(
            ndvi_std is not None and ndvi_std >= _NDVI_STD_VARIABLE
        ),
        "recent_fire_recovery_context": "1-to-6" in fire_class,
        "acoustic_uncertainty": bool(
            human_noise is not None and human_noise >= _HUMAN_NOISE_UNCERTAIN
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Build one case record
# ─────────────────────────────────────────────────────────────────────────────

def _build_case(audio_row: pd.Series, ndvi_df: pd.DataFrame,
                meta_df: pd.DataFrame) -> dict:
    role_canonical = audio_row["role_canonical"]
    site_id        = audio_row["site_id"]

    # ── NDVI ──────────────────────────────────────────────────────────────────
    ndvi_match = (
        ndvi_df[(ndvi_df["role"] == role_canonical) & (ndvi_df["site_id"] == site_id)]
        if not ndvi_df.empty else pd.DataFrame()
    )
    if not ndvi_match.empty:
        nr = ndvi_match.iloc[0]
        ndvi = {
            "mean_ndvi":   round(float(nr["mean_ndvi"]), 4),
            "std_ndvi":    round(float(nr["std_ndvi"]),  4),
            "min_ndvi":    round(float(nr["min_ndvi"]),  4),
            "max_ndvi":    round(float(nr["max_ndvi"]),  4),
            "image_count": int(nr["image_count"]),
            "period_start": str(nr["start"]),
            "period_end":   str(nr["end"]),
        }
    else:
        ndvi = None

    # ── Site metadata ──────────────────────────────────────────────────────────
    meta_match = (
        meta_df[meta_df["SiteID"] == site_id]
        if not meta_df.empty else pd.DataFrame()
    )
    if not meta_match.empty:
        mr = meta_match.iloc[0]
        def _safe(v):
            if pd.isna(v) or str(v).strip() in ("", "Not available", "nan"):
                return None
            return v
        site_meta = {
            "latitude":               _safe(mr.get("Latitude")),
            "longitude":              _safe(mr.get("Longitude")),
            "land_cover_class":       _safe(mr.get("LandCoverClass")),
            "fire_class":             _safe(mr.get("FireClass")),
            "field_veld_age":         _safe(mr.get("FieldVeldAge")),
            "field_aliens_within_20m": _safe(mr.get("FieldAliensWithin20m")),
            "elevation_class":        _safe(mr.get("ElevationClass")),
            "campaign":               _safe(mr.get("Campaign")),
            "recording_num":          _safe(mr.get("RecordingNum")),
            "aviris_coverage":        _safe(mr.get("AVIRIS")),
            "lvis_coverage":          _safe(mr.get("LVIS")),
            "pair_id":                _safe(mr.get("PairID")),
        }
    else:
        site_meta = {}

    # ── Audio aggregates ───────────────────────────────────────────────────────
    def _pick(col_base, stat):
        key = f"{col_base}_{stat}"
        v = audio_row.get(key)
        return round(float(v), 2) if v is not None and pd.notna(v) else None

    audio = {"wav_file_count": int(audio_row.get("wav_file_count", 0))}
    for col in SCORE_COLS:
        audio[col] = {s: _pick(col, s) for s in ("mean", "min", "max")}

    # ── Assets ─────────────────────────────────────────────────────────────────
    assets = _asset_paths(role_canonical, site_id)

    # ── Flat lookup dict for flag computation ──────────────────────────────────
    flag_ctx = {
        "ndvi_mean":    ndvi["mean_ndvi"]  if ndvi else None,
        "ndvi_std":     ndvi["std_ndvi"]   if ndvi else None,
        "fire_class":   site_meta.get("fire_class", ""),
        "bioacoustic_vitality_score_mean":  _pick("bioacoustic_vitality_score", "mean"),
        "human_disturbance_proxy_mean":     _pick("human_disturbance_proxy", "mean"),
    }

    return {
        "role":           role_canonical,
        "site_id":        site_id,
        "site_metadata":  site_meta,
        "audio":          audio,
        "ndvi":           ndvi,
        "assets":         assets,
        "interpretation_flags": _flags(flag_ctx),
        "proxy_disclaimer": PROXY_DISCLAIMER,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(cases: list[dict]) -> None:
    rows = []
    for c in cases:
        flags = c["interpretation_flags"]
        audio = c["audio"]
        ndvi  = c["ndvi"] or {}
        rows.append({
            "role": c["role"],
            "site_id": c["site_id"],
            "wavs": audio["wav_file_count"],
            "vitality_mean": (audio.get("bioacoustic_vitality_score") or {}).get("mean"),
            "richness_mean":  (audio.get("acoustic_richness_score") or {}).get("mean"),
            "human_noise_mean": (audio.get("human_disturbance_proxy") or {}).get("mean"),
            "ndvi_mean": ndvi.get("mean_ndvi"),
            "ndvi_std":  ndvi.get("std_ndvi"),
            "🟢 green_not_alive": flags["green_not_alive_signal"],
            "🌿 spatially_variable": flags["spatially_variable_vegetation"],
            "🔥 recent_fire_ctx": flags["recent_fire_recovery_context"],
            "⚠️  acoustic_uncertain": flags["acoustic_uncertainty"],
        })

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n" + "=" * 90)
    print("FOREST MEMORY — CASE SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    print()
    print("NOTE: All proxy scores are relative within this sample.")
    print("      Interpretation flags are conservative heuristics, not ecological verdicts.")
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading audio features …")
    audio_raw = _load_audio(AUDIO_CSV)
    audio_agg = _aggregate_audio(audio_raw)
    print(f"  {len(audio_raw)} WAV rows → {len(audio_agg)} site aggregates")

    print("Loading NDVI summary …")
    ndvi_df = _load_ndvi(NDVI_CSV)
    if not ndvi_df.empty:
        print(f"  {len(ndvi_df)} NDVI rows loaded")

    print("Loading BioSCape site metadata …")
    site_ids = audio_agg["site_id"].unique().tolist()
    meta_df = _load_metadata(META_CSV, site_ids)
    print(f"  {len(meta_df)} metadata rows matched")

    print("Building cases …")
    cases = [
        _build_case(row, ndvi_df, meta_df)
        for _, row in audio_agg.iterrows()
    ]

    # Canonical role order
    _order = {
        "healthy_baseline": 0, "burned_recovering": 1,
        "invasive_disturbed": 2, "wet_dry_pair_complement": 3,
    }
    cases.sort(key=lambda c: _order.get(c["role"], 99))

    OUT_COMPACT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_COMPACT, "w") as f:
        json.dump(cases, f, separators=(",", ":"), cls=_NumpyEncoder)
    print(f"Saved → {OUT_COMPACT.relative_to(ROOT)}")

    with open(OUT_PRETTY, "w") as f:
        json.dump(cases, f, indent=2, cls=_NumpyEncoder)
    print(f"Saved → {OUT_PRETTY.relative_to(ROOT)}")

    _print_summary(cases)


if __name__ == "__main__":
    main()

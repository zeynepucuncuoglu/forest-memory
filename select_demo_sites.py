"""
BioSCape Demo Site Selector for Forest Memory
Kaggle Gemma 4 Good Hackathon

Selects 3–4 strong demo sites for a multimodal ecosystem resilience demo.
NOTE: LandCoverClass, FireClass, RecordingNum etc. are proxy signals — not
direct biodiversity proofs.
"""

import pandas as pd
import numpy as np

CSV_PATH = "BioSCape_acoustic_sites.csv"
OUT_PATH = "selected_sites.csv"

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

print("=" * 60)
print(f"COLUMNS ({len(df.columns)}):")
for c in df.columns:
    print(f"  {c}")

print(f"\nSHAPE: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Missingness ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSINGNESS / 'Not available' summary:")
not_avail_cols = ["FieldWetlandType", "FieldVeldAge", "FieldAliensWithin20m",
                  "Latitude", "Longitude", "PairID", "WetlandType"]
for col in df.columns:
    n_null = df[col].isna().sum()
    n_na_str = (df[col].astype(str) == "Not available").sum() if df[col].dtype == object else 0
    print(f"  {col:35s}  NaN={n_null:4d}  'Not available'={n_na_str:4d}")

# ── 3. Value counts for key categorical columns ───────────────────────────────
print("\n" + "=" * 60)
for col in ["LandCoverClass", "FireClass", "FieldVeldAge",
            "FieldAliensWithin20m", "Campaign", "ElevationClass"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

# ── 4. Filter candidate pools ─────────────────────────────────────────────────

# Only sites with valid coordinates
has_coords = df["Latitude"].notna() & df["Longitude"].notna()

TARGET_LAND = {"Forest-Woodland", "Shrubland-Fynbos"}

# --- Pool A: Healthy baseline ------------------------------------------------
# No-fire / very old burn, old vegetation, target land cover.
# FieldAliensWithin20m "None" is scored, not hard-filtered, because the strict
# four-way intersection is empty in this dataset.
pool_a = df[
    has_coords
    & df["LandCoverClass"].isin(TARGET_LAND)
    & df["FireClass"].isin(["No data or No Fire", "25+ years"])
    & (df["FieldVeldAge"] == "Old (> 17 yrs)")
].copy()

# --- Pool B: Recently burned / recovering ------------------------------------
# FireClass 1-to-6 yrs, young vegetation, target land cover
pool_b = df[
    has_coords
    & df["LandCoverClass"].isin(TARGET_LAND)
    & (df["FireClass"] == "1-to-6 years")
    & (df["FieldVeldAge"] == "Young (burned, < 6 yrs)")
].copy()

# --- Pool C: Human-disturbed / invasive pressure -----------------------------
# High alien density (any land cover) — Agriculture also acceptable
invasive_vals = {"Scattered to Medium", "Dense to Closed"}
pool_c = df[
    has_coords
    & df["FieldAliensWithin20m"].isin(invasive_vals)
].copy()

# Fallback: Agriculture land cover if pool_c is small
if len(pool_c) < 5:
    pool_c_extra = df[has_coords & (df["LandCoverClass"] == "Agriculture")].copy()
    pool_c = pd.concat([pool_c, pool_c_extra]).drop_duplicates()

# --- Pool D: Wet/Dry paired --------------------------------------------------
# PairID present, two rows with same PairID (Wet + Dry)
paired = df[has_coords & df["PairID"].notna()].copy()
pair_counts = paired.groupby("PairID")["Campaign"].apply(
    lambda x: set(x.str.strip())
)
complete_pairs = pair_counts[pair_counts.apply(
    lambda s: "Wet season" in s and "Dry season" in s
)].index
pool_d = paired[paired["PairID"].isin(complete_pairs)].copy()

# ── 5. Scoring & ranking ──────────────────────────────────────────────────────

def score(pool: pd.DataFrame) -> pd.DataFrame:
    p = pool.copy()
    p["score"] = 0
    # High RecordingNum → richer audio archive (proxy)
    p["score"] += (p["RecordingNum"] / p["RecordingNum"].max()) * 40
    # Has valid PairID > 0 (wet+dry comparison possible)
    p["score"] += (p["PairID"] > 0).astype(int) * 20
    # FieldVeldAge available (not "Not available")
    p["score"] += (p["FieldVeldAge"] != "Not available").astype(int) * 15
    # No invasive aliens — best resilience proxy signal
    aliens = p["FieldAliensWithin20m"].fillna("Not available").astype(str)
    p["score"] += (aliens == "None").astype(int) * 12
    # Aliens data was collected at all (not missing)
    p["score"] += (~aliens.isin(["Not available", "nan"])).astype(int) * 6
    # AVIRIS hyperspectral coverage
    p["score"] += (p["AVIRIS"] == "Y").astype(int) * 10
    # LVIS LiDAR coverage
    p["score"] += (p["LVIS"] == "Y").astype(int) * 5
    return p.sort_values("score", ascending=False)

pool_a = score(pool_a)
pool_b = score(pool_b)
pool_c = score(pool_c)
pool_d = score(pool_d)

print("\n" + "=" * 60)
print(f"POOL SIZES — A(healthy):{len(pool_a)}  B(burned):{len(pool_b)}"
      f"  C(invasive):{len(pool_c)}  D(paired):{len(pool_d)}")

# ── 6. Pick top candidates ────────────────────────────────────────────────────

DISPLAY_COLS = ["SiteID", "LandCoverClass", "FireClass", "FieldVeldAge",
                "FieldAliensWithin20m", "Latitude", "Longitude", "PairID",
                "RecordingNum", "Campaign", "ElevationClass", "score"]

def top(pool, n=5, label=""):
    print(f"\n{'─'*60}")
    print(f"TOP {n} — {label}")
    if pool.empty:
        print("  (empty pool)")
        return pool.head(0)
    print(pool[DISPLAY_COLS].head(n).to_string(index=False))
    return pool.head(n)

top_a = top(pool_a, 5, "POOL A — Healthy / Resilient Baseline")
top_b = top(pool_b, 5, "POOL B — Recently Burned / Recovering")
top_c = top(pool_c, 5, "POOL C — Human-Disturbed / Invasive Pressure")
top_d = top(pool_d, 5, "POOL D — Wet+Dry Paired Sites (top 5 by score)")

# ── 7. Final 4 picks ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SELECTED SITES")
print("=" * 60)

selections = {}

def pick_best(pool, role, exclude_ids=()):
    if pool.empty:
        print(f"  WARNING: No candidates for {role}")
        return None
    cands = pool[~pool["SiteID"].isin(exclude_ids)]
    if cands.empty:
        cands = pool
    row = cands.iloc[0].copy()
    row["demo_role"] = role
    return row

site_a = pick_best(pool_a, "healthy_baseline")
site_b = pick_best(pool_b, "burned_recovering",
                   exclude_ids=[site_a["SiteID"]] if site_a is not None else [])
site_c = pick_best(pool_c, "invasive_disturbed",
                   exclude_ids=[s["SiteID"] for s in [site_a, site_b]
                                if s is not None])
# For paired: find the other-season partner for one of the selected sites.
# If site_b has a valid PairID, find its wet-season counterpart (different SiteID,
# same PairID, opposite Campaign). Fall back to best unrelated paired site.
site_d = None
already_selected_ids = {s["SiteID"] for s in [site_a, site_b, site_c] if s is not None}
if not pool_d.empty:
    found_complement = False
    for candidate in [site_b, site_a]:
        if candidate is None:
            continue
        cpid = candidate.get("PairID")
        ccam = candidate.get("Campaign", "")
        if pd.notna(cpid) and cpid > 0:
            other_campaign = "Wet season" if "Dry" in str(ccam) else "Dry season"
            match = df[
                (df["PairID"] == cpid)
                & (df["Campaign"] == other_campaign)
                & (~df["SiteID"].isin(already_selected_ids))
            ]
            if not match.empty:
                site_d = match.sort_values("RecordingNum", ascending=False).iloc[0].copy()
                site_d["demo_role"] = f"wet_dry_pair_complement (PairID={int(cpid)})"
                found_complement = True
                break
    if not found_complement:
        # Pick best paired site not already selected
        site_d = pick_best(pool_d, "wet_dry_pair_example", exclude_ids=already_selected_ids)

final_sites = [s for s in [site_a, site_b, site_c, site_d] if s is not None]
final_df = pd.DataFrame(final_sites)

REPORT_COLS = ["demo_role", "SiteID", "LandCoverClass", "FireClass",
               "FieldVeldAge", "FieldAliensWithin20m", "Latitude", "Longitude",
               "PairID", "RecordingNum", "Campaign", "ElevationClass", "score"]
available_cols = [c for c in REPORT_COLS if c in final_df.columns]
print(final_df[available_cols].to_string(index=False))

final_df[available_cols].to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

# ── 8. WAV filename mapping ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("WAV FILENAME CONVENTION")
print("SiteID format:  <device>_<begin_date>")
print("WAV format:     <device>_<begin_date>_<recording_datetime>.WAV")
print()
print("To find WAV files for each selected site, filter your audio archive")
print("for files whose name STARTS WITH SiteID (device_date prefix).\n")

for s in final_sites:
    sid = s["SiteID"]
    print(f"  Role: {s['demo_role']}")
    print(f"    SiteID  : {sid}")
    print(f"    WAV glob: {sid}_*.WAV")
    print(f"    Python  : [f for f in wav_files if f.startswith('{sid}_')]")
    print()

print("Example Python snippet to map SiteIDs to WAV files:")
print("""
import glob, os

wav_dir = "data/bioscape/audio/"
wav_files = [os.path.basename(p) for p in glob.glob(wav_dir + "**/*.WAV", recursive=True)]

site_ids = selected_df["SiteID"].tolist()
site_wav_map = {
    sid: [f for f in wav_files if f.startswith(sid + "_")]
    for sid in site_ids
}
for sid, wavs in site_wav_map.items():
    print(f"{sid}: {len(wavs)} WAV files found")
    for w in wavs[:3]:
        print(f"  {w}")
""")

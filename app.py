"""
Forest Memory — Gradio Demo
Multimodal ecological resilience reasoning: LiteRT (YAMNet) + Gemma 4
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import gradio as gr

# ── Load pre-computed data ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent

CASES_PATH  = ROOT / "outputs" / "forest_memory_cases.json"
LITERT_PATH = ROOT / "outputs" / "gemma_reports" / "litert_multimodal_reports.json"
GEMMA3_PATH = ROOT / "outputs" / "gemma_reports" / "gemma_reports.json"

with open(CASES_PATH)  as f: cases_raw   = json.load(f)
with open(LITERT_PATH) as f: litert_raw  = json.load(f)
with open(GEMMA3_PATH) as f: gemma3_raw  = json.load(f)

cases       = {c["role"]: c for c in cases_raw}
litert_rpts = {r["role"]: r for r in litert_raw["reports"]}
gemma3_rpts = {r["role"]: r for r in gemma3_raw["site_reports"]}
synthesis_md    = gemma3_raw["synthesis"].get("raw_response", "")
synthesis_model = gemma3_raw["synthesis"].get("model", "Gemma 4")

ROLE_LABELS = {
    "healthy_baseline":        "Healthy Baseline — s2lam027",
    "burned_recovering":       "Burned Recovering — s2lam047",
    "invasive_disturbed":      "Invasive Disturbed — s2lam051",
    "wet_dry_pair_complement": "Wet/Dry Pair — s2lam096",
}
ROLE_ICONS = {
    "healthy_baseline":        "🌿",
    "burned_recovering":       "🔥",
    "invasive_disturbed":      "⚠️",
    "wet_dry_pair_complement": "💧",
}
ROLE_KEYS        = list(ROLE_LABELS.keys())
DROPDOWN_CHOICES = [ROLE_LABELS[r] for r in ROLE_KEYS]


# ── 1. How it works (static HTML) ─────────────────────────────────────────────

HOW_IT_WORKS_HTML = """
<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin:12px 0 4px 0;">

  <div style="text-align:center; padding:18px 14px;
              background:rgba(25,118,210,0.12);
              border-radius:10px; border-top:3px solid #42a5f5;
              backdrop-filter:blur(8px);">
    <div style="font-size:2em; margin-bottom:6px;">🎧</div>
    <div style="font-weight:700; font-size:1em; margin-bottom:6px; color:#90caf9;">LISTEN — LiteRT Ear</div>
    <div style="font-size:0.88em; color:#ccd6e0; line-height:1.55;">
      YAMNet (3 MB TFLite) classifies 521 sound categories from field recordings —
      bird calls, insect stridulation, wind, rain, human noise.<br><br>
      Runs <strong style="color:#fff;">fully offline on CPU</strong> via LiteRT —
      no GPU, no internet, no cloud. Works on a Raspberry Pi.
    </div>
  </div>

  <div style="text-align:center; padding:18px 14px;
              background:rgba(56,142,60,0.12);
              border-radius:10px; border-top:3px solid #66bb6a;
              backdrop-filter:blur(8px);">
    <div style="font-size:2em; margin-bottom:6px;">🛰️</div>
    <div style="font-weight:700; font-size:1em; margin-bottom:6px; color:#a5d6a7;">LOOK — Satellite Eye</div>
    <div style="font-size:0.88em; color:#ccd6e0; line-height:1.55;">
      EfficientNet-Lite0 (5 MB TFLite) analyses satellite RGB imagery as a
      visual texture feature extractor — also <strong style="color:#fff;">fully offline on CPU</strong>.<br><br>
      NDVI (Sentinel-2) adds the greenness signal.
      Together: offline visual sensing under 10 MB total.
    </div>
  </div>

  <div style="text-align:center; padding:18px 14px;
              background:rgba(245,124,0,0.12);
              border-radius:10px; border-top:3px solid #ffa726;
              backdrop-filter:blur(8px);">
    <div style="font-size:2em; margin-bottom:6px;">🤖</div>
    <div style="font-weight:700; font-size:1em; margin-bottom:6px; color:#ffcc80;">REASON — Gemma 4</div>
    <div style="font-size:0.88em; color:#ccd6e0; line-height:1.55;">
      Gemma 4 fuses acoustic signals, NDVI, fire history, and invasive species data
      into structured ecological reports.<br><br>
      Detects multimodal tension: <em>high greenness + low vitality
      = ecological stress invisible from space.</em>
    </div>
  </div>

</div>
<div style="font-size:0.82em; color:#7a8a99; text-align:center; margin-top:8px;">
  Validated against BioSCape expert classifications (Spearman ρ = 1.00, n = 4) ·
  Prototype built on South African data — designed for deployment in Turkish forests
</div>
"""


# ── 2. Per-site diagnosis cards ────────────────────────────────────────────────

DIAGNOSIS = {
    "healthy_baseline": {
        "bg": "rgba(76,175,80,0.12)", "border": "#4CAF50", "icon": "✅",
        "title": "Ecological Baseline — Healthy",
        "title_color": "#a5d6a7",
        "body": (
            "No recent fire history. Old-growth fynbos shrubland (&gt;17 years). "
            "No invasive species detected within 20 m. "
            "Highest bioacoustic vitality in the sample (<strong style='color:#fff;'>41.1/100</strong>). "
            "NDVI 0.507 — proportional to acoustic vitality. "
            "This site is the ecological ceiling for this monitoring campaign."
        ),
    },
    "burned_recovering": {
        "bg": "rgba(255,152,0,0.12)", "border": "#FF9800", "icon": "🔥",
        "title": "Post-Fire Recovery — Early Succession",
        "title_color": "#ffcc80",
        "body": (
            "Burned <strong style='color:#fff;'>1–6 years ago</strong>. Young fynbos vegetation re-establishing. "
            "Rare invasive species presence. Bioacoustic vitality <strong style='color:#fff;'>35.2/100</strong> — "
            "lower than the baseline, but following the expected recovery arc. "
            "Fauna returns as vegetation rebuilds. NDVI 0.508 and vitality are roughly proportional. "
            "Recovery is on track."
        ),
    },
    "invasive_disturbed": {
        "bg": "rgba(198,40,40,0.15)", "border": "#ef5350", "icon": "⚠️",
        "title": "Ecological Alert — Green But Not Alive",
        "title_color": "#ef9a9a",
        "body": (
            "Burned <strong style='color:#fff;'>6–12 years ago</strong>. Should be more recovered than the 1–6 year site — "
            "but invasive woody species colonised the canopy during the recovery window. "
            "NDVI <strong style='color:#fff;'>0.677</strong> — the highest in the sample, suggesting a lush forest. "
            "Bioacoustic vitality <strong style='color:#fff;'>31.1/100</strong> — <em>lower</em> than the recently-burned site. "
            "<br><br><strong style='color:#ef9a9a;'>This is the core finding:</strong> "
            "satellite greenness hides the invasive collapse. "
            "The forest looks recovered from space. It isn't."
        ),
    },
    "wet_dry_pair_complement": {
        "bg": "rgba(120,144,156,0.12)", "border": "#90a4ae", "icon": "📉",
        "title": "Ecological Stress — Lowest Vitality in Sample",
        "title_color": "#b0bec5",
        "body": (
            "Wet season recording (August 2023). Burned <strong style='color:#fff;'>1–6 years ago</strong>. "
            "Bioacoustic vitality <strong style='color:#fff;'>10.8/100</strong> — the lowest in the sample. "
            "Human disturbance proxy <strong style='color:#fff;'>62/100</strong> — high anthropogenic noise is masking "
            "biological acoustic signals (<em>acoustic uncertainty flag active</em>). "
            "NDVI unavailable: wet-season cloud cover blocked the Sentinel-2 composite. "
            "This site represents the ecological floor in this campaign."
        ),
    },
}


def build_diagnosis_html(role: str) -> str:
    d = DIAGNOSIS[role]
    return f"""
<div style="
  background:{d['bg']};
  border-left:5px solid {d['border']};
  padding:16px 20px;
  border-radius:8px;
  margin:6px 0 14px 0;
  font-family: sans-serif;
  backdrop-filter: blur(8px);
">
  <div style="font-size:1.05em; font-weight:700; margin-bottom:8px; color:{d['title_color']};">
    {d['icon']}&nbsp;&nbsp;{d['title']}
  </div>
  <div style="color:#ccd6e0; line-height:1.65; font-size:0.92em;">
    {d['body']}
  </div>
</div>
"""


# ── 3. Per-site acoustic chart ─────────────────────────────────────────────────

_ACOUSTIC_COLORS = ["#FF7043", "#9C27B0", "#1976D2", "#43A047", "#E53935"]
_ACOUSTIC_LABELS = [
    "Bioacoustic\nVitality",
    "Acoustic\nRichness",
    "Bird\nActivity",
    "Insect\nActivity",
    "Human\nDisturbance",
]
_ACOUSTIC_KEYS = [
    "bioacoustic_vitality_score",
    "acoustic_richness_score",
    "bird_activity_proxy",
    "insect_activity_proxy_score",
    "human_disturbance_proxy",
]
_ACOUSTIC_TIPS = [
    "Overall ecological soundscape health proxy",
    "Spectral diversity — not always correlated with health",
    "High-frequency energy proxy for avian activity",
    "Mid-frequency energy proxy for insect stridulation",
    "Low-frequency / broadband noise from human sources",
]


def build_acoustic_chart(role: str) -> plt.Figure:
    audio = cases.get(role, {}).get("audio", {})

    def _get(col):
        v = (audio.get(col) or {}).get("mean")
        return v if v is not None else 0.0

    values = [_get(k) for k in _ACOUSTIC_KEYS]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#1a1d2e")
    ax.set_facecolor("#1a1d2e")

    bars = ax.barh(
        _ACOUSTIC_LABELS, values,
        color=_ACOUSTIC_COLORS, alpha=0.87, height=0.55,
    )
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 1.5, 103), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=9, color="#222",
        )

    ax.set_xlim(0, 108)
    ax.set_xlabel("Score (0 – 100)", fontsize=10, color="#ccd6e0")
    ax.set_title("Acoustic Proxy Signals", fontsize=11, fontweight="bold", color="#e8eaf6")
    ax.axvline(x=50, color="#2d3250", linestyle="--", lw=1.2)
    ax.tick_params(colors="#ccd6e0")
    ax.yaxis.label.set_color("#ccd6e0")
    ax.grid(axis="x", linestyle="--", alpha=0.15, color="#5c6bc0")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ── 4. Site location map (static) ─────────────────────────────────────────────

_MAP_SITES = [
    {"role": "healthy_baseline",        "lat": -33.247, "lon": 18.194,
     "label": "Healthy Baseline",  "vitality": 41.1, "color": "#4CAF50"},
    {"role": "burned_recovering",       "lat": -34.356, "lon": 18.858,
     "label": "Burned Recovering", "vitality": 35.2, "color": "#FF9800"},
    {"role": "invasive_disturbed",      "lat": -33.949, "lon": 21.058,
     "label": "Invasive Disturbed","vitality": 31.1, "color": "#c62828"},
    {"role": "wet_dry_pair_complement", "lat": -34.356, "lon": 18.860,
     "label": "Wet/Dry Pair",      "vitality": 10.8, "color": "#78909C"},
]


def build_site_map() -> go.Figure:
    fig = go.Figure()
    for s in _MAP_SITES:
        fig.add_trace(go.Scattergeo(
            lat=[s["lat"]],
            lon=[s["lon"]],
            text=[f"<b>{s['label']}</b><br>Vitality: {s['vitality']}/100"],
            hoverinfo="text",
            marker=dict(
                size=10 + s["vitality"] / 4,
                color=s["color"],
                opacity=0.88,
                line=dict(color="white", width=2),
            ),
            name=s["label"],
            mode="markers",
        ))

    fig.update_layout(
        title=dict(
            text="BioSCape Monitoring Sites — Western Cape, South Africa<br>"
                 "<sup>Marker size = bioacoustic vitality · Color = ecological status</sup>",
            x=0.5, xanchor="center", font=dict(size=13),
        ),
        geo=dict(
            scope="africa",
            showland=True,  landcolor="#1e2233",
            showocean=True, oceancolor="#0d1520",
            showcountries=True, countrycolor="#3a4060",
            showcoastlines=True, coastlinecolor="#2d3555",
            showrivers=False,
            bgcolor="#1a1d2e",
            center=dict(lat=-34, lon=19.5),
            projection_scale=12,
        ),
        height=370,
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.05,
                    bgcolor="rgba(15,17,23,0.85)",
                    bordercolor="#2d3250", borderwidth=1,
                    font=dict(color="#ccd6e0")),
        font=dict(color="#ccd6e0"),
        paper_bgcolor="#1a1d2e",
    )
    return fig


# ── 5. Cross-site comparison chart (static) ───────────────────────────────────

def build_comparison_chart() -> plt.Figure:
    short_labels = ["Healthy\nBaseline", "Burned\nRecovering", "Invasive\nDisturbed", "Wet/Dry\nPair"]
    ndvi_vals, vitality_vals = [], []

    for role in ROLE_KEYS:
        case  = cases.get(role, {})
        ndvi  = (case.get("ndvi") or {}).get("mean_ndvi")
        audio = case.get("audio", {})
        vit   = (audio.get("bioacoustic_vitality_score") or {}).get("mean")
        ndvi_vals.append(ndvi if ndvi is not None else np.nan)
        vitality_vals.append(vit / 100 if vit is not None else np.nan)

    x, width = np.arange(len(ROLE_KEYS)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1d2e")
    ax.set_facecolor("#1a1d2e")

    bars_ndvi = ax.bar(x - width/2, ndvi_vals, width,
                       label="NDVI — satellite greenness (0–1)",
                       color="#4CAF50", alpha=0.88, zorder=3)
    bars_vit  = ax.bar(x + width/2, vitality_vals, width,
                       label="Bioacoustic Vitality ÷ 100  (proxy)",
                       color="#FF7043", alpha=0.88, zorder=3)

    for i, v in enumerate(ndvi_vals):
        if np.isnan(v):
            ax.bar(x[i] - width/2, 0.08, width,
                   color="#ccc", alpha=0.7, zorder=3, hatch="////", edgecolor="#aaa")
            ax.text(x[i] - width/2, 0.09, "N/A",
                    ha="center", va="bottom", fontsize=8, color="#888")

    for bar in list(bars_ndvi) + list(bars_vit):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333")

    inv_i = ROLE_KEYS.index("invasive_disturbed")
    ax.annotate("⚠  High NDVI\nLow Vitality",
                xy=(inv_i - width/2, ndvi_vals[inv_i]),
                xytext=(inv_i + 0.85, 0.82),
                fontsize=9, color="#c62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.4))

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10.5, color="#ccd6e0")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0 – 1)", fontsize=11, color="#ccd6e0")
    ax.set_title('NDVI vs Bioacoustic Vitality — all 4 sites\n'
                 '"Green does not always mean ecologically healthy."',
                 fontsize=12, fontweight="bold", pad=12, color="#e8eaf6")
    ax.tick_params(colors="#ccd6e0")
    ax.spines[:].set_color("rgba(255,255,255,0.15)" if False else "#2d3250")
    ax.legend(fontsize=10, loc="upper left",
              facecolor="#1a1d2e", edgecolor="#2d3250",
              labelcolor="#ccd6e0")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0, color="#5c6bc0")
    ax.set_axisbelow(True)
    ax.annotate("† Wet/Dry Pair: NDVI unavailable — wet season cloud cover blocked Sentinel-2 composite",
                xy=(1, -0.07), xycoords="axes fraction",
                ha="right", fontsize=8, color="#5c6a7a")
    plt.tight_layout()
    return fig


# ── 6. Recovery trajectory chart (static) ─────────────────────────────────────

RECOVERY_DATA = [
    {"role": "healthy_baseline",        "label": "Healthy\nBaseline",
     "fire_years": 20,  "vitality": 41.1, "ndvi": 0.507, "invasive": 0},
    {"role": "burned_recovering",       "label": "Burned\nRecovering",
     "fire_years": 3.5, "vitality": 35.2, "ndvi": 0.508, "invasive": 1},
    {"role": "invasive_disturbed",      "label": "Invasive\nDisturbed",
     "fire_years": 9.0, "vitality": 31.1, "ndvi": 0.677, "invasive": 2},
    {"role": "wet_dry_pair_complement", "label": "Wet/Dry\nPair",
     "fire_years": 4.5, "vitality": 10.8, "ndvi": None,  "invasive": 0},
]
_DOT_COLORS = {
    "healthy_baseline":        "#4CAF50",
    "burned_recovering":       "#FF9800",
    "invasive_disturbed":      "#c62828",
    "wet_dry_pair_complement": "#78909C",
}
_INV_SIZE  = {0: 180, 1: 380, 2: 700}
_INV_LABEL = {0: "No invasives", 1: "Rare aliens", 2: "Scattered–Medium aliens"}


def build_recovery_chart() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1d2e")
    ax.set_facecolor("#1a1d2e")

    for d in RECOVERY_DATA:
        ax.scatter(d["fire_years"], d["vitality"],
                   s=_INV_SIZE[d["invasive"]],
                   color=_DOT_COLORS[d["role"]],
                   alpha=0.88, zorder=5,
                   edgecolors="white", linewidths=1.8)
        offset = (-72, 8) if d["role"] == "invasive_disturbed" else (10, 6)
        ax.annotate(d["label"], (d["fire_years"], d["vitality"]),
                    textcoords="offset points", xytext=offset,
                    fontsize=9.5, color="#333")
        if d["ndvi"] is not None:
            ax.annotate(f"NDVI {d['ndvi']:.3f}", (d["fire_years"], d["vitality"]),
                        textcoords="offset points",
                        xytext=(offset[0], offset[1] - 13),
                        fontsize=8, color="#555")

    ax.annotate("", xy=(20, 41), xytext=(3.5, 35),
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.5,
                                linestyle="dashed"), zorder=2)
    ax.text(11, 40.5, "expected\nrecovery →", fontsize=8, color="#607d8b", style="italic")
    ax.annotate("⚠  Invasion breaks\nrecovery trajectory",
                xy=(9.0, 31.1), xytext=(13, 22),
                fontsize=9, color="#c62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.3))

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 52)
    ax.set_xticks([3.5, 9, 20])
    ax.set_xticklabels(["1–6 yrs\npost-fire", "6–12 yrs\npost-fire",
                         ">17 yrs\n(no recent fire)"], fontsize=9.5, color="#ccd6e0")
    ax.set_ylabel("Bioacoustic Vitality (0–100)", fontsize=11, color="#ccd6e0")
    ax.set_title("Recovery trajectory: fire age vs acoustic vitality\n"
                 "Bubble size = invasive species presence",
                 fontsize=11, fontweight="bold", pad=10, color="#e8eaf6")
    ax.tick_params(colors="#ccd6e0")
    ax.grid(axis="y", linestyle="--", alpha=0.2, zorder=0, color="#5c6bc0")
    for inv_level, label in _INV_LABEL.items():
        ax.scatter([], [], s=_INV_SIZE[inv_level], color="#607d8b", alpha=0.7, label=label)
    ax.legend(fontsize=8.5, loc="upper left",
              title="Invasive species", title_fontsize=9,
              facecolor="#1a1d2e", edgecolor="#2d3250", labelcolor="#ccd6e0",
              )
    plt.tight_layout()
    return fig


RECOVERY_EXPLANATION = """
### Why does the Invasive Disturbed site rank worse — despite the most recovery time?

After a wildfire, native fynbos shrubland normally recovers over years to decades.
The **Burned Recovering** site (1–6 years post-fire) shows low but improving vitality — expected early succession.
The **Healthy Baseline** (no recent fire, >17 years) shows the ceiling: vitality 41.1, no invasives.

The **Invasive Disturbed** site should be *between* these — at 6–12 years it has had more time to recover
than the burned site. Instead its vitality (31.1) is **lower**.

**Why?** During the post-fire recovery window, invasive woody species colonised the canopy.
They are photosynthetically active, pushing NDVI to 0.677 — the highest in the sample.
But they suppress native fauna through habitat displacement and altered food webs.
The soundscape records the collapse that satellite imagery hides.

> *"The forest looks recovered from space. It isn't."*
"""


# ── Metrics + Gemma report panels ─────────────────────────────────────────────

def build_metrics_md(role: str) -> str:
    case  = cases.get(role, {})
    audio = case.get("audio", {})
    ndvi  = case.get("ndvi") or {}
    flags = case.get("interpretation_flags", {})
    meta  = case.get("site_metadata", {})

    lines = []

    lines.append("### Site Metadata")
    lines.append(f"- **Site ID:** `{case.get('site_id', '—')}`")
    lines.append(f"- **Land Cover:** {meta.get('land_cover_class', '—')}")
    lines.append(f"- **Fire Class:** {meta.get('fire_class', '—')}")
    lines.append(f"- **Veld Age:** {meta.get('field_veld_age', '—')}")
    lines.append(f"- **Invasive Species (20 m):** {meta.get('field_aliens_within_20m', '—')}")
    lines.append(f"- **Season:** {meta.get('campaign', '—')}")
    lines.append(f"- **WAV files analysed:** {audio.get('wav_file_count', 0)}")
    lines.append("")

    ndvi_mean = ndvi.get("mean_ndvi")
    ndvi_str  = f"`{ndvi_mean:.3f}`" if ndvi_mean is not None else "*not available*"
    lines.append("### Vegetation Signal")
    lines.append(f"**NDVI (Sentinel-2):** {ndvi_str}")
    lines.append("")
    lines.append("> NDVI measures photosynthetic activity — whether plants are green.")
    lines.append("> It does **not** measure ecological function or biodiversity.")
    lines.append("")

    lines.append("### Interpretation Flags")
    flag_defs = {
        "green_not_alive_signal":        ("🟢", "Green-not-alive",
            "High NDVI but low bioacoustic vitality — possible invasive cover"),
        "spatially_variable_vegetation": ("🌿", "Spatially variable veg.",
            "Patchy NDVI — mosaic or uneven regeneration"),
        "recent_fire_recovery_context":  ("🔥", "Recent fire context",
            "1–6 year post-fire window — early succession"),
        "acoustic_uncertainty":          ("⚠️", "Acoustic uncertainty",
            "High human-noise proxy — biological signals may be masked"),
    }
    for k, (icon, name, desc) in flag_defs.items():
        active = flags.get(k, False)
        state  = "**ACTIVE**" if active else "inactive"
        lines.append(f"- {icon} **{name}**: {state} — *{desc}*")

    lines.append("")
    lines.append("---")
    lines.append("*All values are proxy signals — not confirmed species counts or wildfire predictions.*")
    return "\n".join(lines)


def build_litert_report_md(role: str) -> str:
    rpt     = litert_rpts.get(role, {})
    text    = rpt.get("report", "No LiteRT report available.")
    latency = rpt.get("latency_s")
    model   = litert_raw.get("gemma_model", "Gemma 4")
    audio_m = litert_raw.get("litert_audio_model", "YAMNet")
    header  = (f"*Model: **{model}** | Audio: **{audio_m}** | Latency: **{latency:.1f} s***\n\n---\n\n"
               if latency else "")
    return header + text


def build_structured_report_md(role: str) -> str:
    rpt      = gemma3_rpts.get(role, {})
    sections = rpt.get("sections", {})
    latency  = rpt.get("latency_s")
    model    = rpt.get("model", "Gemma 4")

    if not sections:
        raw = rpt.get("raw_response", "No structured report available.")
        return f"*Model: **{model}***\n\n---\n\n{raw}"

    header = (f"*Model: **{model}** | Latency: **{latency:.1f} s***\n\n---\n\n"
              if latency else "")
    lines = [header]
    section_icons = {
        "Vegetation Interpretation":  "🌿",
        "Bioacoustic Interpretation": "🔊",
        "Multimodal Tension Summary": "⚡",
        "Recovery Interpretation":    "📈",
        "Uncertainty Notes":          "⚠️",
    }
    for sec_name, text in sections.items():
        icon = section_icons.get(sec_name, "•")
        lines.append(f"### {icon} {sec_name}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


# ── Main update ────────────────────────────────────────────────────────────────

def update(choice: str):
    role      = ROLE_KEYS[DROPDOWN_CHOICES.index(choice)]
    icon      = ROLE_ICONS[role]
    heading   = f"## {icon} {ROLE_LABELS[role]}"
    diagnosis = build_diagnosis_html(role)
    acoustic  = build_acoustic_chart(role)
    metrics   = build_metrics_md(role)
    litert    = build_litert_report_md(role)
    struct    = build_structured_report_md(role)
    return heading, diagnosis, acoustic, metrics, litert, struct


# ── Gradio UI ──────────────────────────────────────────────────────────────────

CSS = """
/* ── Dark theme ── */
body, .gradio-container, .main, .wrap {
    background: #0f1117 !important;
    color: #dde3ee !important;
}
.block, .gr-box, .gr-panel, .panel {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
}
/* Markdown & prose */
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3, .prose h4,
.markdown-body, .markdown-body p, .markdown-body li,
.markdown-body h1, .markdown-body h2, .markdown-body h3 {
    color: #dde3ee !important;
}
.prose code, .markdown-body code {
    background: rgba(255,255,255,0.08) !important;
    color: #90caf9 !important;
    border-radius: 4px;
    padding: 1px 5px;
}
blockquote, .prose blockquote {
    border-left: 3px solid #42a5f5 !important;
    color: #90a4ae !important;
    background: rgba(66,165,245,0.06) !important;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
}
/* Dropdown / select */
.gr-dropdown, select, input, textarea {
    background: rgba(255,255,255,0.06) !important;
    color: #dde3ee !important;
    border-color: rgba(255,255,255,0.12) !important;
}
/* Tab labels */
.tab-nav button {
    color: #90a4ae !important;
}
.tab-nav button.selected {
    color: #90caf9 !important;
    border-bottom-color: #42a5f5 !important;
}
/* Accordion */
.accordion, details {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,255,255,0.08) !important;
}
/* Tables in markdown */
table { border-collapse: collapse; width: 100%; }
th { background: rgba(66,165,245,0.15) !important; color: #90caf9 !important; }
td, th { border: 1px solid rgba(255,255,255,0.1) !important; padding: 6px 10px; }
tr:nth-child(even) { background: rgba(255,255,255,0.03) !important; }
/* Label text */
label span, .label-wrap span {
    color: #90a4ae !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Forest Memory") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
# 🌲 Forest Memory
### Multimodal Ecological Resilience Reasoning with LiteRT + Gemma 4
*Can we detect ecological collapse before it's visible from space?*

---
    """)

    # ── How it works ──────────────────────────────────────────────────────────
    gr.HTML(HOW_IT_WORKS_HTML)

    gr.Markdown("---")

    # ── Map + comparison chart ────────────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Plot(value=build_site_map(), show_label=False)
        with gr.Column(scale=5):
            gr.Plot(value=build_comparison_chart(), show_label=False)

    # ── Recovery chart + explanation ──────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.Plot(value=build_recovery_chart(), show_label=False)
        with gr.Column(scale=5):
            gr.Markdown(RECOVERY_EXPLANATION)

    gr.Markdown("---\n### Explore individual sites")

    # ── Site selector ─────────────────────────────────────────────────────────
    with gr.Row():
        dropdown = gr.Dropdown(
            choices=DROPDOWN_CHOICES,
            value=DROPDOWN_CHOICES[0],
            label="Select monitoring site",
            scale=2,
        )

    heading_out   = gr.Markdown()
    diagnosis_out = gr.HTML()

    # ── Acoustic chart + Gemma reports ───────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            gr.Markdown("**Acoustic Proxy Signals**")
            acoustic_out = gr.Plot(show_label=False)
            gr.Markdown("**Site Metadata & Flags**")
            metrics_out  = gr.Markdown()
        with gr.Column(scale=6):
            gr.Markdown("**Gemma 4 Ecological Reasoning**")
            with gr.Tabs():
                with gr.Tab("LiteRT Pipeline Report"):
                    litert_out = gr.Markdown()
                with gr.Tab("Structured Analysis (5 sections)"):
                    struct_out = gr.Markdown()

    # ── Accordions ────────────────────────────────────────────────────────────
    with gr.Accordion("Cross-Site Synthesis (Gemma 4, all 4 sites)", open=False):
        gr.Markdown(f"*{synthesis_model}*\n\n---\n\n{synthesis_md}")

    with gr.Accordion("Validation — Do Acoustic Scores Match Expert Classifications?", open=False):
        gr.Markdown("""
### BioSCape Expert Ground Truth vs. Acoustic Proxy Scores

The BioSCape ecology team's site classifications — fire history, vegetation age,
invasive species presence — provide an independent ecological health ranking.

| Site | Expert Rank | Bioacoustic Vitality | NDVI |
|------|:-----------:|:--------------------:|:----:|
| Healthy Baseline (s2lam027) | 4 — best | 41.1 ✅ | 0.507 |
| Burned Recovering (s2lam047) | 3 | 35.2 ✅ | 0.508 |
| **Invasive Disturbed (s2lam051)** | **2 — degraded** | **31.1** 🔴 | **0.677** 🟢 highest |
| Wet/Dry Pair (s2lam096) | 1 — poorest | 10.8 ✅ | — |

**Bioacoustic vitality scores show perfect rank-order agreement with expert ecological
classifications (Spearman ρ = 1.00, n = 4).**

NDVI alone would rank the invasive-disturbed site as the *healthiest* — the exact opposite.
Acoustic AI surfaces the stress that satellite greenness hides.

**Acoustic richness** showed no correlation (ρ = −0.20): soundscape *complexity* alone
is not a reliable health proxy. A site can be acoustically rich due to wind, rain, or
human noise while remaining biologically impoverished.

⚠️ *Limitation: n = 4 sites. This is directional evidence only.
Full validation requires the complete BioSCape dataset (50+ sites).*
        """)

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown("""
---
**Data:** BioSCape BioSoundSCape Acoustic Recordings · Sentinel-2 SR Harmonized (Google Earth Engine)
**Sensing (offline, CPU):** YAMNet 3 MB · EfficientNet-Lite0 5 MB — both via LiteRT, no GPU required
**Reasoning:** Gemma 4 (`gemma-4-26b-a4b-it` / `gemma-4-31b-it`) via Google AI Studio API
**Disclaimer:** All acoustic and spectral values are proxy signals only. Not validated biodiversity metrics.
*Built for the Kaggle Gemma 4 Good Hackathon — Global Resilience + LiteRT tracks.*
    """)

    # ── Event wiring ──────────────────────────────────────────────────────────
    outputs = [heading_out, diagnosis_out, acoustic_out, metrics_out, litert_out, struct_out]
    dropdown.change(fn=update, inputs=dropdown, outputs=outputs)
    demo.load(fn=lambda: update(DROPDOWN_CHOICES[0]), outputs=outputs)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())

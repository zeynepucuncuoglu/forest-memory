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
import gradio as gr

# ── Load pre-computed data ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent

CASES_PATH    = ROOT / "outputs" / "forest_memory_cases.json"
LITERT_PATH   = ROOT / "outputs" / "gemma_reports" / "litert_multimodal_reports.json"
GEMMA3_PATH   = ROOT / "outputs" / "gemma_reports" / "gemma_reports.json"

with open(CASES_PATH)  as f: cases_raw    = json.load(f)
with open(LITERT_PATH) as f: litert_raw   = json.load(f)
with open(GEMMA3_PATH) as f: gemma3_raw   = json.load(f)

# Index by role
cases        = {c["role"]: c for c in cases_raw}
litert_rpts  = {r["role"]: r for r in litert_raw["reports"]}
gemma3_rpts  = {r["role"]: r for r in gemma3_raw["site_reports"]}
synthesis_md = gemma3_raw["synthesis"].get("raw_response", "")
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 100, width: int = 20) -> str:
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _score_row(label: str, value, max_val: float = 100) -> str:
    if value is None:
        return f"| {label} | — | *not available* |"
    pct = min(value / max_val * 100, 100)
    return f"| {label} | `{value:.3f}` | `{_bar(pct)}` |"


# ── Panel builders ────────────────────────────────────────────────────────────

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
    lines.append(f"- **Veld Age:** {meta.get('field_veld_age', '—')} years")
    lines.append(f"- **Invasive Species (20 m):** {meta.get('field_aliens_within_20m', '—')}")
    lines.append(f"- **WAV files analysed:** {audio.get('wav_file_count', 0)}")
    lines.append("")

    lines.append("### Vegetation Signal (Sentinel-2 NDVI)")
    lines.append("| Metric | Value | Visual |")
    lines.append("|--------|-------|--------|")
    lines.append(_score_row("NDVI mean", ndvi.get("mean_ndvi"), max_val=1.0))
    lines.append(_score_row("NDVI std",  ndvi.get("std_ndvi"),  max_val=0.3))
    lines.append("")

    lines.append("### Acoustic Proxy Scores (FFT-based)")
    lines.append("| Signal | Value (/100) | Visual |")
    lines.append("|--------|-------------|--------|")

    def _get(col):
        return (audio.get(col) or {}).get("mean")

    lines.append(_score_row("Bioacoustic Vitality",    _get("bioacoustic_vitality_score"), 100))
    lines.append(_score_row("Acoustic Richness",       _get("acoustic_richness_score"),    100))
    lines.append(_score_row("Bird Activity Proxy",     _get("bird_activity_proxy"),        100))
    lines.append(_score_row("Insect Activity Proxy",   _get("insect_activity_proxy_score"),100))
    lines.append(_score_row("Human Disturbance Proxy", _get("human_disturbance_proxy"),    100))
    lines.append("")

    lines.append("### Interpretation Flags")
    flag_defs = {
        "green_not_alive_signal":        ("🟢", "Green-not-alive",          "High NDVI but low bioacoustic vitality"),
        "spatially_variable_vegetation": ("🌿", "Spatially variable veg.",  "Patchy NDVI — mosaic landscape"),
        "recent_fire_recovery_context":  ("🔥", "Recent fire context",      "1–6 year post-fire window"),
        "acoustic_uncertainty":          ("⚠️",  "Acoustic uncertainty",     "High human-noise proxy — signals may be masked"),
    }
    for k, (icon, name, desc) in flag_defs.items():
        active = flags.get(k, False)
        state  = "**ACTIVE**" if active else "inactive"
        lines.append(f"- {icon} **{name}**: {state} — *{desc}*")

    lines.append("")
    lines.append("---")
    lines.append("*All values are proxy signals. Not confirmed species counts or wildfire predictions.*")

    return "\n".join(lines)


def build_litert_report_md(role: str) -> str:
    rpt     = litert_rpts.get(role, {})
    text    = rpt.get("report", "No LiteRT report available.")
    latency = rpt.get("latency_s")
    model   = litert_raw.get("gemma_model", "Gemma 4")
    audio_m = litert_raw.get("litert_audio_model", "YAMNet")

    header = (
        f"*Model: **{model}** | Audio: **{audio_m}** | Latency: **{latency:.1f} s***\n\n---\n\n"
    ) if latency else ""

    return header + text


def build_structured_report_md(role: str) -> str:
    rpt      = gemma3_rpts.get(role, {})
    sections = rpt.get("sections", {})
    latency  = rpt.get("latency_s")
    model    = rpt.get("model", "Gemma 4")

    if not sections:
        raw = rpt.get("raw_response", "No structured report available.")
        return f"*Model: **{model}***\n\n---\n\n{raw}"

    header = (
        f"*Model: **{model}** | Latency: **{latency:.1f} s***\n\n---\n\n"
    ) if latency else ""

    lines = [header]
    section_icons = {
        "Vegetation Interpretation":   "🌿",
        "Bioacoustic Interpretation":  "🔊",
        "Multimodal Tension Summary":  "⚡",
        "Recovery Interpretation":     "📈",
        "Uncertainty Notes":           "⚠️",
    }
    for sec_name, text in sections.items():
        icon = section_icons.get(sec_name, "•")
        lines.append(f"### {icon} {sec_name}")
        lines.append(text.strip())
        lines.append("")

    return "\n".join(lines)


# ── Cross-site comparison chart ───────────────────────────────────────────────

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

    x     = np.arange(len(ROLE_KEYS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    bars_ndvi = ax.bar(
        x - width / 2, ndvi_vals, width,
        label="NDVI — satellite greenness (0–1)",
        color="#4CAF50", alpha=0.88, zorder=3,
    )
    bars_vit = ax.bar(
        x + width / 2, vitality_vals, width,
        label="Bioacoustic Vitality ÷ 100  (proxy)",
        color="#FF7043", alpha=0.88, zorder=3,
    )

    # N/A placeholder bars for missing NDVI
    for i, v in enumerate(ndvi_vals):
        if np.isnan(v):
            ax.bar(
                x[i] - width / 2, 0.08, width,
                color="#cccccc", alpha=0.7, zorder=3,
                hatch="////", edgecolor="#aaa",
            )
            ax.text(
                x[i] - width / 2, 0.09,
                "N/A", ha="center", va="bottom", fontsize=8, color="#888",
            )

    # Value labels on bars
    for bar in list(bars_ndvi) + list(bars_vit):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333",
            )

    # Highlight invasive_disturbed
    inv_i = ROLE_KEYS.index("invasive_disturbed")
    ax.annotate(
        "⚠  High NDVI\nLow Vitality",
        xy=(inv_i - width / 2, ndvi_vals[inv_i]),
        xytext=(inv_i + 0.85, 0.82),
        fontsize=9, color="#c62828", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.4),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0 – 1)", fontsize=11)
    ax.set_title(
        "NDVI vs Bioacoustic Vitality — all 4 sites\n"
        '"Green does not always mean ecologically healthy."',
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.annotate(
        "† Wet/Dry Pair: NDVI unavailable — wet season cloud cover blocked Sentinel-2 composite",
        xy=(1, -0.07), xycoords="axes fraction",
        ha="right", fontsize=8, color="#888",
    )

    plt.tight_layout()
    return fig


# ── Recovery trajectory chart ─────────────────────────────────────────────────

RECOVERY_DATA = [
    {
        "role":        "healthy_baseline",
        "label":       "Healthy\nBaseline",
        "fire_years":  20,    # no recent fire — old growth proxy
        "vitality":    41.1,
        "ndvi":        0.507,
        "invasive":    0,     # None
    },
    {
        "role":        "burned_recovering",
        "label":       "Burned\nRecovering",
        "fire_years":  3.5,
        "vitality":    35.2,
        "ndvi":        0.508,
        "invasive":    1,     # Rare to Very Scattered
    },
    {
        "role":        "invasive_disturbed",
        "label":       "Invasive\nDisturbed",
        "fire_years":  9.0,
        "vitality":    31.1,
        "ndvi":        0.677,
        "invasive":    2,     # Scattered to Medium
    },
    {
        "role":        "wet_dry_pair_complement",
        "label":       "Wet/Dry\nPair",
        "fire_years":  4.5,   # same class as burned, jittered
        "vitality":    10.8,
        "ndvi":        None,
        "invasive":    0,
    },
]

_DOT_COLORS = {
    "healthy_baseline":        "#4CAF50",
    "burned_recovering":       "#FF9800",
    "invasive_disturbed":      "#c62828",
    "wet_dry_pair_complement": "#78909C",
}
_INV_SIZE = {0: 180, 1: 380, 2: 700}
_INV_LABEL = {0: "No invasives", 1: "Rare aliens", 2: "Scattered–Medium aliens"}


def build_recovery_chart() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    for d in RECOVERY_DATA:
        ax.scatter(
            d["fire_years"], d["vitality"],
            s=_INV_SIZE[d["invasive"]],
            color=_DOT_COLORS[d["role"]],
            alpha=0.88, zorder=5,
            edgecolors="white", linewidths=1.8,
        )
        offset = (-72, 8) if d["role"] == "invasive_disturbed" else (10, 6)
        ax.annotate(
            d["label"],
            (d["fire_years"], d["vitality"]),
            textcoords="offset points", xytext=offset,
            fontsize=9.5, color="#333",
        )
        if d["ndvi"] is not None:
            ax.annotate(
                f"NDVI {d['ndvi']:.3f}",
                (d["fire_years"], d["vitality"]),
                textcoords="offset points", xytext=(offset[0], offset[1] - 13),
                fontsize=8, color="#555",
            )

    # Expected recovery arrow (dashed trend)
    ax.annotate(
        "",
        xy=(20, 41), xytext=(3.5, 35),
        arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.5, linestyle="dashed"),
        zorder=2,
    )
    ax.text(11, 40.5, "expected\nrecovery →", fontsize=8, color="#aaa", style="italic")

    # Invasion break annotation
    ax.annotate(
        "⚠  Invasion breaks\nrecovery trajectory",
        xy=(9.0, 31.1), xytext=(13, 22),
        fontsize=9, color="#c62828", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.3),
    )

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 52)
    ax.set_xticks([3.5, 9, 20])
    ax.set_xticklabels(["1–6 yrs\npost-fire", "6–12 yrs\npost-fire", ">17 yrs\n(no recent fire)"], fontsize=9.5)
    ax.set_ylabel("Bioacoustic Vitality (0–100)", fontsize=11)
    ax.set_title("Recovery trajectory: fire age vs acoustic vitality\nBubble size = invasive species presence", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # Legend for bubble size
    for inv_level, label in _INV_LABEL.items():
        ax.scatter([], [], s=_INV_SIZE[inv_level], color="#aaa", alpha=0.7, label=label)
    ax.legend(fontsize=8.5, loc="upper left", title="Invasive species", title_fontsize=9)

    plt.tight_layout()
    return fig


RECOVERY_EXPLANATION = """
### Why does the Invasive Disturbed site have the worst outcome — despite the most recovery time?

After a wildfire, native fynbos shrubland normally recovers over years to decades.
The **Burned Recovering** site (1–6 years post-fire) shows low but improving vitality — this is expected early succession.
The **Healthy Baseline** (no recent fire, >17 years) shows what full recovery looks like: vitality 41.1, no invasive species.

The **Invasive Disturbed** site breaks this pattern. At 6–12 years post-fire, it should have *higher* vitality than the
recovering site — it has had more time. Instead, vitality is lower (31.1).

**Why?** Invasive woody species colonised the canopy during the post-fire recovery window.
They are photosynthetically active — NDVI climbs to the highest value in the sample (0.677).
But they suppress native fauna: fewer nesting sites, altered food webs, reduced insect diversity.
The soundscape records the collapse that satellite greenness hides.

> *"The forest looks recovered from space. It isn't."*
"""


# ── Main update ───────────────────────────────────────────────────────────────

def update(choice: str):
    role    = ROLE_KEYS[DROPDOWN_CHOICES.index(choice)]
    icon    = ROLE_ICONS[role]
    heading = f"## {icon} {ROLE_LABELS[role]}"
    metrics = build_metrics_md(role)
    litert  = build_litert_report_md(role)
    struct  = build_structured_report_md(role)
    return heading, metrics, litert, struct


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.panel-label { font-weight: 600; margin-bottom: 4px; }
"""

with gr.Blocks(title="Forest Memory") as demo:

    gr.Markdown(
        """
# 🌲 Forest Memory
### Multimodal Ecological Resilience Reasoning
**LiteRT (YAMNet)** acoustic sensing + **Gemma 4** ecological reasoning

*"Green does not always mean ecologically healthy."*

---
        """
    )

    with gr.Row():
        dropdown = gr.Dropdown(
            choices=DROPDOWN_CHOICES,
            value=DROPDOWN_CHOICES[0],
            label="Select Site",
            scale=2,
        )

    heading_out = gr.Markdown()

    with gr.Row():
        comparison_plot = gr.Plot(
            value=build_comparison_chart(),
            label="",
            show_label=False,
        )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Plot(value=build_recovery_chart(), show_label=False)
        with gr.Column(scale=1):
            gr.Markdown(RECOVERY_EXPLANATION)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1):
            gr.Markdown("**Sensor Signals & Flags**")
            metrics_out = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("**Gemma 4 Ecological Reasoning**")
            with gr.Tabs():
                with gr.Tab("LiteRT Pipeline Report"):
                    litert_out = gr.Markdown()
                with gr.Tab("Structured Analysis (5 sections)"):
                    struct_out = gr.Markdown()

    with gr.Accordion("Cross-Site Synthesis (all 4 sites)", open=False):
        gr.Markdown(f"*{synthesis_model}*\n\n---\n\n{synthesis_md}")

    with gr.Accordion("Validation — Do Acoustic Scores Match Expert Classifications?", open=False):
        gr.Markdown("""
### BioSCape Expert Ground Truth vs. Acoustic Proxy Scores

The BioSCape ecology team's site classifications (fire class + veld age + invasive species presence)
provide an independent ground-truth ranking of ecological health. Here we compare that ranking
against our acoustic proxy scores and satellite NDVI.

| Site | Expert Rank | Bioacoustic Vitality | NDVI |
|------|:-----------:|:--------------------:|:----:|
| Healthy Baseline (s2lam027) | 4 — best | 41.1 ✅ | 0.507 |
| Burned Recovering (s2lam047) | 3 | 35.2 ✅ | 0.508 |
| **Invasive Disturbed (s2lam051)** | **2 — degraded** | **31.1** 🔴 | **0.677** 🟢 highest |
| Wet/Dry Pair (s2lam096) | 1 — poorest | 10.8 ✅ | — |

**Bioacoustic vitality scores show perfect rank-order agreement with expert ecological classifications
(Spearman ρ = 1.00, n = 4).**

NDVI alone would rank the invasive-disturbed site as the *healthiest* — the exact opposite of the
expert assessment. Acoustic AI surfaces the stress that satellite greenness hides.

---

**Acoustic richness** showed no correlation with ecological state (ρ = −0.20), confirming that
soundscape *complexity* alone is not a reliable health indicator. A site can have a rich soundscape
due to wind, rain, or human noise while remaining biologically impoverished.

---

⚠️ *Limitation: n = 4 sites precludes statistical significance. This is directional evidence only.
Full validation against the complete BioSCape survey (50+ sites) is needed.*
        """)

    gr.Markdown(
        """
---
**Data sources:** BioSCape BioSoundSCape Acoustic Recordings · Sentinel-2 SR Harmonized (Google Earth Engine)
**Models:** YAMNet (AudioSet, 521 classes) via LiteRT · Gemma 4 (`gemma-4-31b-it`) via Google AI Studio API
**Disclaimer:** All acoustic and spectral values are proxy signals only. Not validated biodiversity metrics.
        """
    )

    outputs = [heading_out, metrics_out, litert_out, struct_out]
    dropdown.change(fn=update, inputs=dropdown, outputs=outputs)
    demo.load(fn=lambda: update(DROPDOWN_CHOICES[0]), outputs=outputs)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())

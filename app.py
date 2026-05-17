"""
Forest Memory — Gradio Demo
Multimodal ecological resilience reasoning: LiteRT (YAMNet) + Gemma 4
"""

import json
from pathlib import Path

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

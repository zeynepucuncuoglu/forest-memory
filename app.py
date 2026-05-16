"""
Forest Memory — Gradio Demo
Multimodal ecological resilience reasoning: LiteRT (YAMNet) + Gemma 4
"""

import json
from pathlib import Path

import gradio as gr

# ── Load pre-computed data ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent

CASES_PATH   = ROOT / "outputs" / "forest_memory_cases.json"
REPORTS_PATH = ROOT / "outputs" / "gemma_reports" / "litert_multimodal_reports.json"

with open(CASES_PATH)   as f: cases_raw   = json.load(f)
with open(REPORTS_PATH) as f: reports_raw = json.load(f)

# Index by role
cases   = {c["role"]: c for c in cases_raw}
reports = {r["role"]: r for r in reports_raw["reports"]}

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

ROLE_KEYS = list(ROLE_LABELS.keys())
DROPDOWN_CHOICES = [ROLE_LABELS[r] for r in ROLE_KEYS]


# ── Helper: build metrics markdown ────────────────────────────────────────────

def _bar(value: float, max_val: float = 100, width: int = 20) -> str:
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _score_row(label: str, value, max_val: float = 100) -> str:
    if value is None:
        return f"| {label} | — | *not available* |"
    pct = min(value / max_val * 100, 100)
    bar = _bar(pct)
    return f"| {label} | `{value:.1f}` | `{bar}` |"


def build_metrics_md(role: str) -> str:
    case   = cases.get(role, {})
    audio  = case.get("audio", {})
    ndvi   = case.get("ndvi") or {}
    flags  = case.get("interpretation_flags", {})
    meta   = case.get("site_metadata", {})
    report = reports.get(role, {})

    lines = []

    # ── Site metadata ──────────────────────────────────────────────────────────
    lines.append("### Site Metadata")
    lines.append(f"- **Site ID:** `{case.get('site_id', '—')}`")
    lines.append(f"- **Land Cover:** {meta.get('land_cover_class', '—')}")
    lines.append(f"- **Fire Class:** {meta.get('fire_class', '—')}")
    lines.append(f"- **Veld Age:** {meta.get('field_veld_age', '—')} years")
    lines.append(f"- **Invasive Species (20 m):** {meta.get('field_aliens_within_20m', '—')}")
    lines.append(f"- **WAV files analysed:** {audio.get('wav_file_count', 0)}")
    lines.append("")

    # ── NDVI ──────────────────────────────────────────────────────────────────
    lines.append("### Vegetation Signal (Sentinel-2 NDVI)")
    lines.append("| Metric | Value | Visual |")
    lines.append("|--------|-------|--------|")
    ndvi_mean = ndvi.get("mean_ndvi")
    ndvi_std  = ndvi.get("std_ndvi")
    lines.append(_score_row("NDVI mean", ndvi_mean, max_val=1.0))
    lines.append(_score_row("NDVI std",  ndvi_std,  max_val=0.3))
    lines.append("")

    # ── Acoustic scores ────────────────────────────────────────────────────────
    lines.append("### Acoustic Proxy Scores (FFT-based)")
    lines.append("| Signal | Value (/100) | Visual |")
    lines.append("|--------|-------------|--------|")

    def _get(col):
        return (audio.get(col) or {}).get("mean")

    lines.append(_score_row("Bioacoustic Vitality",   _get("bioacoustic_vitality_score")))
    lines.append(_score_row("Acoustic Richness",      _get("acoustic_richness_score")))
    lines.append(_score_row("Bird Activity Proxy",    _get("bird_activity_proxy")))
    lines.append(_score_row("Insect Activity Proxy",  _get("insect_activity_proxy_score")))
    lines.append(_score_row("Human Disturbance Proxy", _get("human_disturbance_proxy")))
    lines.append("")

    # ── YAMNet scores ──────────────────────────────────────────────────────────
    yamnet = report.get("litert_yamnet", {})
    if yamnet and "error" not in yamnet:
        lines.append("### YAMNet Audio Scores (LiteRT inference)")
        lines.append("| Class | Score (0–1) | Visual |")
        lines.append("|-------|------------|--------|")
        for key in ("bird_activity_score", "insect_activity_score",
                    "rain_signal_score", "wind_signal_score",
                    "human_noise_score", "silence_score"):
            label = key.replace("_score", "").replace("_", " ").title()
            lines.append(_score_row(label, yamnet.get(key), max_val=1.0))
        lines.append("")

    # ── Interpretation flags ───────────────────────────────────────────────────
    lines.append("### Interpretation Flags")
    flag_defs = {
        "green_not_alive_signal":       ("🟢", "Green-not-alive signal",       "High NDVI but low bioacoustic vitality"),
        "spatially_variable_vegetation":("🌿", "Spatially variable vegetation", "Patchy NDVI — mosaic landscape"),
        "recent_fire_recovery_context": ("🔥", "Recent fire recovery context",  "1–6 year post-fire window"),
        "acoustic_uncertainty":         ("⚠️",  "Acoustic uncertainty",          "High human-noise proxy — signals may be masked"),
    }
    for k, (icon, name, desc) in flag_defs.items():
        active = flags.get(k, False)
        state  = "**ACTIVE**" if active else "inactive"
        lines.append(f"- {icon} **{name}**: {state} — *{desc}*")
    lines.append("")
    lines.append("---")
    lines.append("*All values are proxy signals. Do not interpret as confirmed species counts "
                 "or wildfire predictions. Values are relative within this site sample only.*")

    return "\n".join(lines)


def build_report_md(role: str) -> str:
    report = reports.get(role, {})
    text   = report.get("report", "No report available.")
    latency = report.get("latency_s")
    model   = reports_raw.get("gemma_model", "Gemma 4")
    audio_model = reports_raw.get("litert_audio_model", "YAMNet")

    header = (
        f"*Generated by **{model}** | Audio: **{audio_model}** "
        f"| Latency: **{latency:.1f} s***\n\n---\n\n"
    ) if latency else ""

    return header + text


# ── Main update function ───────────────────────────────────────────────────────

def update(choice: str):
    role = ROLE_KEYS[DROPDOWN_CHOICES.index(choice)]
    icon = ROLE_ICONS[role]
    heading = f"## {icon} {ROLE_LABELS[role]}"
    metrics = build_metrics_md(role)
    report  = build_report_md(role)
    return heading, metrics, report


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#heading { font-size: 1.3em; font-weight: bold; margin-bottom: 4px; }
#tagline  { color: #555; font-style: italic; margin-bottom: 12px; }
.panel-label { font-weight: 600; margin-bottom: 4px; color: #333; }
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

    heading_out = gr.Markdown(elem_id="heading")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.Markdown("**Sensor Signals & Flags**", elem_classes="panel-label")
            metrics_out = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("**Gemma 4 Ecological Reasoning**", elem_classes="panel-label")
            report_out = gr.Markdown()

    gr.Markdown(
        """
---
**Data sources:** BioSCape BioSoundSCape Acoustic Recordings · Sentinel-2 SR Harmonized (Google Earth Engine)
**Models:** YAMNet (AudioSet, 521 classes) via LiteRT · Gemma 4 via Google AI Studio API
**Disclaimer:** All acoustic and spectral values are proxy signals only. Not validated biodiversity metrics.
        """
    )

    # Wire up events
    dropdown.change(fn=update, inputs=dropdown,
                    outputs=[heading_out, metrics_out, report_out])

    # Initial render
    demo.load(fn=lambda: update(DROPDOWN_CHOICES[0]),
              outputs=[heading_out, metrics_out, report_out])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())

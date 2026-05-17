# 🌲 Forest Memory
### Multimodal Ecological Resilience Reasoning with LiteRT + Gemma 4

> *"Green does not always mean ecologically healthy."*

**Kaggle Gemma 4 Good Hackathon** — Global Resilience Track + LiteRT Track

---

## The Story Behind This Project

In the summers of 2021 and 2024, wildfires burned through hundreds of thousands of hectares of Turkish forests. Watching the satellite imagery of the destruction, I kept thinking: by the time a crisis is visible from space, it's already too late.

I wanted to build something that could **listen to a forest** — not just photograph it. A tool that detects ecological stress before it becomes disaster. Something cheap enough for a ranger in a remote area, with no internet, no cloud, no expensive infrastructure.

The problem: Turkey has no acoustic monitoring data for its forests. To test this idea, I needed a dataset that didn't exist in my own country yet.

That's why I turned to **BioSCape** (Biodiversity Survey of the Cape, South Africa) — one of the world's most comprehensive passive acoustic monitoring campaigns. BioSCape gave me real data to prove the concept works.

**Forest Memory is a prototype built on borrowed data, aimed at forests that have never been listened to.** When the resources allow, the goal is to deploy this in Turkish forests — and eventually anywhere that needs it.

---

## The Core Finding

Satellite NDVI (vegetation greenness) has a critical blind spot: invasive plants look green. A forest taken over by invasive species can score the highest greenness index while its native wildlife has entirely collapsed.

Across 4 BioSCape monitoring sites:

| Site | Expert Rank | Bioacoustic Vitality | NDVI |
|------|:-----------:|:--------------------:|:----:|
| Healthy Baseline | 4 — best | 41.1 ✅ | 0.507 |
| Burned Recovering | 3 | 35.2 ✅ | 0.508 |
| **Invasive Disturbed** | **2 — degraded** | **31.1** 🔴 | **0.677** 🟢 highest |
| Wet/Dry Pair | 1 — poorest | 10.8 ✅ | — |

Bioacoustic vitality scores show perfect rank-order agreement with BioSCape expert classifications (Spearman ρ = 1.00, n = 4). NDVI alone would rank the most degraded site as the healthiest.

---

## How It Works

```
LiteRT Ear  → YAMNet (3MB TFLite) classifies bird, insect, wind, human noise
LiteRT Eye  → Satellite RGB pixel analysis + NDVI (Sentinel-2)
Metadata    → BioSCape fire history, veld age, invasive species data

All signals → Gemma 4 → Ecological resilience report
```

**LiteRT** runs YAMNet locally — no internet, no API, 3MB model. This is what makes the tool deployable in the field: a $50 passive recorder + Raspberry Pi + LiteRT = a complete edge ecological sensor.

**Gemma 4** fuses acoustic, visual, and metadata signals into structured ecological reports — flagging multimodal tension (high NDVI, low acoustic vitality), recovery trajectories, and uncertainty.

---

## Why This Matters

The forests that need monitoring most are the ones furthest from data infrastructure. Field rangers in rural Turkey, the Amazon, or remote Borneo don't have reliable cloud access. Conservation NGOs in low-income countries can't afford continuous API costs.

This tool is designed to work where it's needed:
- Sensing layer: fully offline via LiteRT
- On-device Gemma (E2B/E4B) = fully offline reasoning (next step)
- Total hardware cost: under $100

---

## Project Structure

```
notebooks/
├── 01_audio_audit.ipynb          # WAV → FFT acoustic proxy scores
├── 03_gemma_ecological_reasoning.ipynb  # Gemma 4 structured analysis
└── 04_litert_edge.ipynb          # LiteRT (YAMNet) + Gemma 4 pipeline
src/
├── audio_signals.py              # FFT feature extraction
├── scoring.py                    # Proxy score computation
├── litert_sensing.py             # LiteRT ear + satellite eye
└── build_forest_memory_cases.py  # Multimodal case builder
outputs/gemma_reports/
├── gemma_reports.json            # Notebook 03 structured reports
└── litert_multimodal_reports.json  # Notebook 04 LiteRT reports
app.py                            # Gradio demo
```

---

## Running the Demo

```bash
pip install -r requirements.txt
python app.py  # → http://localhost:7860
```

---

## Scientific Constraints

All outputs use hedged language: *"proxy signal suggests"*, *"may indicate"*, *"consistent with"*. No exact species counts, no wildfire predictions, no definitive collapse diagnoses. Validated directionally against BioSCape expert classifications (n = 4; full statistical validation requires 50+ sites).

---

## Future Work

- Validate against full BioSCape dataset (50+ sites)
- Deploy on-device Gemma 4 E2B/E4B via LiteRT for fully offline reasoning
- Pilot in Turkish forests with locally collected acoustic data
- Extend to savanna, wetland, and tropical forest biomes

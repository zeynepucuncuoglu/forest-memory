# 🌲 Forest Memory
### Multimodal Ecological Resilience Reasoning with LiteRT + Gemma 4

> *"Green does not always mean ecologically healthy."*

**Kaggle Gemma 4 Good Hackathon submission** — using on-device AI (LiteRT/YAMNet) and large language model reasoning (Gemma 4) to detect hidden ecological stress in South African biodiversity hotspots.

---

## The Problem

Conservation teams rely heavily on satellite NDVI (vegetation greenness index) to assess ecosystem health. But NDVI has a critical blind spot: **invasive plant species can produce high greenness scores while completely suppressing native fauna.**

A site can look perfectly healthy from space — and be acoustically dead.

## The Core Finding

Across 4 BioSCape monitoring sites in the Cape Floristic Region:

| Site | Expert Ecological Rank | Bioacoustic Vitality | NDVI |
|------|----------------------|----------------------|------|
| Healthy Baseline (s2lam027) | 4 — best | 41.1 ✅ | 0.507 |
| Burned Recovering (s2lam047) | 3 | 35.2 ✅ | 0.508 |
| **Invasive Disturbed (s2lam051)** | **2 — degraded** | **31.1** 🔴 | **0.677** 🟢 highest |
| Wet/Dry Pair (s2lam096) | 1 — poorest | 10.8 ✅ | — |

The invasive-disturbed site has the **highest NDVI** (most "green" from space) but the **lowest bioacoustic vitality** (most acoustically dead on the ground). Bioacoustic vitality scores show perfect rank-order agreement with BioSCape expert site classifications (Spearman ρ = 1.00, n = 4).

NDVI would have ranked this site as the healthiest. Acoustic AI reveals the truth.

---

## Technical Approach

```
WAV recordings (BioSCape)
    ├── scipy FFT → acoustic proxy scores (vitality, richness, bird/insect activity)
    └── LiteRT (YAMNet TFLite) → ecological audio class scores
                                  (bird, insect, rain, wind, human noise)

Sentinel-2 imagery (Google Earth Engine)
    └── NDVI extraction (500 m buffer, dry season 2023)

BioSCape site metadata
    └── fire class, veld age, invasive species presence

All signals → Gemma 4 (gemma-4-31b-it) → ecological resilience report
```

### LiteRT as Sensory Organs

- **Ear:** YAMNet (AudioSet, 521 classes) runs via `ai_edge_litert` — classifies 0.96-second audio patches, aggregates bird, insect, rain, wind, and human-noise proxy scores
- **Eye:** Sentinel-2 RGB pixel statistics (green dominance, burned-area signal, spatial heterogeneity) complement NDVI from Earth Engine

### Gemma 4 as Reasoning Brain

Two reasoning pipelines:

1. **LiteRT Pipeline** (`04_litert_edge.ipynb`) — YAMNet scores + NDVI + FFT features → compact multimodal report
2. **Structured Analysis** (`03_gemma_ecological_reasoning.ipynb`) — 5-section structured report per site: Vegetation Interpretation, Bioacoustic Interpretation, Multimodal Tension, Recovery Interpretation, Uncertainty Notes → followed by a cross-site synthesis

---

## Datasets

1. **BioSCape BioSoundSCape Acoustic Recordings** — passive acoustic monitoring, Cape Floristic Region, South Africa
2. **BioSCape Acoustic Site Metadata** — expert site classifications (fire class, veld age, invasive species presence)
3. **Sentinel-2 SR Harmonized** via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`) — dry season 2023 composites

---

## Scientific Constraints

All acoustic and spectral values are **proxy signals** only:
- ✅ Ecosystem vitality estimation, proxy-based reasoning, uncertainty-aware interpretation
- ❌ Exact species counts, confirmed species detections, wildfire prediction, ecosystem collapse diagnosis

Proxy language is enforced throughout: *"proxy signal suggests"*, *"acoustic data may indicate"*, *"consistent with"*, *"uncertainty remains"*.

---

## Project Structure

```
forest-memory/
├── notebooks/
│   ├── 01_audio_audit.ipynb          # WAV → FFT acoustic proxy scores
│   ├── 03_gemma_ecological_reasoning.ipynb  # Gemma 4 structured analysis
│   └── 04_litert_edge.ipynb          # LiteRT (YAMNet) + Gemma 4 pipeline
├── src/
│   ├── audio_signals.py              # FFT feature extraction
│   ├── scoring.py                    # Proxy score computation
│   ├── litert_sensing.py             # LiteRT ear + satellite eye
│   └── build_forest_memory_cases.py  # Multimodal case builder
├── outputs/
│   ├── forest_memory_cases.json      # Merged multimodal cases
│   └── gemma_reports/
│       ├── gemma_reports.json        # Notebook 03 structured reports
│       └── litert_multimodal_reports.json  # Notebook 04 LiteRT reports
├── app.py                            # Gradio demo
└── requirements.txt
```

---

## Validation

Bioacoustic vitality scores align perfectly in rank order with BioSCape expert ecological classifications (Spearman ρ = 1.00, n = 4). Acoustic richness showed no correlation (ρ = −0.20), confirming that soundscape complexity alone does not indicate ecosystem health.

**Limitations:** n = 4 sites precludes statistical significance. Future validation against the full BioSCape survey (50+ sites) is needed.

---

## Running the Demo

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

---

## Future Work

- Validate against full BioSCape dataset (50+ sites) for statistical power
- Add a LiteRT image classification model for satellite RGB (MobileNet/EfficientNet) as a true on-device visual sensor
- Extend to other biomes (savanna, wetland, forest)
- Deploy as edge tool for field rangers without internet connectivity

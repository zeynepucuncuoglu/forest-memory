# Forest Memory — Technical Writeup
## Kaggle Gemma 4 Good Hackathon | Global Resilience Track | LiteRT Track

---

## 1. Why I Built This

In 2021 and 2024, wildfires burned through hundreds of thousands of hectares of Turkish forests. By the time satellite imagery confirmed the crisis, it was already too late. But wildfire is just one threat — invasive colonization, drought, and habitat degradation all degrade a forest silently, long before they become visible from space.

This raised a question I couldn't let go of: **could we detect ecological stress before it becomes visible from space?**

NDVI, the greenness index used by most monitoring systems, has a fundamental blind spot: it measures photosynthetic activity, not ecological function. A forest taken over by invasive plants appears perfectly green from orbit while its soundscape has collapsed. I wanted to build a tool that listens to the forest — not just photographs it.

**Turkey has no acoustic monitoring infrastructure for its forests.** To test this idea, I turned to **BioSCape** — the Biodiversity Survey of the Cape, South Africa — one of the world's most comprehensive passive acoustic monitoring campaigns. Forest Memory is a prototype built on BioSCape data, designed for deployment in forests that have never been listened to.

---

## 2. The Core Problem

Conservation monitoring systems rely on NDVI as the primary vegetation health signal. NDVI is powerful but insufficient: it tells you whether photosynthesis is happening, not whether a living ecosystem is functioning.

Our findings across 4 BioSCape sites make this concrete:

| Site | Expert Ecological Rank | Bioacoustic Vitality | NDVI |
|------|:---:|:---:|:---:|
| Healthy Baseline (s2lam027) | 4 — best | 41.1 | 0.507 |
| Burned Recovering (s2lam047) | 3 | 35.2 | 0.508 |
| **Invasive Disturbed (s2lam051)** | **2 — degraded** | **31.1** 🔴 | **0.677** 🟢 |
| Wet/Dry Pair (s2lam096) | 1 — poorest | 10.8 | — |

The invasive-disturbed site is the **greenest from space** and shows the **lowest bioacoustic vitality proxy** despite having the most post-fire recovery time. Six to twelve years after fire, invasive woody species colonised the canopy. They push NDVI up. They push native fauna out.

NDVI would have ranked this as the healthiest site. The soundscape tells the truth.

---

## 3. Technical Architecture

Forest Memory uses two AI layers working together:

```
LiteRT Ear  (YAMNet, 3MB TFLite) ──→ acoustic proxy scores
LiteRT Eye  (pixel analysis)     ──→ vegetation proxy signals
NDVI        (Sentinel-2 / EE)   ──→ greenness index
Metadata    (BioSCape survey)   ──→ fire history, veld age, invasives

All signals ──→ Gemma 4 ──→ ecological resilience report
```

### 3.1 LiteRT Sensing Layer

**YAMNet (the ear):** Google's AudioSet classifier runs via `ai_edge_litert` — fully local, no cloud dependency. WAV recordings are split into 0.96-second patches, each classified across 521 AudioSet classes. The primary ecological health signal — bioacoustic vitality — is derived from FFT spectral features, not from YAMNet class labels directly. YAMNet category scores (bird-range, insect-range, human noise) are supplementary exploratory signals passed to Gemma as additional context. All are presented as proxy signals, not species detections.

**EfficientNet-Lite0 (the eye):** A 5 MB TFLite model runs on the same Raspberry Pi CPU — also fully offline via `ai_edge_litert`. Used as a texture feature extractor on satellite RGB GeoTIFFs: raw logit activations encode visual texture patterns (texture energy, spread, high-activation ratio) without relying on ImageNet class labels, which are not meaningful for aerial imagery. Sentinel-2 NDVI, computed via Google Earth Engine, provides the greenness signal.

### 3.2 Gemma 4 Reasoning Layer

Two complementary pipelines:

Both pipelines use retrieval-augmented generation: before each site report, the system retrieves cross-site statistics (NDVI range, vitality range, per-site rankings) from a pre-computed feature store and injects them into the prompt. This grounds Gemma's output in relative ecological context — the model reasons about each site knowing where it stands among all four.

**Pipeline A — LiteRT Multimodal Report** (`04_litert_edge.ipynb`): YAMNet scores, EfficientNet texture features, NDVI, FFT acoustics, and site metadata are fused with retrieved cross-site context. Gemma 4 (`gemma-4-26b-a4b-it`) produces a 4-section grounded report per site.

**Pipeline B — Agentic Multi-Step Analysis** (`03_gemma_ecological_reasoning.ipynb`): A structured 5-section analysis (Vegetation, Bioacoustic, Multimodal Tension, Recovery, Uncertainty) is generated per site, then a second Gemma call synthesises all four reports into a cross-site comparison — a two-step agentic reasoning chain.

Both pipelines use `google.genai` SDK with retry logic and temperature = 0.3 for consistent, conservative outputs.

### 3.3 Scientific Constraints

All outputs use strictly hedged language: *"proxy signal suggests"*, *"may indicate"*, *"consistent with"*, *"uncertainty remains"*. The system never claims exact species counts, never predicts wildfire probability, and never diagnoses ecosystem collapse as definitive. These constraints are enforced at the prompt level.

---

## 4. Validation

Bioacoustic vitality scores show **perfect rank-order agreement** with a metadata-derived ecological severity ordering across the four selected sites — based on fire history, vegetation age, and invasive species pressure from the BioSCape survey records (Spearman ρ = 1.00, n = 4).

Critically, acoustic richness showed **no correlation** (ρ = −0.20), confirming that soundscape complexity alone is not a reliable health proxy. A site can be acoustically complex for non-biological reasons — wind, rain, human noise. Vitality, not richness, tracks ecological state.

**Honest limitation:** n = 4 sites. No statistical significance is claimed. This is directional evidence supporting the approach. Full validation requires the complete BioSCape dataset (50+ sites with ground truth).

---

## 5. Why LiteRT Matters for This Problem

The forests that need monitoring most are furthest from data infrastructure — field rangers in rural Turkey or remote Borneo have no reliable cloud, and conservation NGOs cannot afford continuous API costs.

**The sensing layer runs fully offline on CPU — no GPU required.** YAMNet (3 MB) and EfficientNet-Lite0 (5 MB) both run via `ai_edge_litert`, total footprint under 10 MB. A passive recorder, a Raspberry Pi, and these two LiteRT models equal a complete ecological sensor — total hardware cost under $100.

The reasoning layer currently uses Gemma 4 via API. A field ranger collects data offline, uploads when connectivity is available. On-device Gemma 4 E2B/E4B via LiteRT is the next step.

---

## 6. Results Summary

Across 8 generated reports (4 sites × 2 pipelines), Gemma 4 consistently distinguished early-recovery from invasion-suppression patterns, flagged acoustic uncertainty at the wet-season site, and applied proxy language throughout.

The key generated insight (Gemma 4, Multimodal Tension, invasive-disturbed site):
> *"A significant divergence exists between the high satellite-derived greenness and the low bioacoustic vitality. Such a gap is often associated with sites where invasive species dominate the biomass."*

A satellite operator looking at NDVI 0.677 would classify this site as thriving. Gemma — grounded in retrieved cross-site context and sensor data — flags it as ecologically stressed.

---

## 7. Future Work & Vision

- Validate against full BioSCape dataset (50+ sites) for statistical power
- Deploy Gemma 4 E2B/E4B via LiteRT for fully offline reasoning
- Pilot in Turkish forests with locally collected acoustic data
- Real-time alert mode: YAMNet anomaly detection + LoRaWAN radio on the same $100 hardware — batch monitoring becomes real-time early warning

Turkey loses forests every summer. The infrastructure to listen to them does not exist yet.

---

## 8. Reproducibility

All notebooks run on Kaggle (T4 GPU, CPU sufficient for notebooks 01 and 03).
Dataset: `zeynepucuncuoglu/forest-memory-data`
API key: `GOOGLE_API_KEY` stored as Kaggle Secret
YAMNet: auto-downloaded on first run (~3MB)

```bash
# Local demo
pip install -r requirements.txt
python app.py  # → http://localhost:7860
```

---

## 9. References & Data Sources

**Dataset**
- BioSCape / BioSoundSCape — Biodiversity Survey of the Cape, South Africa. Passive acoustic monitoring campaign, Western Cape. [bioscape.io](https://www.bioscape.io)

**Models**
- YAMNet — Google AudioSet audio classifier. TensorFlow Hub. [tfhub.dev/google/yamnet/1](https://tfhub.dev/google/yamnet/1)
- EfficientNet-Lite0 — TensorFlow Lite image model, used as texture feature extractor. [tfhub.dev/tensorflow/lite-model/efficientnet/lite0/uint8/2](https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/uint8/2)
- Gemma 4 — Google DeepMind. Accessed via Google AI Studio API (`gemma-4-26b-a4b-it`, `gemma-4-31b-it`).
- LiteRT (`ai_edge_litert`) — Google edge inference runtime. [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)

**Satellite Data**
- Sentinel-2 Surface Reflectance — European Space Agency / Copernicus. Accessed via Google Earth Engine.
- Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, 202, 18–27.

**Acoustic Methods**
- Gemmeke, J. F., et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events. *ICASSP 2017*.
- Bioacoustic vitality proxy derived from FFT spectral features (spectral centroid, flatness, high-frequency energy ratio).

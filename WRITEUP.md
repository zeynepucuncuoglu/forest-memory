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

**Pipeline A — LiteRT Multimodal Report** (`04_litert_edge.ipynb`): YAMNet scores, NDVI, FFT acoustics, and site metadata are fused into a single prompt. Gemma 4 (`gemma-4-26b-a4b-it`) produces a 4-section report per site, with cross-site NDVI and vitality context embedded so the model understands relative rankings.

**Pipeline B — Structured 5-Section Analysis** (`03_gemma_ecological_reasoning.ipynb`): A more detailed prompt produces structured sections — Vegetation Interpretation, Bioacoustic Interpretation, Multimodal Tension, Recovery Interpretation, Uncertainty Notes — followed by a cross-site synthesis comparing all four sites.

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

The forests that need monitoring most are the ones furthest from data infrastructure. A field ranger in rural Turkey, the Congo Basin, or remote Borneo does not have reliable cloud connectivity. Conservation NGOs in low-income countries cannot afford continuous API costs.

**The sensing layer runs fully offline on CPU — no GPU required.**

- **YAMNet (3 MB TFLite):** classifies 521 AudioSet sound categories from WAV recordings. Runs on a Raspberry Pi CPU in real time.
- **EfficientNet-Lite0 (5 MB TFLite):** analyses satellite RGB imagery as a texture feature extractor. Also runs on CPU — no GPU, no internet, no cloud.
- Both models run via `ai_edge_litert`. Total model footprint: under 10 MB.

The same code that runs on a developer's laptop runs identically on a $35 Raspberry Pi in a forest. A passive recorder, a Raspberry Pi, and these two LiteRT models equal a complete ecological sensor — total hardware cost under $100.

The reasoning layer currently uses Gemma 4 via API. This is an honest limitation: a field ranger can collect acoustic and visual data offline, then upload it for reasoning when connectivity is available (monthly resupply runs, satellite uplink). On-device Gemma 4 E2B/E4B via LiteRT is the next step — the architecture is designed for it.

---

## 6. Results Summary

Across 8 generated reports (4 sites × 2 pipelines), Gemma 4 consistently:
- Correctly integrated invasive-pressure metadata with NDVI and acoustic tension to flag the disturbed site
- Distinguished early-recovery acoustic patterns from invasion-suppression patterns
- Flagged acoustic uncertainty from high human-noise proxy at the wet-season site
- Applied proxy language throughout without over-claiming

The key generated insight at the invasive-disturbed site (Gemma 4, Multimodal Tension section):
> *"A significant divergence exists between the high satellite-derived greenness and the low bioacoustic vitality. This tension suggests that the vegetatively productive state indicated by the NDVI may not be supporting a proportional level of faunal activity. Such a gap is often associated with sites where invasive species dominate the biomass."*

And from the Uncertainty Notes section:
> *"NDVI may be reflecting invasive biomass rather than native forest health."*

These are the sentences that matter. A satellite operator looking at NDVI 0.677 would classify this site as thriving. Gemma correctly integrates the provided invasive-pressure metadata with NDVI and acoustic tension — surfacing a stress signal that NDVI alone would miss.

---

## 7. Future Work & Vision

**Immediate next steps:**
- Validate against full BioSCape dataset (50+ sites) for statistical power
- Deploy Gemma 4 E2B/E4B LiteRT for fully offline reasoning
- Pilot in Turkish forests with locally collected acoustic data

**Real-time alert mode — same hardware, different firmware:**

The current system operates in batch mode: sensors collect data offline, then upload when connectivity is available. The same $100 hardware stack could support a real-time alert mode: YAMNet detects acoustic anomalies (crackling, sudden silence), a small fire-detection TFLite model monitors camera input, and a sleep-and-wake LTE or LoRaWAN radio sends alerts only when an anomaly is confirmed. A single LoRaWAN gateway covers 15 km of forest at under $3/month per node. This is not a feature of the current prototype — it is the logical next deployment mode for the same LiteRT sensing architecture.

**The larger vision:**

Forest Memory is a continuous ecological health monitor — for any forest, any threat. Wildfire is one disturbance. Invasive colonization, drought, and disease all degrade a soundscape before they become visible from space. Turkey loses forests every summer. The infrastructure to listen to them does not exist yet. Forest Memory is designed to be cheap enough to deploy anywhere it is needed.

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

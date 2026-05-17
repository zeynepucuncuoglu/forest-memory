# Forest Memory — Technical Writeup
## Kaggle Gemma 4 Good Hackathon | Global Resilience Track | LiteRT Track

---

## 1. Why I Built This

In the summers of 2021 and 2024, I watched wildfires tear through Turkey's forests on the news. Hundreds of thousands of hectares burned. The images from space showed the destruction clearly — but by the time satellite imagery confirmed a crisis, it was already too late.

This raised a question I couldn't let go of: **could we detect ecological stress before it becomes visible from space?**

The challenge is that satellite imagery, specifically NDVI (the greenness index used by most monitoring systems), has a fundamental blind spot. It measures photosynthetic activity — not ecological function. A forest recovering after fire, or one taken over by invasive plants, can appear perfectly green from orbit while its native wildlife has collapsed entirely. The forest looks fine. It isn't.

I wanted to build a tool that listens to the forest — not just photographs it.

The second problem: **Turkey has no acoustic monitoring infrastructure for its forests.** No long-term passive audio recordings, no standardized sensor network, no labeled ecological dataset. To test this idea, I needed data that didn't exist in my own country yet.

That's why I turned to **BioSCape** — the Biodiversity Survey of the Cape, South Africa. BioSCape is one of the world's most comprehensive passive acoustic monitoring campaigns, covering the Cape Floristic Region, a global biodiversity hotspot facing the same pressures as Turkish forests: wildfire, invasive species, and climate-driven vegetation shifts. It gave me the data I needed to prove the concept.

**Forest Memory is a prototype.** It was built on BioSCape because that's where the data exists. When the resources become available, the goal is to deploy this exact pipeline in Turkish forests — where it's needed most.

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

The invasive-disturbed site is the **greenest from space** and the **most ecologically dead on the ground.** Six to twelve years after fire, invasive woody species colonised the canopy. They push NDVI up. They push native fauna out.

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

**YAMNet (the ear):** Google's AudioSet classifier runs via `ai_edge_litert` — fully local, no cloud dependency. WAV recordings are split into 0.96-second patches, each classified across 521 AudioSet classes. We aggregate into ecological categories: bird activity, insect activity, rain, wind, human noise. These are proxy signals — not species detections — and are presented as such throughout.

**Satellite analysis (the eye):** Sentinel-2 NDVI was computed via Google Earth Engine (500m buffer, dry season 2023 composites). RGB pixel statistics provide additional texture signals.

### 3.2 Gemma 4 Reasoning Layer

Two complementary pipelines:

**Pipeline A — LiteRT Multimodal Report** (`04_litert_edge.ipynb`): YAMNet scores, NDVI, FFT acoustics, and site metadata are fused into a single prompt. Gemma 4 (`gemma-4-26b-a4b-it`) produces a 4-section report per site, with cross-site NDVI and vitality context embedded so the model understands relative rankings.

**Pipeline B — Structured 5-Section Analysis** (`03_gemma_ecological_reasoning.ipynb`): A more detailed prompt produces structured sections — Vegetation Interpretation, Bioacoustic Interpretation, Multimodal Tension, Recovery Interpretation, Uncertainty Notes — followed by a cross-site synthesis comparing all four sites.

Both pipelines use `google.genai` SDK with retry logic and temperature = 0.3 for consistent, conservative outputs.

### 3.3 Scientific Constraints

All outputs use strictly hedged language: *"proxy signal suggests"*, *"may indicate"*, *"consistent with"*, *"uncertainty remains"*. The system never claims exact species counts, never predicts wildfire probability, and never diagnoses ecosystem collapse as definitive. These constraints are enforced at the prompt level.

---

## 4. Validation

The BioSCape survey team's own site classifications — fire history, vegetation age, invasive species presence — provide an independent ecological health ranking. Bioacoustic vitality scores show **perfect rank-order agreement** with this expert ranking (Spearman ρ = 1.00, n = 4).

Critically, acoustic richness showed **no correlation** (ρ = −0.20), confirming that soundscape complexity alone is not a reliable health proxy. A site can be acoustically complex for non-biological reasons — wind, rain, human noise. Vitality, not richness, tracks ecological state.

**Honest limitation:** n = 4 sites. No statistical significance is claimed. This is directional evidence supporting the approach. Full validation requires the complete BioSCape dataset (50+ sites with ground truth).

---

## 5. Why LiteRT Matters for This Problem

The forests that need monitoring most are the ones furthest from data infrastructure. A field ranger in rural Turkey, the Congo Basin, or remote Borneo does not have reliable cloud connectivity. Conservation NGOs in low-income countries cannot afford continuous API costs.

LiteRT makes the sensing layer genuinely deployable:
- YAMNet: 3MB TFLite, runs on a Raspberry Pi or Android phone
- No internet required for audio classification
- Battery-powered recorder + LiteRT device = a complete edge sensor

The current deployment uses Gemma 4 via API for the reasoning layer. On-device Gemma (E2B/E4B LiteRT) is the next step — the architecture is designed for it, and the notebook includes the LiteRT offline inference path. This brings the full reasoning capability offline, making the tool viable in environments with no connectivity at all.

---

## 6. Results Summary

Across 8 generated reports (4 sites × 2 pipelines), Gemma 4 consistently:
- Identified the multimodal tension at the invasive-disturbed site without explicit instruction
- Distinguished early-recovery acoustic patterns from invasion-suppression patterns
- Flagged acoustic uncertainty from high human-noise proxy at the wet-season site
- Applied proxy language throughout without over-claiming

The key generated insight at the invasive-disturbed site (Gemma 4, Multimodal Tension section):
> *"A significant divergence exists between the high satellite-derived greenness and the low bioacoustic vitality. This tension suggests that the vegetatively productive state indicated by the NDVI may not be supporting a proportional level of faunal activity. Such a gap is often associated with sites where invasive species dominate the biomass."*

And from the Uncertainty Notes section:
> *"NDVI may be reflecting invasive biomass rather than native forest health."*

These are the sentences that matter. A satellite operator looking at NDVI 0.677 would classify this site as thriving. The multimodal system flags it as ecologically compromised — without being told to look for invasive species.

---

## 7. Future Work & Vision

**Immediate next steps:**
- Validate against full BioSCape dataset (50+ sites) for statistical power
- Add MobileNet TFLite as on-device visual classifier (completing the LiteRT eye)
- Deploy Gemma 4 E2B/E4B LiteRT for fully offline reasoning

**The larger vision:**
Turkey loses forests to wildfire every summer. The infrastructure to detect early stress — unusual acoustic silence, shifts in bird activity proxies, NDVI divergence from acoustic signals — does not exist in Turkish forests yet. Forest Memory is designed to be cheap enough to deploy anywhere: a passive recorder costs under $100, satellite imagery is free via Earth Engine, LiteRT models are free and run on consumer hardware.

The prototype works. The data was borrowed from South Africa because that's where the monitoring infrastructure exists. The goal is to bring this capability to the forests that have never been listened to.

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

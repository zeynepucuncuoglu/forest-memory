# Forest Memory — Technical Writeup
## Kaggle Gemma 4 Good Hackathon

---

## 1. Motivation

Satellite remote sensing has transformed conservation biology — but it has a well-known limitation. The Normalized Difference Vegetation Index (NDVI), the most widely used proxy for vegetation health, measures **photosynthetic activity**, not **ecological function**. Invasive plant species, structurally recovering vegetation after fire, and monoculture plantations can all produce high NDVI while supporting a fraction of the native biodiversity of an intact ecosystem.

This project asks: **can passive acoustic monitoring close this gap?**

Sound is produced by living organisms. A site with birds, insects, frogs, and wind-rustling native shrubs produces a characteristically complex soundscape. A site dominated by invasive aliens — even a very green one — tends to be acoustically simplified. By combining acoustic AI with satellite signals and reasoning through a large language model, we can surface ecological stress that remote sensing alone misses.

---

## 2. Datasets

### 2.1 BioSCape BioSoundSCape Acoustic Recordings
Passive acoustic monitoring deployed across the Cape Floristic Region, South Africa — one of the world's six floral kingdoms and a global biodiversity hotspot. Recorders were placed at sites spanning a gradient of ecological conditions: intact fynbos, recently burned areas, and sites under pressure from invasive alien plants.

For this project we selected 4 representative sites (11 WAV files total) covering the full gradient:

| Role | Site ID | Land Cover | Fire Class | Veld Age | Invasives |
|------|---------|-----------|-----------|----------|-----------|
| healthy_baseline | s2lam027_231127 | Shrubland-Fynbos | No fire | Old (>17 yr) | None recorded |
| burned_recovering | s2lam047_231107 | Shrubland-Fynbos | 1–6 yr | Young (<6 yr) | Rare |
| invasive_disturbed | s2lam051_231030 | Forest-Woodland | 6–12 yr | Intermediate | Scattered–Medium |
| wet_dry_pair | s2lam096_230823 | Shrubland-Fynbos | 1–6 yr | Unknown | Unknown |

### 2.2 Sentinel-2 SR Harmonized (Google Earth Engine)
Median composites of Sentinel-2 surface reflectance imagery over the 2023 dry season (June–November), computed within a 500 m buffer around each acoustic monitoring site. NDVI calculated as (NIR − Red) / (NIR + Red) using bands B8 and B4.

### 2.3 BioSCape Site Metadata
Expert ecological classifications provided by the BioSCape survey team: fire history, vegetation age class, presence/density of invasive alien plants within 20 m of the recorder, land cover class. These serve as our ground-truth reference for validation.

---

## 3. Pipeline

### 3.1 Acoustic Feature Extraction (`01_audio_audit.ipynb`)

Each WAV file is processed with `scipy.fft` (no librosa dependency — chosen for Kaggle compatibility). We compute:

- **Bioacoustic Vitality Score (0–100):** energy in the 2–8 kHz band, normalized. This range encompasses the majority of bird and insect vocalizations in fynbos ecosystems.
- **Acoustic Richness Score (0–100):** spectral entropy across the full frequency range — a proxy for soundscape complexity.
- **Bird Activity Proxy:** band energy in 1.5–8 kHz.
- **Insect Activity Proxy:** band energy in 5–12 kHz.
- **Human Disturbance Proxy:** energy in 0–500 Hz (vehicle noise, equipment hum, low-frequency wind artifacts).

Multiple WAV files per site are aggregated (mean, min, max) to produce a single site-level acoustic profile.

### 3.2 LiteRT Acoustic Sensing (`04_litert_edge.ipynb`, `src/litert_sensing.py`)

YAMNet (Google, AudioSet, 521 classes) is run via `ai_edge_litert` — the official Google on-device inference library. The model is downloaded as a 3 MB `.tflite` file and executed locally, requiring no cloud inference for the audio analysis step.

Audio is resampled to 16 kHz mono, split into 0.96-second patches, and each patch is classified independently. Scores are averaged across all patches. AudioSet class indices are mapped to ecological categories:

| Ecological Category | AudioSet Class Indices |
|--------------------|----------------------|
| bird_activity | 0–22 (Bird vocalizations) |
| insect_activity | 71–75 (Insect, Cricket, Bee…) |
| rain_signal | 287–289 |
| wind_signal | 361–362 |
| human_noise | speech, vehicle, machinery classes |
| silence | 494 |

This constitutes the **LiteRT "ear"** — on-device acoustic sensing without cloud dependency.

### 3.3 Multimodal Case Assembly (`src/build_forest_memory_cases.py`)

All signals are merged into a structured JSON case per site: acoustic proxy scores, NDVI statistics, site metadata, satellite asset paths, and four interpretation flags derived from rule-based thresholds:

- `green_not_alive_signal`: NDVI ≥ 0.45 AND bioacoustic vitality ≤ 40 — the primary stress indicator
- `spatially_variable_vegetation`: NDVI std ≥ 0.10 — patchy, heterogeneous recovery
- `recent_fire_recovery_context`: FireClass contains "1-to-6" — early post-fire window
- `acoustic_uncertainty`: human disturbance proxy ≥ 45 — biological signals may be masked

### 3.4 Gemma 4 Ecological Reasoning

Two complementary reasoning pipelines:

**Pipeline A — LiteRT Multimodal Report (`04_litert_edge.ipynb`)**
YAMNet ecological class scores are combined with NDVI and FFT proxy scores into a single prompt. Gemma 4 (`gemma-4-31b-it`) produces a 4-section report per site: Vegetation Signal, Acoustic Signal, Multimodal Tension, Recovery Outlook.

**Pipeline B — Structured 5-Section Analysis (`03_gemma_ecological_reasoning.ipynb`)**
A more detailed prompt elicits a structured 5-section report: Vegetation Interpretation, Bioacoustic Interpretation, Multimodal Tension Summary, Recovery Interpretation, Uncertainty Notes. A final cross-site synthesis prompt compares all 4 sites and derives a comparative resilience narrative.

Both pipelines use `google.genai` SDK with retry logic (3 attempts, 5-second backoff) and conservative generation parameters (temperature = 0.3, max_output_tokens = 600–800).

---

## 4. Results

### 4.1 The Green-Not-Alive Finding

| Site | NDVI (satellite greenness) | Bioacoustic Vitality | Expert Rank |
|------|--------------------------|----------------------|-------------|
| healthy_baseline | 0.507 | **41.1** | 4 (best) |
| burned_recovering | 0.508 | 35.2 | 3 |
| **invasive_disturbed** | **0.677** | **31.1** | 2 (degraded) |
| wet_dry_pair | — | 10.8 | 1 (poorest) |

The invasive-disturbed site is the greenest site by NDVI — and the most ecologically degraded site by both acoustic measure and expert classification. 6–12 years post-fire, invasive woody species have colonised the canopy, suppressing native fynbos understorey and the fauna that depends on it.

### 4.2 Validation Against Expert Classifications

We used BioSCape expert metadata (fire class + veld age + invasive presence) to derive an ordinal ecological health ranking for the 4 sites. Bioacoustic vitality scores show **perfect rank-order agreement** with this ranking (Spearman ρ = 1.00, n = 4).

By contrast, acoustic richness showed no meaningful correlation (ρ = −0.20), confirming that soundscape complexity is not a reliable proxy for ecological health — a site can have a diverse soundscape for non-biological reasons (wind, human noise, rain).

**Limitation:** n = 4 precludes statistical significance. The p-value cannot be interpreted with this sample size. This result is directional evidence, not a validated metric. Full validation against the BioSCape dataset (50+ sites) is required.

### 4.3 Gemma 4 Report Quality

Across 8 generated reports (4 sites × 2 pipelines), Gemma 4 consistently:
- Applied proxy-hedged language ("may indicate", "consistent with", "uncertainty remains")
- Identified the multimodal tension at the invasive_disturbed site without explicit instruction
- Distinguished between early-recovery acoustic patterns (burned_recovering) and suppression patterns (invasive_disturbed)
- Flagged the wet_dry_pair site's acoustic uncertainty due to high human-noise proxy (62.4/100)

Sample — invasive_disturbed Multimodal Tension (Gemma 4):
> *"A significant divergence exists between the high satellite-derived greenness and the low bioacoustic vitality proxies. This is consistent with the green_not_alive_signal flag, suggesting that the detected vegetation may not support a functioning wildlife acoustic community."*

---

## 5. Limitations

- **Small sample:** 4 sites, 11 WAV files. No statistical significance claims are valid.
- **Proxy signals only:** YAMNet was not trained on fynbos species. AudioSet class indices are mapped heuristically to ecological categories. FFT band scores are not species detections.
- **Temporal mismatch:** NDVI is a dry-season composite; audio recordings span different campaign dates. The wet_dry_pair site was recorded in wet season, making direct acoustic comparison unreliable.
- **No LiteRT image model:** The "eye" currently uses pixel statistics (rasterio/PIL) rather than a TFLite image classification model. NDVI from Earth Engine is the primary vegetation signal.
- **API dependency:** Gemma 4 reasoning requires Google AI Studio API access. Edge deployment of Gemma is not yet demonstrated in this submission.

---

## 6. Future Work

- Validate against full BioSCape dataset (50+ sites) for statistically meaningful correlation
- Add MobileNet/EfficientNet TFLite model as on-device satellite image classifier (completing the LiteRT "eye")
- Deploy YAMNet + lightweight Gemma on edge hardware (Raspberry Pi / Android) for field use without internet
- Extend to savanna, wetland, and forest biomes with appropriate acoustic class mappings
- Integrate with real-time acoustic monitoring streams for continuous ecosystem tracking

---

## 7. Reproducibility

All notebooks are designed to run on Kaggle (T4 GPU, no GPU required for notebooks 01 and 03). Dataset path: `/kaggle/input/datasets/zeynepucuncuoglu/forest-memory-data/forest-memory/`.

API key required: `GOOGLE_API_KEY` stored as Kaggle Secret.

YAMNet TFLite model: auto-downloaded from Google Storage on first run (~3 MB).

```bash
# Local demo
pip install -r requirements.txt
python app.py
```

"""
litert_sensing.py — Forest Memory LiteRT sensing layer

LiteRT acts as two sensory organs:
  Ear: YAMNet (audio classification model) → produces bird, insect, rain,
       and human-noise proxy scores from WAV files.
  Eye: Extracts pixel statistics from RGB satellite imagery
       (green dominance, burned-area signal, brightness).

The outputs of these two layers are passed as text to Gemma 4 (the reasoning brain).

NOTE: YAMNet outputs are audio-class proxies — not direct species detections.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# ── YAMNet model ──────────────────────────────────────────────────────────────
YAMNET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite/task_library/audio_classification/android/"
    "lite-model_yamnet_classification_tflite_1.tflite"
)
YAMNET_DEFAULT_PATH = Path(__file__).parent.parent / "gemma" / "yamnet.tflite"

# YAMNet AudioSet class indices → ecological category
# Full list: https://raw.githubusercontent.com/tensorflow/models/master/
#             research/audioset/yamnet/yamnet_class_map.csv
ECOLOGICAL_MAP: dict[str, list[int]] = {
    "bird_activity":    list(range(0, 23)),     # Bird, Bird vocalization, ...
    "insect_activity":  list(range(71, 76)),    # Insect, Cricket, Bee, ...
    "rain_signal":      [287, 288, 289],        # Rain, Drizzle, ...
    "wind_signal":      [361, 362],             # Wind, Rustling leaves
    "frog_amphibian":   [81, 82, 83],           # Frog, Tree frog, ...
    "human_noise":      [0, 1, 132, 133, 134,  # Speech, Shout, Vehicle...
                         300, 301, 302, 303,
                         304, 305, 306],
    "silence":          [494],                  # Silence
}

YAMNET_SAMPLE_RATE = 16_000   # YAMNet expects 16 kHz mono
YAMNET_PATCH_SECONDS = 0.96   # Each YAMNet frame = 0.96 s


def _download_yamnet(dest: Path) -> Path:
    """Download YAMNet TFLite model if not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  Downloading YAMNet → {dest} (~3 MB) …", flush=True)
        urllib.request.urlretrieve(YAMNET_URL, dest)
        print("  Download complete.")
    return dest


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Rational resample using scipy."""
    if orig_sr == target_sr:
        return y
    g = gcd(orig_sr, target_sr)
    return resample_poly(y, target_sr // g, orig_sr // g).astype(np.float32)


def _load_interpreter(model_path: Path):
    """Load a LiteRT Interpreter (ai_edge_litert → tflite_runtime → tf.lite fallback)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def classify_audio_yamnet(
    wav_path: str | Path,
    yamnet_path: Optional[str | Path] = None,
) -> dict:
    """
    Classify a WAV file using YAMNet.

    Parameters
    ----------
    wav_path    : path to .wav file
    yamnet_path : path to YAMNet .tflite file (None → auto-download)

    Returns
    -------
    dict:
      bird_activity_score    0–1  (proxy — not a species detection)
      insect_activity_score  0–1
      rain_signal_score      0–1
      wind_signal_score      0–1
      human_noise_score      0–1
      silence_score          0–1
      n_frames               int  number of 0.96 s frames analysed
      sample_rate_used       int  (always 16000)
    """
    wav_path = Path(wav_path)
    model_path = Path(yamnet_path) if yamnet_path else YAMNET_DEFAULT_PATH
    model_path = _download_yamnet(model_path)

    # ── Load audio and convert to mono / 16 kHz ──────────────────────────────
    y_raw, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    y_mono = y_raw.mean(axis=1) if y_raw.ndim > 1 else y_raw
    y_16k = _resample(y_mono, sr, YAMNET_SAMPLE_RATE)

    # ── LiteRT inference ──────────────────────────────────────────────────────
    interp = _load_interpreter(model_path)
    inp_detail  = interp.get_input_details()[0]
    out_detail  = interp.get_output_details()[0]

    patch_len = int(YAMNET_SAMPLE_RATE * YAMNET_PATCH_SECONDS)
    n_classes = out_detail["shape"][-1]   # 521
    all_scores = []

    # Split audio into 0.96 s patches and run inference on each
    for start in range(0, len(y_16k) - patch_len + 1, patch_len):
        patch = y_16k[start : start + patch_len]
        interp.set_tensor(inp_detail["index"], patch.reshape(inp_detail["shape"]))
        interp.invoke()
        scores = interp.get_tensor(out_detail["index"]).flatten()
        all_scores.append(scores)

    if not all_scores:
        return {k: 0.0 for k in list(ECOLOGICAL_MAP.keys()) + ["n_frames", "sample_rate_used"]}

    mean_scores = np.mean(all_scores, axis=0)   # (521,)

    # ── Aggregate into ecological categories ─────────────────────────────────
    result: dict = {}
    for category, indices in ECOLOGICAL_MAP.items():
        valid = [i for i in indices if i < len(mean_scores)]
        result[f"{category}_score"] = float(np.mean(mean_scores[valid])) if valid else 0.0

    result["n_frames"]         = len(all_scores)
    result["sample_rate_used"] = YAMNET_SAMPLE_RATE
    return result


# ── Satellite RGB image analysis ──────────────────────────────────────────────

def analyze_satellite_rgb(tif_path: str | Path) -> dict:
    """
    Extract pixel statistics from an RGB GeoTIFF.

    When a full image-classification model is not available, direct pixel
    analysis provides useful proxy signals:
      - green_dominance  → proxy for photosynthetic activity
      - burned_area_signal → proxy for red/near-infrared ratio elevation
      - brightness       → overall reflectance

    Returns
    -------
    dict:
      green_dominance_proxy  0–1   green channel dominance
      burned_area_signal     0–1   red channel elevation (burn proxy)
      mean_brightness        0–1   overall brightness
      pixel_std              float pixel heterogeneity (green channel)
      data_source            str
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        return {
            "green_dominance_proxy": None,
            "burned_area_signal":    None,
            "mean_brightness":       None,
            "pixel_std":             None,
            "data_source":           "file_not_found",
        }

    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            data = src.read()   # (bands, H, W)

        # Assign R/G/B by band count
        if data.shape[0] >= 3:
            r = data[0].astype(np.float32)
            g = data[1].astype(np.float32)
            b = data[2].astype(np.float32)
        else:
            # Single band — treat all channels as equal
            r = g = b = data[0].astype(np.float32)

        # Normalise to 0–1 per channel
        for arr in [r, g, b]:
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr[:] = (arr - arr_min) / (arr_max - arr_min)

        total = r + g + b + 1e-6
        green_dom   = float(np.mean(g / total))
        burned_sig  = float(np.mean(r / total))
        brightness  = float(np.mean((r + g + b) / 3))
        pixel_std   = float(np.std(g))

        return {
            "green_dominance_proxy": round(green_dom,  4),
            "burned_area_signal":    round(burned_sig, 4),
            "mean_brightness":       round(brightness, 4),
            "pixel_std":             round(pixel_std,  4),
            "data_source":           tif_path.name,
        }

    except ImportError:
        # Fall back to PIL if rasterio is unavailable
        try:
            from PIL import Image
            img = Image.open(tif_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            total = r + g + b + 1e-6
            return {
                "green_dominance_proxy": round(float(np.mean(g / total)), 4),
                "burned_area_signal":    round(float(np.mean(r / total)), 4),
                "mean_brightness":       round(float(np.mean((r+g+b)/3)), 4),
                "pixel_std":             round(float(np.std(g)), 4),
                "data_source":           tif_path.name,
            }
        except Exception as e:
            return {
                "green_dominance_proxy": None,
                "burned_area_signal":    None,
                "mean_brightness":       None,
                "pixel_std":             None,
                "data_source":           f"error:{e}",
            }


def sense_case(case: dict, audio_dir: Path, sat_dir: Path,
               yamnet_path: Optional[Path] = None) -> dict:
    """
    Run the LiteRT sensing layer for one Forest Memory case.
    Runs YAMNet on all WAV files and averages the scores.
    Runs pixel analysis on the RGB TIF.

    Returns
    -------
    case dict with an added "litert_sensing" key containing:
      audio_yamnet:  dict  (averaged YAMNet scores across WAV files)
      satellite_rgb: dict  (RGB pixel statistics)
    """
    role     = case["role"]
    site_id  = case["site_id"]
    assets   = case.get("assets", {})

    # ── YAMNet: run on all WAV files, then average ────────────────────────────
    wav_files = assets.get("audio_files", [])
    yamnet_results = []
    for rel_path in wav_files:
        full = audio_dir.parent.parent / rel_path
        if not full.exists():
            full = audio_dir / role / Path(rel_path).name
        if full.exists():
            try:
                scores = classify_audio_yamnet(full, yamnet_path)
                yamnet_results.append(scores)
            except Exception as e:
                print(f"    YAMNet error ({full.name}): {e}")

    if yamnet_results:
        numeric_keys = [k for k in yamnet_results[0] if k not in ("n_frames", "sample_rate_used")]
        yamnet_avg = {
            k: round(float(np.mean([r[k] for r in yamnet_results])), 4)
            for k in numeric_keys
        }
        yamnet_avg["n_files"] = len(yamnet_results)
    else:
        yamnet_avg = {"error": "no_wav_files_found"}

    # ── Satellite RGB analysis ────────────────────────────────────────────────
    rgb_tif_rel = assets.get("rgb_tif")
    if rgb_tif_rel:
        rgb_tif_path = audio_dir.parent.parent / rgb_tif_rel
        rgb_features = analyze_satellite_rgb(rgb_tif_path)
    else:
        rgb_features = {"data_source": "not_available"}

    return {
        **case,
        "litert_sensing": {
            "audio_yamnet":   yamnet_avg,
            "satellite_rgb":  rgb_features,
        },
    }

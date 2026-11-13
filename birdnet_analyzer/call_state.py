from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CALL_STATE_LABELS = (
    "alarm",
    "territorial",
    "courtship",
    "contact",
    "distress",
    "neutral_song",
)

CALL_STATE_LABELS_TO_HUMAN = {
    "alarm": "Alarm / warning",
    "territorial": "Territorial display",
    "courtship": "Courtship / mating",
    "contact": "Contact / social",
    "distress": "Distress / stress",
    "neutral_song": "Neutral song / routine vocalization",
}


@dataclass(frozen=True)
class CallStatePrediction:
    species: str
    call_state: str
    confidence: float
    interpretation: str
    notes: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_audio(path: str) -> tuple[np.ndarray, float]:
    import librosa

    signal, sample_rate = librosa.load(path, sr=None, mono=True)
    return np.asarray(signal, dtype=np.float32), float(sample_rate)


def _estimate_features(path: str) -> dict[str, float]:
    import librosa

    signal, sample_rate = _load_audio(path)
    if signal.size == 0:
        return {
            "duration": 0.0,
            "rms_mean": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "tempo": 0.0,
            "zero_crossing_rate": 0.0,
            "energy_flux": 0.0,
        }

    duration = signal.size / sample_rate
    rms = librosa.feature.rms(y=signal)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sample_rate)
    pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
    pitch_mean = float(np.nanmean(pitch_values)) if pitch_values.size else 0.0
    pitch_std = float(np.nanstd(pitch_values)) if pitch_values.size else 0.0

    try:
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sample_rate)
        tempo_value = float(tempo)
    except Exception:
        tempo_value = 0.0

    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    energy_flux = float(np.mean(np.abs(np.diff(signal)))) if signal.size > 1 else 0.0

    return {
        "duration": float(duration),
        "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
        "spectral_centroid": float(np.nanmean(spectral_centroid)) if spectral_centroid.size else 0.0,
        "spectral_bandwidth": float(np.nanmean(spectral_bandwidth)) if spectral_bandwidth.size else 0.0,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "tempo": tempo_value,
        "zero_crossing_rate": float(np.nanmean(zcr)) if zcr.size else 0.0,
        "energy_flux": energy_flux,
    }


def _heuristic_call_state(features: dict[str, float]) -> tuple[str, float, str]:
    """Rule-based prototype for bird call-state inference.

    This is intentionally a transparent, interpretable placeholder. It is not a
    scientific claim about avian emotion. It classifies call morphology into a set of
    behavioral states that can later be replaced by a trained classifier on a labeled
    dataset.
    """
    duration = _safe_float(features.get("duration"), 0.0)
    centroid = _safe_float(features.get("spectral_centroid"), 0.0)
    bandwidth = _safe_float(features.get("spectral_bandwidth"), 0.0)
    pitch_mean = _safe_float(features.get("pitch_mean"), 0.0)
    pitch_std = _safe_float(features.get("pitch_std"), 0.0)
    tempo = _safe_float(features.get("tempo"), 0.0)
    zcr = _safe_float(features.get("zero_crossing_rate"), 0.0)
    energy_flux = _safe_float(features.get("energy_flux"), 0.0)

    if tempo > 2.5 and zcr > 0.08 and energy_flux > 0.03:
        label = "alarm"
        confidence = 0.78
        interpretation = "Short, sharp, rhythmically dense vocalization consistent with alarm or warning behavior."
    elif duration > 1.5 and pitch_std > 250 and centroid > 1500:
        label = "territorial"
        confidence = 0.72
        interpretation = "Long, high-energy pattern suggests territorial signaling or repeated nest defense calls."
    elif pitch_mean > 1800 and duration < 1.5 and tempo > 1.5:
        label = "courtship"
        confidence = 0.68
        interpretation = "Higher-pitched, rhythmic and patterned vocalization consistent with courtship or mating display."
    elif bandwidth < 700 and duration < 1.0 and pitch_std < 250:
        label = "contact"
        confidence = 0.71
        interpretation = "Stable, narrow-band call likely used for social contact or group coordination."
    elif energy_flux > 0.05 and pitch_std > 300 and duration < 1.2:
        label = "distress"
        confidence = 0.69
        interpretation = "High-variance, intense call structure suggesting stress or distress behavior."
    else:
        label = "neutral_song"
        confidence = 0.64
        interpretation = "Regular song-like pattern without strong alarm or territorial markers; likely routine vocalization."

    # Smooth the confidence using a tiny, explainable clamp.
    confidence = max(0.5, min(0.95, confidence))
    return label, confidence, interpretation


def infer_call_state(audio_path: str, species_name: str | None = None) -> CallStatePrediction:
    """Generate the species + likely call-state interpretation for a bird recording.

    The species is discovered with the BirdNET inference pipeline already used by this
    repository. The call-state label is inferred by a transparent, rule-based prototype
    that maps acoustic morphology to behavior-like labels. This prototype is intended as
    a training scaffold for a richer model trained on labeled bird-call examples.
    """
    from birdnet_analyzer.analyze import analyze

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        predictions = analyze(
            audio_input=audio_path,
            output=None,
            _return_only=True,
            model="birdnet",
            birdnet="3.0",
            min_conf=0.1,
            locale="en_us",
            n_workers=1,
            n_producers=1,
            batch_size=1,
            split_tables=False,
            save_params=False,
        )
    except Exception:
        predictions = None

    species = species_name or "Unknown species"
    if predictions is not None and predictions.inputs:
        df = predictions.to_dataframe()
        if not df.empty:
            species = str(df.iloc[0]["species_name"]).replace("_", " ")

    features = _estimate_features(audio_path)
    label, confidence, interpretation = _heuristic_call_state(features)
    return CallStatePrediction(
        species=species,
        call_state=label,
        confidence=float(confidence),
        interpretation=interpretation,
        notes=(
            "Prototype classifier: call state is inferred from acoustic morphology, not"
            " direct emotion recognition."
        ),
    )


def format_call_state(prediction: CallStatePrediction) -> dict[str, str | float]:
    return {
        "species": prediction.species,
        "call_state": CALL_STATE_LABELS_TO_HUMAN.get(prediction.call_state, prediction.call_state),
        "confidence": round(prediction.confidence, 3),
        "interpretation": prediction.interpretation,
        "notes": prediction.notes,
    }


__all__ = [
    "CALL_STATE_LABELS",
    "CALL_STATE_LABELS_TO_HUMAN",
    "CallStatePrediction",
    "format_call_state",
    "infer_call_state",
]

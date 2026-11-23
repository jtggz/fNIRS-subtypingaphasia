"""HbO feature extraction from preprocessed continuous channel recordings.

Inputs are kept outside the public repository. Event intervals are zero-based,
start-inclusive, end-exclusive and must have been reviewed before analysis.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, detrend, filtfilt
from scipy.special import gammaln

N_ROI = 22
UPPER = np.triu_indices(N_ROI, 1)


def load_run(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    n = mat["nirsdata"]
    signal = np.asarray(n.oxyData, dtype=float)
    valid = np.asarray(n.exception_channel).reshape(-1) == 0
    period = float(np.asarray(n.T).item())
    if signal.ndim != 2 or signal.shape[1] != len(valid) or period <= 0:
        raise ValueError(f"Invalid run: {path}")
    if not np.isfinite(signal[:, valid]).all():
        raise ValueError(f"Nonfinite valid channel: {path}")
    return signal, period, valid


def roi_mean(channels: np.ndarray, valid: np.ndarray, membership: np.ndarray) -> np.ndarray:
    if channels.shape[1] != len(membership) or len(valid) != len(membership):
        raise ValueError("ROI map/channel mismatch")
    out = []
    for roi in range(1, N_ROI + 1):
        use = (membership == roi) & valid
        if not use.any():
            raise ValueError(f"ROI {roi} has no valid channels")
        out.append(channels[:, use].mean(axis=1))
    return np.column_stack(out)


def _nextpow2_one35(n: int) -> int:
    if n < 1:
        raise ValueError("Empty signal")
    if n < 16:
        return 1 << int(math.ceil(math.log2(n))) if n > 1 else 1
    for candidate in range(n, (1 << int(math.ceil(math.log2(n)))) + 1):
        for multiplier in (1, 15, 3, 5):
            quotient, remainder = divmod(candidate, multiplier)
            if not remainder and quotient > 0 and quotient & (quotient - 1) == 0:
                return candidate
    raise RuntimeError("No suitable FFT length")


def malff(roi: np.ndarray, period: float) -> np.ndarray:
    n = len(roi)
    padded = _nextpow2_one35(n)
    low = math.ceil(0.01 * padded * period + 1) - 1
    high = math.trunc(0.08 * padded * period + 1)
    if low >= high:
        raise ValueError("ALFF band has no frequency bins")
    amplitude = 2 * np.abs(np.fft.fft(detrend(roi, axis=0), n=padded, axis=0)) / n
    raw = amplitude[low:high].mean(axis=0)
    if np.any(raw <= 0):
        raise ValueError("Invalid ALFF")
    return raw / raw.mean()


def fc(roi: np.ndarray) -> np.ndarray:
    r = np.corrcoef(roi, rowvar=False)[UPPER]
    if not np.isfinite(r).all() or np.any(np.abs(r) >= 1):
        raise ValueError("Undefined or perfect-correlation FC")
    return np.arctanh(r)


def _gamma_density(x: np.ndarray, shape: float, rate: float) -> np.ndarray:
    out = np.zeros_like(x)
    positive = x > 0
    out[positive] = np.exp((shape - 1) * np.log(x[positive]) + shape * np.log(rate)
                           - rate * x[positive] - gammaln(shape))
    return out


def hrf(period: float, shift: float = 0.0) -> np.ndarray:
    dt = period / 16
    x = np.arange(0, np.floor(32 / dt) + 1, dtype=float) - shift / dt
    micro = _gamma_density(x, 6, dt) - _gamma_density(x, 16, dt) / 6
    sampled = micro[(np.arange(0, np.floor(32 / period) + 1) * 16).astype(int)]
    return sampled / sampled.sum()


def matched_filter(x: np.ndarray, period: float) -> np.ndarray:
    fs = 1 / period
    if fs / 2 <= 0.08:
        raise ValueError("Sampling frequency too low")
    for cutoff, kind in ((0.08, "low"), (0.01, "high")):
        b, a = butter(3, cutoff, btype=kind, fs=fs)
        x = filtfilt(b, a, x, axis=0, padtype="odd", padlen=3 * (max(len(a), len(b)) - 1))
    return x


def design(n: int, period: float, events: pd.DataFrame, conditions: tuple[str, ...],
           derivative: bool = False) -> np.ndarray:
    required = {"condition", "start", "end"}
    if not required <= set(events.columns):
        raise ValueError(f"Events require {required}")
    columns = []
    canonical = hrf(period)
    shifted = hrf(period, 1.0)
    temporal = canonical - shifted
    temporal -= canonical * (canonical @ temporal) / (canonical @ canonical)
    for condition in conditions:
        box = np.zeros(n)
        selected = events.loc[events.condition == condition]
        if selected.empty:
            raise ValueError(f"Missing {condition} events")
        for row in selected.itertuples():
            start, end = int(row.start), int(row.end)
            if not 0 <= start < end <= n:
                raise ValueError("Event interval outside run")
            box[start:end] = 1
        columns.append(np.convolve(box, canonical)[:n])
        if derivative:
            columns.append(np.convolve(box, temporal)[:n])
    filtered = matched_filter(np.column_stack(columns), period)
    return np.column_stack((filtered, np.ones(n)))


def activation_beta(channels: np.ndarray, valid: np.ndarray, membership: np.ndarray,
                    period: float, events: pd.DataFrame, conditions: tuple[str, ...],
                    interval: tuple[int, int] | None = None) -> np.ndarray:
    x = design(len(channels), period, events, conditions)
    start, end = interval or (0, len(channels))
    beta = np.linalg.lstsq(x[start:end], channels[start:end, valid], rcond=None)[0][:-1]
    return roi_mean(beta, np.ones(valid.sum(), dtype=bool), membership[valid])


def background_fc(channels: np.ndarray, valid: np.ndarray, membership: np.ndarray,
                  period: float, events: pd.DataFrame, conditions: tuple[str, ...],
                  interval: tuple[int, int] | None = None) -> np.ndarray:
    x = design(len(channels), period, events, conditions, derivative=True)
    start, end = interval or (0, len(channels))
    y = channels[start:end, valid]
    x = x[start:end]
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return fc(roi_mean(residual, np.ones(valid.sum(), dtype=bool), membership[valid]))


def extract_subject(folder: Path, events: pd.DataFrame, membership: np.ndarray) -> dict[str, np.ndarray]:
    runs = {name: load_run(folder / f"{name}.mat") for name in ("rest", "naming", "fluency")}
    rest, period, valid = runs["rest"]
    rest_roi = roi_mean(rest, valid, membership)[:int(np.ceil(180 / period))]
    if len(rest_roi) * period < 180:
        raise ValueError("Rest recording shorter than 180 seconds")
    rest_fc = fc(rest_roi)
    intrinsic = np.r_[malff(rest_roi, period), rest_fc]
    conditions = {"naming": ("Naming",), "fluency": ("Phonemic", "Semantic")}
    betas = {}
    task_fc = {}
    for name in ("naming", "fluency"):
        signal, dt, mask = runs[name]
        rows = events.loc[events.run == name]
        counts = rows.condition.value_counts().to_dict()
        valid_counts = ((name == "naming" and counts in ({"Naming": 4}, {"Naming": 5})) or
                        (name == "fluency" and counts in ({"Phonemic": 5, "Semantic": 4},
                                                           {"Phonemic": 5, "Semantic": 5})))
        if not valid_counts:
            raise ValueError(f"Unexpected {name} event blocks")
        start = int(rows.analysis_start.iloc[0]) if "analysis_start" in rows else 0
        end = int(rows.analysis_end.iloc[0]) if "analysis_end" in rows else len(signal)
        if not 0 <= start < end <= len(signal):
            raise ValueError(f"Invalid {name} analysis interval")
        interval = (start, end)
        betas[name] = activation_beta(signal, mask, membership, dt, rows, conditions[name], interval)
        task_fc[name] = background_fc(signal, mask, membership, dt, rows, conditions[name], interval)
    naming = betas["naming"][0]
    phonemic, semantic = betas["fluency"]
    activation = np.r_[(naming + phonemic + semantic) / 3,
                       naming - (phonemic + semantic) / 2, semantic - phonemic]
    delta_n = task_fc["naming"] - rest_fc
    delta_f = task_fc["fluency"] - rest_fc
    reconfiguration = np.r_[(delta_n + delta_f) / 2, delta_n - delta_f]
    left = UPPER[0] % 2 == 0
    right = UPPER[0] % 2 == 1
    # ROI indices alternate left/right for each anatomical region.
    groups = (left & (UPPER[1] % 2 == 0), right & (UPPER[1] % 2 == 1),
              (UPPER[0] % 2) != (UPPER[1] % 2))
    summaries = []
    for context in (rest_fc, delta_n, delta_f):
        for group in groups:
            summaries.extend((context[group].mean(), np.abs(context[group]).mean()))
    return {"intrinsic": intrinsic, "activation": activation,
            "reconfiguration": reconfiguration,
            "profiling": np.r_[intrinsic[:22], naming, phonemic, semantic, summaries],
            "edge_profiles": np.r_[rest_fc, delta_n, delta_f]}

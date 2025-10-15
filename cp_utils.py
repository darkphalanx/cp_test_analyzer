"""
Critical Power Analysis – Refactored Core Functions
Author: Jordi & GPT‑5
Refactor: Unified lowercase column naming and normalization helper
"""

import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
#  Shared column normalization
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistent internal use."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        if c in ["power (w)", "power (watts)"]:
            rename_map[c] = "power"
        elif c == "power (w/kg)":
            rename_map[c] = "power_wkg"
        elif c in ["distance", "distance (m)", "stryd distance (m)"]:
            rename_map[c] = "watch distance (meters)"
        elif c in ["time", "elapsed time (s)"]:
            rename_map[c] = "timestamp"

    df = df.rename(columns=rename_map)
    return df

# ============================================================
#  CSV Handling
# ============================================================

def load_csv_auto(file):
    import io
    sample = file.getvalue().decode("utf-8", errors="ignore")[:500]
    sep = ";" if sample.count(";") > sample.count(",") else ","
    file.seek(0)
    df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8", errors="ignore")), sep=sep)
    return normalize_columns(df)

# ============================================================
#  Core Calculations
# ============================================================

def best_avg_power(df, window_sec, sampling_rate=1):
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")
    window_samples = int(window_sec * sampling_rate)
    rolling = df["power"].rolling(window_samples, min_periods=window_samples).mean()
    best_idx = rolling.idxmax()
    best_power = rolling.max()
    start_idx = max(0, best_idx - window_samples + 1)
    return best_power, start_idx, best_idx


def extend_best_segment(df, start_idx, end_idx, best_power, max_extend=60):
    df = normalize_columns(df)
    current_end = end_idx
    for _ in range(max_extend):
        if current_end + 1 >= len(df):
            break
        current_end += 1
        ext_pow = df.loc[start_idx:current_end, "power"].mean()
        if ext_pow < best_power:
            current_end -= 1
            break
        best_power = ext_pow
    duration = current_end - start_idx + 1
    return best_power, start_idx, current_end, duration


def compute_cp_linear(p1, t1, p2, t2):
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return cp, w_prime


def best_power_for_distance(df, distance_m):
    df = normalize_columns(df)
    if "watch distance (meters)" not in df.columns:
        raise ValueError("Missing column: watch distance (meters)")
    df["dist"] = pd.to_numeric(df["watch distance (meters)"], errors="coerce").ffill()
    dist_array = df["dist"].to_numpy()
    power_array = df["power"].to_numpy()
    best_power, best_start, best_end = 0, 0, 0
    j = 0
    for i in range(len(df)):
        target = dist_array[i] + distance_m
        while j < len(df) and dist_array[j] < target - 1.0:
            j += 1
        if j >= len(df):
            break
        avg_pow = power_array[i:j].mean()
        if avg_pow > best_power:
            best_power, best_start, best_end = avg_pow, i, j
    if best_power == 0:
        raise RuntimeError("No valid distance window found.")
    return best_power, best_start, best_end

# ============================================================
#  Models and Derived Metrics
# ============================================================

def compute_cp_5k_range(p):
    profiles = {"Aerobic": 0.985, "Balanced": 0.975, "Anaerobic": 0.965}
    return {label: p * f for label, f in profiles.items()}


def running_effectiveness(distance_m, duration_s, power_w, weight_kg):
    if duration_s <= 0 or power_w <= 0:
        return None
    velocity = distance_m / duration_s
    return (velocity * weight_kg) / power_w

# ============================================================
#  Segment Detection (mainly used in Segment Analysis)
# ============================================================

def prepare_power_series(df, sampling_rate=1, smooth_window_sec=5):
    df = normalize_columns(df)
    window = int(smooth_window_sec * sampling_rate)
    df["smooth_power"] = df["power"].rolling(window=window, center=True, min_periods=1).mean()
    return pd.to_numeric(df["smooth_power"], errors="coerce").to_numpy()


def average_min_max(power_array, chunk_size=10, sampling_rate=1):
    samples = int(chunk_size * sampling_rate)
    mins, maxs = [], []
    for i in range(0, len(power_array), samples):
        chunk = power_array[i:i + samples]
        if len(chunk) == 0:
            continue
        mins.append(np.min(chunk))
        maxs.append(np.max(chunk))
    return np.mean(mins), np.mean(maxs)


def merge_similar_segments(segments, merge_gap_sec=90, merge_diff_pct=0.03):
    if not segments:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start_elapsed"] - prev["end_elapsed"]
        diff = abs(seg["avg_power"] - prev["avg_power"]) / prev["avg_power"]
        if gap <= merge_gap_sec and diff <= merge_diff_pct:
            combined = {
                **prev,
                "end_idx": seg["end_idx"],
                "end_elapsed": seg["end_elapsed"],
                "duration_s": prev["duration_s"] + gap + seg["duration_s"],
                "avg_power": np.mean([prev["avg_power"], seg["avg_power"]]),
                "min_power": min(prev["min_power"], seg["min_power"]),
                "max_power": max(prev["max_power"], seg["max_power"]),
                "distance_m": prev["distance_m"] + seg["distance_m"],
                "pace_per_km": (prev["pace_per_km"] + seg["pace_per_km"]) / 2,
                "cv_%": np.mean([prev["cv_%"], seg["cv_%"]]),
            }
            merged[-1] = combined
        else:
            merged.append(seg)
    return merged


def detect_stable_segments_rolling(
    df,
    max_std_ratio=0.05,
    smooth_window_sec=6,
    allowed_spike_sec=5,
    sampling_rate=1,
    pause_threshold_w=5,
    pause_min_sec=3,
    min_speed_mps=0.5,
    show_diagnostics=True,
):
    df = normalize_columns(df)
    for col in ["power", "watch distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    window = int(smooth_window_sec * sampling_rate)
    roll_mean = df["power"].rolling(window=window, min_periods=1).mean()
    roll_std = df["power"].rolling(window=window, min_periods=1).std()
    stability_ok = (roll_std / roll_mean).fillna(0) <= max_std_ratio

    smooth_power = roll_mean.to_numpy()
    raw_power = df["power"].to_numpy()
    dist = pd.to_numeric(df["watch distance (meters)"], errors="coerce").ffill().to_numpy()
    speed = np.concatenate(([0.0], np.diff(dist)))
    times = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]

    segments, start, spike_count, pause_run = [], None, 0, 0
    spike_limit = int(allowed_spike_sec * sampling_rate)
    pause_need = int(pause_min_sec * sampling_rate)

    def close_segment(end_idx, reason):
        nonlocal start, spike_count
        if start is None:
            return
        end = max(end_idx, start)
        dur = (end - start + 1) / sampling_rate
        if dur <= 0:
            return
        seg_pow = smooth_power[start:end + 1]
        avg_p = float(np.nanmean(seg_pow))
        min_p = float(np.nanmin(seg_pow))
        max_p = float(np.nanmax(seg_pow))
        cv = float(100 * np.nanstd(seg_pow) / avg_p) if avg_p > 0 else None
        distance_m = float(dist[end] - dist[start]) if end < len(dist) else float("nan")
        pace = (dur / (distance_m / 1000.0)) if distance_m > 0 else None
        segments.append({
            "start_idx": start,
            "end_idx": end,
            "start_elapsed": (times.iloc[start] - t0).total_seconds(),
            "end_elapsed": (times.iloc[end] - t0).total_seconds(),
            "duration_s": dur,
            "avg_power": avg_p,
            "min_power": min_p,
            "max_power": max_p,
            "distance_m": distance_m,
            "pace_per_km": pace,
            "cv_%": cv,
            "end_reason": reason,
        })
        start, spike_count = None, 0

    n = len(df)
    for i in range(n):
        low_p = raw_power[i] < pause_threshold_w
        very_slow = speed[i] <= min_speed_mps
        paused = low_p and very_slow

        if paused:
            pause_run += 1
            if start is not None and pause_run == pause_need:
                end_idx = i - pause_run - spike_count
                if end_idx >= start:
                    close_segment(end_idx, f"stop/pause ≥{pause_min_sec}s")
            continue
        else:
            if pause_run > 0:
                if start is None:
                    pause_run = 0
                else:
                    spike_count += pause_run
                    pause_run = 0

        if stability_ok.iloc[i]:
            if start is None:
                start = i
            spike_count = 0
        else:
            if start is not None:
                spike_count += 1
                if spike_count > spike_limit:
                    end_idx = i - spike_count
                    close_segment(end_idx, f"instability spike > {allowed_spike_sec}s")

    if start is not None:
        end_idx = n - 1 - (spike_count + (pause_run if pause_run < pause_need else 0))
        end_idx = max(end_idx, start)
        close_segment(end_idx, "end of file")

    if show_diagnostics:
        try:
            eps = 1e-6
            safe_mean = roll_mean.where(roll_mean.abs() > eps, np.nan)
            diag_ratio = (roll_std / safe_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            stable_mask = (diag_ratio <= max_std_ratio)
            vals = stable_mask.values.astype(np.uint8)
            if vals.size:
                changes = np.where(np.diff(vals) != 0)[0] + 1
                boundaries = np.concatenate(([0], changes, [len(vals)]))
                lengths = np.diff(boundaries)
                states = vals[boundaries[:-1]]
                longest_true = lengths[states == 1].max() if np.any(states == 1) else 0
            else:
                longest_true = 0
            st.caption(f"Stability diagnostics — {100*stable_mask.mean():.1f}% ≤ {int(max_std_ratio*100)}% variability; longest stable streak: {int(longest_true)}s.")
        except Exception:
            pass

    return sorted(segments, key=lambda x: x["start_elapsed"])

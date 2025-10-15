"""
Critical Power Analysis – Simplified Core Functions (without segment analysis)
Author: Jordi & GPT-5
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
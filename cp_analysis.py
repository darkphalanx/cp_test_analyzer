#!/usr/bin/env python3
"""
Critical Power Analysis Script v3.0
Author: Jordi & GPT-5
Usage:
    python cp_analysis.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ---------- Helper Functions ---------- #

def load_csv_auto(file_path):
    """Load CSV with correct separator and decimal handling."""
    try:
        df = pd.read_csv(file_path, sep=",", decimal=".")
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=";", decimal=",")
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}")
    return df


def best_avg_power(df, window_sec, sampling_rate=1):
    """Find best rolling average power for a given duration (seconds)."""
    window_samples = int(window_sec * sampling_rate)
    rolling = df["power"].rolling(window_samples, min_periods=window_samples).mean()
    best_idx = rolling.idxmax()
    best_power = rolling.max()
    start_idx = max(0, best_idx - window_samples + 1)
    return best_power, start_idx, best_idx


def extend_best_segment(df, start_idx, end_idx, best_power, max_extend=60):
    """Extend a segment up to max_extend seconds if avg power stays >= best_power."""
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
    duration = int(current_end - start_idx + 1)
    return best_power, start_idx, current_end, duration


def compute_cp_linear(p1, t1, p2, t2):
    """Linear CP model (2-point)."""
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return cp, w_prime


def best_power_for_distance(df, distance_m):
    """Find best avg power over specified distance using Watch Distance (meters)."""
    if "Watch Distance (meters)" not in df.columns:
        raise ValueError("Column 'Watch Distance (meters)' not found in CSV.")

    # Clean and prepare distance
    df["dist"] = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill()
    df["dist"] = df["dist"].astype(float)

    # Sanity check
    dist_max = df["dist"].max()
    print(f"Loaded Watch Distance range: 0 → {dist_max:.1f} m")

    if dist_max < distance_m * 0.9:
        print("Warning: activity shorter than target distance.")

    best_power = 0
    best_start = 0
    best_end = 0
    n = len(df)

    dist_array = df["dist"].to_numpy()
    power_array = df["power"].to_numpy()

    # Efficient sliding window
    j = 0
    for i in range(n):
        target = dist_array[i] + distance_m
        while j < n and dist_array[j] < target - 1.0:
            j += 1
        if j >= n:
            break
        avg_pow = power_array[i:j].mean()
        if avg_pow > best_power:
            best_power = avg_pow
            best_start, best_end = i, j

    if best_power == 0:
        raise RuntimeError("Could not find valid distance window. Check distance scaling.")

    return best_power, best_start, best_end


def compute_cp_exponential(p, t, k=0.018, p_max=None):
    """
    Estimate Critical Power (CP) from a single test using the exponential model:
        P = CP + (Pmax - CP)e^{-k t}
    If Pmax not provided, assume Pmax ≈ P * 1.18 (typical for 3-min max vs 20-min effort).
    """
    if p_max is None:
        # Assume a realistic short-duration peak about 18% above current effort
        p_max = p * 1.18

    exp_term = np.exp(-k * t)
    cp = (p - p_max * exp_term) / (1 - exp_term)
    return cp

def compute_cp_5k_range(p):
    """
    Estimate Critical Power (CP) range from a 5 K time trial average power.
    """
    profiles = {
        "Aerobic": 0.985,     # slow fatigue
        "Balanced": 0.975,    # moderate fatigue
        "Anaerobic": 0.965,   # fast fatigue
    }
    return {label: p * factor for label, factor in profiles.items()}

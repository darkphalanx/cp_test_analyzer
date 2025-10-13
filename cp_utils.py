"""
Critical Power Analysis – Core Functions
Author: Jordi & GPT-5
Used by: Streamlit front-end (streamlit_app.py)
"""

import pandas as pd
import numpy as np

# ---------- CSV Handling ---------- #

def load_csv_auto(file):
    """Load CSV file and detect separator automatically."""
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file, sep=",")
    return df, ("," if "," in open(file.name).read(200) else ";")

# ---------- Core Calculations ---------- #

def best_avg_power(df, window_sec, sampling_rate=1):
    """Find best average power for a given duration."""
    window_samples = int(window_sec * sampling_rate)
    rolling = df["power"].rolling(window_samples, min_periods=window_samples).mean()
    best_idx = rolling.idxmax()
    best_power = rolling.max()
    start_idx = max(0, best_idx - window_samples + 1)
    return best_power, start_idx, best_idx


def extend_best_segment(df, start_idx, end_idx, best_power, max_extend=60):
    """
    Extend a detected best segment if slightly longer
    durations yield equal or higher average power.
    """
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
    """Compute CP and W′ using the linear 3/12-minute model."""
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return cp, w_prime


def best_power_for_distance(df, distance_m):
    """Find best avg power over a given distance (uses Watch Distance or Stryd Distance)."""
    # --- Detect column ---
    dist_col = None
    for col in df.columns:
        if "watch distance" in col.lower():
            dist_col = col
            break
        if "stryd distance" in col.lower():
            dist_col = col
            break
    if dist_col is None:
        raise ValueError(
            "No distance column found (expected 'Watch Distance' or 'Stryd Distance')."
        )

    # --- Normalize decimals and clean values ---
    df[dist_col] = (
        df[dist_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("[^0-9.]", "", regex=True)  # remove possible units or text
    )
    df[dist_col] = pd.to_numeric(df[dist_col], errors="coerce").ffill()
    df["dist"] = df[dist_col].astype(float)
    df = df.reset_index(drop=True)

    # --- Sanity check ---
    if df["dist"].max() < 1000:
        raise ValueError(
            f"Distance values look too small (max {df['dist'].max():.1f} m). "
            "Check if your CSV uses comma decimals or wrong column."
        )

    # --- Find best rolling segment ---
    best_power = 0
    start_idx = end_idx = 0
    for i in range(len(df)):
        target = df.loc[i, "dist"] + distance_m
        j = df["dist"].searchsorted(target, side="right") - 1
        if j <= i:
            continue
        avg_pow = df.loc[i:j, "power"].mean()
        if avg_pow > best_power:
            best_power = avg_pow
            start_idx, end_idx = i, j

    return best_power, start_idx, end_idx


# ---------- 5K Scaling Model ---------- #

def compute_cp_5k_range(p):
    """
    Estimate CP range from a 5 K time trial using empirical scaling factors.
    Represents different fatigue profiles.
    """
    profiles = {
        "Aerobic": 0.985,     # Slow fatigue (endurance-focused)
        "Balanced": 0.975,    # Typical runner
        "Anaerobic": 0.965,   # Fast fatigue (power-focused)
    }
    return {label: p * factor for label, factor in profiles.items()}

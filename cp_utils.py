"""
Critical Power Analysis – Core Functions
Author: Jordi & GPT-5
Used by: Streamlit front-end (streamlit_app.py)
"""

import pandas as pd
import numpy as np

# ---------- CSV Handling ---------- #

def load_csv_auto(file):
    """
    Load a CSV uploaded through Streamlit and automatically detect the separator.
    Works safely with UTF-8 encoding and both comma or semicolon delimited files.
    """
    import io

    # Read a short preview to detect delimiter
    sample = file.getvalue().decode("utf-8", errors="ignore")[:500]
    sep = ";" if sample.count(";") > sample.count(",") else ","

    # Reset pointer and read full file
    file.seek(0)
    df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8", errors="ignore")), sep=sep)

    return df


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
    """
    Find the best average power over the specified distance (meters),
    using the 'Watch Distance (meters)' column. This is preferred when
    the Stryd footpod is correctly configured as the distance source.
    """
    import numpy as np
    import pandas as pd

    # --- Validate ---
    if "Watch Distance (meters)" not in df.columns:
        raise ValueError("Column 'Watch Distance (meters)' not found in CSV.")

    # --- Prepare distance data ---
    df["dist"] = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill().astype(float)

    # --- Sanity check ---
    dist_max = df["dist"].max()
    print(f"Loaded Watch Distance range: 0 → {dist_max:.1f} m")
    if dist_max < distance_m * 0.9:
        print("Warning: activity shorter than target distance.")

    # --- Efficient sliding window search ---
    best_power = 0
    best_start = 0
    best_end = 0
    n = len(df)

    dist_array = df["dist"].to_numpy()
    power_array = df["power"].to_numpy()

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

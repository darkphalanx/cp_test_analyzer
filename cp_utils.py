"""
Critical Power Analysis – Core Functions
Author: Jordi & GPT-5
Used by: Streamlit front-end (streamlit_app.py)
"""

import pandas as pd
import numpy as np
import streamlit as st
import datetime

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
        st.warning("⚠️ Activity appears shorter than the target distance.")

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

# ---------- Segment Detection & Running Effectiveness ---------- #

def detect_segments(
    df,
    target_power,
    tolerance=0.05,
    min_duration_sec=300,
    sampling_rate=1,
    max_gap_sec=5,
    smooth_window_sec=30,
):
    """
    Detect continuous segments where average power stays within ±tolerance of target_power,
    allowing brief gaps (max_gap_sec) and applying optional smoothing.

    Args:
        df: DataFrame with 'power', 'Watch Distance (meters)', and 'timestamp' columns.
        target_power: Target wattage.
        tolerance: ± fraction (e.g., 0.05 = 5 %).
        min_duration_sec: Minimum segment duration in seconds.
        sampling_rate: Samples per second (default 1 Hz).
        max_gap_sec: Seconds allowed outside range before breaking a segment.
        smooth_window_sec: Rolling-average smoothing window (seconds).
    """
    import pandas as pd
    import datetime

    # --- Validate columns
    for col in ["power", "Watch Distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # --- Rolling-average smoothing to reduce second-to-second spikes
    window = int(smooth_window_sec * sampling_rate)
    df["smooth_power"] = (
        df["power"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )

    # --- Adaptive tolerance scaling
    # Loosen tolerance for long steady runs (>15 min)
    if min_duration_sec >= 1800:       # >30 min
        tolerance *= 1.5
    elif min_duration_sec >= 900:      # 15–30 min
        tolerance *= 1.2

    power = pd.to_numeric(df["smooth_power"], errors="coerce").to_numpy()
    dist = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill().to_numpy()
    times = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]

    lower = target_power * (1 - tolerance)
    upper = target_power * (1 + tolerance)
    in_zone = (power >= lower) & (power <= upper)

    segments = []
    start = None
    gap_count = 0
    max_gap = int(max_gap_sec * sampling_rate)

    for i, val in enumerate(in_zone):
        if val:
            if start is None:
                start = i
            gap_count = 0
        elif start is not None:
            gap_count += 1
            if gap_count > max_gap:
                end = i - gap_count
                duration = (end - start + 1) / sampling_rate
                if duration >= min_duration_sec:
                    avg_power = power[start:end + 1].mean()
                    avg_min_power, avg_max_power = average_min_max(
                        power[start:end + 1], chunk_size=10, sampling_rate=sampling_rate
                    )
                    distance_m = dist[end] - dist[start]
                    pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                    elapsed_start = str(datetime.timedelta(seconds=int((times.iloc[start] - t0).total_seconds())))
                    elapsed_end = str(datetime.timedelta(seconds=int((times.iloc[end] - t0).total_seconds())))
                    segments.append({
                        "start_idx": start,
                        "end_idx": end,
                        "elapsed_start": elapsed_start,
                        "elapsed_end": elapsed_end,
                        "duration_s": duration,
                        "avg_power": avg_power,
                        "min_power": avg_min_power,
                        "max_power": avg_max_power,
                        "distance_m": distance_m,
                        "pace_per_km": pace_per_km
                    })
                start = None
                gap_count = 0

    # --- Handle segment that continues to end
    if start is not None:
        end = len(in_zone) - 1
        duration = (end - start + 1) / sampling_rate
        if duration >= min_duration_sec:
            avg_power = power[start:end + 1].mean()
            avg_min_power, avg_max_power = average_min_max(
                power[start:end + 1], chunk_size=10, sampling_rate=sampling_rate
            )
            distance_m = dist[end] - dist[start]
            pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
            elapsed_start = str(datetime.timedelta(seconds=int((times.iloc[start] - t0).total_seconds())))
            elapsed_end = str(datetime.timedelta(seconds=int((times.iloc[end] - t0).total_seconds())))
            segments.append({
                "start_idx": start,
                "end_idx": end,
                "elapsed_start": elapsed_start,
                "elapsed_end": elapsed_end,
                "duration_s": duration,
                "avg_power": avg_power,
                "min_power": avg_min_power,
                "max_power": avg_max_power,
                "distance_m": distance_m,
                "pace_per_km": pace_per_km
            })

    return sorted(segments, key=lambda x: x["elapsed_start"])


def detect_stable_power_segments(
    df,
    max_std_ratio=0.05,
    min_duration_sec=300,
    sampling_rate=1,
    max_gap_sec=5,
):
    """
    Automatically detect steady-state power segments based on low variability.
    Allows brief (max_gap_sec) instability periods before ending a segment.

    Args:
        df: DataFrame containing 'power', 'Watch Distance (meters)', and 'timestamp'.
        max_std_ratio: Maximum allowed ratio (std / mean) within a window (default 5%).
        min_duration_sec: Minimum segment duration (seconds).
        sampling_rate: Samples per second (default 1 Hz).
        max_gap_sec: How many seconds of instability are tolerated (default = 5 s).
    """
    import pandas as pd
    import numpy as np
    import datetime

    for col in ["power", "Watch Distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    power = pd.to_numeric(df["power"], errors="coerce").to_numpy()
    dist = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill().to_numpy()
    times = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]

    window = int(min_duration_sec * sampling_rate)
    max_gap = int(max_gap_sec * sampling_rate)
    n = len(df)

    segments = []
    i = 0
    while i + window < n:
        segment = power[i:i + window]
        mean_p = segment.mean()
        std_p = segment.std()
        if mean_p > 0 and (std_p / mean_p) <= max_std_ratio:
            j = i + window
            gap_count = 0
            while j < n:
                ext_segment = power[i:j]
                mean_ext = ext_segment.mean()
                std_ext = ext_segment.std()
                if mean_ext > 0 and (std_ext / mean_ext) <= max_std_ratio:
                    j += 1
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > max_gap:
                        break
                    j += 1
            duration = (j - i) / sampling_rate
            if duration >= min_duration_sec:
                avg_power = power[i:j].mean()
                avg_min_power, avg_max_power = average_min_max(power[i:j], chunk_size=10, sampling_rate=sampling_rate)
                distance_m = dist[j - 1] - dist[i]
                pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                elapsed_start = str(datetime.timedelta(seconds=int((times.iloc[i] - t0).total_seconds())))
                elapsed_end = str(datetime.timedelta(seconds=int((times.iloc[j - 1] - t0).total_seconds())))
                segments.append({
                    "start_idx": i,
                    "end_idx": j,
                    "elapsed_start": elapsed_start,
                    "elapsed_end": elapsed_end,
                    "duration_s": duration,
                    "avg_power": avg_power,
                    "min_power": avg_min_power,
                    "max_power": avg_max_power,
                    "distance_m": distance_m,
                    "pace_per_km": pace_per_km
                })
            i = j
        else:
            i += window // 2

    return sorted(segments, key=lambda x: x["elapsed_start"])


def running_effectiveness(distance_m, duration_s, power_w, weight_kg):
    """
    Compute Running Effectiveness (RE).

    RE = (velocity * weight_kg) / power_w
    where velocity = distance_m / duration_s.

    Returns:
        Running Effectiveness (float)
    """
    if duration_s <= 0 or power_w <= 0:
        return None
    velocity = distance_m / duration_s  # m/s
    return (velocity * weight_kg) / power_w

def average_min_max(power_array, chunk_size=10, sampling_rate=1):
    """
    Compute average of minimums and maximums over fixed-duration chunks.

    Args:
        power_array: np.array of power values.
        chunk_size: size of each chunk in seconds.
        sampling_rate: samples per second.
    Returns:
        (avg_min, avg_max)
    """
    import numpy as np

    samples_per_chunk = int(chunk_size * sampling_rate)
    n = len(power_array)
    mins, maxs = [], []
    for i in range(0, n, samples_per_chunk):
        chunk = power_array[i:i + samples_per_chunk]
        if len(chunk) == 0:
            continue
        mins.append(np.min(chunk))
        maxs.append(np.max(chunk))
    return (np.mean(mins), np.mean(maxs))

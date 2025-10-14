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

# ============================================================
#  Manual (target/tolerance-based) detection
# ============================================================
def detect_segments(
    df,
    target_power,
    tolerance=0.05,
    min_duration_sec=300,
    sampling_rate=1,
    max_gap_sec=5,
    smooth_window_sec=5,
):
    # --- Column checks
    for col in ["power", "Watch Distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # --- Adaptive tolerance scaling
    if min_duration_sec >= 1800:
        tolerance *= 1.5
    elif min_duration_sec >= 900:
        tolerance *= 1.2

    power = prepare_power_series(df, sampling_rate, smooth_window_sec)
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
                    avg_min_power, avg_max_power = average_min_max(power[start:end + 1], 10, sampling_rate)
                    cv_pct = 100 * np.std(power[start:end + 1]) / avg_power
                    distance_m = dist[end] - dist[start]
                    pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                    start_elapsed = (times.iloc[start] - t0).total_seconds()
                    end_elapsed = (times.iloc[end] - t0).total_seconds()
                    segments.append({
                        "start_idx": start,
                        "end_idx": end,
                        "start_elapsed": start_elapsed,
                        "end_elapsed": end_elapsed,
                        "duration_s": duration,
                        "avg_power": avg_power,
                        "min_power": avg_min_power,
                        "max_power": avg_max_power,
                        "distance_m": distance_m,
                        "pace_per_km": pace_per_km,
                        "cv_%": cv_pct,
                    })
                start = None
                gap_count = 0

    # --- Tail segment
    if start is not None:
        end = len(in_zone) - 1
        duration = (end - start + 1) / sampling_rate
        if duration >= min_duration_sec:
            avg_power = power[start:end + 1].mean()
            avg_min_power, avg_max_power = average_min_max(power[start:end + 1], 10, sampling_rate)
            cv_pct = 100 * np.std(power[start:end + 1]) / avg_power
            distance_m = dist[end] - dist[start]
            pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
            start_elapsed = (times.iloc[start] - t0).total_seconds()
            end_elapsed = (times.iloc[end] - t0).total_seconds()
            segments.append({
                "start_idx": start,
                "end_idx": end,
                "start_elapsed": start_elapsed,
                "end_elapsed": end_elapsed,
                "duration_s": duration,
                "avg_power": avg_power,
                "min_power": avg_min_power,
                "max_power": avg_max_power,
                "distance_m": distance_m,
                "pace_per_km": pace_per_km,
                "cv_%": cv_pct,
            })

    # --- Merge similar segments
    return merge_similar_segments(sorted(segments, key=lambda x: x["start_elapsed"]))


# ============================================================
#  Auto (stability-based) detection
# ============================================================
def detect_stable_power_segments(
    df,
    max_std_ratio=0.05,
    min_duration_sec=300,
    sampling_rate=1,
    max_gap_sec=5,
    smooth_window_sec=5,
):
    for col in ["power", "Watch Distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # --- Adaptive std-ratio scaling
    global_std_ratio = df["power"].std() / df["power"].mean()
    if global_std_ratio < 0.08:
        max_std_ratio = min(max_std_ratio, 0.06)
    else:
        max_std_ratio = max(max_std_ratio, 0.07)

    power = prepare_power_series(df, sampling_rate, smooth_window_sec)
    dist = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill().to_numpy()
    times = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]

    window_len = int(min_duration_sec * sampling_rate)
    max_gap = int(max_gap_sec * sampling_rate)
    n = len(df)
    segments = []
    i = 0

    while i + window_len < n:
        segment = power[i:i + window_len]
        mean_p = segment.mean()
        std_p = segment.std()

        if mean_p > 0 and (std_p / mean_p) <= max_std_ratio:
            j = i + window_len
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
                avg_min_power, avg_max_power = average_min_max(power[i:j], 10, sampling_rate)
                cv_pct = 100 * np.std(power[i:j]) / avg_power
                distance_m = dist[j - 1] - dist[i]
                pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                start_elapsed = (times.iloc[i] - t0).total_seconds()
                end_elapsed = (times.iloc[j - 1] - t0).total_seconds()
                segments.append({
                    "start_idx": i,
                    "end_idx": j,
                    "start_elapsed": start_elapsed,
                    "end_elapsed": end_elapsed,
                    "duration_s": duration,
                    "avg_power": avg_power,
                    "min_power": avg_min_power,
                    "max_power": avg_max_power,
                    "distance_m": distance_m,
                    "pace_per_km": pace_per_km,
                    "cv_%": cv_pct,
                })
            i = j
        else:
            i += window_len // 2

    return merge_similar_segments(sorted(segments, key=lambda x: x["start_elapsed"]))

# ============================================================
#  Utility: merge consecutive similar segments
# ============================================================
def merge_similar_segments(segments, merge_gap_sec=90, merge_diff_pct=0.03):
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start_elapsed"] - prev["end_elapsed"]
        power_diff = abs(seg["avg_power"] - prev["avg_power"]) / prev["avg_power"]

        if gap <= merge_gap_sec and power_diff <= merge_diff_pct:
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

# ============================================================
#  Utility: average of minima and maxima over time chunks
# ============================================================
def average_min_max(power_array, chunk_size=10, sampling_rate=1):
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

# ============================================================
#  Shared pre-processing (used by both detectors)
# ============================================================
def prepare_power_series(df, sampling_rate=1, smooth_window_sec=5):
    window = int(smooth_window_sec * sampling_rate)
    df["smooth_power"] = (
        df["power"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    return pd.to_numeric(df["smooth_power"], errors="coerce").to_numpy()
    
 #----------------------------------------------------------------
def detect_target_segments_rolling(
    df,
    target_power,
    tolerance=0.05,
    smooth_window_sec=5,
    min_duration_sec=300,
    sampling_rate=1,
):
    """
    Detect continuous segments where the rolling-average power over the last X seconds
    stays within ± tolerance of the target power.

    Args:
        df: DataFrame with columns 'power', 'Watch Distance (meters)', 'timestamp'
        target_power: target power in watts
        tolerance: allowed ± fraction (e.g. 0.05 = ±5 %)
        smooth_window_sec: rolling-average window length (seconds)
        min_duration_sec: minimum segment duration in seconds
        sampling_rate: samples per second (default 1 Hz)
    """
    import pandas as pd
    import numpy as np
    import datetime

    for col in ["power", "Watch Distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # --- Rolling-average smoothing
    window = int(smooth_window_sec * sampling_rate)
    df["smooth_power"] = (
        df["power"].rolling(window=window, min_periods=1).mean()
    )

    power = pd.to_numeric(df["smooth_power"], errors="coerce").to_numpy()
    dist = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill().to_numpy()
    times = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]

    lower = target_power * (1 - tolerance)
    upper = target_power * (1 + tolerance)
    in_zone = (power >= lower) & (power <= upper)

    segments = []
    start = None

    for i, val in enumerate(in_zone):
        if val:
            if start is None:
                start = i
        elif start is not None:
            # segment ends when we exit the zone
            end = i - 1
            duration = (end - start + 1) / sampling_rate
            if duration >= min_duration_sec:
                seg_power = power[start:end + 1]
                avg_power = np.mean(seg_power)
                min_power = np.min(seg_power)
                max_power = np.max(seg_power)
                cv_pct = 100 * np.std(seg_power) / avg_power
                distance_m = dist[end] - dist[start]
                pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                start_elapsed = (times.iloc[start] - t0).total_seconds()
                end_elapsed = (times.iloc[end] - t0).total_seconds()
                segments.append({
                    "start_idx": start,
                    "end_idx": end,
                    "start_elapsed": start_elapsed,
                    "end_elapsed": end_elapsed,
                    "duration_s": duration,
                    "avg_power": avg_power,
                    "min_power": min_power,
                    "max_power": max_power,
                    "distance_m": distance_m,
                    "pace_per_km": pace_per_km,
                    "cv_%": cv_pct,
                })
            start = None

    # --- Handle segment continuing to end
    if start is not None:
        end = len(in_zone) - 1
        duration = (end - start + 1) / sampling_rate
        if duration >= min_duration_sec:
            seg_power = power[start:end + 1]
            avg_power = np.mean(seg_power)
            min_power = np.min(seg_power)
            max_power = np.max(seg_power)
            cv_pct = 100 * np.std(seg_power) / avg_power
            distance_m = dist[end] - dist[start]
            pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
            start_elapsed = (times.iloc[start] - t0).total_seconds()
            end_elapsed = (times.iloc[end] - t0).total_seconds()
            segments.append({
                "start_idx": start,
                "end_idx": end,
                "start_elapsed": start_elapsed,
                "end_elapsed": end_elapsed,
                "duration_s": duration,
                "avg_power": avg_power,
                "min_power": min_power,
                "max_power": max_power,
                "distance_m": distance_m,
                "pace_per_km": pace_per_km,
                "cv_%": cv_pct,
            })

    # Sort and return
    return sorted(segments, key=lambda x: x["start_elapsed"])

def detect_stable_segments_rolling(
    df,
    max_std_ratio=0.05,
    smooth_window_sec=6,
    allowed_spike_sec=5,     # max consecutive seconds outside variability
    min_duration_sec=300,
    sampling_rate=1,
    pause_threshold_w=5,     # stops (<5 W) end segment immediately
):
    """
    Detect low-variability (stable) segments using a rolling window.
    - Stability: rolling_std / rolling_mean <= max_std_ratio
    - Allowed spikes: up to 'allowed_spike_sec' consecutive unstable samples
    - Stops: power < pause_threshold_w ends segment immediately (separate segment)
    - Adds 'end_reason' to each segment ("stop/pause ~Xs", "spike > Xs", "end of file")
    """
    import pandas as pd, numpy as np, datetime

    # Validate
    for col in ["power", "Watch Distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    window = int(smooth_window_sec * sampling_rate)
    roll_mean = df["power"].rolling(window=window, min_periods=1).mean()
    roll_std  = df["power"].rolling(window=window, min_periods=1).std()
    stability_ok = (roll_std / roll_mean).fillna(0) <= max_std_ratio

    smooth_power = roll_mean.to_numpy()
    raw_power    = pd.to_numeric(df["power"], errors="coerce").to_numpy()
    dist         = pd.to_numeric(df["Watch Distance (meters)"], errors="coerce").ffill().to_numpy()
    times        = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]

    segments = []
    start = None
    spike_count = 0
    spike_limit = int(allowed_spike_sec * sampling_rate)

    n = len(df)
    for i in range(n):
        # If this is a stop, end current segment immediately
        if raw_power[i] < pause_threshold_w:
            # estimate pause length (peek ahead, does not change iteration)
            k = i
            while k < n and raw_power[k] < pause_threshold_w:
                k += 1
            pause_len_sec = (k - i) / sampling_rate

            if start is not None:
                end = i - 1
                end = max(end - spike_count, start)  # drop trailing spikes
                duration = (end - start + 1) / sampling_rate
                if duration >= min_duration_sec:
                    seg_pow = smooth_power[start:end+1]
                    avg_p = float(np.mean(seg_pow))
                    min_p = float(np.min(seg_pow))
                    max_p = float(np.max(seg_pow))
                    cv_pct = float(100 * np.std(seg_pow) / avg_p) if avg_p > 0 else None
                    distance_m = float(dist[end] - dist[start])
                    pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                    segments.append({
                        "start_idx": start,
                        "end_idx": end,
                        "start_elapsed": (times.iloc[start] - t0).total_seconds(),
                        "end_elapsed": (times.iloc[end] - t0).total_seconds(),
                        "duration_s": duration,
                        "avg_power": avg_p,
                        "min_power": min_p,
                        "max_power": max_p,
                        "distance_m": distance_m,
                        "pace_per_km": pace_per_km,
                        "cv_%": cv_pct,
                        "end_reason": f"stop/pause ~{int(round(pause_len_sec))}s",
                    })
            # reset state; stop creates a hard split
            start = None
            spike_count = 0
            continue

        # Not a stop; evaluate stability
        if stability_ok.iloc[i]:
            if start is None:
                start = i
            spike_count = 0
        else:
            # spike (unstable second)
            if start is not None:
                spike_count += 1
                if spike_count > spike_limit:
                    end = i - spike_count  # last stable index
                    duration = (end - start + 1) / sampling_rate
                    if duration >= min_duration_sec:
                        seg_pow = smooth_power[start:end+1]
                        avg_p = float(np.mean(seg_pow))
                        min_p = float(np.min(seg_pow))
                        max_p = float(np.max(seg_pow))
                        cv_pct = float(100 * np.std(seg_pow) / avg_p) if avg_p > 0 else None
                        distance_m = float(dist[end] - dist[start])
                        pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
                        segments.append({
                            "start_idx": start,
                            "end_idx": end,
                            "start_elapsed": (times.iloc[start] - t0).total_seconds(),
                            "end_elapsed": (times.iloc[end] - t0).total_seconds(),
                            "duration_s": duration,
                            "avg_power": avg_p,
                            "min_power": min_p,
                            "max_power": max_p,
                            "distance_m": distance_m,
                            "pace_per_km": pace_per_km,
                            "cv_%": cv_pct,
                            "end_reason": f"instability spike > {allowed_spike_sec}s",
                        })
                    start = None
                    spike_count = 0
            # else: not in a segment yet; wait for stability to start

    # Trailing segment (if we ended inside a segment)
    if start is not None:
        end = n - 1 - spike_count  # drop trailing spikes
        end = max(end, start)
        duration = (end - start + 1) / sampling_rate
        if duration >= min_duration_sec:
            seg_pow = smooth_power[start:end+1]
            avg_p = float(np.mean(seg_pow))
            min_p = float(np.min(seg_pow))
            max_p = float(np.max(seg_pow))
            cv_pct = float(100 * np.std(seg_pow) / avg_p) if avg_p > 0 else None
            distance_m = float(dist[end] - dist[start])
            pace_per_km = (duration / (distance_m / 1000)) if distance_m > 0 else None
            segments.append({
                "start_idx": start,
                "end_idx": end,
                "start_elapsed": (times.iloc[start] - t0).total_seconds(),
                "end_elapsed": (times.iloc[end] - t0).total_seconds(),
                "duration_s": duration,
                "avg_power": avg_p,
                "min_power": min_p,
                "max_power": max_p,
                "distance_m": distance_m,
                "pace_per_km": pace_per_km,
                "cv_%": cv_pct,
                "end_reason": "end of file",
            })

    return sorted(segments, key=lambda x: x["start_elapsed"])


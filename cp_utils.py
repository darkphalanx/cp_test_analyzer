
"""
Critical Power Analysis – Core Functions (3/12, 5K, PDC & Stable Blocks)
Author: Jordi & GPT-5

This module normalizes input DataFrame columns to lowercase and expects:
- "power" (W) or "power_wkg" (W/kg, to be converted by caller)
- "timestamp" (datetime or seconds since epoch; caller can parse)
- "watch distance (meters)" (preferred) or can be absent (some funcs handle)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ============================================================
#  Shared column normalization
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and common variants for consistent internal use."""
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

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

# ============================================================
#  CSV Handling
# ============================================================

def load_csv_auto(file) -> pd.DataFrame:
    """
    Load a CSV uploaded through Streamlit and automatically detect the separator.
    Works safely with UTF-8 encoding and both comma or semicolon delimited files.
    """
    import io
    if hasattr(file, "getvalue"):
        sample = file.getvalue().decode("utf-8", errors="ignore")[:500]
        sep = ";" if sample.count(";") > sample.count(",") else ","
        file.seek(0)
        df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8", errors="ignore")), sep=sep)
    else:
        # Fallback for file paths / file-like objects used outside Streamlit
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(file, sep=",")
    return normalize_columns(df)

# ============================================================
#  Core Calculations
# ============================================================

def best_avg_power(df: pd.DataFrame, window_sec: int, sampling_rate: int = 1):
    """Find best average power for a given duration (seconds)."""
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")

    window_samples = int(max(1, window_sec) * sampling_rate)
    rolling = df["power"].rolling(window_samples, min_periods=window_samples).mean()
    best_idx = int(rolling.idxmax())
    best_power = float(rolling.max())
    start_idx = max(0, best_idx - window_samples + 1)
    return best_power, start_idx, best_idx


def extend_best_segment(df: pd.DataFrame, start_idx: int, end_idx: int, best_power: float, max_extend: int = 60):
    """
    Extend a detected best segment if slightly longer durations yield equal or higher average power.
    """
    df = normalize_columns(df)
    current_end = int(end_idx)
    best_power = float(best_power)

    for _ in range(max(0, int(max_extend))):
        if current_end + 1 >= len(df):
            break
        current_end += 1
        ext_pow = float(df.loc[start_idx:current_end, "power"].mean())
        if ext_pow < best_power:
            current_end -= 1
            break
        best_power = ext_pow

    duration = int(current_end - start_idx + 1)
    return best_power, start_idx, current_end, duration


def compute_cp_linear(p1: float, t1: int, p2: float, t2: int):
    """Compute CP and W′ using the linear 3/12-minute model."""
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return float(cp), float(w_prime)


def best_power_for_distance(df: pd.DataFrame, distance_m: float):
    """
    Find best average power over the specified distance (meters).
    Uses 'watch distance (meters)' if present.
    """
    df = normalize_columns(df)
    dist_col = None
    for col in df.columns:
        if "watch distance" in col:
            dist_col = col
            break
        if "stryd distance" in col:
            dist_col = col
            break

    if dist_col is None:
        raise ValueError("No distance column found (expected 'Watch Distance (meters)' or 'Stryd Distance').")

    df["dist"] = pd.to_numeric(df[dist_col], errors="coerce").ffill().astype(float)

    best_power = 0.0
    start_idx = end_idx = 0
    n = len(df)

    # Two-pointer search using the monotonic distance column
    j = 0
    for i in range(n):
        target = df.loc[i, "dist"] + distance_m
        # advance j until we reach/overstep target
        while j < n and df.loc[j, "dist"] < target - 1.0:
            j += 1
        if j >= n or j <= i:
            continue
        avg_pow = float(df.loc[i:j, "power"].mean())
        if avg_pow > best_power:
            best_power = avg_pow
            start_idx, end_idx = i, j
    if best_power <= 0.0:
        raise RuntimeError("No valid distance window found. Check distance scaling in the CSV.")
    return best_power, start_idx, end_idx

# ============================================================
#  Models and Derived Metrics
# ============================================================

def compute_cp_5k_range(p: float) -> dict[str, float]:
    """Estimate CP range from a 5K TT using empirical scaling factors (different fatigue profiles)."""
    profiles = {"Aerobic": 0.985, "Balanced": 0.975, "Anaerobic": 0.965}
    return {label: p * f for label, f in profiles.items()}


def running_effectiveness(distance_m: float, duration_s: float, power_w: float, weight_kg: float | None):
    """
    Compute Running Effectiveness (RE) = (velocity * weight_kg) / power_w
    where velocity = distance_m / duration_s.
    """
    if duration_s <= 0 or power_w <= 0 or not weight_kg:
        return None
    velocity = distance_m / duration_s  # m/s
    return (velocity * weight_kg) / power_w

# ============================================================
#  Power Duration Curve (PDC)
# ============================================================

def compute_power_duration_curve(df: pd.DataFrame, max_duration_s: int = 3600, points: int = 60) -> pd.DataFrame:
    """
    Compute a power–duration curve using log-spaced durations between 5 s and max_duration_s.
    Returns a DataFrame with columns: duration_s, best_power_w.
    """
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")

    points = max(10, int(points))
    max_duration_s = int(max(5, max_duration_s))
    # log-spaced durations, unique and clipped to at least 5s
    durations = np.unique(np.clip(np.round(np.logspace(np.log10(5), np.log10(max_duration_s), num=points)).astype(int), 5, None))

    out = []
    for d in durations:
        window = int(d)
        roll = df["power"].rolling(window, min_periods=window).mean()
        m = float(roll.max()) if not pd.isna(roll.max()) else np.nan
        out.append((int(d), m))

    return pd.DataFrame(out, columns=["duration_s", "best_power_w"]).dropna()

# ============================================================
#  Stable Power Blocks (variability-based, simple & fast)
# ============================================================

def detect_stable_blocks(
    df: pd.DataFrame,
    max_std_ratio: float = 0.05,
    min_duration_sec: int = 60,
    smooth_window_sec: int = 5,
    sampling_rate: int = 1,
    weight_kg: float | None = None,
):
    """
    Identify stable-power blocks where rolling_std/rolling_mean <= max_std_ratio.
    Returns a list of dicts with stats:
    start_idx, end_idx, duration_s, avg_power, distance_m, pace_per_km, RE, start_elapsed, end_elapsed.
    """
    df = normalize_columns(df)
    for col in ["power", "watch distance (meters)", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    window = int(max(1, smooth_window_sec) * sampling_rate)
    roll_mean = df["power"].rolling(window=window, min_periods=1).mean()
    roll_std  = df["power"].rolling(window=window, min_periods=1).std()
    ratio = (roll_std / roll_mean).fillna(0)
    stable = (ratio <= max_std_ratio).to_numpy()

    dist = pd.to_numeric(df["watch distance (meters)"], errors="coerce").ffill().to_numpy(dtype=float)
    times = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    t0 = times.iloc[0]
    smooth_power = roll_mean.to_numpy(dtype=float)

    blocks = []
    n = len(df)
    start = None

    for i in range(n):
        if stable[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                dur = (end - start + 1) / sampling_rate
                if dur >= min_duration_sec:
                    avg_p = float(np.nanmean(smooth_power[start:end + 1]))
                    distance_m = float(dist[end] - dist[start])
                    pace_per_km = (dur / (distance_m / 1000.0)) if distance_m > 0 else None
                    RE = running_effectiveness(distance_m, dur, avg_p, weight_kg)
                    blocks.append({
                        "start_idx": start,
                        "end_idx": end,
                        "start_elapsed": (times.iloc[start] - t0).total_seconds(),
                        "end_elapsed": (times.iloc[end] - t0).total_seconds(),
                        "duration_s": dur,
                        "avg_power": avg_p,
                        "distance_m": distance_m,
                        "pace_per_km": pace_per_km,
                        "RE": RE,
                    })
                start = None

    # tail block
    if start is not None:
        end = n - 1
        dur = (end - start + 1) / sampling_rate
        if dur >= min_duration_sec:
            avg_p = float(np.nanmean(smooth_power[start:end + 1]))
            distance_m = float(dist[end] - dist[start])
            pace_per_km = (dur / (distance_m / 1000.0)) if distance_m > 0 else None
            RE = running_effectiveness(distance_m, dur, avg_p, weight_kg)
            blocks.append({
                "start_idx": start,
                "end_idx": end,
                "start_elapsed": (times.iloc[start] - t0).total_seconds(),
                "end_elapsed": (times.iloc[end] - t0).total_seconds(),
                "duration_s": dur,
                "avg_power": avg_p,
                "distance_m": distance_m,
                "pace_per_km": pace_per_km,
                "RE": RE,
            })

    return sorted(blocks, key=lambda x: x["start_elapsed"])

"""
Critical Power Analysis – Core Functions (3/12, 5K, PDC & Stable Blocks)
Author: Jordi & GPT-5
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# ============================================================
#  Shared column normalization
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    import io
    if hasattr(file, "getvalue"):
        sample = file.getvalue().decode("utf-8", errors="ignore")[:500]
        sep = ";" if sample.count(";") > sample.count(",") else ","
        file.seek(0)
        df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8", errors="ignore")), sep=sep)
    else:
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(file, sep=",")
    return normalize_columns(df)


# ============================================================
#  Core Calculations
# ============================================================

def best_avg_power(df: pd.DataFrame, window_sec: int, sampling_rate: int = 1):
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
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return float(cp), float(w_prime)


def best_power_for_distance(df: pd.DataFrame, distance_m: float):
    df = normalize_columns(df)
    dist_col = None
    for col in df.columns:
        if "watch distance" in col or "stryd distance" in col:
            dist_col = col
            break
    if dist_col is None:
        raise ValueError("No distance column found")

    df["dist"] = pd.to_numeric(df[dist_col], errors="coerce").ffill().astype(float)

    best_power = 0.0
    start_idx = end_idx = 0
    n = len(df)
    j = 0
    for i in range(n):
        target = df.loc[i, "dist"] + distance_m
        while j < n and df.loc[j, "dist"] < target - 1.0:
            j += 1
        if j >= n or j <= i:
            continue
        avg_pow = float(df.loc[i:j, "power"].mean())
        if avg_pow > best_power:
            best_power = avg_pow
            start_idx, end_idx = i, j
    if best_power <= 0.0:
        raise RuntimeError("No valid distance window found.")
    return best_power, start_idx, end_idx


# ============================================================
#  Derived Metrics
# ============================================================

def compute_cp_5k_range(p: float) -> dict[str, float]:
    profiles = {"Aerobic": 0.985, "Balanced": 0.975, "Anaerobic": 0.965}
    return {label: p * f for label, f in profiles.items()}


def running_effectiveness(distance_m: float, duration_s: float, power_w: float, weight_kg: float | None):
    if duration_s <= 0 or power_w <= 0 or not weight_kg:
        return None
    velocity = distance_m / duration_s
    return (velocity * weight_kg) / power_w


# ============================================================
#  Power Duration Curve
# ============================================================

def compute_power_duration_curve(df: pd.DataFrame, max_duration_s: int = 3600, step: int = 5):
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")

    durations = np.arange(5, max_duration_s + 1, step)
    powers = []
    for d in durations:
        roll = df["power"].rolling(d, min_periods=d).mean()
        powers.append(float(roll.max()) if roll.notna().any() else np.nan)
    return pd.DataFrame({"duration_s": durations, "best_power_w": powers}).dropna()


# ============================================================
#  New: Test Detection & Segment Extraction
# ============================================================

def detect_best_test_segments(df: pd.DataFrame, expected_durations=(180, 720), tolerance: float = 0.5):
    """
    Find the best power segments around expected durations (± tolerance),
    then extend them if the average power remains stable.
    """

    df = normalize_columns(df).copy()
    if "power" not in df.columns:
        raise ValueError("Missing power column")

    df["power_smooth"] = df["power"].rolling(3, min_periods=1).mean()

    segments = []
    for dur in expected_durations:
        lower, upper = int(dur * (1 - tolerance)), int(dur * (1 + tolerance))
        best_pow, best_dur, s_idx, e_idx = 0, 0, 0, 0
        for d in range(lower, upper + 1):
            roll = df["power_smooth"].rolling(d, min_periods=d).mean()
            if roll.notna().any() and roll.max() > best_pow:
                best_pow = float(roll.max())
                best_dur = d
                e_idx = int(roll.idxmax())
                s_idx = max(0, e_idx - d + 1)
        if best_pow > 0:
            # directly call extend_best_segment (same file)
            best_pow, s_idx, e_idx, new_dur = extend_best_segment(
                df, s_idx, e_idx, best_pow, max_extend=int(dur * 0.2)
            )
            segments.append({
                "target_dur": dur,
                "found_dur": new_dur,
                "avg_power": best_pow,
                "start_idx": s_idx,
                "end_idx": e_idx
            })
    return segments

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
    roll_std = df["power"].rolling(window=window, min_periods=1).std()
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


def infer_test_type_from_pdc(df: pd.DataFrame):
    """
    Infer whether the uploaded run resembles a 3/12-min test, a 5K TT, or a 20-min test.
    Uses simple duration-power ratios.
    """
    pdc = compute_power_duration_curve(df, max_duration_s=1800, step=5)
    if pdc.empty:
        return "Unknown"

    p5 = pdc.loc[pdc["duration_s"].between(160, 200), "best_power_w"].max(skipna=True)
    p12 = pdc.loc[pdc["duration_s"].between(660, 780), "best_power_w"].max(skipna=True)
    p20 = pdc.loc[pdc["duration_s"].between(1140, 1260), "best_power_w"].max(skipna=True)

    if pd.notna(p5) and pd.notna(p12):
        return "3/12-min Test"
    if pd.notna(p20):
        return "20-min TT"
    if pdc["duration_s"].max() > 1000:
        return "5K / Long TT"
    return "Unknown"

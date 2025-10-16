
"""
Critical Power Utilities v8
- Exact-duration + smart-extend best effort (optimized)
- Distance- or duration-based 5K analysis
- Stable block detection
- PDC via cumulative sums
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# ------------------------------
# Column normalization & CSV
# ------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c in ["power (w)", "power (watts)"]:
            rename_map[c] = "power"
        elif c == "power (w/kg)":
            rename_map[c] = "power_wkg"
        elif c in ["watch distance", "watch distance (m)", "distance", "distance (m)", "distance_m"]:
            rename_map[c] = "watch distance (meters)"
        elif c in ["stryd distance", "stryd distance (m)", "distance stryd (m)"]:
            rename_map[c] = "stryd distance (m)"
        elif c in ["time", "elapsed time (s)", "timestamp (s)"]:
            rename_map[c] = "timestamp"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

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

# ------------------------------
# Core math
# ------------------------------
def compute_cp_linear(p1: float, t1: int, p2: float, t2: int):
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return float(cp), float(w_prime)

def compute_cp_5k_range(p: float) -> dict[str, float]:
    profiles = {"Aerobic": 0.985, "Balanced": 0.975, "Anaerobic": 0.965}
    return {label: p * f for label, f in profiles.items()}

def running_effectiveness(distance_m: float, duration_s: float, power_w: float, weight_kg: float | None):
    if duration_s <= 0 or power_w <= 0 or not weight_kg:
        return None
    velocity = distance_m / duration_s  # m/s
    return (velocity * weight_kg) / power_w

# ------------------------------
# PDC (optimized with cumulative sums)
# ------------------------------
def compute_power_duration_curve(df: pd.DataFrame, max_duration_s: int = 3600, step: int = 5) -> pd.DataFrame:
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")
    s = pd.to_numeric(df["power"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cumsum = np.cumsum(np.insert(s, 0, 0.0))
    durations = np.arange(5, int(max_duration_s) + 1, int(max(1, step)))
    powers = []
    valid_durs = []
    for d in durations:
        if len(s) >= d:
            window_sums = cumsum[d:] - cumsum[:-d]
            powers.append(float(window_sums.max() / d))
            valid_durs.append(d)
    return pd.DataFrame({"duration_s": valid_durs, "best_power_w": powers}).dropna()

# ------------------------------
# Distance series helper
# ------------------------------
def _get_distance_series(df: pd.DataFrame) -> pd.Series | None:
    df = normalize_columns(df)
    dist = None
    if "watch distance (meters)" in df.columns:
        s = pd.to_numeric(df["watch distance (meters)"], errors="coerce").ffill()
        if s.notna().any() and float(s.max() - s.min()) > 0:
            dist = s
    if dist is None and "stryd distance (m)" in df.columns:
        s2 = pd.to_numeric(df["stryd distance (m)"], errors="coerce").ffill()
        if s2.notna().any() and float(s2.max() - s2.min()) > 0:
            dist = s2
    return dist

# ------------------------------
# Best-effort (exact duration then extend forward if avg stays >= base)
# ------------------------------
def find_best_effort(df: pd.DataFrame, duration_s: int, equality_tolerance: float = 0):
    """
    Find the best effort of at least `duration_s` seconds.
    1. Finds the highest average power over the exact duration.
    2. Extends forward as long as the new average power remains
       >= the original average (within tolerance).
    Does NOT include the second that causes a drop.
    """
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing power column")

    p = pd.to_numeric(df["power"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(p)
    if n < duration_s:
        raise RuntimeError("Activity too short for requested duration.")

    # Precompute cumulative sums for O(1) window averages
    cumsum = np.cumsum(np.insert(p, 0, 0.0))
    window_sums = cumsum[duration_s:] - cumsum[:-duration_s]
    rolling_means = window_sums / duration_s

    best_idx = int(np.argmax(rolling_means))
    start_idx = best_idx
    end_idx = best_idx + duration_s - 1

    base_sum = window_sums[best_idx]
    base_len = duration_s
    base_avg = base_sum / base_len

    # ✅ Extend forward only while the new mean stays >= base avg
    while end_idx < n - 1:
        next_mean = (base_sum + p[end_idx + 1]) / (base_len + 1)
        if next_mean + base_avg * equality_tolerance >= base_avg:
            base_sum += p[end_idx + 1]
            base_len += 1
            end_idx += 1
            base_avg = next_mean
        else:
            break

    return {
        "target_dur": int(duration_s),
        "found_dur": int(base_len),
        "avg_power": float(base_avg),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
    }


# ------------------------------
# Best-effort (distance-based, prefer watch distance then stryd)
# ------------------------------
def find_best_distance_effort(df: pd.DataFrame, distance_m: float, equality_tolerance: float = 0.001):
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing power column")
    dist_s = _get_distance_series(df)
    if dist_s is None:
        raise ValueError("No usable distance column found.")
    dist = dist_s.astype(float).to_numpy()
    p = pd.to_numeric(df["power"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(p)
    best_mean = -1.0
    best_pair = (0, 0)

    j = 0
    for i in range(n):
        target = dist[i] + distance_m
        while j < n and dist[j] < target - 0.5:
            j += 1
        if j >= n or j <= i:
            continue
        m = p[i:j+1].mean()
        if m > best_mean:
            best_mean = m
            best_pair = (i, j)

    if best_mean < 0:
        raise RuntimeError("No valid distance window found." )

    start, end = best_pair
    base_sum = p[start:end+1].sum()
    base_len = end - start + 1
    base_avg = base_sum / base_len

    # extend forward while mean doesn't get worse
    while end < n - 1:
        new_sum = base_sum + p[end + 1]
        new_len = base_len + 1
        new_mean = new_sum / new_len
        if new_mean + base_avg * equality_tolerance >= base_avg:
            base_sum, base_len, base_avg = new_sum, new_len, new_mean
            end += 1
        else:
            break

    found_distance = float(dist[end] - dist[start])
    return {
        "target_distance": float(distance_m),
        "found_distance": found_distance,
        "found_dur": int(base_len),
        "avg_power": float(base_avg),
        "start_idx": int(start),
        "end_idx": int(end),
    }

# ------------------------------
# Stable blocks (segment analysis)
# ------------------------------
def detect_stable_blocks(
    df: pd.DataFrame,
    max_std_ratio: float = 0.05,
    min_duration_sec: int = 60,
    smooth_window_sec: int = 5,
    sampling_rate: int = 1,
    weight_kg: float | None = None,
):
    df = normalize_columns(df)
    if "watch distance (meters)" not in df.columns and "stryd distance (m)" in df.columns:
        df["watch distance (meters)"] = pd.to_numeric(df["stryd distance (m)"], errors="coerce").ffill()

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

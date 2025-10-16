
from __future__ import annotations
import pandas as pd
import numpy as np

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

def compute_cp_linear(p1: float, t1: int, p2: float, t2: int):
    cp = (p1 * t1 - p2 * t2) / (t1 - t2)
    w_prime = (p1 - p2) * t1 * t2 / (t2 - t1)
    return float(cp), float(w_prime)

def compute_power_duration_curve(df: pd.DataFrame, max_duration_s: int = 3600, step: int = 5) -> pd.DataFrame:
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")
    s = df["power"].to_numpy(dtype=float)
    cumsum = np.cumsum(np.insert(s, 0, 0.0))
    durations = np.arange(5, int(max_duration_s) + 1, int(max(1, step)))
    powers = [(cumsum[d:] - cumsum[:-d]).max() / d for d in durations if len(s) >= d]
    return pd.DataFrame({"duration_s": durations[:len(powers)], "best_power_w": powers}).dropna()

def find_best_effort(df: pd.DataFrame, duration_s: int, equality_tolerance: float = 0.001):
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing power column")
    p = pd.to_numeric(df["power"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(p)
    if n < duration_s:
        raise RuntimeError("Activity too short for requested duration.")

    cumsum = np.cumsum(np.insert(p, 0, 0.0))
    window_sums = cumsum[duration_s:] - cumsum[:-duration_s]
    rolling_means = window_sums / duration_s
    best_idx = int(np.argmax(rolling_means))
    best_mean = float(rolling_means[best_idx])
    start_idx = best_idx
    end_idx = best_idx + duration_s - 1

    base_avg = best_mean
    base_sum = window_sums[best_idx]
    base_len = duration_s

    while end_idx < n - 1:
        new_sum = base_sum + p[end_idx + 1]
        new_len = base_len + 1
        new_mean = new_sum / new_len
        if new_mean + base_avg * equality_tolerance >= base_avg:
            base_sum = new_sum
            base_len = new_len
            end_idx += 1
            base_avg = new_mean
        else:
            break

    return {
        "target_dur": int(duration_s),
        "found_dur": int(base_len),
        "avg_power": float(base_avg),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
    }

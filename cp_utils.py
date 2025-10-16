
"""
Critical Power Utilities v5
Manual-duration 3/12 (or custom) CP analysis.
"""
import pandas as pd
import numpy as np

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c in ["power (w)", "power (watts)"]:
            rename_map[c] = "power"
        elif c == "power (w/kg)":
            rename_map[c] = "power_wkg"
        elif c in ["distance", "distance (m)", "watch distance (meters)", "stryd distance (m)"]:
            rename_map[c] = "watch distance (meters)"
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
        df = pd.read_csv(file, sep=None, engine="python")
    return normalize_columns(df)

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
    velocity = distance_m / duration_s
    return (velocity * weight_kg) / power_w

def compute_power_duration_curve(df: pd.DataFrame, max_duration_s: int = 3600, step: int = 5) -> pd.DataFrame:
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing column: power")
    durations = np.arange(5, int(max_duration_s) + 1, int(max(1, step)))
    powers = []
    s = df["power"].reset_index(drop=True)
    for d in durations:
        rm = s.rolling(d, min_periods=d).mean()
        powers.append(float(rm.max()) if rm.notna().any() else np.nan)
    return pd.DataFrame({"duration_s": durations, "best_power_w": powers}).dropna()

def find_best_effort(df: pd.DataFrame, duration_s: int, tolerance: float = 0.1):
    """Find the best average power for a target duration, extending if slightly longer keeps same or higher avg power."""
    df = normalize_columns(df)
    if "power" not in df.columns:
        raise ValueError("Missing power column")
    s = df["power"].rolling(duration_s, min_periods=duration_s).mean()
    best_idx = int(s.idxmax())
    best_power = float(s.max())
    start_idx = max(0, best_idx - duration_s + 1)
    end_idx = best_idx

    # extend forward if longer durations keep same or higher mean
    power_arr = df["power"].to_numpy(dtype=float)
    while end_idx < len(power_arr) - 1:
        new_mean = power_arr[start_idx:end_idx + 2].mean()
        if new_mean >= best_power * (1 - tolerance/5):
            end_idx += 1
            best_power = new_mean
        else:
            break
    found_dur = end_idx - start_idx + 1
    return {
        "target_dur": duration_s,
        "found_dur": found_dur,
        "avg_power": best_power,
        "start_idx": start_idx,
        "end_idx": end_idx
    }

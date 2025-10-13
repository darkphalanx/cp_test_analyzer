#!/usr/bin/env python3
"""
Critical Power Analysis Script v3.0
Author: Jordi & GPT-5
Usage:
    python cp_analysis.py
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import timedelta

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ---------- Settings Storage ---------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "cp_settings.json")

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


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


def compute_cp_exponential(p, t, k, p_max=None):
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



def detect_test_type(df):
    """Detect whether this activity looks like a 3/12-min test or a 5k race."""
    power = df["power"].rolling(5, center=True, min_periods=1).mean().to_numpy()

    # Use a relative threshold: 60% of peak power to capture both intervals
    threshold = 0.6 * np.max(power)
    above = power > threshold

    # Find contiguous high-power segments
    segments = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            duration = end - start
            segments.append((start, end, duration))
            start = None
    if start is not None:
        segments.append((start, len(power) - 1, len(power) - 1 - start))

    # Filter short noise segments
    segments = [s for s in segments if s[2] >= 60]

    # Merge very close segments (<60 s apart)
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            gap = seg[0] - prev[1]
            if gap < 60:
                merged[-1] = (prev[0], seg[1], seg[1] - prev[0])
            else:
                merged.append(seg)
    segments = merged

    total_time = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    total_distance = 0
    if "Watch Distance (meters)" in df.columns:
        total_distance = df["Watch Distance (meters)"].max()
    elif "Stryd Distance (meters)" in df.columns:
        total_distance = df["Stryd Distance (meters)"].max()

    print(f"File summary: {total_time/60:.1f} min | {total_distance:.0f} m total")

    if len(segments) >= 2:
        durations = [s[2] for s in segments]
        short = any(120 <= d <= 300 for d in durations)
        long = any(540 <= d <= 900 for d in durations)
        if short and long:
            print("Detected pattern of 3+12 minute intervals → 3/12-minute test.")
            return "3_12"

    print("Detected single sustained effort → 5k-type time trial.")
    return "5k"




# ---------- Main Script ---------- #

def main():
    print("\n" + "=" * 60)
    print(" CRITICAL POWER ANALYSIS v3.0 ".center(60, "="))
    print("=" * 60)

    # --- Ask for CSV file path ---
    file_path = input("\nEnter CSV file path: ").strip().strip('"')
    if not os.path.exists(file_path):
        print("Error: File not found.\n")
        return

    # --- Load or ask for body weight ---
    settings = load_settings()
    prev_weight = settings.get("weight")

    print("\n" + "-" * 60)
    if prev_weight:
        raw = input(f"Enter your body weight (kg) [default {prev_weight}]: ").strip()
        if raw:
            weight = float(raw)
            settings["weight"] = weight
        else:
            weight = prev_weight
    else:
        weight = float(input("Enter your body weight (kg): ").strip())
        settings["weight"] = weight
    save_settings(settings)
    print("-" * 60)

    # --- Load CSV ---
    df = load_csv_auto(file_path)

    # --- Prepare data ---
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    df["timestamp"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    power_col = [c for c in df.columns if "power" in c.lower()][0]
    df["power"] = df[power_col]
    if "w/kg" in power_col.lower():
        df["power"] *= weight
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Detect test type ---
    print("\n" + "=" * 60)
    print(" TEST TYPE DETECTION ".center(60))
    print("=" * 60)
    test_type = detect_test_type(df)
    confirm = input("Is this correct? [Y/n]: ").strip().lower()

    if confirm == "n":
        print("\nSelect test type manually:")
        print("  [1] 3/12-minute test (linear model)")
        print("  [2] 5k time trial (exponential model)")
        manual = input("Choice: ").strip()
        mode = manual if manual in ["1", "2"] else ("2" if test_type == "5k" else "1")
    else:
        mode = "2" if test_type == "5k" else "1"


    # ---- 3/12-minute Test ---- #
    if mode == "1":
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)
        ext3 = extend_best_segment(df, s3, e3, best3)
        ext12 = extend_best_segment(df, s12, e12, best12)
        cp, w_prime = compute_cp_linear(ext3[0], 180, ext12[0], 720)

        print("\n[ 3-MINUTE SEGMENT ]")
        print(f"  Avg Power : {ext3[0]:.1f} W")
        print(f"  Duration  : {ext3[3]} s")
        print(f"  Start     : {df.loc[ext3[1],'timestamp']}")
        print(f"  End       : {df.loc[ext3[2],'timestamp']}")

        print("\n[ 12-MINUTE SEGMENT ]")
        print(f"  Avg Power : {ext12[0]:.1f} W")
        print(f"  Duration  : {ext12[3]} s")
        print(f"  Start     : {df.loc[ext12[1],'timestamp']}")
        print(f"  End       : {df.loc[ext12[2],'timestamp']}")

        print("\n[ ESTIMATED CRITICAL POWER ]")
        print(f"  CP        : {cp:.1f} W")
        print(f"  W′ (anaerobic work capacity) : {w_prime/1000:.2f} kJ")

        print("\n" + "-" * 55)
        print(" Tip: Use these values to update your Stryd manual CP.")
        print("-" * 55)

        if HAS_PLOT:
            durations = np.array([180, 720])
            powers = np.array([ext3[0], ext12[0]])
            plt.figure(figsize=(7, 4))
            plt.scatter(durations / 60, powers, color="red", label="Test Points")
            plt.plot(durations / 60, cp + (w_prime / durations), label="CP Line")
            plt.xlabel("Duration (minutes)")
            plt.ylabel("Power (W)")
            plt.title("Critical Power Model (Linear)")
            plt.legend()
            plt.grid(True)
            plt.show()

    # ---- 5k Time Trial ---- #
    elif mode == "2":
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k, max_extend=100)
        t5k = int(ext5k[3])
        avg_pow = ext5k[0]
        cp_est = compute_cp_exponential(avg_pow, t5k, k=0.0135)

        diff_w = avg_pow - cp_est
        diff_pct = diff_w / avg_pow * 100

        total_time = str(timedelta(seconds=int(t5k)))
        pace_per_km = timedelta(seconds=int(t5k / 5))

        print("\n[ BEST 5K SEGMENT ]")
        print(f"  Avg Power : {avg_pow:.1f} W")
        print(f"  Duration  : {total_time}  ({pace_per_km} per km)")
        print(f"  Start     : {df.loc[ext5k[1],'timestamp']}")
        print(f"  End       : {df.loc[ext5k[2],'timestamp']}")

        print("\n[ ESTIMATED CRITICAL POWER ]")
        print(f"  CP        : {cp_est:.1f} W")
        print(f"  Diff vs 5k Power : -{diff_w:.1f} W  ({-diff_pct:.1f} %)")

        print("\n" + "-" * 55)
        print(" Note: CP is derived from exponential model tuned to")
        print("       Palladino's SuperPowerCalculator (≤ 40 min efforts).")
        print("-" * 55)

    print("\nAnalysis complete.\n")



if __name__ == "__main__":
    main()

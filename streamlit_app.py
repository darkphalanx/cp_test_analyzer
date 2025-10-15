
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from docs import render_documentation
from cp_utils import (
    load_csv_auto,
    best_avg_power,
    best_power_for_distance,
    extend_best_segment,
    compute_cp_linear,
    compute_cp_5k_range,
)

st.set_page_config(page_title="Critical Power Analysis Tool", layout="wide")

# --- Title ---
st.title("⚡ Critical Power Analysis Tool")
st.caption("Analyze your running power data to estimate Critical Power (CP) and W′.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    stryd_weight = st.number_input(
        "Stryd weight (kg)",
        min_value=30.0,
        max_value=150.0,
        step=0.1,
        value=76.0,
        help="If your CSV has power in W/kg, we'll convert to Watts using this.",
    )

    test_choice = st.radio(
        "Select test type:",
        ["3/12-minute CP Test", "5K Time Trial"],
        horizontal=False
    )

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis")

def _find_column(df: pd.DataFrame, contains: str) -> str | None:
    cols = [c for c in df.columns if contains in c.lower()]
    return cols[0] if cols else None

def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # try typical names
    cand = None
    for key in ["timestamp", "time", "date", "datetime"]:
        c = _find_column(df, key)
        if c is not None:
            cand = c
            break
    if cand is None:
        st.error("No time-like column found. Columns are: " + ", ".join(map(str, df.columns)))
        st.stop()
    ts = pd.to_datetime(df[cand], errors="coerce", utc=False)
    if ts.isna().all():
        # try if it's in seconds
        ts = pd.to_datetime(df[cand], unit="s", errors="coerce", utc=False)
    if ts.isna().all():
        st.error(f"Could not parse time column '{cand}'.")
        st.stop()
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def _ensure_power(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    pcol = _find_column(df, "power")
    if pcol is None:
        st.error("No power column found in CSV.")
        st.stop()
    series = pd.to_numeric(df[pcol], errors="coerce")
    if "w/kg" in pcol.lower() or "wkg" in pcol.lower():
        series = series * float(weight)
    df = df.assign(power=series).dropna(subset=["power"]).reset_index(drop=True)
    return df

def _find_distance_col(df: pd.DataFrame) -> str | None:
    for key in ["watch distance", "stryd distance", "distance (m)", "distance_m"]:
        c = _find_column(df, key)
        if c is not None:
            return c
    return None

# --- Main ---
if run_analysis:
    if file is None:
        st.warning("Please upload a CSV file before running the analysis.")
        st.stop()

    # Load and normalize
    df = load_csv_auto(file)
    df = _ensure_timestamp(df)
    df = _ensure_power(df, stryd_weight)

    st.markdown("## 📊 Analysis Results")

    # 3/12 Test
    if "3/12" in test_choice:
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)

        ext3 = extend_best_segment(df, s3, e3, best3)
        ext12 = extend_best_segment(df, s12, e12, best12)

        cp, w_prime = compute_cp_linear(ext3[0], 180, ext12[0], 720)

        # Distance & pace (if available)
        dist_col = _find_distance_col(df)
        if dist_col:
            dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            dist3 = float(dist_series.iloc[ext3[2]] - dist_series.iloc[ext3[1]])
            dist12 = float(dist_series.iloc[ext12[2]] - dist_series.iloc[ext12[1]])
        else:
            dist3 = np.nan
            dist12 = np.nan

        dur3 = int(ext3[3])
        dur12 = int(ext12[3])
        pace3 = timedelta(seconds=int(dur3 / (dist3 / 1000))) if pd.notna(dist3) and dist3 > 0 else None
        pace12 = timedelta(seconds=int(dur12 / (dist12 / 1000))) if pd.notna(dist12) and dist12 > 0 else None

        st.subheader("Segment Details")
        seg_df = pd.DataFrame({
            "Segment": ["3-minute", "12-minute"],
            "Distance (m)": [f"{dist3:.0f}" if pd.notna(dist3) else "–", f"{dist12:.0f}" if pd.notna(dist12) else "–"],
            "Duration": [str(timedelta(seconds=dur3)), str(timedelta(seconds=dur12))],
            "Pace (/km)": [str(pace3) if pace3 else "–", str(pace12) if pace12 else "–"],
            "Avg Power (W)": [f"{ext3[0]:.1f}", f"{ext12[0]:.1f}"],
        })
        st.dataframe(seg_df, use_container_width=True)

        st.subheader("Critical Power Results")
        st.write(f"**Critical Power (CP):** {cp:.1f} W")
        st.write(f"**W′:** {w_prime/1000:.2f} kJ")

    # 5K TT
    else:
        # best segment over 5k
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
        t5k = int(ext5k[3])
        avg_pow = float(ext5k[0])

        dist_col = _find_distance_col(df)
        if dist_col:
            dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            actual_distance = float(dist_series.iloc[ext5k[2]] - dist_series.iloc[ext5k[1]])
        else:
            actual_distance = 5000.0

        pace_per_km = timedelta(seconds=int(t5k / (actual_distance / 1000)))

        st.subheader("Segment Details")
        seg_df = pd.DataFrame({
            "Segment": ["5 km Time Trial"],
            "Distance (m)": [f"{actual_distance:.0f}"],
            "Duration": [str(timedelta(seconds=t5k))],
            "Pace (/km)": [str(pace_per_km)],
            "Avg Power (W)": [f"{avg_pow:.1f}"],
        })
        st.dataframe(seg_df, use_container_width=True)

        cp_results = compute_cp_5k_range(avg_pow)

        st.subheader("Critical Power Profiles")
        st.markdown("Each profile represents a different **fatigue characteristic**.")

        # preserve dict insertion order for rows
        profiles = list(cp_results.items())
        cp_table = pd.DataFrame({
            "Profile": [k for k, _ in profiles],
            "CP (W)": [f"{v:.1f}" for _, v in profiles],
            "Scaling": ["98.5%", "97.5%", "96.5%"],
            "Trait": ["Endurance-focused", "Typical distance runner", "Power-focused"],
        })
        st.dataframe(cp_table, use_container_width=True, hide_index=True)

        cp_min = min(cp_results.values())
        cp_max = max(cp_results.values())
        cp_mid = list(cp_results.values())[1]  # Balanced
        st.markdown(f"**Estimated CP range:** {cp_min:.1f} – {cp_max:.1f} W (typical ≈ {cp_mid:.1f} W)")

st.markdown("---")
render_documentation()

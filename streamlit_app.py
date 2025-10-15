@ -1,188 +1,247 @@

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import hashlib
import plotly.graph_objects as go

from cp_utils import (
    load_csv_auto,
    best_avg_power,
    best_power_for_distance,
    extend_best_segment,
    compute_cp_linear,
    compute_cp_5k_range,
    compute_power_duration_curve,
    detect_stable_blocks,
    running_effectiveness,
)
from docs import render_documentation

def fmt_sec_hms(total_s: int) -> str:
    total_s = int(total_s)
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def fmt_pace_mmss(sec_val):
    if not sec_val:
        return "–"
    try:
        total = int(sec_val)
    except Exception:
        return "–"
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"

def fmt_short_axis(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if sec < 3600:
        return f"{m}m" if s == 0 else f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h" if m == 0 else f"{h}h{m:02d}m"

@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes):
    import io
    df = load_csv_auto(io.BytesIO(file_bytes))
    return df

@st.cache_data(show_spinner=False)
def compute_pdc_cached(file_hash: str, max_dur: int, pdc_step: int, power_series: pd.Series) -> pd.DataFrame:
    durations = np.arange(5, max_dur + 1, pdc_step, dtype=int)
    s = power_series.reset_index(drop=True)
    pows = []
    for d in durations:
        rm = s.rolling(d, min_periods=d).mean()
        m = float(rm.max()) if rm.notna().any() else np.nan
        pows.append(m)
    return pd.DataFrame({"duration_s": durations, "best_power_w": pows}).dropna()

@st.cache_data(show_spinner=False)
def detect_stable_blocks_cached(file_hash: str, max_std: float, min_block: int, smooth_window: int, stryd_weight: float, df: pd.DataFrame):
    return detect_stable_blocks(
        df,
        max_std_ratio=max_std,
        min_duration_sec=min_block,
        smooth_window_sec=smooth_window,
        weight_kg=float(stryd_weight) if stryd_weight else None,
    )

def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding: 1.6rem;
            background-color: #ffffff;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        ">
            <h2 style="color:{color}; margin-bottom:0.7rem;">🏁 {title}</h2>
            <p style="font-size:1.7rem;font-weight:800;color:{color};margin:0;">{main_value}</p>
            <p style="font-size:1.15rem;color:#333;margin-top:0.4rem;">{subtext}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])

    stryd_weight = st.number_input(
        "⚖️ Stryd Weight (kg)",
        min_value=40.0,
        max_value=120.0,
        value=76.0,
        step=0.1,
        help="Your body weight used by Stryd to calculate running power.",
    )

    test_choice = st.radio(
        "Choose Analysis Type",
        ["Critical Power Test (3/12)", "5K Test", "Power Duration Curve"],
        index=0,
    )

    if test_choice == "Power Duration Curve":
        st.subheader("PDC Settings")
        max_dur = st.slider("Max Duration (s)", 60, 7200, 3600, 30)
        pdc_res = st.radio("Curve point resolution", ["Every 1s", "Every 5s"], index=1, horizontal=True)
        pdc_step = 1 if pdc_res == "Every 1s" else 5
        st.subheader("Stable Block Settings")
        max_std = st.slider("Power Variability Threshold (%)", 2, 10, 5) / 100.0
        min_block = st.slider("Min Block Duration (s)", 10, 600, 60, 5)
        smooth_window = st.slider("Smoothing Window (s)", 1, 15, 5)

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis")

if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file before running the analysis.")
        st.stop()

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    df = load_csv_cached(uploaded_file.getvalue())

    if "power_wkg" in df.columns:
        df["power"] = df["power_wkg"] * stryd_weight
    elif "power" not in df.columns:
        st.error("No valid power column found in file.")
        st.stop()

    if "timestamp" not in df.columns:
        time_col = next((c for c in df.columns if "time" in c), None)
        if time_col:
            df["timestamp"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
        else:
            st.error("Timestamp column not found.")
            st.stop()

    if "watch distance (meters)" not in df.columns:
        df["watch distance (meters)"] = 0

    df = df.sort_values("timestamp").reset_index(drop=True)

    st.markdown("## 📊 Analysis Results")

    if "3/12" in test_choice:
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)
        ext3 = extend_best_segment(df, s3, e3, best3)
        ext12 = extend_best_segment(df, s12, e12, best12)
        cp, w_prime = compute_cp_linear(ext3[0], 180, ext12[0], 720)

        df["dist"] = pd.to_numeric(df["watch distance (meters)"], errors="coerce").ffill()
        dist3 = df.loc[ext3[2], "dist"] - df.loc[ext3[1], "dist"]
        dist12 = df.loc[ext12[2], "dist"] - df.loc[ext12[1], "dist"]

        dur3 = int(ext3[3])
        dur12 = int(ext12[3])
        pace3 = timedelta(seconds=int(dur3 / (dist3 / 1000))) if dist3 > 0 else None
        pace12 = timedelta(seconds=int(dur12 / (dist12 / 1000))) if dist12 > 0 else None

        st.subheader("Segment Details")
        seg_data = {
            "Segment": ["3-minute", "12-minute"],
            "Distance (m)": [f"{dist3:.0f}", f"{dist12:.0f}"],
            "Duration": [str(timedelta(seconds=dur3)), str(timedelta(seconds=dur12))],
            "Pace (/km)": [str(pace3) if pace3 else "–", str(pace12) if pace12 else "–"],
            "Avg Power (W)": [f"{ext3[0]:.1f}", f"{ext12[0]:.1f}"],
        }
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

        show_result_card("Critical Power (3/12 Test)", f"{cp:.1f} W", f"W′ = {w_prime/1000:.2f} kJ", color="#1a73e8")
        # ----- CP-derived power zones & Individual Interval Targets -----
        st.markdown("### 🧭 Power Zones & Individual Interval Targets")

        def _zones_from_cp(cp_val: float):
            bands = [
                ("Z1 • Endurance", 0.65, 0.80),
                ("Z2 • Moderate", 0.80, 0.90),
                ("Z3 • Threshold", 0.90, 1.00),
                ("Z4 • Interval", 1.00, 1.15),
                ("Z5 • Anaerobic", 1.15, 1.35),
            ]
            rows = []
            for name, lo, hi in bands:
                rows.append({
                    "Zone": name,
                    "Range (W)": f"{cp_val*lo:.0f} – {cp_val*hi:.0f}",
                    "% of CP": f"{int(lo*100)}–{int(hi*100)}%",
                })
            return pd.DataFrame(rows)

        zdf = _zones_from_cp(cp)
        st.dataframe(zdf, use_container_width=True, hide_index=True)

        with st.expander("⚙️ Interval Target Calculator (CP/W′ model)"):
            colA, colB, colC, colD = st.columns(4)
            with colA:
                rep_dur = st.number_input("Rep duration (s)", min_value=20, max_value=1800, value=180, step=5)
            with colB:
                frac_dep = st.slider("W′ depletion per rep (%)", 5, 40, 20, help="What fraction of W′ to spend each rep.")
            with colC:
                reps = st.number_input("Reps", min_value=1, max_value=40, value=6, step=1)
            with colD:
                rec_dur = st.number_input("Recovery (s)", min_value=15, max_value=600, value=120, step=5)

            frac = frac_dep / 100.0
            target_power = cp + (frac * w_prime) / max(1, rep_dur)
            tte_at_target = (w_prime / max(1e-6, (target_power - cp)))
            total_wprime_used = frac * w_prime * reps

            it_tbl = pd.DataFrame([{
                "CP (W)": f"{cp:.0f}",
                "W′ (J)": f"{w_prime:.0f}",
                "Rep (s)": f"{int(rep_dur)}",
                "Target Power (W)": f"{target_power:.0f}",
                "Model TTE at Target": str(timedelta(seconds=int(tte_at_target))),
                "Total W′ Used": f"{total_wprime_used/1000:.2f} kJ ({int(frac*100)}% × {reps})",
            }])
            st.dataframe(it_tbl, use_container_width=True, hide_index=True)

            notes = []
            if target_power < cp:
                notes.append("Target is below CP (adjust inputs).")
            if total_wprime_used > 1.05 * w_prime:
                notes.append("Plan spends >100% of W′ in total — ensure recovery is sufficient.")
            if rep_dur > tte_at_target * 1.1:
                notes.append("Rep duration exceeds model TTE at target — reduce power or duration.")
            if notes:
                st.caption(" • ".join(notes))


    elif "5K" in test_choice:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from cp_utils import (
    load_csv_auto,
    best_avg_power,
    best_power_for_distance,
    extend_best_segment,
    compute_cp_linear,
    compute_cp_5k_range,
    running_effectiveness,
    detect_stable_segments_rolling,
)
from docs import render_documentation

# ============================================================
#  UI Helper: Styled Result Card
# ============================================================

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

# ============================================================
#  Streamlit Page Setup
# ============================================================

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
        ["Critical Power Test (3/12)", "5K Test", "Segment Analysis"],
        index=0,
    )

    if test_choice == "Segment Analysis":
        sensitivity = st.radio("Detection Sensitivity", ["Low", "Medium", "High"], index=1)
        if sensitivity == "Low":
            smooth_window, max_std, allowed_spike = 8, 0.06, 8
        elif sensitivity == "High":
            smooth_window, max_std, allowed_spike = 4, 0.035, 3
        else:
            smooth_window, max_std, allowed_spike = 6, 0.045, 5

        with st.expander("⚙️ Advanced", expanded=False):
            smooth_window = st.slider("📈 Smoothing Window (sec)", 1, 15, smooth_window)
            max_std = st.slider("📊 Power Variability Threshold (%)", 2, 10, int(max_std * 100)) / 100
            allowed_spike = st.slider("⚠️ Allowed Spike Duration (sec)", 0, 30, allowed_spike)

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis")

# ============================================================
#  Data Loading and Normalization
# ============================================================

if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file before running the analysis.")
        st.stop()

    df = load_csv_auto(uploaded_file)

    # Handle power conversion if file contains w/kg
    if "power_wkg" in df.columns:
        df["power"] = df["power_wkg"] * stryd_weight
    elif "power" not in df.columns:
        st.error("No valid power column found in file.")
        st.stop()

    if "timestamp" not in df.columns:
        # Try to parse a numeric timestamp column
        time_col = next((c for c in df.columns if "time" in c), None)
        if time_col:
            df["timestamp"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
        else:
            st.error("Timestamp column not found.")
            st.stop()

    if "watch distance (meters)" not in df.columns:
        df["watch distance (meters)"] = 0

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ============================================================
    #  Analysis Logic
    # ============================================================

    st.markdown("## 📊 Analysis Results")

    # ------------------- 3/12-Minute Critical Power Test ------------------- #
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
        st.dataframe(pd.DataFrame(seg_data), width="stretch")

        show_result_card(
            "Critical Power (3/12 Test)",
            f"{cp:.1f} W",
            f"W′ = {w_prime/1000:.2f} kJ",
            color="#1a73e8",
        )

    # ------------------- 5K Time Trial ------------------- #
    elif "5K" in test_choice:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
        t5k = int(ext5k[3])
        avg_pow = ext5k[0]

        df["dist"] = pd.to_numeric(df["watch distance (meters)"], errors="coerce").ffill()
        actual_distance = df.loc[ext5k[2], "dist"] - df.loc[ext5k[1], "dist"]
        pace_per_km = timedelta(seconds=int(t5k / (actual_distance / 1000)))

        st.subheader("Segment Details")
        seg_data = {
            "Segment": ["5 km Time Trial"],
            "Distance (m)": [f"{actual_distance:.0f}"],
            "Duration": [str(timedelta(seconds=t5k))],
            "Pace (/km)": [str(pace_per_km)],
            "Avg Power (W)": [f"{avg_pow:.1f}"],
        }
        st.dataframe(pd.DataFrame(seg_data), width="stretch")

        cp_results = compute_cp_5k_range(avg_pow)
        cp_table = pd.DataFrame({
            "Profile": ["Aerobic", "Balanced", "Anaerobic"],
            "CP (W)": [f"{cp:.1f}" for cp in cp_results.values()],
            "Scaling": ["98.5%", "97.5%", "96.5%"],
            "Trait": ["Endurance-focused", "Typical distance runner", "Power-focused"],
        })

        st.subheader("Critical Power Profiles")
        st.dataframe(cp_table, width="stretch", hide_index=True)

        cp_min, cp_max, cp_mid = min(cp_results.values()), max(cp_results.values()), list(cp_results.values())[1]
        show_result_card(
            "Estimated Critical Power Range (5K Time Trial)",
            f"{cp_min:.1f} – {cp_max:.1f} W",
            f"Typical profile ≈ {cp_mid:.1f} W",
            color="#ff8800",
        )

    # ------------------- Segment Analysis ------------------- #
    elif "Segment Analysis" in test_choice:
        segments = detect_stable_segments_rolling(
            df,
            max_std_ratio=max_std,
            smooth_window_sec=smooth_window,
            allowed_spike_sec=allowed_spike,
        )

        if not segments:
            st.warning("No stable segments found with the current settings.")
            st.stop()

        for seg in segments:
            seg["RE"] = running_effectiveness(seg["distance_m"], seg["duration_s"], seg["avg_power"], stryd_weight)

        seg_df = pd.DataFrame([
            {
                "Start": str(timedelta(seconds=int(seg["start_elapsed"]))),
                "End": str(timedelta(seconds=int(seg["end_elapsed"]))),
                "Duration": str(timedelta(seconds=int(seg["duration_s"]))),
                "Min Power (W)": f"{seg['min_power']:.1f}",
                "Avg Power (W)": f"{seg['avg_power']:.1f}",
                "Max Power (W)": f"{seg['max_power']:.1f}",
                "CV %": f"{seg['cv_%']:.2f}" if seg.get("cv_%") is not None else "–",
                "Distance (m)": f"{seg['distance_m']:.0f}",
                "Pace (/km)": str(timedelta(seconds=int(seg["pace_per_km"]))) if seg["pace_per_km"] else "–",
                "RE": f"{seg['RE']:.3f}" if seg["RE"] else "–",
                "End Reason": seg.get("end_reason", "–"),
            }
            for seg in segments
        ])

        st.subheader("Detected Segments")
        st.dataframe(seg_df, width="stretch")

        avg_re = np.nanmean([seg.get("RE") for seg in segments if seg.get("RE") is not None])
        show_result_card(
            "Running Effectiveness (total)",
            f"{avg_re:.3f}" if avg_re else "–",
            "Typical values: 0.98–1.05",
            color="#16a34a",
        )

st.markdown("---")
render_documentation()

import streamlit as st
import pandas as pd
from cp_utils import (
    load_csv_auto, best_avg_power, best_power_for_distance,
    extend_best_segment, compute_cp_linear, compute_cp_5k_range,
    detect_segments, running_effectiveness, detect_stable_power_segments
)

from datetime import timedelta
from docs import render_documentation
import numpy as np

# --- Helper: Styled Result Card ---
def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    """
    Display a stylized result card for key outcomes.
    - title: main heading (e.g. "Critical Power (3/12 Test)")
    - main_value: emphasized numeric result
    - subtext: secondary info (e.g. W′)
    - color: accent color for highlight (default = blue)
    """
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding: 1.6rem;
            background-color: #ffffff;  /* white card for contrast in dark/light mode */
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        ">
            <h2 style="color:{color}; margin-bottom:0.7rem;">🏁 {title}</h2>
            <p style="
                font-size:1.7rem;
                font-weight:800;
                color:{color};
                margin:0;
            ">
                {main_value}
            </p>
            <p style="
                font-size:1.15rem;
                color:#333;
                margin-top:0.4rem;
            ">
                {subtext}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Critical Power Analysis Tool", layout="wide")
# --- App Title ---
st.title("⚡ Critical Power Analysis Tool")
st.caption("Analyze your running power data to estimate Critical Power (CP) and W′.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Settings")
    file = st.file_uploader("📂 Upload Stryd CSV file", type=["csv"])

    stryd_weight = st.number_input(
        "Stryd weight (kg)",
        min_value=30.0,
        max_value=150.0,
        step=0.1,
        value=None,
        placeholder="Enter the weight configured in your Stryd app"
    )

    test_choice = st.radio(
        "Select test type:",
        [
            "3/12-minute CP Test",
            "5K Time Trial",
            "Segment Analysis (Running Effectiveness)"
        ],
        horizontal=False
    )

    if "Segment Analysis" in test_choice:
        st.markdown("---")
        auto_mode = st.toggle("🔍 Auto-detect stable segments", value=True)
        st.subheader("Segment Detection Settings")
        max_gap = st.number_input("Allow brief gap (sec)", 0, 120, 10)

        if auto_mode:
            max_std = st.slider("Max Power Variability (%)", 1, 10, 5) / 100
            min_duration = st.number_input("⏱️ Minimum Duration (minutes)", 3, 60, 10) * 60
            smooth_window = st.slider("Smoothing window (sec)", 1, 10, 5)
        else:
            target_power = st.number_input("🎯 Target Power (W)", 100.0, 600.0, step=1.0)
            tolerance = st.slider("± Power Tolerance (%)", 1, 10, 5) / 100
            # Display calculated range (auto-updates with target & tolerance)
            if target_power and tolerance:
                lower = target_power * (1 - tolerance)
                upper = target_power * (1 + tolerance)
                st.markdown(
                    f"<span style='font-size: 0.8em; color: gray;'>Range: {lower:.1f} W – {upper:.1f} W</span>",
                    unsafe_allow_html=True
                )
            min_duration = st.number_input("⏱️ Minimum Duration (minutes)", 3, 60, 10) * 60

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis")

# --- Main Screen ---
if run_analysis:
    # --- Validation ---
    if file is None:
        st.warning("Please upload a CSV file before running the analysis.")
        st.stop()
    if stryd_weight is None or stryd_weight == 0:
        st.warning("Please enter your Stryd weight before running the analysis.")
        st.stop()

    # --- Load and prepare data ---
    df = load_csv_auto(file)

    time_col = [c for c in df.columns if "time" in c.lower()][0]
    df["timestamp"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    power_col = [c for c in df.columns if "power" in c.lower()][0]
    df["power"] = df[power_col]

    if "w/kg" in power_col.lower():
        df["power"] = df["power"] * stryd_weight

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Analysis Results ---
    st.markdown("## 📊 Analysis Results")

    # ==============================================================
    # 3/12-Minute Critical Power Test
    # ==============================================================
    if "3/12" in test_choice:
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)
        ext3 = extend_best_segment(df, s3, e3, best3)
        ext12 = extend_best_segment(df, s12, e12, best12)
        cp, w_prime = compute_cp_linear(ext3[0], 180, ext12[0], 720)

        # --- Helper for distance column ---
        dist_col = None
        for c in df.columns:
            if "watch distance" in c.lower() or "stryd distance" in c.lower():
                dist_col = c
                break

        # --- Segment 3 min ---
        if dist_col:
            df["dist"] = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            dist3 = df.loc[ext3[2], "dist"] - df.loc[ext3[1], "dist"]
        else:
            dist3 = np.nan

        dur3 = int(ext3[3])
        pace3 = timedelta(seconds=int(dur3 / (dist3 / 1000))) if pd.notna(dist3) and dist3 > 0 else None

        # --- Segment 12 min ---
        if dist_col:
            dist12 = df.loc[ext12[2], "dist"] - df.loc[ext12[1], "dist"]
        else:
            dist12 = np.nan

        dur12 = int(ext12[3])
        pace12 = timedelta(seconds=int(dur12 / (dist12 / 1000))) if pd.notna(dist12) and dist12 > 0 else None

        # --- Display unified segment table ---
        st.subheader("Segment Details")
        seg_data = {
            "Segment": ["3-minute", "12-minute"],
            "Distance (m)": [f"{dist3:.0f}" if pd.notna(dist3) else "–", f"{dist12:.0f}" if pd.notna(dist12) else "–"],
            "Duration": [str(timedelta(seconds=dur3)), str(timedelta(seconds=dur12))],
            "Pace (/km)": [str(pace3) if pace3 else "–", str(pace12) if pace12 else "–"],
            "Avg Power (W)": [f"{ext3[0]:.1f}", f"{ext12[0]:.1f}"],
        }
        st.dataframe(pd.DataFrame(seg_data), width='stretch')

        # --- Display computed CP/W' ---
        show_result_card(
            "Critical Power (3/12 Test)",
            f"{cp:.1f} W",
            f"W′ = {w_prime/1000:.2f} kJ",
            color="#1a73e8"  # blue accent
        )

    # ==============================================================
    # 5K Time Trial
    # ==============================================================
    elif "5K" in test_choice:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
        t5k = int(ext5k[3])
        avg_pow = ext5k[0]

        # --- Determine distance column ---
        dist_col = None
        for c in df.columns:
            if "watch distance" in c.lower() or "stryd distance" in c.lower():
                dist_col = c
                break

        if dist_col:
            df["dist"] = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            actual_distance = df.loc[ext5k[2], "dist"] - df.loc[ext5k[1], "dist"]
        else:
            actual_distance = 5000

        pace_per_km = timedelta(seconds=int(t5k / (actual_distance / 1000)))

        # --- Segment details table ---
        st.subheader("Segment Details")
        seg_data = {
            "Segment": ["5 km Time Trial"],
            "Distance (m)": [f"{actual_distance:.0f}"],
            "Duration": [str(timedelta(seconds=t5k))],
            "Pace (/km)": [str(pace_per_km)],
            "Avg Power (W)": [f"{avg_pow:.1f}"],
        }
        st.dataframe(pd.DataFrame(seg_data), width='stretch')

        # --- Critical Power estimate across fatigue profiles (empirical model) ---
        cp_results = compute_cp_5k_range(avg_pow)

        st.subheader("Critical Power Profiles")
        st.markdown("""
        Each profile represents a different **fatigue characteristic**.  
        Choose the one that best matches your physiology:
        """)

        # Compact, clean table
        cp_table = pd.DataFrame(
            {
                "Profile": ["Aerobic", "Balanced", "Anaerobic"],
                "CP (W)": [f"{cp:.1f}" for cp in cp_results.values()],
                "Scaling": ["98.5%", "97.5%", "96.5%"],
                "Trait": [
                    "Endurance-focused",
                    "Typical distance runner",
                    "Power-focused",
                ],
            }
        )

        st.dataframe(cp_table, width='stretch', hide_index=True)

        # Range summary
        cp_min = min(cp_results.values())
        cp_max = max(cp_results.values())
        cp_mid = list(cp_results.values())[1]  # Balanced profile

        show_result_card(
            "Estimated Critical Power Range (5K Time Trial)",
            f"{cp_min:.1f} – {cp_max:.1f} W",
            f"Typical profile ≈ {cp_mid:.1f} W",
            color="#ff8800"  # orange accent
        )
        
    # ==============================================================
    # Segment Analysis (Running Effectiveness)
    # ==============================================================
    elif "Segment Analysis" in test_choice:
        if 'auto_mode' in locals() and auto_mode:
            segments = detect_stable_power_segments(df, max_std_ratio=max_std, min_duration_sec=min_duration, smooth_window_sec=smooth_window)
        else:
            segments = detect_segments(df, target_power=target_power, tolerance=tolerance, min_duration_sec=min_duration)

        if not segments:
            st.warning("No stable power segments found within the specified range and duration.")
            st.stop()

        for seg in segments:
            seg["RE"] = running_effectiveness(
                seg["distance_m"], seg["duration_s"], seg["avg_power"], stryd_weight
            )

        seg_df = pd.DataFrame([
            {
                "Start": str(datetime.timedelta(seconds=int(seg["start_elapsed"]))),
                "End": str(datetime.timedelta(seconds=int(seg["end_elapsed"]))),
                "Duration": str(timedelta(seconds=int(seg["duration_s"]))),
                "Min Power (W)": f"{seg['min_power']:.1f}",
                "Avg Power (W)": f"{seg['avg_power']:.1f}",
                "Max Power (W)": f"{seg['max_power']:.1f}",
                "Distance (m)": f"{seg['distance_m']:.0f}",
                "Pace (/km)": str(timedelta(seconds=int(seg["pace_per_km"]))) if seg["pace_per_km"] else "–",
                "Running Effectiveness": f"{seg['RE']:.3f}" if seg["RE"] else "–"
            }
            for seg in segments
        ])

        st.subheader("Detected Segments")
        st.dataframe(seg_df, width="stretch")

        avg_re = np.nanmean([seg["RE"] for seg in segments if seg["RE"]])
        show_result_card(
            "Average Running Effectiveness",
            f"{avg_re:.3f}",
            "Typical values: 0.98 – 1.05 for most runners",
            color="#16a34a"
        )

st.markdown("---")
render_documentation()

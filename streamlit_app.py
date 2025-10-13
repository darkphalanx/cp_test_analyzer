import streamlit as st
import pandas as pd
from cp_analysis import (
    load_csv_auto, best_avg_power, best_power_for_distance,
    extend_best_segment, compute_cp_linear, compute_cp_exponential
)
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

# --- App Title ---
st.title("⚡ Critical Power Analysis Tool")
st.caption("Analyze your running power data to estimate Critical Power (CP) and W′.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Settings")
    file = st.file_uploader("📂 Upload CSV file", type=["csv"])

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
            "3/12-minute Critical Power Test (linear model)",
            "5K Time Trial (exponential model)"
        ],
        horizontal=False
    )

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

    # --- Results header ---
    st.markdown("## 📊 Analysis Results")

    # --- 3/12-minute Test ---
    if "3/12" in test_choice:
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)
        ext3 = extend_best_segment(df, s3, e3, best3)
        ext12 = extend_best_segment(df, s12, e12, best12)
        cp, w_prime = compute_cp_linear(ext3[0], 180, ext12[0], 720)

        actual_dur3 = int(ext3[3])
        actual_dur12 = int(ext12[3])

        st.write(f"**3-min segment:** {ext3[0]:.1f} W  (actual duration {timedelta(seconds=actual_dur3)})")
        st.write(f"**12-min segment:** {ext12[0]:.1f} W  (actual duration {timedelta(seconds=actual_dur12)})")
        st.write(f"**Critical Power:** {cp:.1f} W")
        st.write(f"**W′:** {w_prime/1000:.2f} kJ")

        fig, ax = plt.subplots()
        durations = np.array([180, 720])
        powers = np.array([ext3[0], ext12[0]])
        ax.scatter(durations / 60, powers, color="red")
        ax.plot(durations / 60, cp + (w_prime / durations))
        ax.set_xlabel("Duration (min)")
        ax.set_ylabel("Power (W)")
        ax.set_title("Linear CP Model")
        ax.set_ylim(min(powers) - 10, max(powers) + 10)
        st.pyplot(fig)

    # --- 5K Time Trial ---
    else:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
        t5k = int(ext5k[3])
        avg_pow = ext5k[0]
        cp_est = compute_cp_exponential(avg_pow, t5k)
        diff = avg_pow - cp_est
        total_time_str = str(timedelta(seconds=int(t5k)))
        pace_per_km = timedelta(seconds=int(t5k / 5))

        st.subheader("Results – 5K Time Trial")
        st.write(f"**Total time:** {total_time_str}  ({pace_per_km} per km)")
        st.write(f"**5K avg power:** {avg_pow:.1f} W")
        st.write(f"**Critical Power:** {cp_est:.1f} W  (−{diff:.1f} W, {diff/avg_pow*100:.1f} %)")

    st.info("💡 Tip: Use consistent test conditions (course, weather, shoes) for reliable comparisons.")

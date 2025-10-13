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
            "3/12-minute CP Test",
            "5K Time Trial"
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
        st.dataframe(pd.DataFrame(seg_data))

        # --- Display computed CP/W' ---
        st.subheader("Critical Power Results")
        st.write(f"**Critical Power (CP):** {cp:.1f} W")
        st.write(f"**W′:** {w_prime/1000:.2f} kJ")

        # --- Plot CP line ---
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

    # ==============================================================
    # 5K Time Trial
    # ==============================================================
    else:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
        t5k = int(ext5k[3])
        avg_pow = ext5k[0]
        cp_est = compute_cp_exponential(avg_pow, t5k)
        diff = avg_pow - cp_est

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
        st.dataframe(pd.DataFrame(seg_data))

        # --- Critical Power estimate ---
        st.subheader("Critical Power Results")
        st.write(f"**Estimated Critical Power (CP):** {cp_est:.1f} W")
        st.write(f"**Difference from avg power:** {diff:.1f} W ({diff/avg_pow*100:.1f} %)")

# ==============================================================
# 📘 Help & Documentation
# ==============================================================

with st.expander("ℹ️ How this tool works (click to expand)"):
    st.markdown("""
### 🧠 Overview
This tool analyzes your running power data from Stryd or Garmin to estimate your **Critical Power (CP)** —  
the maximum power you can theoretically sustain indefinitely — and your **W′ (anaerobic work capacity)**.

You can calculate CP using two supported protocols:
""")

    st.markdown("#### 1️⃣ 3/12-minute Critical Power Test (Linear Model)")
    st.markdown("""
- Finds the highest average power over 3 minutes and 12 minutes.  
- If a slightly longer segment has equal or higher power, it’s included automatically.  
- Uses the *linear CP model*:  
""")
    st.latex(r"P = CP + \frac{W'}{t}")
    st.markdown("""
- From the two test points (3 min & 12 min), the tool solves for CP and W′.
""")

    st.markdown("#### 2️⃣ 5 K Time Trial (Exponential Model)")
    st.markdown("""
- Finds the segment of ~5,000 m with the highest average power.  
- If a slightly longer section yields equal or higher power, it’s extended automatically.  
- Uses an *exponential fatigue model*:  
""")
    st.latex(r"P = CP + (P_{max} - CP)e^{-k t}")
    st.markdown("""
- A constant **k = 0.018** is used by default (based on Steve Palladino’s methodology).
""")

    st.markdown("---")
    st.markdown("""
### ⚙️ How segments are detected
- The uploaded file is scanned for rolling windows of fixed length (e.g., 180 s or 720 s).  
- The window with the highest average power is selected as the **best effort**.  
- The algorithm then checks if extending the segment slightly (up to 60 s by default) keeps the average power equal or higher — if so, it expands that segment.
""")

    st.markdown("---")
    st.markdown("""
### 📈 Outputs explained
| Metric | Description |
|---------|-------------|
| **Distance** | Actual covered distance within the detected best segment. |
| **Duration** | Actual elapsed time for that segment. |
| **Pace** | Average pace for that segment. |
| **Average Power** | Mean running power in watts for that segment. |
| **CP (Critical Power)** | Theoretical threshold you can sustain indefinitely (aerobic limit). |
| **W′** | Finite anaerobic work capacity above CP, expressed in kilojoules (kJ). |
| **Power above CP** | How much harder your effort was compared to CP — typically 2–5 % higher for a 5 K race. |
""")

    st.markdown("---")
    st.markdown("""
### 💡 Practical notes
- For reliable CP results, perform tests on flat terrain in similar conditions.  
- Use the same **Stryd weight** value as configured in your pod/app.  
- The 3/12 test is ideal for frequent CP recalculations; the 5 K trial for performance validation.  
- CP is dynamic — it adapts with training and recovery.

---

*Based on the work of Steve Palladino’s Power Project and Stryd’s Critical Power methodology.*
""")

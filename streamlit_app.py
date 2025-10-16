
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import plotly.graph_objects as go

from cp_utils import (
    load_csv_auto,
    compute_cp_linear,
    compute_cp_5k_range,
    compute_power_duration_curve,
    running_effectiveness,
    find_best_effort
)

st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])
    stryd_weight = st.number_input("⚖️ Stryd Weight (kg)", min_value=40.0, max_value=120.0, value=76.0, step=0.1)
    test_choice = st.radio("Choose Analysis Type", ["Critical Power Test", "5K Test"], index=0)
    if test_choice == "Critical Power Test":
        short_min = st.number_input("Short Test Duration (min)", 1.0, 10.0, 3.0, 0.5)
        long_min = st.number_input("Long Test Duration (min)", 5.0, 30.0, 12.0, 0.5)
    run_analysis = st.button("🚀 Run Analysis")

def find_col_contains(df, key):
    key = key.lower()
    for c in df.columns:
        if key in str(c).lower():
            return c
    return None

if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    df = load_csv_auto(uploaded_file)

    power_col = find_col_contains(df, "power")
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    ts_col = find_col_contains(df, "timestamp") or find_col_contains(df, "time")
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "power"]).sort_values("timestamp").reset_index(drop=True)

    st.markdown("## 📊 Analysis Results")

    if "Critical Power" in test_choice:
        short_seg = find_best_effort(df, int(short_min * 60))
        long_seg = find_best_effort(df, int(long_min * 60))
        segments = [short_seg, long_seg]

        # PDC plot
        pdc_df = compute_power_duration_curve(df, max_duration_s=int(long_min * 120), step=5)
        fig_pdc = go.Figure()
        fig_pdc.add_trace(go.Scatter(x=pdc_df["duration_s"], y=pdc_df["best_power_w"], mode="lines", name="PDC"))
        for seg in segments:
            fig_pdc.add_vline(x=seg["found_dur"], line=dict(color="orange", dash="dash"),
                              annotation_text=f"{seg['found_dur']}s")
        fig_pdc.update_layout(title="Power Duration Curve", template="plotly_white")
        st.plotly_chart(fig_pdc, use_container_width=True)

        # Power-over-time
        elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.4, name="Power"))
        for i, seg in enumerate(segments):
            color = f"rgba(255,165,0,{0.3 + 0.2*i})"
            x0, x1 = float(elapsed_s.iloc[seg["start_idx"]]), float(elapsed_s.iloc[seg["end_idx"]])
            fig_time.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0)
        fig_time.update_layout(title="Detected Segments", template="plotly_white")
        st.plotly_chart(fig_time, use_container_width=True)

        # Table and CP calc
        dist_col = find_col_contains(df, "distance")
        rows = []
        for i, seg in enumerate(segments):
            start, end = seg["start_idx"], seg["end_idx"]
            distance_m = None
            if dist_col:
                dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
                if dist_series.notna().any():
                    distance_m = float(dist_series.iloc[end] - dist_series.iloc[start])
            RE = None
            if distance_m and distance_m > 0:
                RE = running_effectiveness(distance_m, seg["found_dur"], seg["avg_power"], stryd_weight)
            rows.append({
                "Segment": ["Short", "Long"][i],
                "Duration (s)": seg["found_dur"],
                "Avg Power (W)": round(seg["avg_power"], 1),
                "Distance (m)": f"{distance_m:.0f}" if distance_m else "–",
                "RE": f"{RE:.3f}" if RE else "–"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        p1, t1 = short_seg["avg_power"], short_seg["found_dur"]
        p2, t2 = long_seg["avg_power"], long_seg["found_dur"]
        cp, w_prime = compute_cp_linear(p1, t1, p2, t2)
        st.success(f"**Critical Power:** {cp:.1f} W | **W′:** {w_prime/1000:.2f} kJ")

    else:
        st.info("5K mode not yet adapted for manual durations.")

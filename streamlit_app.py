
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from cp_utils import load_csv_auto, compute_cp_linear, compute_power_duration_curve, find_best_effort

st.set_page_config(page_title="Critical Power Analyzer v7", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    stryd_weight = st.number_input("Stryd Weight (kg)", 40.0, 120.0, 76.0, 0.1)
    short_min = st.number_input("Short Test Duration (min)", 1.0, 10.0, 3.0, 0.5)
    long_min  = st.number_input("Long Test Duration (min)", 5.0, 30.0, 12.0, 0.5)
    run = st.button("🚀 Run Analysis")

if run:
    if not uploaded_file:
        st.warning("Upload a CSV file first."); st.stop()

    df = load_csv_auto(uploaded_file)
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0.0)

    short_seg = find_best_effort(df, int(short_min * 60))
    long_seg = find_best_effort(df, int(long_min * 60))

    st.markdown("## Power Duration Curve")
    pdc = compute_power_duration_curve(df, max_duration_s=int(long_min * 120))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pdc["duration_s"], y=pdc["best_power_w"], mode="lines+markers", name="PDC"))
    for label, seg in [("Short", short_seg), ("Long", long_seg)]:
        fig.add_vline(x=seg["found_dur"], line=dict(color="orange", width=2, dash="dash"),
                      annotation_text=f"{label} {seg['found_dur']}s")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Detected Segments")
    elapsed = np.arange(len(df))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=elapsed, y=df["power"], mode="lines", opacity=0.5, name="Power"))
    for label, seg in [("Short", short_seg), ("Long", long_seg)]:
        fig2.add_vrect(x0=seg["start_idx"], x1=seg["end_idx"], fillcolor="orange", opacity=0.3)
        fig2.add_annotation(x=(seg["start_idx"]+seg["end_idx"])/2, y=df["power"].max(), text=label, showarrow=False)
    st.plotly_chart(fig2, use_container_width=True)

    cp, wp = compute_cp_linear(short_seg["avg_power"], short_seg["found_dur"], long_seg["avg_power"], long_seg["found_dur"])
    st.success(f"**Critical Power:** {cp:.1f} W | **W′:** {wp/1000:.2f} kJ")

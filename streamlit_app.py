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
    find_best_effort,
    find_best_distance_effort,
    detect_stable_blocks,
)

st.set_page_config(page_title="Critical Power Analyzer v8", page_icon="⚡", layout="wide")

# ------------------------------
# Helper functions
# ------------------------------
def find_col_contains(df, key):
    key = key.lower()
    for c in df.columns:
        if key in str(c).lower():
            return c
    return None

def format_duration(seconds: int) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"

def format_seconds(seconds):
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])
    stryd_weight = st.number_input("⚖️ Stryd Weight (kg)", min_value=40.0, max_value=120.0, value=76.0, step=0.1)

    mode = st.radio("Choose Analysis Type", ["Critical Power Test", "5K Test", "Segment Analysis"], index=0)

    if mode == "Critical Power Test":
        short_min = st.number_input("Short Test Duration (min)", 1.0, 10.0, 3.0, 0.5)
        long_min  = st.number_input("Long Test Duration (min)", 5.0, 30.0, 12.0, 0.5)

    elif mode == "5K Test":
        fivek_mode = st.radio("5K target by", ["Distance", "Duration"], index=0, horizontal=True)
        if fivek_mode == "Distance":
            fivek_distance = st.number_input("Target Distance (m)", min_value=3000, max_value=21097, value=5000, step=100)
        else:
            fivek_minutes = st.number_input("Target Duration (min)", min_value=10.0, max_value=60.0, value=20.0, step=0.5)

    else:
        st.subheader("Stable Block Settings")
        max_std = st.slider("Power Variability Threshold (%)", 2, 10, 5) / 100.0
        min_block = st.slider("Min Block Duration (s)", 10, 600, 60, 5)
        smooth_window = st.slider("Smoothing Window (s)", 1, 15, 5)

    run_analysis = st.button("🚀 Run Analysis")

# ------------------------------
# Main
# ------------------------------
if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    df = load_csv_auto(uploaded_file)
    power_col = find_col_contains(df, "power")
    if power_col is None:
        st.error("No power column found.")
        st.stop()
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    if "w/kg" in power_col or "wkg" in power_col:
        df["power"] = df["power"] * float(stryd_weight)

    ts_col = find_col_contains(df, "timestamp") or find_col_contains(df, "time")
    if ts_col is None:
        st.error("No timestamp column found.")
        st.stop()
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp", "power"]).sort_values("timestamp").reset_index(drop=True)

    st.markdown("## 📊 Analysis Results")

    # ---------- Critical Power Test ----------
    if mode == "Critical Power Test":
        short_seg = find_best_effort(df, int(short_min * 60))
        long_seg  = find_best_effort(df, int(long_min  * 60))
        segments = [("Short", short_seg), ("Long", long_seg)]

        pdc_df = compute_power_duration_curve(df, max_duration_s=max(int(long_min*120), 1800), step=5)
        fig_pdc = go.Figure()
        fig_pdc.add_trace(go.Scatter(x=pdc_df["duration_s"], y=pdc_df["best_power_w"], mode="lines+markers", name="PDC"))
        for label, seg in segments:
            fig_pdc.add_vline(x=seg["found_dur"], line=dict(color="orange", width=2, dash="dash"),
                              annotation_text=f"{label} {format_duration(seg['found_dur'])}", annotation_position="top right")
        tickvals = pdc_df["duration_s"][::len(pdc_df)//10 or 1]
        fig_pdc.update_xaxes(title="Duration (hh:mm:ss)", tickvals=tickvals, ticktext=[format_seconds(s) for s in tickvals])
        st.plotly_chart(fig_pdc, use_container_width=True)

        elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.4, name="Power"))
        for i, (label, seg) in enumerate(segments):
            color = f"rgba(255,165,0,{0.25 + 0.2*i})"
            x0, x1 = float(elapsed_s.iloc[seg["start_idx"]]), float(elapsed_s.iloc[seg["end_idx"]])
            fig_time.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0)
            fig_time.add_annotation(x=(x0+x1)/2, y=float(df["power"].max()), text=label, showarrow=False, yanchor="bottom")
        tickvals = elapsed_s[::len(elapsed_s)//10 or 1]
        fig_time.update_xaxes(title="Elapsed Time (hh:mm:ss)", tickvals=tickvals, ticktext=[format_seconds(s) for s in tickvals])
        st.plotly_chart(fig_time, use_container_width=True)

        dist_col = (find_col_contains(df, "watch distance") or find_col_contains(df, "distance (m)") or find_col_contains(df, "stryd distance (m)") or find_col_contains(df, "distance_m") or find_col_contains(df, "distance"))
        rows = []
        for label, seg in segments:
            start, end = seg["start_idx"], seg["end_idx"]
            distance_m = np.nan
            if dist_col:
                dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
                if dist_series.notna().any():
                    distance_m = float(dist_series.iloc[end] - dist_series.iloc[start])
            RE = None
            if not np.isnan(distance_m) and distance_m > 0:
                RE = running_effectiveness(distance_m, seg["found_dur"], seg["avg_power"], stryd_weight)
            rows.append({"Segment": label, "Duration": format_duration(seg["found_dur"]), "Avg Power (W)": f"{seg['avg_power']:.1f}", "Distance (m)": f"{distance_m:.0f}" if not np.isnan(distance_m) else "–", "RE": f"{RE:.3f}" if RE else "–"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        cp, w_prime = compute_cp_linear(short_seg["avg_power"], short_seg["found_dur"], long_seg["avg_power"],  long_seg["found_dur"])
        st.success(f"**Critical Power:** {cp:.1f} W | **W′:** {w_prime/1000:.2f} kJ")

    # ---------- 5K Test ----------
    elif mode == "5K Test":
        dist_col = (find_col_contains(df, "watch distance") or find_col_contains(df, "distance (m)") or find_col_contains(df, "stryd distance (m)") or find_col_contains(df, "distance_m") or find_col_contains(df, "distance"))
        if fivek_mode == "Distance":
            if dist_col is not None:
                dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
                total_dist = float(dist_series.max() - dist_series.min())
                if total_dist < fivek_distance:
                    st.warning(f"⚠️ Activity shorter than target distance ({total_dist:.0f} m available, {fivek_distance:.0f} m requested). Using full activity distance instead.")
                    fivek_distance = total_dist
            seg = find_best_distance_effort(df, float(fivek_distance))
            label = f"5 K (distance – capped to {fivek_distance:.0f} m)"
        else:
            target_s = int(fivek_minutes * 60)
            n = len(df)
            if n < target_s:
                st.warning(f"⚠️ Activity shorter than target duration ({n}s available, {target_s}s requested). Using full activity length instead.")
                target_s = n
            seg = find_best_effort(df, target_s)
            label = f"5 K (duration – capped to {format_duration(target_s)})"

        segments = [(label, seg)]
        pdc_df = compute_power_duration_curve(df, max_duration_s=3600, step=5)
        fig_pdc = go.Figure()
        fig_pdc.add_trace(go.Scatter(x=pdc_df["duration_s"], y=pdc_df["best_power_w"], mode="lines+markers", name="PDC"))
        for label, sdict in segments:
            fig_pdc.add_vline(x=sdict["found_dur"], line=dict(color="orange", width=2, dash="dash"), annotation_text=f"{format_duration(sdict['found_dur'])}", annotation_position="top right")
        tickvals = pdc_df["duration_s"][::len(pdc_df)//10 or 1]
        fig_pdc.update_xaxes(title="Duration (hh:mm:ss)", tickvals=tickvals, ticktext=[format_seconds(s) for s in tickvals])
        st.plotly_chart(fig_pdc, use_container_width=True)

        elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.4, name="Power"))
        color = "rgba(255,165,0,0.35)"
        x0, x1 = float(elapsed_s.iloc[seg["start_idx"]]), float(elapsed_s.iloc[seg["end_idx"]])
        fig_time.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0)
        fig_time.add_annotation(x=(x0+x1)/2, y=float(df["power"].max()), text=label, showarrow=False, yanchor="bottom")
        tickvals = elapsed_s[::len(elapsed_s)//10 or 1]
        fig_time.update_xaxes(title="Elapsed Time (hh:mm:ss)", tickvals=tickvals, ticktext=[format_seconds(s) for s in tickvals])
        st.plotly_chart(fig_time, use_container_width=True)

        start, end = seg["start_idx"], seg["end_idx"]
        distance_m = np.nan
        if dist_col:
            dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            if dist_series.notna().any():
                distance_m = float(dist_series.iloc[end] - dist_series.iloc[start])
        RE = None
        if not np.isnan(distance_m) and distance_m > 0:
            RE = running_effectiveness(distance_m, seg["found_dur"], seg["avg_power"], stryd_weight)
        st.dataframe(pd.DataFrame([{ "Segment": label, "Duration": format_duration(seg["found_dur"]), "Avg Power (W)": f"{seg['avg_power']:.1f}", "Distance (m)": f"{distance_m:.0f}" if not np.isnan(distance_m) else "–", "RE": f"{RE:.3f}" if RE else "–" }]), use_container_width=True)

        cp_range = compute_cp_5k_range(seg["avg_power"])
        cp_min, cp_mid, cp_max = min(cp_range.values()), list(cp_range.values())[1], max(cp_range.values())
        st.info(f"**Estimated CP (range):** {cp_min:.1f} – {cp_max:.1f} W  |  Typical ≈ {cp_mid:.1f} W")

    # ---------- Segment Analysis ----------
    else:
        pdc_df = compute_power_duration_curve(df, max_duration_s=3600, step=5)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdc_df["duration_s"], y=pdc_df["best_power_w"], mode="lines+markers", name="PDC"))
        tickvals = pdc_df["duration_s"][::len(pdc_df)//10 or 1]
        fig.update_xaxes(title="Duration (hh:mm:ss)", tickvals=tickvals, ticktext=[format_seconds(s) for s in tickvals])
        st.plotly_chart(fig, use_container_width=True)

        blocks = detect_stable_blocks(df, max_std_ratio=max_std, min_duration_sec=min_block, smooth_window_sec=smooth_window, weight_kg=stryd_weight)
        if not blocks:
            st.info("No stable blocks found.")
        else:
            rows = []
            for b in blocks:
                pace = f"{int(b['pace_per_km']//60):02d}:{int(b['pace_per_km']%60):02d}" if b['pace_per_km'] else "–"
                rows.append({"Duration": format_duration(b["duration_s"]), "Avg Power (W)": f"{b['avg_power']:.1f}", "Distance (m)": f"{b['distance_m']:.0f}", "Pace (/km)": pace, "RE": f"{b['RE']:.3f}" if b.get("RE") else "–"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
            smooth_power = df["power"].rolling(window=max(1, smooth_window), min_periods=1).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.3, name="Power"))
            fig2.add_trace(go.Scatter(x=elapsed_s, y=smooth_power, mode="lines", name=f"Smoothed ({smooth_window}s)"))

            palette = ["LightGreen", "LightSkyBlue", "LightSalmon", "Khaki", "Plum", "LightPink", "PaleTurquoise", "Wheat"]
            for idx, b in enumerate(blocks):
                x0, x1 = float(elapsed_s.iloc[b["start_idx"]]), float(elapsed_s.iloc[b["end_idx"]])
                fig2.add_vrect(x0=x0, x1=x1, fillcolor=palette[idx % len(palette)], opacity=0.25, line_width=0)
                fig2.add_annotation(x=(x0+x1)/2, y=float(df["power"].max()), text=f"B{idx+1}", showarrow=False, yanchor="bottom")
            tickvals = elapsed_s[::len(elapsed_s)//10 or 1]
            fig2.update_xaxes(title="Elapsed Time (hh:mm:ss)", tickvals=tickvals, ticktext=[format_seconds(s) for s in tickvals])
            st.plotly_chart(fig2, use_container_width=True)

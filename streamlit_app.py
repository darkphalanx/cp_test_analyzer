
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import hashlib
import plotly.graph_objects as go

from cp_utils import (
    load_csv_auto,
    compute_cp_linear,
    compute_cp_5k_range,
    detect_best_test_segments,
    detect_stable_blocks,
    compute_power_duration_curve,
    running_effectiveness,
)

# ===========================
# Formatting helpers
# ===========================
def fmt_sec_hms(total_s: int) -> str:
    total_s = int(total_s)
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def fmt_short_axis(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if sec < 3600:
        return f"{m}m" if s == 0 else f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h" if m == 0 else f"{h}h{m:02d}m"

def fmt_pace_mmss(sec_val):
    if sec_val is None:
        return "–"
    try:
        total = int(sec_val)
    except Exception:
        return "–"
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"

def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    st.markdown(
        f"""
        <div style="text-align:center;padding:1.2rem;background-color:#ffffff;
        border-radius:10px;border:2px solid #e0e0e0;box-shadow:0 2px 10px rgba(0,0,0,0.06);
        margin-top:1rem;">
            <h3 style="color:{color};margin-bottom:0.5rem;">{title}</h3>
            <p style="font-size:1.6rem;font-weight:800;color:{color};margin:0;">{main_value}</p>
            <p style="font-size:1.05rem;color:#333;margin-top:0.35rem;">{subtext}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================
# Caching
# ===========================
@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes):
    import io
    return load_csv_auto(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def compute_pdc_cached(file_hash: str, max_dur: int, step: int, power_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"power": power_series})
    return compute_power_duration_curve(df, max_duration_s=max_dur, step=step)

@st.cache_data(show_spinner=False)
def detect_stable_blocks_cached(file_hash: str, max_std: float, min_block: int, smooth_window: int, weight: float, df: pd.DataFrame):
    return detect_stable_blocks(
        df,
        max_std_ratio=max_std,
        min_duration_sec=min_block,
        smooth_window_sec=smooth_window,
        weight_kg=float(weight) if weight else None,
    )

# ===========================
# UI Layout
# ===========================
st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv'])
    stryd_weight = st.number_input("⚖️ Stryd Weight (kg)", min_value=40.0, max_value=120.0, value=76.0, step=0.1)
    test_choice = st.radio("Choose Analysis Type", ["Critical Power Test (3/12)", "5K Test", "Segment Analysis"], index=0)

    if test_choice == "Segment Analysis":
        st.subheader("Stable Block Settings")
        max_std = st.slider("Power Variability Threshold (%)", 2, 10, 5) / 100.0
        min_block = st.slider("Min Block Duration (s)", 10, 600, 60, 5)
        smooth_window = st.slider("Smoothing Window (s)", 1, 15, 5)
    else:
        max_std = min_block = smooth_window = None

    run_analysis = st.button("🚀 Run Analysis")

# ===========================
# Helpers
# ===========================
def find_col_contains(df: pd.DataFrame, key: str):
    key = key.lower()
    for c in df.columns:
        if key in str(c).lower():
            return c
    return None

# ===========================
# Main Logic
# ===========================
if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    df = load_csv_cached(uploaded_file.getvalue())

    power_col = find_col_contains(df, "power")
    if power_col is None:
        st.error("No power column found.")
        st.stop()
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    if "w/kg" in str(power_col).lower() or "wkg" in str(power_col).lower():
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

    # =====================================================
    # === 3/12 TEST MODE ==================================
    # =====================================================
    if "3/12" in test_choice:
        # Detect two efforts (short: 2–4 min, long: 9–12 min)
        segments = detect_best_test_segments(df, expected_durations=(180, 720), tolerance=0.333)  # ±33% covers 2–4 and 9–12

        # PDC
        pdc_df = compute_pdc_cached(file_hash, 1800, 5, df["power"])
        fig_pdc = go.Figure()
        fig_pdc.add_trace(go.Scatter(
            x=pdc_df["duration_s"], y=pdc_df["best_power_w"],
            mode="lines+markers", name="PDC",
            hovertemplate="Duration %{x}s<br>Power %{y:.1f}W<extra></extra>"
        ))
        for seg in segments:
            fig_pdc.add_vline(
                x=seg["found_dur"],
                line=dict(color="orange", width=2, dash="dash"),
                annotation_text=f"{seg['found_dur']}s",
                annotation_position="top right"
            )
        fig_pdc.update_layout(title="Power Duration Curve", template="plotly_white", height=400)
        st.plotly_chart(fig_pdc, use_container_width=True)

        # Segments overlay
        elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.4, name="Power"))
        for i, seg in enumerate(segments):
            color = f"rgba(255,165,0,{0.25 + 0.2*i})"
            x0, x1 = float(elapsed_s.iloc[seg["start_idx"]]), float(elapsed_s.iloc[seg["end_idx"]])
            fig_time.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0)
            fig_time.add_annotation(x=(x0+x1)/2, y=float(df["power"].max()), text=f"Seg {i+1}", showarrow=False, yanchor="bottom")
        fig_time.update_layout(title="Detected 3/12 Segments", template="plotly_white", height=400)
        st.plotly_chart(fig_time, use_container_width=True)

        # Segment stats table
        dist_col = find_col_contains(df, "watch distance") or find_col_contains(df, "distance (m)") or find_col_contains(df, "distance_m") or find_col_contains(df, "distance")
        rows = []
        for i, seg in enumerate(segments):
            start, end = seg["start_idx"], seg["end_idx"]
            duration = int(seg["found_dur"])
            power = float(seg["avg_power"])
            distance_m = np.nan
            if dist_col:
                dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
                if dist_series.notna().any():
                    distance_m = float(dist_series.iloc[end] - dist_series.iloc[start])
            RE = None
            if not np.isnan(distance_m) and distance_m > 0:
                RE = running_effectiveness(distance_m, duration, power, stryd_weight)
            rows.append({
                "Segment": f"{i+1}",
                "Duration (s)": duration,
                "Duration (mm:ss)": fmt_pace_mmss(duration),
                "Avg Power (W)": f"{power:.1f}",
                "Distance (m)": f"{distance_m:.0f}" if not np.isnan(distance_m) else "–",
                "RE": f"{RE:.3f}" if RE else "–",
            })
        st.subheader("Segment Statistics")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # CP Calculation when we have 2 segments
        if len(segments) >= 2:
            p1, t1 = segments[0]["avg_power"], segments[0]["found_dur"]
            p2, t2 = segments[1]["avg_power"], segments[1]["found_dur"]
            cp, w_prime = compute_cp_linear(p1, t1, p2, t2)
            show_result_card("Critical Power (3/12)", f"{cp:.1f} W", f"W′ = {w_prime/1000:.2f} kJ", "#1a73e8")

    # =====================================================
    # === 5K TEST MODE ====================================
    # =====================================================
    elif "5K" in test_choice:
        segments = detect_best_test_segments(df, expected_durations=(1200,), tolerance=0.25)  # ~15–25 min window
        pdc_df = compute_pdc_cached(file_hash, 3600, 5, df["power"])
        fig_pdc = go.Figure()
        fig_pdc.add_trace(go.Scatter(
            x=pdc_df["duration_s"], y=pdc_df["best_power_w"],
            mode="lines+markers", name="PDC",
            hovertemplate="Duration %{x}s<br>Power %{y:.1f}W<extra></extra>"
        ))
        for seg in segments:
            fig_pdc.add_vline(
                x=seg["found_dur"],
                line=dict(color="orange", width=2, dash="dash"),
                annotation_text=f"{seg['found_dur']}s",
                annotation_position="top right"
            )
        fig_pdc.update_layout(title="Power Duration Curve", template="plotly_white", height=400)
        st.plotly_chart(fig_pdc, use_container_width=True)

        elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.4, name="Power"))
        for i, seg in enumerate(segments):
            color = f"rgba(255,165,0,{0.25 + 0.2*i})"
            x0, x1 = float(elapsed_s.iloc[seg["start_idx"]]), float(elapsed_s.iloc[seg["end_idx"]])
            fig_time.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0)
            fig_time.add_annotation(x=(x0+x1)/2, y=float(df["power"].max()), text=f"Seg {i+1}", showarrow=False, yanchor="bottom")
        fig_time.update_layout(title="Detected 5K/Long TT Segments", template="plotly_white", height=400)
        st.plotly_chart(fig_time, use_container_width=True)

        # CP estimation from long segment power
        if len(segments) >= 1:
            avg_pow = float(segments[0]["avg_power"])
            cp_range = compute_cp_5k_range(avg_pow)
            cp_min, cp_mid, cp_max = min(cp_range.values()), list(cp_range.values())[1], max(cp_range.values())
            show_result_card("Estimated CP range", f"{cp_min:.1f}–{cp_max:.1f} W", f"Typical ≈ {cp_mid:.1f} W", "#ff8800")

    # =====================================================
    # === SEGMENT ANALYSIS MODE ===========================
    # =====================================================
    else:
        st.subheader("🏃 Segment Analysis")
        pdc_df = compute_pdc_cached(file_hash, 3600, 5, df["power"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pdc_df["duration_s"], y=pdc_df["best_power_w"],
            mode="lines+markers", name="PDC",
            hovertemplate="Duration %{x}s<br>Power %{y:.1f}W<extra></extra>"
        ))
        fig.update_layout(title="Power Duration Curve", template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

        max_std = max_std or 0.05
        min_block = min_block or 60
        smooth_window = smooth_window or 5

        blocks = detect_stable_blocks_cached(file_hash, max_std, min_block, smooth_window, stryd_weight, df)
        if not blocks:
            st.info("No stable blocks found.")
        else:
            rows = []
            for b in blocks:
                rows.append({
                    "Duration": str(timedelta(seconds=int(b['duration_s']))),
                    "Avg Power (W)": f"{b['avg_power']:.1f}",
                    "Distance (m)": f"{b['distance_m']:.0f}",
                    "Pace (/km)": fmt_pace_mmss(b['pace_per_km']) if b['pace_per_km'] else "–",
                    "RE": f"{b['RE']:.3f}" if b.get("RE") else "–",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
            smooth_power = df["power"].rolling(window=max(1, smooth_window), min_periods=1).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.3, name="Power"))
            fig2.add_trace(go.Scatter(x=elapsed_s, y=smooth_power, mode="lines", name=f"Smoothed ({smooth_window}s)"))

            palette = ["LightGreen", "LightSkyBlue", "LightSalmon", "Khaki", "Plum", "LightPink", "PaleTurquoise", "Wheat"]
            for idx, b in enumerate(blocks):
                x0, x1 = float(elapsed_s.iloc[b['start_idx']]), float(elapsed_s.iloc[b['end_idx']])
                fig2.add_vrect(x0=x0, x1=x1, fillcolor=palette[idx % len(palette)], opacity=0.25, line_width=0)
                fig2.add_annotation(x=(x0+x1)/2, y=float(df["power"].max()), text=f"B{idx+1}", showarrow=False, yanchor="bottom")

            fig2.update_layout(title="Power over Time (Stable Blocks)", template="plotly_white", height=460)
            st.plotly_chart(fig2, use_container_width=True)

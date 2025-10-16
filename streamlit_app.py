import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import hashlib
import plotly.graph_objects as go

from cp_utils import (
    load_csv_auto,
    best_avg_power,
    extend_best_segment,
    compute_cp_linear,
    compute_cp_5k_range,
    detect_best_test_segments,
    infer_test_type_from_pdc,
    compute_power_duration_curve,
)
from docs import render_documentation


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


# ===========================
# UI Layout
# ===========================
st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])
    stryd_weight = st.number_input("⚖️ Stryd Weight (kg)", min_value=40.0, max_value=120.0, value=76.0, step=0.1)
    test_choice = st.radio("Choose Analysis Type", ["Critical Power Test (3/12)", "5K Test", "Power Duration Curve"], index=0)
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

    # Detect test type automatically
    inferred_type = infer_test_type_from_pdc(df)
    st.caption(f"🧠 Detected test type: **{inferred_type}**")

    # =========================
    # 3/12 and 5K with PDC
    # =========================
    if "3/12" in test_choice or "5K" in test_choice:
        # Compute PDC
        pdc_df = compute_pdc_cached(file_hash, 1800 if "3/12" in test_choice else 3600, 5, df["power"])

        # Find segments
        if "3/12" in test_choice:
            segments = detect_best_test_segments(df, expected_durations=(180, 720))
        else:
            segments = detect_best_test_segments(df, expected_durations=(1200,))  # around 20 min for 5k/20min test

        # --- PDC chart
        fig_pdc = go.Figure()
        fig_pdc.add_trace(
            go.Scatter(
                x=pdc_df["duration_s"], y=pdc_df["best_power_w"],
                mode="lines+markers", name="PDC",
                hovertemplate="Duration %{x}s<br>Power %{y:.1f}W<extra></extra>"
            )
        )
        for seg in segments:
            fig_pdc.add_vline(
                x=seg["found_dur"],
                line=dict(color="orange", width=2, dash="dash"),
                annotation_text=f"{seg['found_dur']}s",
                annotation_position="top right"
            )
        fig_pdc.update_layout(
            title="Power Duration Curve",
            xaxis_title="Duration (s)",
            yaxis_title="Best Power (W)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_pdc, use_container_width=True)

        # --- Power over time chart
        elapsed_s = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", name="Power", opacity=0.4))
        for i, seg in enumerate(segments):
            color = f"rgba(255,165,0,{0.25 + 0.2*i})"
            x0, x1 = float(elapsed_s.iloc[seg["start_idx"]]), float(elapsed_s.iloc[seg["end_idx"]])
            fig_time.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.3, layer="below", line_width=0)
            fig_time.add_annotation(x=(x0+x1)/2, y=max(df["power"]), text=f"Seg {i+1}", showarrow=False, yanchor="bottom")
        fig_time.update_layout(
            title="Detected Test Segments",
            xaxis_title="Elapsed Time (s)",
            yaxis_title="Power (W)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # --- Compute CP if 3/12
        if "3/12" in test_choice and len(segments) >= 2:
            p1, t1 = segments[0]["avg_power"], segments[0]["found_dur"]
            p2, t2 = segments[1]["avg_power"], segments[1]["found_dur"]
            cp, w_prime = compute_cp_linear(p1, t1, p2, t2)
            show_result_card("Critical Power (3/12)", f"{cp:.1f} W", f"W′ = {w_prime/1000:.2f} kJ", "#1a73e8")

        # --- 5K estimation
        elif "5K" in test_choice and len(segments) >= 1:
            avg_pow = segments[0]["avg_power"]
            cp_range = compute_cp_5k_range(avg_pow)
            cp_min, cp_mid, cp_max = min(cp_range.values()), list(cp_range.values())[1], max(cp_range.values())
            show_result_card("Estimated CP range", f"{cp_min:.1f}–{cp_max:.1f} W", f"Typical ≈ {cp_mid:.1f} W", "#ff8800")

    # =========================
    # PDC Standalone Mode
    # =========================
    else:
        pdc_df = compute_pdc_cached(file_hash, 3600, 5, df["power"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pdc_df["duration_s"], y=pdc_df["best_power_w"],
                mode="lines+markers", name="PDC",
                hovertemplate="Duration %{x}s<br>Power %{y:.1f}W<extra></extra>"
            )
        )
        fig.update_layout(
            title="Power Duration Curve",
            xaxis_title="Duration (s)",
            yaxis_title="Best Power (W)",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
render_documentation()


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
    detect_stable_blocks,
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

def fmt_pace_mmss(sec_val):
    if sec_val is None:
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

# ===========================
# Caching
# ===========================
@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes):
    import io
    return load_csv_auto(io.BytesIO(file_bytes))

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

# ===========================
# UI Helpers
# ===========================
def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center;padding:1.2rem;background-color:#ffffff;
        border-radius:10px;border:2px solid #e0e0e0;box-shadow:0 2px 10px rgba(0,0,0,0.06);
        margin-top:1rem;">
            <h3 style="color:{color};margin-bottom:0.5rem;">{title}</h3>
            <p style="font-size:1.6rem;font-weight:800;color:{color};margin:0;">{main_value}</p>
            <p style="font-size:1.05rem;color:#333;margin-top:0.35rem;">{subtext}</p>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# Sidebar & Page
# ===========================
st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])
    stryd_weight = st.number_input("⚖️ Stryd Weight (kg)", min_value=40.0, max_value=120.0, value=76.0, step=0.1)
    test_choice = st.radio("Choose Analysis Type", ["Critical Power Test (3/12)", "5K Test", "Power Duration Curve"], index=0)

    if test_choice == "Power Duration Curve":
        st.subheader("PDC Settings")
        max_dur = st.slider("Max Duration (s)", 60, 7200, 3600, 30)
        pdc_res = st.radio("Curve point resolution", ["Every 1s", "Every 5s"], index=1, horizontal=True)
        pdc_step = 1 if pdc_res == "Every 1s" else 5

        st.subheader("Stable Block Settings")
        max_std = st.slider("Power Variability Threshold (%)", 2, 10, 5) / 100.0
        min_block = st.slider("Min Block Duration (s)", 10, 600, 60, 5)
        smooth_window = st.slider("Smoothing Window (s)", 1, 15, 5)

    run_analysis = st.button("🚀 Run Analysis")

# ===========================
# Main logic
# ===========================
def find_col_contains(df: pd.DataFrame, key: str):
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
    df = load_csv_cached(uploaded_file.getvalue())

    # Power column
    power_col = find_col_contains(df, "power")
    if power_col is None:
        st.error("No power column found.")
        st.stop()
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    if "w/kg" in str(power_col).lower() or "wkg" in str(power_col).lower():
        df["power"] = df["power"] * float(stryd_weight)

    # Timestamp
    ts_col = find_col_contains(df, "timestamp") or find_col_contains(df, "time")
    if ts_col is None:
        st.error("No timestamp column found.")
        st.stop()
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp", "power"]).sort_values("timestamp").reset_index(drop=True)

    # Distance (optional)
    dist_col = find_col_contains(df, "watch distance") or find_col_contains(df, "distance (m)") or find_col_contains(df, "distance_m")

    st.markdown("## 📊 Analysis Results")

    # ==== 3/12 Test ====
    if "3/12" in test_choice:
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)
        ext3 = extend_best_segment(df, s3, e3, best3)
        ext12 = extend_best_segment(df, s12, e12, best12)
        cp, w_prime = compute_cp_linear(ext3[0], 180, ext12[0], 720)

        # Optional distance/pace
        dist3 = dist12 = np.nan
        if dist_col is not None:
            dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            if dist_series.notna().any():
                dist3 = float(dist_series.iloc[ext3[2]] - dist_series.iloc[ext3[1]])
                dist12 = float(dist_series.iloc[ext12[2]] - dist_series.iloc[ext12[1]])

        dur3 = int(ext3[3]); dur12 = int(ext12[3])
        pace3 = timedelta(seconds=int(dur3 / (dist3 / 1000))) if pd.notna(dist3) and dist3 > 0 else None
        pace12 = timedelta(seconds=int(dur12 / (dist12 / 1000))) if pd.notna(dist12) and dist12 > 0 else None

        st.subheader("Segment Details")
        seg_df = pd.DataFrame({
            "Segment": ["3-minute", "12-minute"],
            "Distance (m)": [f"{dist3:.0f}" if pd.notna(dist3) else "–", f"{dist12:.0f}" if pd.notna(dist12) else "–"],
            "Duration": [str(timedelta(seconds=dur3)), str(timedelta(seconds=dur12))],
            "Pace (/km)": [str(pace3) if pace3 else "–", str(pace12) if pace12 else "–"],
            "Avg Power (W)": [f"{ext3[0]:.1f}", f"{ext12[0]:.1f}"],
        })
        st.dataframe(seg_df, use_container_width=True)

        show_result_card("Critical Power (3/12 Test)", f"{cp:.1f} W", f"W′ = {w_prime/1000:.2f} kJ", "#1a73e8")

    # ==== 5K Test ====
    elif "5K" in test_choice:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
        t5k = int(ext5k[3]); avg_pow = float(ext5k[0])

        # Distance / pace
        actual_distance = 5000.0
        if dist_col is not None:
            dist_series = pd.to_numeric(df[dist_col], errors="coerce").ffill()
            if dist_series.notna().any():
                actual_distance = float(dist_series.iloc[ext5k[2]] - dist_series.iloc[ext5k[1]])
        pace_per_km = timedelta(seconds=int(t5k / (actual_distance / 1000))) if actual_distance > 0 else None

        st.subheader("Segment Details")
        seg_df = pd.DataFrame({
            "Segment": ["5 km Time Trial"],
            "Distance (m)": [f"{actual_distance:.0f}"],
            "Duration": [str(timedelta(seconds=t5k))],
            "Pace (/km)": [str(pace_per_km) if pace_per_km else "–"],
            "Avg Power (W)": [f"{avg_pow:.1f}"],
        })
        st.dataframe(seg_df, use_container_width=True)

        cp_results = compute_cp_5k_range(avg_pow)
        profiles = list(cp_results.items())
        cp_table = pd.DataFrame({
            "Profile": [k for k, _ in profiles],
            "CP (W)": [f"{v:.1f}" for _, v in profiles],
            "Scaling": ["98.5%", "97.5%", "96.5%"],
            "Trait": ["Endurance-focused", "Typical distance runner", "Power-focused"],
        })
        st.subheader("Critical Power Profiles")
        st.dataframe(cp_table, use_container_width=True, hide_index=True)

        cp_min = min(cp_results.values()); cp_max = max(cp_results.values()); cp_mid = list(cp_results.values())[1]
        show_result_card("Estimated CP range (5K Time Trial)", f"{cp_min:.1f} – {cp_max:.1f} W", f"Typical profile ≈ {cp_mid:.1f} W", "#ff8800")

    # ==== PDC + Stable Blocks ====
    else:
        # PDC
        pdc_df = compute_pdc_cached(file_hash, max_dur, pdc_step, df["power"])
        pdc_df["duration_label"] = [fmt_sec_hms(s) for s in pdc_df["duration_s"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pdc_df["duration_s"], y=pdc_df["best_power_w"],
            mode="lines+markers", name="PDC",
            customdata=pdc_df["duration_label"],
            hovertemplate="Duration: %{customdata}<br>Power: %{y:.1f} W<extra></extra>"
        ))
        max_x = int(pdc_df["duration_s"].max()) if len(pdc_df) else 3600
        base_ticks = [0, 5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
        tickvals = [t for t in base_ticks if t <= max_x]
        ticktext = [fmt_short_axis(t) for t in tickvals]
        fig.update_layout(title="Power Duration Curve", xaxis_title="Duration", yaxis_title="Best Average Power (W)",
                          template="plotly_white", height=420, margin=dict(t=60, r=20, l=50, b=60))
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0, max_x])
        st.plotly_chart(fig, use_container_width=True)

        # Stable blocks
        blocks = detect_stable_blocks_cached(file_hash, max_std, min_block, smooth_window, stryd_weight, df)
        if not blocks:
            st.info("No stable blocks found.")
        else:
            tbl_rows = []
            for b in blocks:
                tbl_rows.append({
                    "Duration": str(timedelta(seconds=int(b['duration_s']))),
                    "Avg Power (W)": f"{b['avg_power']:.1f}",
                    "Distance (m)": f"{b['distance_m']:.0f}",
                    "Pace (/km)": fmt_pace_mmss(b['pace_per_km']) if b['pace_per_km'] else "–",
                    "RE": f"{b['RE']:.3f}" if b.get("RE") else "–",
                })
            tbl = pd.DataFrame(tbl_rows)
            tbl.reset_index(drop=True, inplace=True)
            tbl.index.name = "Block"

            st.subheader("Stable Blocks")
            st.dataframe(tbl, use_container_width=True)

            # Overlay
            ts = df["timestamp"]
            if pd.api.types.is_datetime64_any_dtype(ts):
                elapsed_s = (ts - ts.iloc[0]).dt.total_seconds()
            else:
                ts_num = pd.to_numeric(ts, errors="coerce")
                if ts_num.notna().any():
                    elapsed_s = ts_num - float(ts_num.iloc[0])
                else:
                    elapsed_s = pd.Series(range(len(df)), index=df.index, dtype=float)
            elapsed_s = (elapsed_s - float(elapsed_s.min())).astype(float)

            smooth_power = df["power"].rolling(window=max(1, smooth_window), min_periods=1).mean()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", opacity=0.3, name="Power"))
            fig2.add_trace(go.Scatter(x=elapsed_s, y=smooth_power, mode="lines", name=f"Smoothed ({smooth_window}s)"))

            shapes = []
            annotations = []
            palette = ["LightGreen", "LightSkyBlue", "LightSalmon", "Khaki", "Plum", "LightPink", "PaleTurquoise", "Wheat"]
            for idx, b in enumerate(blocks):
                x0, x1 = float(elapsed_s.iloc[b['start_idx']]), float(elapsed_s.iloc[b['end_idx']])
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1,
                                   fillcolor=palette[idx % len(palette)], opacity=0.25, line_width=0))
                annotations.append(dict(x=(x0+x1)/2, y=1.02, xref="x", yref="paper", text=str(idx), showarrow=False))

            max_x2 = float(elapsed_s.max()) if len(elapsed_s) else 3600
            base_ticks2 = [0, 5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
            tickvals2 = [t for t in base_ticks2 if t <= max_x2]
            ticktext2 = [fmt_short_axis(int(t)) for t in tickvals2] if tickvals2 else None

            fig2.update_layout(title="Power over Time (Stable Blocks)", xaxis_title="Elapsed Time", yaxis_title="Power (W)",
                               template="plotly_white", height=460, shapes=shapes, annotations=annotations,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               margin=dict(t=60, r=20, l=50, b=60))
            fig2.update_xaxes(type="linear", tickmode="array", tickvals=tickvals2, ticktext=ticktext2, range=[0, max_x2])

            st.plotly_chart(fig2, use_container_width=True)

            st.caption(f"Cache key → file={file_hash[:8]} • PDC step={pdc_step}s • blocks: std≤{int(max_std*100)}% • min={min_block}s • smooth={smooth_window}s")

st.markdown("---")
render_documentation()

@ -1,6 +1,11 @@

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import hashlib
import plotly.graph_objects as go

from cp_utils import (
    load_csv_auto,
    best_avg_power,
@ -13,12 +18,58 @@ from cp_utils import (
    running_effectiveness,
)
from docs import render_documentation
import plotly.graph_objects as go
import hashlib

# ============================================================
#  UI Helper: Styled Result Card
# ============================================================
def fmt_sec_hms(total_s: int) -> str:
    total_s = int(total_s)
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def fmt_pace_mmss(sec_val):
    if not sec_val:
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

@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes):
    import io
    df = load_csv_auto(io.BytesIO(file_bytes))
    return df

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

def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    st.markdown("---")
@ -41,100 +92,177 @@ def show_result_card(title: str, main_value: str, subtext: str = "", color: str
        unsafe_allow_html=True,
    )

# ---------- Caching helpers ----------
st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes):
    """Cache CSV parsing by file bytes (so the same file doesn't re-parse)."""
    import io
    from cp_utils import load_csv_auto
    return load_csv_auto(io.BytesIO(file_bytes))
with st.sidebar:
    st.header("⚙️ Analysis Settings")

@st.cache_data(show_spinner=False)
def compute_pdc_cached(file_hash: str, max_dur: int, pdc_step: int, power_series: pd.Series) -> pd.DataFrame:
    import numpy as np
    pdc_df = compute_pdc_cached(file_hash=file_hash, max_dur=max_dur, pdc_step=pdc_step, power_series=df["power"])
    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])

        # Human-readable duration labels (mm:ss or hh:mm:ss)
        def _fmt_sec_to_label(total_s: int) -> str:
            total_s = int(total_s)
            m, s = divmod(total_s, 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"
        pdc_df["duration_label"] = [_fmt_sec_to_label(s) for s in pdc_df["duration_s"]]
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
        ["Critical Power Test (3/12)", "5K Test", "Power Duration Curve"],
        index=0,
    )

    if test_choice == "Power Duration Curve":
        st.subheader("PDC Settings")
        max_dur = st.slider("Max Duration (s)", 60, 7200, 3600, 30)
        pdc_res = st.radio("Curve point resolution", ["Every 1s", "Every 5s"], index=1, horizontal=True)
        pdc_step = 1 if pdc_res == "Every 1s" else 5
        st.subheader("Stable Block Settings")
        max_std = st.slider("Power Variability Threshold (%)", 2, 10, 5) / 100.0
        min_block = st.slider("Min Block Duration (s)", 10, 600, 60, 5)
        smooth_window = st.slider("Smoothing Window (s)", 1, 15, 5)

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis")

if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file before running the analysis.")
        st.stop()

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    df = load_csv_cached(uploaded_file.getvalue())

    if "power_wkg" in df.columns:
        df["power"] = df["power_wkg"] * stryd_weight
    elif "power" not in df.columns:
        st.error("No valid power column found in file.")
        st.stop()

    if "timestamp" not in df.columns:
        time_col = next((c for c in df.columns if "time" in c), None)
        if time_col:
            df["timestamp"] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
        else:
            st.error("Timestamp column not found.")
            st.stop()

    if "watch distance (meters)" not in df.columns:
        df["watch distance (meters)"] = 0

    df = df.sort_values("timestamp").reset_index(drop=True)

    st.markdown("## 📊 Analysis Results")

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
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

        show_result_card("Critical Power (3/12 Test)", f"{cp:.1f} W", f"W′ = {w_prime/1000:.2f} kJ", color="#1a73e8")

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
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

        cp_results = compute_cp_5k_range(avg_pow)
        cp_table = pd.DataFrame({
            "Profile": ["Aerobic", "Balanced", "Anaerobic"],
            "CP (W)": [f"{cp:.1f}" for cp in cp_results.values()],
            "Scaling": ["98.5%", "97.5%", "96.5%"],
            "Trait": ["Endurance-focused", "Typical distance runner", "Power-focused"],
        })

        st.subheader("Critical Power Profiles")
        st.dataframe(cp_table, use_container_width=True, hide_index=True)

        cp_min, cp_max, cp_mid = min(cp_results.values()), max(cp_results.values()), list(cp_results.values())[1]
        show_result_card("Estimated Critical Power Range (5K Time Trial)", f"{cp_min:.1f} – {cp_max:.1f} W", f"Typical profile ≈ {cp_mid:.1f} W", color="#ff8800")

    elif "Power Duration Curve" in test_choice:
        # PDC via cache (per 1s or per 5s)
        pdc_df = compute_pdc_cached(file_hash=file_hash, max_dur=max_dur, pdc_step=pdc_step, power_series=df["power"])
        pdc_df["duration_label"] = [fmt_sec_hms(s) for s in pdc_df["duration_s"]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pdc_df["duration_s"],
                y=pdc_df["best_power_w"],
                mode="lines+markers",
                name="PDC",
                customdata=pdc_df["duration_label"],
                hovertemplate="Duration: %{customdata}<br>Power: %{y:.1f} W<extra></extra>",
            )
        )

        # Dynamic tick labels for durations
        fig.add_trace(go.Scatter(
            x=pdc_df["duration_s"], y=pdc_df["best_power_w"], mode="lines+markers", name="PDC",
            customdata=pdc_df["duration_label"],
            hovertemplate="Duration: %{customdata}<br>Power: %{y:.1f} W<extra></extra>"
        ))

        max_x = int(pdc_df["duration_s"].max()) if len(pdc_df) else 3600
        base_ticks = [5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
        base_ticks = [0, 5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
        tickvals = [t for t in base_ticks if t <= max_x]
        ticktext = [fmt_short_axis(t) for t in tickvals]

        def _fmt_short(sec: int) -> str:
            if sec < 60:
                return f"{sec}s"
            m, s = divmod(sec, 60)
            if sec < 3600:
                return f"{m}m" if s == 0 else f"{m}m{s:02d}s"
            h, m = divmod(m, 60)
            return f"{h}h" if m == 0 else f"{h}h{m:02d}m"

        ticktext = [_fmt_short(t) for t in tickvals]

        fig.update_layout(
            title="Power Duration Curve",
            xaxis_title="Duration",
            yaxis_title="Best Average Power (W)",
            # linear axis fits the per-1s/5s resolution nicely; switch to log if desired later
            xaxis_type="linear",
            template="plotly_white",
            height=420,
            margin=dict(t=60, r=20, l=50, b=60),
        )
        fig.update_layout(title="Power Duration Curve", xaxis_title="Duration", yaxis_title="Best Average Power (W)",
                          xaxis_type="linear", template="plotly_white", height=420, margin=dict(t=60, r=20, l=50, b=60))
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0, max_x])

        st.plotly_chart(fig, use_container_width=True)

        # Stable blocks
        blocks = detect_stable_blocks_cached(
            file_hash=file_hash,
            max_std=max_std,
            min_block=min_block,
            smooth_window=smooth_window,
            stryd_weight=float(stryd_weight) if stryd_weight else None,
            df=df
        ) if stryd_weight else None,
        )
        # Stable blocks via cache
        blocks = detect_stable_blocks_cached(file_hash=file_hash, max_std=max_std, min_block=min_block, smooth_window=smooth_window, stryd_weight=float(stryd_weight) if stryd_weight else None, df=df)

        if not blocks:
            st.info("No stable blocks found with the current settings.")
        else:
            tbl = pd.DataFrame([
                {
                    "Block": idx,
                    "Duration": str(timedelta(seconds=int(b["duration_s"]))),
                    "Avg Power (W)": f"{b['avg_power']:.1f}",
                    "Distance (m)": f"{b['distance_m']:.0f}",
                    "Pace (/km)": str(timedelta(seconds=int(b["pace_per_km"]))) if b["pace_per_km"] else "–",
                    "Pace (/km)": fmt_pace_mmss(b["pace_per_km"]) if b["pace_per_km"] else "–",
                    "RE": f"{b['RE']:.3f}" if b.get("RE") else "–",
                }
                for idx, b in enumerate(blocks, start=1)
                for b in blocks
            ])
            tbl.reset_index(drop=True, inplace=True)
            tbl.index.name = "Block"

            st.subheader("Stable Blocks")
            st.dataframe(tbl, use_container_width=True)

            # Visual overlay: power vs time with shaded stable blocks
            # Build elapsed seconds robustly (timestamp may be datetime or numeric)
            # Overlay (linear axis)
            ts = df["timestamp"]
            if pd.api.types.is_datetime64_any_dtype(ts):
                elapsed_s = (ts - ts.iloc[0]).dt.total_seconds()
@ -144,7 +272,6 @@ def compute_pdc_cached(file_hash: str, max_dur: int, pdc_step: int, power_series
                    elapsed_s = ts_num - float(ts_num.iloc[0])
                else:
                    elapsed_s = pd.Series(range(len(df)), index=df.index, dtype=float)
            # start at 0 (linear axis)
            elapsed_s = (elapsed_s - float(elapsed_s.min())).astype(float)

            smooth_power_series = df["power"].rolling(window=max(1, smooth_window), min_periods=1).mean()
@ -153,68 +280,29 @@ def compute_pdc_cached(file_hash: str, max_dur: int, pdc_step: int, power_series
            fig2.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", name="Power (W)", opacity=0.35))
            fig2.add_trace(go.Scatter(x=elapsed_s, y=smooth_power_series, mode="lines", name=f"Smoothed ({smooth_window}s)"))

            # Color each block differently and add number-only labels
            palette = ["LightGreen", "LightSkyBlue", "LightSalmon", "Khaki", "Plum", "LightPink", "PaleTurquoise", "Wheat"]
            shapes = []
            annotations = []
            for idx, b in enumerate(blocks, start=1):
            for idx, b in enumerate(blocks):  # 0-based to match table index
                x0 = float(elapsed_s.iloc[b["start_idx"]])
                x1 = float(elapsed_s.iloc[b["end_idx"]])
                shapes.append({
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": palette[(idx-1) % len(palette)],
                    "opacity": 0.25,
                    "line": {"width": 0},
                })
                annotations.append(dict(
                    x=(x0 + x1) / 2,
                    y=1.01,
                    xref="x",
                    yref="paper",
                    text=f"{idx}",
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                ))

            # Dynamic tick labels (linear, same style as PDC)
                shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": x0, "x1": x1, "y0": 0, "y1": 1, "fillcolor": palette[idx % len(palette)], "opacity": 0.25, "line": {"width": 0}})
                annotations.append(dict(x=(x0 + x1) / 2, y=1.01, xref="x", yref="paper", text=f"{idx}", showarrow=False, font=dict(size=12), align="center"))

            max_x = float(elapsed_s.max()) if len(elapsed_s) else 3600
            base_ticks = [0, 5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
            tickvals = [t for t in base_ticks if t <= max_x]
            def _fmt_short(sec: int) -> str:
                if sec < 60:
                    return f"{sec}s"
                m, s = divmod(sec, 60)
                if sec < 3600:
                    return f"{m}m" if s == 0 else f"{m}m{s:02d}s"
                h, m = divmod(m, 60)
                return f"{h}h" if m == 0 else f"{h}h{m:02d}m"
            ticktext = [_fmt_short(int(t)) for t in tickvals] if tickvals else None

            fig2.update_layout(
                title="Power over Time (Stable Blocks)",
                xaxis_title="Elapsed Time",
                yaxis_title="Power (W)",
                template="plotly_white",
                height=460,
                shapes=shapes,
                annotations=annotations,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, r=20, l=50, b=60),
            )
            ticktext = [fmt_short_axis(int(t)) for t in tickvals] if tickvals else None

            fig2.update_layout(title="Power over Time (Stable Blocks)", xaxis_title="Elapsed Time", yaxis_title="Power (W)",
                               template="plotly_white", height=460, shapes=shapes, annotations=annotations,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               margin=dict(t=60, r=20, l=50, b=60))
            fig2.update_xaxes(type="linear", tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0, max_x])

            st.plotly_chart(fig2, use_container_width=True)

            st.caption(
                f"Cache key → file={file_hash[:8]} • PDC step={pdc_step}s • blocks: std≤{int(max_std*100)}% • min={min_block}s • smooth={smooth_window}s"
            )
            st.caption(f"Cache key → file={file_hash[:8]} • PDC step={pdc_step}s • blocks: std≤{int(max_std*100)}% • min={min_block}s • smooth={smooth_window}s")

    st.markdown("---")
st.markdown("---")
render_documentation()
@ -1,5 +1,7 @@

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from cp_utils import (
    load_csv_auto,
@ -15,10 +17,6 @@ from cp_utils import (
from docs import render_documentation
import plotly.graph_objects as go

# ============================================================
#  UI Helper: Styled Result Card
# ============================================================

def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    st.markdown("---")
    st.markdown(
@ -40,37 +38,18 @@ def show_result_card(title: str, main_value: str, subtext: str = "", color: str
        unsafe_allow_html=True,
    )

# ============================================================
#  Streamlit Page Setup
# ============================================================

st.set_page_config(page_title="Critical Power Analyzer", page_icon="⚡", layout="wide")

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    uploaded_file = st.file_uploader("📁 Upload CSV (Garmin/Stryd export)", type=["csv"])

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
    stryd_weight = st.number_input("⚖️ Stryd Weight (kg)", min_value=40.0, max_value=120.0, value=76.0, step=0.1)
    test_choice = st.radio("Choose Analysis Type", ["Critical Power Test (3/12)", "5K Test", "Power Duration Curve"], index=0)

    if test_choice == "Power Duration Curve":
        st.subheader("PDC Settings")
        max_dur = st.slider("Max Duration (s)", 60, 7200, 3600, 30)
        # Resolution selector: per 1s or 5s
        pdc_res = st.radio("Curve point resolution", ["Every 1s", "Every 5s"], index=0, horizontal=True)
        pdc_res = st.radio("Curve point resolution", ["Every 1s", "Every 5s"], index=1, horizontal=True)
        pdc_step = 1 if pdc_res == "Every 1s" else 5
        st.subheader("Stable Block Settings")
        max_std = st.slider("Power Variability Threshold (%)", 2, 10, 5) / 100.0
@ -80,10 +59,6 @@ with st.sidebar:
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis")

# ============================================================
#  Data Loading and Normalization
# ============================================================

if run_analysis:
    if uploaded_file is None:
        st.warning("Please upload a CSV file before running the analysis.")
@ -91,7 +66,6 @@ if run_analysis:

    df = load_csv_auto(uploaded_file)

    # Handle power conversion if file contains w/kg
    if "power_wkg" in df.columns:
        df["power"] = df["power_wkg"] * stryd_weight
    elif "power" not in df.columns:
@ -113,7 +87,6 @@ if run_analysis:

    st.markdown("## 📊 Analysis Results")

    # ------------------- 3/12-Minute Critical Power Test ------------------- #
    if "3/12" in test_choice:
        best3, s3, e3 = best_avg_power(df, 180)
        best12, s12, e12 = best_avg_power(df, 720)
@ -140,14 +113,8 @@ if run_analysis:
        }
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

        show_result_card(
            "Critical Power (3/12 Test)",
            f"{cp:.1f} W",
            f"W′ = {w_prime/1000:.2f} kJ",
            color="#1a73e8",
        )
        show_result_card("Critical Power (3/12 Test)", f"{cp:.1f} W", f"W′ = {w_prime/1000:.2f} kJ", color="#1a73e8")

    # ------------------- 5K Time Trial ------------------- #
    elif "5K" in test_choice:
        best5k, s5k, e5k = best_power_for_distance(df, 5000)
        ext5k = extend_best_segment(df, s5k, e5k, best5k)
@ -180,20 +147,11 @@ if run_analysis:
        st.dataframe(cp_table, use_container_width=True, hide_index=True)

        cp_min, cp_max, cp_mid = min(cp_results.values()), max(cp_results.values()), list(cp_results.values())[1]
        show_result_card(
            "Estimated Critical Power Range (5K Time Trial)",
            f"{cp_min:.1f} – {cp_max:.1f} W",
            f"Typical profile ≈ {cp_mid:.1f} W",
            color="#ff8800",
        )

    # ------------------- Power Duration Curve & Stable Blocks ------------------- #
        show_result_card("Estimated Critical Power Range (5K Time Trial)", f"{cp_min:.1f} – {cp_max:.1f} W", f"Typical profile ≈ {cp_mid:.1f} W", color="#ff8800")

    elif "Power Duration Curve" in test_choice:
        # PDC
        # Build PDC either per-second or per-5s using linear durations
        import numpy as np
        # PDC per 1s/5s
        durations = np.arange(5, max_dur + 1, pdc_step, dtype=int)
        # compute best rolling mean for each duration
        pows = []
        for d in durations:
            rm = df["power"].rolling(d, min_periods=d).mean()
@ -201,7 +159,6 @@ if run_analysis:
            pows.append(m)
        pdc_df = pd.DataFrame({"duration_s": durations, "best_power_w": pows}).dropna()

        # Human-readable duration labels (mm:ss or hh:mm:ss)
        def _fmt_sec_to_label(total_s: int) -> str:
            total_s = int(total_s)
            m, s = divmod(total_s, 60)
@ -210,20 +167,10 @@ if run_analysis:
        pdc_df["duration_label"] = [_fmt_sec_to_label(s) for s in pdc_df["duration_s"]]

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
        fig.add_trace(go.Scatter(x=pdc_df["duration_s"], y=pdc_df["best_power_w"], mode="lines+markers", name="PDC", customdata=pdc_df["duration_label"], hovertemplate="Duration: %{customdata}<br>Power: %{y:.1f} W<extra></extra>"))

        max_x = int(pdc_df["duration_s"].max()) if len(pdc_df) else 3600
        base_ticks = [5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
        base_ticks = [0, 5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
        tickvals = [t for t in base_ticks if t <= max_x]

        def _fmt_short(sec: int) -> str:
@ -234,51 +181,42 @@ if run_analysis:
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
        fig.update_layout(title="Power Duration Curve", xaxis_title="Duration", yaxis_title="Best Average Power (W)", xaxis_type="linear", template="plotly_white", height=420, margin=dict(t=60, r=20, l=50, b=60))
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0, max_x])

        st.plotly_chart(fig, use_container_width=True)

        # Stable blocks
        blocks = detect_stable_blocks(
            df,
            max_std_ratio=max_std,
            min_duration_sec=min_block,
            smooth_window_sec=smooth_window,
            weight_kg=float(stryd_weight) if stryd_weight else None,
        )
        blocks = detect_stable_blocks(df, max_std_ratio=max_std, min_duration_sec=min_block, smooth_window_sec=smooth_window, weight_kg=float(stryd_weight) if stryd_weight else None)

        if not blocks:
            st.info("No stable blocks found with the current settings.")
        else:
            def _fmt_pace_mmss(sec_val):
                if not sec_val: return "–"
                try: total = int(sec_val)
                except Exception: return "–"
                m, s = divmod(total, 60)
                return f"{m:02d}:{s:02d}"

            tbl = pd.DataFrame([
                {
                    "Block": idx,
                    "Duration": str(timedelta(seconds=int(b["duration_s"]))),
                    "Avg Power (W)": f"{b['avg_power']:.1f}",
                    "Distance (m)": f"{b['distance_m']:.0f}",
                    "Pace (/km)": str(timedelta(seconds=int(b["pace_per_km"]))) if b["pace_per_km"] else "–",
                    "Pace (/km)": _fmt_pace_mmss(b["pace_per_km"]) if b["pace_per_km"] else "–",
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
@ -288,7 +226,6 @@ if run_analysis:
                    elapsed_s = ts_num - float(ts_num.iloc[0])
                else:
                    elapsed_s = pd.Series(range(len(df)), index=df.index, dtype=float)
            # start at 0 (linear axis)
            elapsed_s = (elapsed_s - float(elapsed_s.min())).astype(float)

            smooth_power_series = df["power"].rolling(window=max(1, smooth_window), min_periods=1).mean()
@ -297,64 +234,30 @@ if run_analysis:
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
                if sec < 60: return f"{sec}s"
                m, s = divmod(sec, 60)
                if sec < 3600:
                    return f"{m}m" if s == 0 else f"{m}m{s:02d}s"
                if sec < 3600: return f"{m}m" if s == 0 else f"{m}m{s:02d}s"
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
            fig2.update_layout(title="Power over Time (Stable Blocks)", xaxis_title="Elapsed Time", yaxis_title="Power (W)", template="plotly_white", height=460, shapes=shapes, annotations=annotations, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=60, r=20, l=50, b=60))
            fig2.update_xaxes(type="linear", tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0, max_x])

            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
st.markdown("---")
render_documentation()
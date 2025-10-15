import streamlit as st
import pandas as pd
from datetime import timedelta
from cp_utils import (
    load_csv_auto,
    best_avg_power,
    best_power_for_distance,
    extend_best_segment,
    compute_cp_linear,
    compute_cp_5k_range,
    compute_power_duration_curve,
    detect_stable_blocks,
    running_effectiveness,
)
from docs import render_documentation
import plotly.graph_objects as go
import hashlib

# ============================================================
#  UI Helper: Styled Result Card
# ============================================================

def show_result_card(title: str, main_value: str, subtext: str = "", color: str = "#0b5394"):
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding: 1.6rem;
            background-color: #ffffff;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 1.5rem;
        ">
            <h2 style="color:{color}; margin-bottom:0.7rem;">🏁 {title}</h2>
            <p style="font-size:1.7rem;font-weight:800;color:{color};margin:0;">{main_value}</p>
            <p style="font-size:1.15rem;color:#333;margin-top:0.4rem;">{subtext}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Caching helpers ----------

@st.cache_data(show_spinner=False)
def load_csv_cached(file_bytes: bytes):
    """Cache CSV parsing by file bytes (so the same file doesn't re-parse)."""
    import io
    from cp_utils import load_csv_auto
    return load_csv_auto(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def compute_pdc_cached(file_hash: str, max_dur: int, pdc_step: int, power_series: pd.Series) -> pd.DataFrame:
    import numpy as np
    pdc_df = compute_pdc_cached(file_hash=file_hash, max_dur=max_dur, pdc_step=pdc_step, power_series=df["power"])

        # Human-readable duration labels (mm:ss or hh:mm:ss)
        def _fmt_sec_to_label(total_s: int) -> str:
            total_s = int(total_s)
            m, s = divmod(total_s, 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"
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
        max_x = int(pdc_df["duration_s"].max()) if len(pdc_df) else 3600
        base_ticks = [5, 10, 15, 30, 45, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 5400, 7200, 10800, 14400]
        tickvals = [t for t in base_ticks if t <= max_x]

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
                    "RE": f"{b['RE']:.3f}" if b.get("RE") else "–",
                }
                for idx, b in enumerate(blocks, start=1)
            ])
            st.subheader("Stable Blocks")
            st.dataframe(tbl, use_container_width=True)

            # Visual overlay: power vs time with shaded stable blocks
            # Build elapsed seconds robustly (timestamp may be datetime or numeric)
            ts = df["timestamp"]
            if pd.api.types.is_datetime64_any_dtype(ts):
                elapsed_s = (ts - ts.iloc[0]).dt.total_seconds()
            else:
                ts_num = pd.to_numeric(ts, errors="coerce")
                if ts_num.notna().any():
                    elapsed_s = ts_num - float(ts_num.iloc[0])
                else:
                    elapsed_s = pd.Series(range(len(df)), index=df.index, dtype=float)
            # start at 0 (linear axis)
            elapsed_s = (elapsed_s - float(elapsed_s.min())).astype(float)

            smooth_power_series = df["power"].rolling(window=max(1, smooth_window), min_periods=1).mean()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=elapsed_s, y=df["power"], mode="lines", name="Power (W)", opacity=0.35))
            fig2.add_trace(go.Scatter(x=elapsed_s, y=smooth_power_series, mode="lines", name=f"Smoothed ({smooth_window}s)"))

            # Color each block differently and add number-only labels
            palette = ["LightGreen", "LightSkyBlue", "LightSalmon", "Khaki", "Plum", "LightPink", "PaleTurquoise", "Wheat"]
            shapes = []
            annotations = []
            for idx, b in enumerate(blocks, start=1):
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
            fig2.update_xaxes(type="linear", tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[0, max_x])

            st.plotly_chart(fig2, use_container_width=True)

            st.caption(
                f"Cache key → file={file_hash[:8]} • PDC step={pdc_step}s • blocks: std≤{int(max_std*100)}% • min={min_block}s • smooth={smooth_window}s"
            )

    st.markdown("---")
render_documentation()

import streamlit as st

def render_documentation():
    """Render the documentation expander in Streamlit."""
    with st.expander("ℹ️ About & Documentation", expanded=False):
        st.markdown(r"""
## ⚡ Critical Power Analysis – How It Works

This tool estimates your **Critical Power (CP)** — the highest power you can sustain without fatigue rapidly accumulating — based on **running power data** from a `.csv` file exported from Stryd.

---

### 🧮 1️⃣ Critical Power Models

#### **A. 3/12-Minute Test (Linear Model)**
You perform two all-out efforts:
- **3 minutes** (short, anaerobic effort)
- **12 minutes** (long, aerobic effort)

The tool finds the **highest average power** over those durations and uses the **linear CP model**:

    st.latex(r"P = CP + \frac{W′}{t}")

where:
- *P* = mean power (W)  
- *t* = duration (s)  
- *CP* = Critical Power (W)  
- *W′* = finite anaerobic work capacity (J)

From these two data points, both **CP** and **W′** are solved directly.

---

#### **B. 5 K Time Trial (Empirical Model)**
If you upload a 5 K time trial file, the tool identifies the **highest-power 5 000 m segment** (using the *watch distance* field, not elapsed time).

The 5 K average power is then converted to CP using **empirical fatigue scaling factors**, inspired by Steve Palladino’s Power Project calculator:

| Runner Type | Typical CP vs 5 K Power | Description |
|--------------|------------------------|--------------|
| **Highly aerobic / ultra runner (slow fatigue)** | ~ 98.5 % | Exceptional endurance, power stays close to race power. |
| **Balanced distance runner (typical)** | ~ 97.5 % | Well-trained athlete, typical 5 K – marathon profile. |
| **Anaerobic / sprinter (fast fatigue)** | ~ 96.5 % | Strong short-term power, fatigues faster on long efforts. |

This gives a **CP range** rather than a single value — just like Palladino’s calculator — showing realistic variability between athlete types.

---

### 📊 2️⃣ Segment Selection

- The app automatically finds the **best rolling window** for each test duration (e.g. best 3 min, 12 min, or 5 000 m).
- If a slightly longer segment yields the same or higher average power, it **extends** the window automatically.
- Results include actual distance, duration, pace, and mean power.

---

### ⚙️ 3️⃣ Units & Input Notes

- Upload your `.csv` file (comma-separated).  
- Power data can be in **watts** or **W/kg**.  
  - If it’s in *W/kg*, enter your **Stryd weight** (the weight configured in the Stryd app).  
- Distances are read from “Watch Distance (meters)” or “Stryd Distance (meters)”.

---

### 📘 4️⃣ Interpretation

| Metric | What it tells you |
|---------|------------------|
| **Critical Power (CP)** | Sustainable threshold — approximately 40–60 min race power. |
| **W′** | Anaerobic work capacity (in kJ) — energy you can expend above CP before exhaustion. |
| **Pace / km** | Equivalent running pace for the detected segment. |
| **Power above CP (% difference)** | How much harder the test effort was than your sustainable threshold. |

---

### 🧠 5️⃣ Quick Reference – Typical Relationships
| Event | Typical Duration | % of CP | Purpose |
|--------|------------------|---------|----------|
| 1 K repeat | 3–4 min | 110–120 % | Anaerobic tolerance |
| 5 K race | 18–22 min | 103–107 % | Threshold calibration |
| 10 K race | 38–45 min | 100–102 % | CP-level effort |
| Half marathon | 80–100 min | 92–96 % | Endurance benchmark |

---

**💡 Tip:**  
Use the *3/12-minute test* when you can execute structured intervals,  
and the *5 K test* when you prefer a real-race simulation.  
Either method gives a valid CP estimate — just ensure you run them at maximal, steady effort.
        """)

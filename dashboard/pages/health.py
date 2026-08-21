from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "generated" / "final_prediction.csv"


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def status_key(value: str) -> str:
    value = str(value).strip().lower().replace("_", " ")
    if value in {"observation", "at risk", "at-risk", "atrisk"}:
        return "At Risk"
    if value == "diseased":
        return "Diseased"
    return "Healthy"


def status_color(value: str) -> str:
    return {"Healthy": "#17803d", "At Risk": "#d97706", "Diseased": "#c2413b"}[status_key(value)]


def gauge(title: str, value: float, minimum: float, maximum: float, suffix: str, color: str) -> go.Figure:
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 24, "color": "#17324d"}},
        title={"text": title, "font": {"size": 14, "color": "#526579"}},
        gauge={
            "axis": {"range": [minimum, maximum], "tickfont": {"color": "#526579"}},
            "bar": {"color": color},
            "bgcolor": "#edf3f7",
            "borderwidth": 0,
        },
    ))
    figure.update_layout(height=220, margin={"l": 20, "r": 20, "t": 45, "b": 10}, paper_bgcolor="white", font_color="#17324d")
    return figure


def show_health():
    st.markdown("# Cow health")
    st.caption("Live health overview from the latest completed multi-modal prediction run.")

    try:
        data = load_predictions().copy()
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error(f"Prediction data is not available: {error}")
        return
    if data.empty:
        st.warning("No cow predictions are available yet.")
        return

    data["Status"] = data["Final_Status"].map(status_key)
    status_counts = data["Status"].value_counts()
    kpis = [
        ("Total cows", len(data), "#2266a8"),
        ("Healthy", int(status_counts.get("Healthy", 0)), "#17803d"),
        ("At risk", int(status_counts.get("At Risk", 0)), "#d97706"),
        ("Diseased", int(status_counts.get("Diseased", 0)), "#c2413b"),
        ("Average confidence", f"{data['Confidence'].mean() * 100:.1f}%", "#2266a8"),
    ]
    columns = st.columns(5)
    for column, (label, value, color) in zip(columns, kpis):
        with column:
            st.markdown(f"<div class='kpi-card' style='border-top-color:{color}'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("### Herd health register")
    search = st.text_input("Search by cow ID, status, or behavior", placeholder="Search cows", key="health_search")
    if search:
        mask = data.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)
        filtered = data[mask]
    else:
        filtered = data
    display = filtered[["Cow_ID", "Status", "Risk_Score", "Dominant_Behavior", "Heart_Rate", "Body_Temperature", "Confidence", "Recommendation"]].rename(columns={
        "Cow_ID": "Cow ID", "Status": "Final status", "Risk_Score": "Risk score", "Dominant_Behavior": "Dominant behavior", "Heart_Rate": "Heart rate", "Body_Temperature": "Temperature", "Confidence": "Confidence", "Recommendation": "Recommendation",
    })
    styled = display.style.map(lambda value: f"color: {status_color(value)}; font-weight: 700" if value in {"Healthy", "At Risk", "Diseased"} else "", subset=["Final status"])
    st.dataframe(styled, hide_index=True, width="stretch", column_config={
        "Risk score": st.column_config.NumberColumn(format="%.0f"),
        "Heart rate": st.column_config.NumberColumn("Heart rate (bpm)", format="%.2f"),
        "Temperature": st.column_config.NumberColumn("Temperature (C)", format="%.2f"),
        "Confidence": st.column_config.NumberColumn(format="%.1f%%", min_value=0, max_value=1),
    })

    selected_id = st.selectbox("Open cow profile", options=data["Cow_ID"].tolist(), key="health_cow")
    cow = data.loc[data["Cow_ID"] == selected_id].iloc[0]
    st.markdown(f"### Profile: {selected_id} <span class='status-pill' style='background:{status_color(cow['Status'])}'>{cow['Status']}</span>", unsafe_allow_html=True)
    behavior, sensors, health = st.columns(3)
    with behavior:
        with st.container(border=True):
            st.markdown("**Behavior summary**")
            for label, field in [("Standing", "Standing_%"), ("Walking", "Walking_%"), ("Eating", "Eating_%"), ("Lying", "Lying_%"), ("Rumination", "Rumination_%")]:
                st.write(f"{label}: {cow[field]:.1f}%")
            st.caption(f"Dominant: {str(cow['Dominant_Behavior']).title()}")
    with sensors:
        with st.container(border=True):
            st.markdown("**Sensor summary**")
            st.write(f"Heart rate: {cow['Heart_Rate']:.2f} bpm")
            st.write(f"Body temperature: {cow['Body_Temperature']:.2f} C")
            st.write(f"Humidity: {cow['Humidity']:.1f}%")
            st.write(f"Ambient temperature: {cow['Ambient_Temperature']:.1f} C")
            st.write(f"THI: {cow['THI']:.1f}")
    with health:
        with st.container(border=True):
            st.markdown("**Health summary**")
            st.metric("Risk score", f"{cow['Risk_Score']:.0f}")
            st.metric("Confidence", f"{cow['Confidence'] * 100:.1f}%")
            st.write(f"Risk level: {cow['Risk_Level']}")
    with st.expander("Recommendation", icon=":material/assignment:"):
        for recommendation in str(cow["Recommendation"]).split("|"):
            st.write(f"- {recommendation.strip()}")

    st.markdown("### Cow signals")
    chart_columns = st.columns(4)
    charts = [
        ("Heart rate", float(cow["Heart_Rate"]), 35, 120, " bpm", "#2266a8"),
        ("Temperature", float(cow["Body_Temperature"]), 35, 42, " C", "#d97706"),
        ("Risk score", float(cow["Risk_Score"]), 0, max(30, float(data["Risk_Score"].max())), "", status_color(cow["Status"])),
        ("Confidence", float(cow["Confidence"]) * 100, 0, 100, "%", "#17803d"),
    ]
    for column, chart in zip(chart_columns, charts):
        with column:
            st.plotly_chart(gauge(*chart), width="stretch", config={"displayModeBar": False})

    st.markdown("<style>.kpi-card{background:#fff;border:1px solid #dbe6ee;border-top:4px solid;border-radius:12px;padding:18px;box-shadow:0 4px 16px rgba(23,50,77,.07)}.kpi-label{color:#526579;font-size:.85rem}.kpi-value{color:#17324d;font-size:1.65rem;font-weight:700;margin-top:5px}.status-pill{color:#fff;border-radius:999px;padding:4px 10px;font-size:.75rem;margin-left:8px}</style>", unsafe_allow_html=True)

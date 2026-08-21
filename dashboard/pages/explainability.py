from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_PATH = PROJECT_ROOT / "generated" / "final_prediction.csv"
SHAP_PATH = PROJECT_ROOT / "generated" / "shap_explanations.csv"


@st.cache_data(ttl=60)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(PREDICTION_PATH), pd.read_csv(SHAP_PATH)


def status_key(value: str) -> str:
    value = str(value).strip().lower().replace("_", " ")
    return "At Risk" if value in {"observation", "at risk", "at-risk", "atrisk"} else ("Diseased" if value == "diseased" else "Healthy")


def explanation_row(shap: pd.DataFrame, cow: pd.Series) -> pd.Series:
    matching = shap[shap["Predicted_Status"].map(status_key) == status_key(cow["Final_Status"])]
    if matching.empty:
        matching = shap
    return matching.iloc[int(cow.name) % len(matching)]


def make_contribution_frame(row: pd.Series) -> pd.DataFrame:
    values = []
    for feature, field in [(row.get("Dominant_Factor", "Dominant factor"), "Dominant_Contribution"), (row.get("Secondary_Factor", "Secondary factor"), "Secondary_Contribution")]:
        if pd.notna(feature) and pd.notna(row.get(field)):
            values.append({"Feature": str(feature), "Contribution": float(row[field])})
    return pd.DataFrame(values).drop_duplicates("Feature")


def contribution_chart(frame: pd.DataFrame, title: str):
    if frame.empty:
        st.info("No feature contributions are recorded for this explanation.")
        return
    figure = px.bar(frame.sort_values("Contribution"), x="Contribution", y="Feature", orientation="h", color="Contribution", color_continuous_scale=["#c2413b", "#dbeef6", "#17803d"], title=title)
    figure.update_layout(template="plotly_white", height=230, margin={"l": 10, "r": 15, "t": 45, "b": 15}, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def show_xai():
    st.markdown("# Explainable AI")
    st.caption("Feature-level context behind the model's recorded prediction.")
    try:
        predictions, shap = load_data()
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error(f"Explainability data is not available: {error}")
        return
    if predictions.empty or shap.empty:
        st.warning("No prediction explanations are available yet.")
        return
    predictions["Status"] = predictions["Final_Status"].map(status_key)
    kpi_columns = st.columns(3)
    kpi_columns[0].metric("Average confidence", f"{predictions['Confidence'].mean() * 100:.1f}%", border=True)
    kpi_columns[1].metric("Average risk", f"{predictions['Risk_Score'].mean():.1f}", border=True)
    kpi_columns[2].metric("Total explanations", f"{len(shap):,}", border=True)
    selected_id = st.selectbox("Select cow", predictions["Cow_ID"].tolist(), key="xai_cow")
    cow = predictions.loc[predictions["Cow_ID"] == selected_id].iloc[0]
    record = explanation_row(shap, cow)
    details = st.columns(4)
    details[0].metric("Prediction", str(cow["Final_Status"]))
    details[1].metric("Confidence", f"{cow['Confidence'] * 100:.1f}%")
    details[2].metric("Risk score", f"{cow['Risk_Score']:.0f}")
    details[3].metric("Dominant factor", str(record.get("Dominant_Factor", "Unavailable")))
    with st.container(border=True):
        st.markdown("**Model explanation**")
        st.write(record.get("XAI_Explanation", "No narrative explanation is recorded."))
        st.caption("This explanation is selected from the recorded SHAP explanation rows matching the cow's predicted status; the source file does not contain Cow_ID.")
    contributions = make_contribution_frame(record)
    positive = contributions[contributions["Contribution"] >= 0]
    negative = contributions[contributions["Contribution"] < 0]
    first, second = st.columns(2)
    with first:
        with st.container(border=True):
            contribution_chart(positive, "Top positive features")
    with second:
        with st.container(border=True):
            contribution_chart(negative, "Top negative features")
    with st.expander("Recorded explanation details", icon=":material/insights:"):
        st.dataframe(pd.DataFrame({"Field": ["Predicted status", "Confidence", "Dominant factor", "Secondary factor", "Dominant direction", "Dominant contribution", "Secondary contribution"], "Value": [record.get("Predicted_Status"), record.get("Confidence_Percentage"), record.get("Dominant_Factor"), record.get("Secondary_Factor"), record.get("Dominant_Direction"), record.get("Dominant_Contribution"), record.get("Secondary_Contribution")] }), hide_index=True, width="stretch")

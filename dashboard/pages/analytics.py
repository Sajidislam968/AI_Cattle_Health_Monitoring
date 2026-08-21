from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "generated" / "final_prediction.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
X_TEST_PATH = PROJECT_ROOT / "processed" / "X_test.csv"
Y_TEST_PATH = PROJECT_ROOT / "processed" / "y_test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
STATUS_COLORS = {"Healthy": "#17803d", "At Risk": "#d97706", "Diseased": "#c2413b"}
ROC_LABELS = {0: "Healthy", 1: "At Risk", 2: "Diseased"}


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(ttl=3600)
def load_xgboost_roc() -> tuple[pd.DataFrame, pd.DataFrame]:
    model = joblib.load(MODEL_PATH)
    x_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0].to_numpy()
    probabilities = model.predict_proba(x_test)
    classes = [int(value) for value in model.classes_]
    binarized_labels = cast(Any, label_binarize(y_test, classes=classes))
    encoded_labels = binarized_labels.toarray() if hasattr(binarized_labels, "toarray") else binarized_labels
    curve_rows = []
    auc_rows = []
    for index, class_id in enumerate(classes):
        false_positive_rate, true_positive_rate, _ = roc_curve(encoded_labels[:, index], probabilities[:, index])
        class_auc = auc(false_positive_rate, true_positive_rate)
        curve_rows.extend({
            "False positive rate": false_positive_rate_value,
            "True positive rate": true_positive_rate_value,
            "Class": ROC_LABELS.get(class_id, f"Class {class_id}"),
        } for false_positive_rate_value, true_positive_rate_value in zip(false_positive_rate, true_positive_rate))
        auc_rows.append({"Class": ROC_LABELS.get(class_id, f"Class {class_id}"), "AUC": class_auc})
    curve_data = pd.DataFrame(curve_rows)
    auc_data = pd.DataFrame(auc_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    curve_data.to_csv(RESULTS_DIR / "xgboost_roc_curve.csv", index=False)
    auc_data.to_csv(RESULTS_DIR / "xgboost_roc_auc.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    for class_name, class_data in curve_data.groupby("Class"):
        axis.plot(class_data["False positive rate"], class_data["True positive rate"], label=f"{class_name} (AUC = {auc_data.loc[auc_data['Class'] == class_name, 'AUC'].iloc[0]:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="#9aaabd", label="Random classifier")
    axis.set_title("XGBoost Multiclass ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "xgboost_roc_curve.png", dpi=300)
    plt.close(figure)
    return curve_data, auc_data


def normalize_status(value: str) -> str:
    value = str(value).strip().lower().replace("_", " ")
    return "At Risk" if value in {"observation", "at risk", "at-risk", "atrisk"} else ("Diseased" if value == "diseased" else "Healthy")


def base_layout(figure, height=350):
    figure.update_layout(template="plotly_white", height=height, margin={"l": 20, "r": 20, "t": 45, "b": 20}, font={"color": "#17324d"}, legend={"orientation": "h", "y": -0.2})
    return figure


def chart_card(title, figure):
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        st.plotly_chart(base_layout(figure), width="stretch", config={"displayModeBar": False})


def show_analytics():
    st.markdown("# Analytics")
    st.caption("Aggregated view of behavior, physiology, environment, and model confidence.")
    try:
        data = load_predictions().copy()
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        st.error(f"Prediction data is not available: {error}")
        return
    if data.empty:
        st.warning("No analytics data is available yet.")
        return
    data["Status"] = data["Final_Status"].map(normalize_status)
    status = data["Status"].value_counts().rename_axis("Status").reset_index(name="Cows")
    behaviors = data["Dominant_Behavior"].astype(str).str.title().value_counts().rename_axis("Behavior").reset_index(name="Cows")
    behavior_columns = {"Standing_%": "Standing", "Walking_%": "Walking", "Eating_%": "Eating", "Lying_%": "Lying", "Rumination_%": "Rumination"}
    behavior_avg = pd.DataFrame({"Behavior": list(behavior_columns.values()), "Average": [data[column].mean() for column in behavior_columns]})

    first, second = st.columns(2)
    with first:
        chart_card("Health status", px.pie(status, names="Status", values="Cows", hole=.48, color="Status", color_discrete_map=STATUS_COLORS))
    with second:
        chart_card("Dominant behavior", px.pie(behaviors, names="Behavior", values="Cows", hole=.48, color="Behavior", color_discrete_sequence=["#2266a8", "#36a269", "#d97706", "#5b7c99", "#9aaabd"]))
    first, second = st.columns(2)
    with first:
        chart_card("Average behavior percentages", px.bar(behavior_avg, x="Behavior", y="Average", color="Average", color_continuous_scale=["#dbeef6", "#2266a8"], labels={"Average": "Average (%)"}))
    with second:
        chart_card("Risk score distribution", px.histogram(data, x="Risk_Score", nbins=12, color="Status", color_discrete_map=STATUS_COLORS, labels={"Risk_Score": "Risk score"}))
    first, second = st.columns(2)
    with first:
        chart_card("Confidence distribution", px.histogram(data, x=data["Confidence"] * 100, nbins=10, labels={"x": "Confidence (%)"}, color_discrete_sequence=["#17803d"]))
    with second:
        chart_card("Heart rate distribution", px.histogram(data, x="Heart_Rate", nbins=12, labels={"Heart_Rate": "Heart rate (bpm)"}, color_discrete_sequence=["#2266a8"]))
    first, second = st.columns(2)
    with first:
        chart_card("Body temperature distribution", px.histogram(data, x="Body_Temperature", nbins=12, labels={"Body_Temperature": "Temperature (C)"}, color_discrete_sequence=["#d97706"]))
    with second:
        chart_card("THI distribution", px.histogram(data, x="THI", nbins=12, labels={"THI": "THI"}, color_discrete_sequence=["#5b7c99"]))
    numeric = data.select_dtypes(include="number").drop(columns=[column for column in ["standing", "walking", "eating", "lying", "rumination", "Unknown", "Total_Frames"] if column in data], errors="ignore")
    with st.container(border=True):
        st.markdown("**Correlation heatmap**")
        chart_card("", px.imshow(numeric.corr(numeric_only=True), text_auto=True, color_continuous_scale="RdBu_r", aspect="auto"))
    try:
        roc_data, auc_data = load_xgboost_roc()
    except (FileNotFoundError, ValueError, ImportError) as error:
        st.warning(f"XGBoost ROC data is not available: {error}")
    else:
        st.markdown("### XGBoost ROC curve")
        st.caption("One-vs-rest ROC curves calculated from the saved XGBoost model and held-out test set.")
        auc_columns = st.columns(len(auc_data))
        for column, (_, auc_row) in zip(auc_columns, auc_data.iterrows()):
            with column:
                st.metric(f"{auc_row['Class']} AUC", f"{auc_row['AUC']:.3f}", border=True)
        roc_figure = px.line(roc_data, x="False positive rate", y="True positive rate", color="Class", color_discrete_map=STATUS_COLORS, hover_data={"False positive rate": ":.3f", "True positive rate": ":.3f"})
        roc_figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"color": "#9aaabd", "dash": "dash"})
        chart_card("", roc_figure)
    first, second = st.columns(2)
    with first:
        chart_card("Risk score vs confidence", px.scatter(data, x=data["Confidence"] * 100, y="Risk_Score", color="Status", color_discrete_map=STATUS_COLORS, hover_name="Cow_ID", labels={"x": "Confidence (%)", "Risk_Score": "Risk score"}))
    with second:
        chart_card("Temperature vs heart rate", px.scatter(data, x="Body_Temperature", y="Heart_Rate", color="Status", color_discrete_map=STATUS_COLORS, hover_name="Cow_ID", labels={"Body_Temperature": "Temperature (C)", "Heart_Rate": "Heart rate (bpm)"}))
    st.download_button(":material/download: Download prediction CSV", data=load_predictions().to_csv(index=False), file_name="final_prediction.csv", mime="text/csv", width="content")
